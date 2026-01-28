# pyright: reportGeneralTypeIssues=false
"""
Vector Store for Research Knowledge Base

Manages document embeddings and retrieval using ChromaDB.
Supports multiple collections:
- academic_papers: Peer-reviewed literature
- research_notes: User's own notes and annotations
- web_sources: Grey literature, reports, news
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, cast

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class ResearchVectorStore:
    """
    Vector database for the research knowledge base.

    Example:
        store = ResearchVectorStore("./data/chroma_db")

        # Add a paper
        store.add_paper(
            paper_id="abc123",
            chunks=["chunk1...", "chunk2..."],
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            metadata={"title": "...", "year": 2023, "authors": ["..."]}
        )

        # Search
        results = store.search(query_embedding, collection="papers", n_results=5)
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        embedding_function: Optional[Any] = None,
        reranker: Optional[Any] = None,
        rerank_top_k: Optional[int] = None,
    ):
        """
        Initialize the vector store.

        Args:
            persist_dir: Directory to persist the database
            embedding_function: Optional ChromaDB embedding function
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Optional reranker (e.g., BGE cross-encoder)
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Initialize collections with cosine similarity
        self.papers: Any = cast(
            Any,
            self.client.get_or_create_collection(
                name="academic_papers", metadata={"hnsw:space": "cosine"}
            ),
        )

        self.notes: Any = cast(
            Any,
            self.client.get_or_create_collection(
                name="research_notes", metadata={"hnsw:space": "cosine"}
            ),
        )

        self.web_sources: Any = cast(
            Any,
            self.client.get_or_create_collection(
                name="web_sources", metadata={"hnsw:space": "cosine"}
            ),
        )

        logger.info(f"Initialized vector store at {self.persist_dir}")

    def _get_collection(self, collection: str):
        """Get collection by name."""
        collections = {
            "papers": self.papers,
            "notes": self.notes,
            "web_sources": self.web_sources,
        }
        if collection not in collections:
            raise ValueError(
                f"Unknown collection: {collection}. Use: {list(collections.keys())}"
            )
        return collections[collection]

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata for ChromaDB.

        ChromaDB only supports str, int, float, bool values.
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                sanitized[key] = ", ".join(str(v) for v in value)
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        return sanitized

    def add_paper(
        self,
        paper_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Add paper chunks to the knowledge base.

        Args:
            paper_id: Unique identifier for the paper
            chunks: Text chunks from the paper
            embeddings: Embedding vectors for each chunk
            metadata: Paper metadata (title, year, authors, etc.)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if not chunks:
            logger.warning(f"No chunks to add for paper {paper_id}")
            return

        # Add timestamp - use timezone-aware datetime
        base_metadata = {
            **metadata,
            "paper_id": paper_id,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }

        # Sanitize metadata
        base_metadata = self._sanitize_metadata(base_metadata)

        # Create IDs and metadata for each chunk
        ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{**base_metadata, "chunk_index": i} for i in range(len(chunks))]
        embeddings_any: Any = embeddings
        metadatas_any: Any = metadatas

        # Add to collection
        self.papers.add(
            ids=ids,
            embeddings=embeddings_any,
            documents=chunks,
            metadatas=metadatas_any,
        )  # type: ignore[arg-type]

        logger.info(f"Added paper {paper_id} with {len(chunks)} chunks")

    def add_note(
        self,
        note_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Add a research note to the knowledge base.

        Args:
            note_id: Unique identifier for the note
            content: Note text content
            embedding: Embedding vector
            metadata: Note metadata (title, tags, etc.)
        """
        # Add timestamp
        note_metadata = {
            **metadata,
            "note_id": note_id,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }

        # Sanitize metadata
        note_metadata = self._sanitize_metadata(note_metadata)

        self.notes.add(
            ids=[note_id],
            embeddings=cast(Any, [embedding]),
            documents=[content],
            metadatas=cast(Any, [note_metadata]),
        )  # type: ignore[arg-type]

        logger.info(f"Added note {note_id}")

    def add_web_source(
        self,
        source_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Add web content to the knowledge base.

        Args:
            source_id: Unique identifier for the web source
            chunks: Text chunks from the page
            embeddings: Embedding vectors for each chunk
            metadata: Source metadata (url, title, etc.)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if not chunks:
            logger.warning(f"No chunks to add for web source {source_id}")
            return

        # Add timestamp
        base_metadata = {
            **metadata,
            "source_id": source_id,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }

        # Sanitize metadata
        base_metadata = self._sanitize_metadata(base_metadata)

        # Create IDs and metadata for each chunk
        ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{**base_metadata, "chunk_index": i} for i in range(len(chunks))]
        embeddings_any: Any = embeddings
        metadatas_any: Any = metadatas

        self.web_sources.add(
            ids=ids,
            embeddings=embeddings_any,
            documents=chunks,
            metadatas=metadatas_any,
        )  # type: ignore[arg-type]

        logger.info(f"Added web source {source_id} with {len(chunks)} chunks")

    def add_document(
        self,
        collection: str,
        document_id: str,
        content: str,
        embedding: Optional[List[float]],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Add a single document to a collection.

        Args:
            collection: "papers", "notes", or "web_sources"
            document_id: Unique document ID
            content: Document text
            embedding: Embedding vector (required when no embedding function)
            metadata: Document metadata
        """
        if embedding is None:
            raise ValueError("Embedding required to add document")

        coll = self._get_collection(collection)
        sanitized = self._sanitize_metadata(metadata)
        coll.add(
            ids=[document_id],
            embeddings=cast(Any, [embedding]),
            documents=[content],
            metadatas=cast(Any, [sanitized]),
        )  # type: ignore[arg-type]

    def search(
        self,
        query_embedding: List[float],
        collection: str = "papers",
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
        query_text: Optional[str] = None,
        reranker: Optional[Any] = None,
        rerank_top_k: Optional[int] = None,
    ) -> Dict:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            collection: "papers", "notes", or "web_sources"
            n_results: Maximum results to return
            filter_dict: Metadata filters (e.g., {"year": {"$gte": 2020}})

        Returns:
            Dict with ids, documents, metadatas, distances
        """
        coll = self._get_collection(collection)

        results = cast(
            Dict[str, Any],
            coll.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_dict,
                include=["documents", "metadatas", "distances"],
            ),
        )

        # Flatten results (query returns nested lists)
        flat = {
            "ids": (results.get("ids") or [[]])[0],
            "documents": (results.get("documents") or [[]])[0],
            "metadatas": (results.get("metadatas") or [[]])[0],
            "distances": (results.get("distances") or [[]])[0],
        }

        effective_reranker = reranker or self.reranker
        effective_top_k = rerank_top_k or self.rerank_top_k

        if effective_reranker and query_text and flat["documents"]:
            reranked = effective_reranker.rerank(
                query_text, flat["documents"], top_k=effective_top_k
            )
            indices = [item.index for item in reranked]
            flat = {
                "ids": [flat["ids"][i] for i in indices],
                "documents": [flat["documents"][i] for i in indices],
                "metadatas": [flat["metadatas"][i] for i in indices],
                "distances": [flat["distances"][i] for i in indices],
                "rerank_scores": [item.score for item in reranked],
            }

        return flat

    def search_by_metadata(
        self,
        collection: str,
        filter_dict: Dict[str, Any],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Fetch documents by metadata filter.

        Args:
            collection: "papers", "notes", or "web_sources"
            filter_dict: Metadata filter dict
            limit: Maximum results to return

        Returns:
            List of dicts with id, document, metadata
        """
        coll = self._get_collection(collection)
        results = cast(
            Dict[str, Any],
            coll.get(
                where=filter_dict, limit=limit, include=["documents", "metadatas"]
            ),
        )
        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        output = []
        for i, doc_id in enumerate(ids):
            output.append(
                {
                    "id": doc_id,
                    "document": documents[i] if i < len(documents) else None,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
            )
        return output

    def search_all(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_dict: Optional[Dict] = None,
    ) -> Dict[str, Dict]:
        """
        Search across all collections.

        Args:
            query_embedding: Query vector
            n_results: Maximum results per collection
            filter_dict: Metadata filters

        Returns:
            Dict mapping collection name to search results
        """
        return {
            "papers": self.search(query_embedding, "papers", n_results, filter_dict),
            "notes": self.search(query_embedding, "notes", n_results, filter_dict),
            "web_sources": self.search(
                query_embedding, "web_sources", n_results, filter_dict
            ),
        }

    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """
        Get all chunks and metadata for a paper.

        Args:
            paper_id: Paper ID

        Returns:
            Dict with chunks and metadata, or None if not found
        """
        results = cast(
            Dict[str, Any],
            self.papers.get(
                where={"paper_id": paper_id}, include=["documents", "metadatas"]
            ),
        )
        result_ids = results.get("ids") or []
        documents = cast(List[str], results.get("documents") or [])
        metadatas = cast(List[Dict[str, Any]], results.get("metadatas") or [])

        if not result_ids:
            return None

        # Sort chunks by index
        chunks_with_meta = list(zip(documents, metadatas))
        chunks_with_meta.sort(key=lambda x: x[1].get("chunk_index", 0))

        return {
            "paper_id": paper_id,
            "chunks": [c[0] for c in chunks_with_meta],
            "metadata": chunks_with_meta[0][1] if chunks_with_meta else {},
            "num_chunks": len(chunks_with_meta),
        }

    def delete_paper(self, paper_id: str) -> bool:
        """
        Remove a paper from the knowledge base.

        Args:
            paper_id: Paper ID to delete

        Returns:
            True if paper was deleted, False if not found
        """
        # Get all chunk IDs for this paper
        results = cast(Dict[str, Any], self.papers.get(where={"paper_id": paper_id}))
        result_ids = results.get("ids") or []

        if not result_ids:
            return False

        self.papers.delete(ids=result_ids)
        logger.info(f"Deleted paper {paper_id} ({len(result_ids)} chunks)")
        return True

    def delete_note(self, note_id: str) -> bool:
        """Remove a note from the knowledge base."""
        try:
            self.notes.delete(ids=[note_id])
            logger.info(f"Deleted note {note_id}")
            return True
        except Exception:
            return False

    def delete_web_source(self, source_id: str) -> bool:
        """Remove a web source from the knowledge base."""
        results = cast(
            Dict[str, Any], self.web_sources.get(where={"source_id": source_id})
        )
        result_ids = results.get("ids") or []

        if not result_ids:
            return False

        self.web_sources.delete(ids=result_ids)
        logger.info(f"Deleted web source {source_id}")
        return True

    def get_stats(self) -> Dict[str, int]:
        """Get knowledge base statistics."""
        # Count unique papers/sources by looking at metadata
        paper_chunks = self.papers.count()
        note_count = self.notes.count()
        web_chunks = self.web_sources.count()

        # Estimate unique papers (get sample to count unique paper_ids)
        unique_papers = 0
        if paper_chunks > 0:
            all_papers = cast(Dict[str, Any], self.papers.get(include=["metadatas"]))
            all_metadatas = all_papers.get("metadatas") or []
            unique_papers = len(set(m.get("paper_id", "") for m in all_metadatas))

        unique_web = 0
        if web_chunks > 0:
            all_web = cast(Dict[str, Any], self.web_sources.get(include=["metadatas"]))
            all_web_metadatas = all_web.get("metadatas") or []
            unique_web = len(set(m.get("source_id", "") for m in all_web_metadatas))

        return {
            "total_papers": unique_papers,
            "total_paper_chunks": paper_chunks,
            "total_notes": note_count,
            "total_web_sources": unique_web,
            "total_web_chunks": web_chunks,
            "total_chunks": paper_chunks + note_count + web_chunks,
        }

    def list_papers(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        List papers in the knowledge base.

        Args:
            limit: Maximum papers to return
            offset: Number of papers to skip

        Returns:
            List of paper metadata dicts
        """
        # Get all paper metadata
        all_results = cast(Dict[str, Any], self.papers.get(include=["metadatas"]))
        all_metadatas = all_results.get("metadatas") or []

        if not all_metadatas:
            return []

        # Get unique papers
        seen_ids = set()
        papers = []

        for meta in all_metadatas:
            paper_id = meta.get("paper_id", "")
            if paper_id and paper_id not in seen_ids:
                seen_ids.add(paper_id)
                papers.append(
                    {
                        "paper_id": paper_id,
                        "title": meta.get("title", "Unknown"),
                        "year": meta.get("year"),
                        "authors": meta.get("authors", ""),
                        "added_at": meta.get("added_at", ""),
                    }
                )

        # Sort by added_at (newest first)
        papers.sort(key=lambda x: x.get("added_at", ""), reverse=True)

        # Apply pagination
        return papers[offset : offset + limit]

    def list_papers_detailed(self, limit: int = 1000, offset: int = 0) -> List[Dict]:
        """
        List papers with extended metadata.

        Args:
            limit: Maximum papers to return
            offset: Number of papers to skip

        Returns:
            List of paper metadata dicts
        """
        all_results = cast(Dict[str, Any], self.papers.get(include=["metadatas"]))
        all_metadatas = all_results.get("metadatas") or []

        if not all_metadatas:
            return []

        seen_ids = set()
        papers = []

        for meta in all_metadatas:
            paper_id = meta.get("paper_id", "")
            if paper_id and paper_id not in seen_ids:
                seen_ids.add(paper_id)
                papers.append(
                    {
                        "paper_id": paper_id,
                        "title": meta.get("title", "Unknown"),
                        "year": meta.get("year"),
                        "authors": meta.get("authors", ""),
                        "added_at": meta.get("added_at", ""),
                        "citations": meta.get("citations")
                        or meta.get("citation_count"),
                        "venue": meta.get("venue", ""),
                        "fields": meta.get("fields", ""),
                        "source": meta.get("source", ""),
                        "researcher": meta.get("researcher", ""),
                        "ingest_source": meta.get("ingest_source", ""),
                        "doi": meta.get("doi", ""),
                    }
                )

        papers.sort(key=lambda x: x.get("added_at", ""), reverse=True)
        return papers[offset : offset + limit]

    def list_notes(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """List notes in the knowledge base."""
        results = cast(
            Dict[str, Any], self.notes.get(include=["metadatas", "documents"])
        )
        result_metadatas = cast(List[Dict[str, Any]], results.get("metadatas") or [])
        result_documents = cast(List[str], results.get("documents") or [])
        result_ids = cast(List[str], results.get("ids") or [])

        if not result_metadatas:
            return []

        notes = []
        for i, meta in enumerate(result_metadatas):
            notes.append(
                {
                    "note_id": meta.get("note_id", result_ids[i]),
                    "title": meta.get("title", "Untitled"),
                    "preview": result_documents[i][:100] + "..."
                    if len(result_documents[i]) > 100
                    else result_documents[i],
                    "added_at": meta.get("added_at", ""),
                    "tags": meta.get("tags", ""),
                }
            )

        notes.sort(key=lambda x: x.get("added_at", ""), reverse=True)
        return notes[offset : offset + limit]

    def clear_collection(self, collection: str) -> None:
        """Clear all documents from a collection."""
        coll = self._get_collection(collection)
        # Get all IDs and delete
        all_ids = cast(Dict[str, Any], coll.get()).get("ids") or []
        if all_ids:
            coll.delete(ids=all_ids)
            logger.info(f"Cleared {len(all_ids)} documents from {collection}")

    def reset(self) -> None:
        """Reset the entire database (delete all data)."""
        self.client.reset()
        # Recreate collections
        self.papers = self.client.get_or_create_collection(
            name="academic_papers", metadata={"hnsw:space": "cosine"}
        )
        self.notes = self.client.get_or_create_collection(
            name="research_notes", metadata={"hnsw:space": "cosine"}
        )
        self.web_sources = self.client.get_or_create_collection(
            name="web_sources", metadata={"hnsw:space": "cosine"}
        )
        logger.info("Reset vector store")
