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
from typing import List, Dict, Optional, Any, Union

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
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize the vector store.

        Args:
            persist_dir: Directory to persist the database
            embedding_function: Optional ChromaDB embedding function
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize collections with cosine similarity
        self.papers = self.client.get_or_create_collection(
            name="academic_papers",
            metadata={"hnsw:space": "cosine"}
        )

        self.notes = self.client.get_or_create_collection(
            name="research_notes",
            metadata={"hnsw:space": "cosine"}
        )

        self.web_sources = self.client.get_or_create_collection(
            name="web_sources",
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Initialized vector store at {self.persist_dir}")

    def _get_collection(self, collection: str):
        """Get collection by name."""
        collections = {
            "papers": self.papers,
            "notes": self.notes,
            "web_sources": self.web_sources
        }
        if collection not in collections:
            raise ValueError(f"Unknown collection: {collection}. Use: {list(collections.keys())}")
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
        metadata: Dict[str, Any]
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
            "added_at": datetime.now(timezone.utc).isoformat()
        }

        # Sanitize metadata
        base_metadata = self._sanitize_metadata(base_metadata)

        # Create IDs and metadata for each chunk
        ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {**base_metadata, "chunk_index": i}
            for i in range(len(chunks))
        ]

        # Add to collection
        self.papers.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        logger.info(f"Added paper {paper_id} with {len(chunks)} chunks")

    def add_note(
        self,
        note_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
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
            "added_at": datetime.now(timezone.utc).isoformat()
        }

        # Sanitize metadata
        note_metadata = self._sanitize_metadata(note_metadata)

        self.notes.add(
            ids=[note_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[note_metadata]
        )

        logger.info(f"Added note {note_id}")

    def add_web_source(
        self,
        source_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, Any]
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
            "added_at": datetime.now(timezone.utc).isoformat()
        }

        # Sanitize metadata
        base_metadata = self._sanitize_metadata(base_metadata)

        # Create IDs and metadata for each chunk
        ids = [f"{source_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {**base_metadata, "chunk_index": i}
            for i in range(len(chunks))
        ]

        self.web_sources.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        logger.info(f"Added web source {source_id} with {len(chunks)} chunks")

    def search(
        self,
        query_embedding: List[float],
        collection: str = "papers",
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
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

        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict,
            include=["documents", "metadatas", "distances"]
        )

        # Flatten results (query returns nested lists)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def search_all(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
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
            "web_sources": self.search(query_embedding, "web_sources", n_results, filter_dict)
        }

    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """
        Get all chunks and metadata for a paper.

        Args:
            paper_id: Paper ID

        Returns:
            Dict with chunks and metadata, or None if not found
        """
        results = self.papers.get(
            where={"paper_id": paper_id},
            include=["documents", "metadatas"]
        )

        if not results["ids"]:
            return None

        # Sort chunks by index
        chunks_with_meta = list(zip(
            results["documents"],
            results["metadatas"]
        ))
        chunks_with_meta.sort(key=lambda x: x[1].get("chunk_index", 0))

        return {
            "paper_id": paper_id,
            "chunks": [c[0] for c in chunks_with_meta],
            "metadata": chunks_with_meta[0][1] if chunks_with_meta else {},
            "num_chunks": len(chunks_with_meta)
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
        results = self.papers.get(
            where={"paper_id": paper_id}
        )

        if not results["ids"]:
            return False

        self.papers.delete(ids=results["ids"])
        logger.info(f"Deleted paper {paper_id} ({len(results['ids'])} chunks)")
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
        results = self.web_sources.get(
            where={"source_id": source_id}
        )

        if not results["ids"]:
            return False

        self.web_sources.delete(ids=results["ids"])
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
            all_papers = self.papers.get(include=["metadatas"])
            unique_papers = len(set(
                m.get("paper_id", "") for m in all_papers["metadatas"]
            ))

        unique_web = 0
        if web_chunks > 0:
            all_web = self.web_sources.get(include=["metadatas"])
            unique_web = len(set(
                m.get("source_id", "") for m in all_web["metadatas"]
            ))

        return {
            "total_papers": unique_papers,
            "total_paper_chunks": paper_chunks,
            "total_notes": note_count,
            "total_web_sources": unique_web,
            "total_web_chunks": web_chunks,
            "total_chunks": paper_chunks + note_count + web_chunks
        }

    def list_papers(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        List papers in the knowledge base.

        Args:
            limit: Maximum papers to return
            offset: Number of papers to skip

        Returns:
            List of paper metadata dicts
        """
        # Get all paper metadata
        all_results = self.papers.get(include=["metadatas"])

        if not all_results["metadatas"]:
            return []

        # Get unique papers
        seen_ids = set()
        papers = []

        for meta in all_results["metadatas"]:
            paper_id = meta.get("paper_id", "")
            if paper_id and paper_id not in seen_ids:
                seen_ids.add(paper_id)
                papers.append({
                    "paper_id": paper_id,
                    "title": meta.get("title", "Unknown"),
                    "year": meta.get("year"),
                    "authors": meta.get("authors", ""),
                    "added_at": meta.get("added_at", "")
                })

        # Sort by added_at (newest first)
        papers.sort(key=lambda x: x.get("added_at", ""), reverse=True)

        # Apply pagination
        return papers[offset:offset + limit]

    def list_notes(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """List notes in the knowledge base."""
        results = self.notes.get(include=["metadatas", "documents"])

        if not results["metadatas"]:
            return []

        notes = []
        for i, meta in enumerate(results["metadatas"]):
            notes.append({
                "note_id": meta.get("note_id", results["ids"][i]),
                "title": meta.get("title", "Untitled"),
                "preview": results["documents"][i][:100] + "..." if len(results["documents"][i]) > 100 else results["documents"][i],
                "added_at": meta.get("added_at", ""),
                "tags": meta.get("tags", "")
            })

        notes.sort(key=lambda x: x.get("added_at", ""), reverse=True)
        return notes[offset:offset + limit]

    def clear_collection(self, collection: str) -> None:
        """Clear all documents from a collection."""
        coll = self._get_collection(collection)
        # Get all IDs and delete
        all_ids = coll.get()["ids"]
        if all_ids:
            coll.delete(ids=all_ids)
            logger.info(f"Cleared {len(all_ids)} documents from {collection}")

    def reset(self) -> None:
        """Reset the entire database (delete all data)."""
        self.client.reset()
        # Recreate collections
        self.papers = self.client.get_or_create_collection(
            name="academic_papers",
            metadata={"hnsw:space": "cosine"}
        )
        self.notes = self.client.get_or_create_collection(
            name="research_notes",
            metadata={"hnsw:space": "cosine"}
        )
        self.web_sources = self.client.get_or_create_collection(
            name="web_sources",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Reset vector store")
