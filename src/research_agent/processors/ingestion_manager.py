"""
Ingestion Manager

Manages autonomous knowledge base building:
- Evaluates source quality and relevance
- Handles user approval workflow
- Tracks what's been ingested
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SourceQuality(Enum):
    """Quality tiers for sources."""
    HIGH = "high"        # Peer-reviewed, highly cited
    MEDIUM = "medium"    # Working papers, reports from known orgs
    LOW = "low"          # Blog posts, news (still valuable for context)
    UNCERTAIN = "uncertain"


@dataclass
class IngestionCandidate:
    """A source being considered for ingestion."""
    source_id: str
    title: str
    source_type: str  # "paper", "report", "web"
    quality_score: SourceQuality
    relevance_score: float  # 0-1
    reasons: List[str]
    metadata: Dict


class IngestionManager:
    """
    Manages what gets added to the knowledge base.

    Supports two modes:
    - Manual: User approves each source
    - Auto: High-quality relevant sources added automatically

    Example:
        manager = IngestionManager(vector_store, llm, auto_ingest=False)

        # Evaluate a source
        candidate = await manager.evaluate_source(paper, user_query)

        # Get formatted offer for user
        offer = manager.format_ingestion_offer([candidate])

        # Process user's decision
        await manager.process_decision("1", candidate)
    """

    def __init__(
        self,
        vector_store,
        llm,
        embedder=None,
        auto_ingest: bool = False,
        auto_threshold: float = 0.85
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.embedder = embedder
        self.auto_ingest = auto_ingest
        self.auto_threshold = auto_threshold
        self.pending_review: List[IngestionCandidate] = []

    async def evaluate_source(
        self,
        source: Dict,
        user_query: str
    ) -> IngestionCandidate:
        """
        Evaluate whether a source should be ingested.

        Args:
            source: Source metadata from search
            user_query: User's research query (for relevance)

        Returns:
            IngestionCandidate with scores and reasons
        """
        # Check for duplicates first
        is_duplicate = await self.check_duplicate(source)
        if is_duplicate:
            return IngestionCandidate(
                source_id=source.get("paper_id", source.get("id", "")),
                title=source.get("title", "Unknown"),
                source_type=source.get("source", "unknown"),
                quality_score=SourceQuality.UNCERTAIN,
                relevance_score=0.0,
                reasons=["Already in knowledge base"],
                metadata=source,
            )

        # Assess quality
        quality = self._assess_quality(source)

        # Assess relevance
        relevance = await self._assess_relevance(source, user_query)

        # Build reasons
        reasons = []
        if quality == SourceQuality.HIGH:
            reasons.append("High-quality peer-reviewed source")
        elif quality == SourceQuality.MEDIUM:
            reasons.append("Academic source")

        citations = source.get("citation_count", source.get("citations", 0))
        if citations and citations > 100:
            reasons.append(f"Highly cited ({citations} citations)")
        elif citations and citations > 20:
            reasons.append(f"Well cited ({citations} citations)")

        if relevance > 0.8:
            reasons.append("Highly relevant to your query")
        elif relevance > 0.5:
            reasons.append("Relevant to your query")

        if not reasons:
            reasons.append("May be useful for context")

        candidate = IngestionCandidate(
            source_id=source.get("paper_id", source.get("id", "")),
            title=source.get("title", "Unknown"),
            source_type=source.get("source", "unknown"),
            quality_score=quality,
            relevance_score=relevance,
            reasons=reasons,
            metadata=source,
        )

        # Auto-ingest if enabled and meets threshold
        if self.auto_ingest and relevance >= self.auto_threshold and quality in [SourceQuality.HIGH, SourceQuality.MEDIUM]:
            await self._ingest_source(candidate)
        else:
            self.pending_review.append(candidate)

        return candidate

    def _assess_quality(self, source: Dict) -> SourceQuality:
        """Heuristic quality assessment."""
        source_type = source.get("source", "")
        citations = source.get("citation_count", source.get("citations", 0))

        if source_type in ["semantic_scholar", "openalex"]:
            if citations and citations > 50:
                return SourceQuality.HIGH
            return SourceQuality.MEDIUM
        elif source_type == "web":
            return SourceQuality.LOW

        return SourceQuality.UNCERTAIN

    async def _assess_relevance(
        self,
        source: Dict,
        user_query: str
    ) -> float:
        """Use LLM to assess relevance to user's research."""
        title = source.get("title", "")
        abstract = source.get("abstract", "")[:500] if source.get("abstract") else ""

        if not title and not abstract:
            return 0.3  # Low default for sources with no info

        prompt = f"""Rate the relevance of this academic paper to the research query.
Respond with ONLY a single decimal number between 0.0 and 1.0.

Research query: {user_query}

Paper title: {title}
Abstract: {abstract}

Relevance score (0.0-1.0):"""

        try:
            response = await self.llm.generate(prompt, max_tokens=10)
            # Extract the first number from response
            import re
            match = re.search(r"(\d+\.?\d*)", response.strip())
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp to 0-1
        except Exception:
            pass

        return 0.5  # Default if LLM fails

    async def check_duplicate(self, source: Dict) -> bool:
        """Check if source already exists in knowledge base."""
        # Check by DOI first (most reliable)
        doi = source.get("doi")
        if doi:
            existing = self.vector_store.search_by_metadata(
                collection="papers",
                filter_dict={"doi": doi},
                limit=1
            )
            if existing:
                return True

        # Check by paper_id
        paper_id = source.get("paper_id", source.get("id"))
        if paper_id:
            existing = self.vector_store.search_by_metadata(
                collection="papers",
                filter_dict={"paper_id": paper_id},
                limit=1
            )
            if existing:
                return True

        # Fallback: check by title similarity
        title = source.get("title", "")
        if title and len(title) > 10:
            if self.embedder is not None:
                try:
                    if hasattr(self.embedder, "embed_query"):
                        query_embedding = self.embedder.embed_query(title)
                    elif hasattr(self.embedder, "embed"):
                        query_embedding = self.embedder.embed(title)
                    else:
                        query_embedding = self.embedder.encode(title)

                    if hasattr(query_embedding, "tolist"):
                        query_embedding = query_embedding.tolist()

                    results = self.vector_store.search(
                        query_embedding=query_embedding,
                        collection="papers",
                        n_results=3
                    )
                    for meta in results.get("metadatas", []):
                        existing_title = meta.get("title", "") if meta else ""
                        if existing_title and self._title_similarity(title, existing_title) > 0.9:
                            return True
                except Exception:
                    pass
            else:
                try:
                    papers = self.vector_store.list_papers(limit=500)
                    for paper in papers:
                        existing_title = paper.get("title", "")
                        if existing_title and self._title_similarity(title, existing_title) > 0.9:
                            return True
                except Exception:
                    pass

        return False

    def _title_similarity(self, title1: str, title2: str) -> float:
        """Simple title similarity check."""
        t1 = title1.lower().strip()
        t2 = title2.lower().strip()

        # Exact match
        if t1 == t2:
            return 1.0

        # Check if one is substring of other
        if t1 in t2 or t2 in t1:
            return 0.95

        # Simple word overlap
        words1 = set(t1.split())
        words2 = set(t2.split())
        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def format_ingestion_offer(
        self,
        candidates: List[IngestionCandidate]
    ) -> str:
        """
        Format candidates for user review.

        Returns a message listing sources with quality indicators.
        """
        worthy = [c for c in candidates if c.relevance_score > 0.5]

        if not worthy:
            return "No new sources worth adding to your knowledge base."

        emoji_map = {
            SourceQuality.HIGH: "⭐",
            SourceQuality.MEDIUM: "📄",
            SourceQuality.LOW: "📰",
            SourceQuality.UNCERTAIN: "❓"
        }

        lines = ["**📥 Sources to consider adding:**\n"]

        for i, c in enumerate(worthy, 1):
            emoji = emoji_map.get(c.quality_score, "❓")
            lines.append(f"{i}. {emoji} **{c.title}**")
            lines.append(f"   - Relevance: {c.relevance_score:.0%}")
            lines.append(f"   - {', '.join(c.reasons)}\n")

        lines.append("Reply with numbers to add (e.g., '1, 3') or 'all' / 'none'")

        return "\n".join(lines)

    async def process_decision(
        self,
        decision: str,
        candidates: List[IngestionCandidate]
    ) -> Dict:
        """
        Process user's ingestion decision.

        Args:
            decision: "all", "none", or comma-separated numbers
            candidates: List of candidates offered

        Returns:
            Dict with ingested count and details
        """
        decision = decision.strip().lower()
        ingested = []
        skipped = []

        if decision == "none":
            skipped = candidates
        elif decision == "all":
            for candidate in candidates:
                if candidate.relevance_score > 0:  # Skip duplicates
                    result = await self._ingest_source(candidate)
                    if result:
                        ingested.append(candidate)
                    else:
                        skipped.append(candidate)
                else:
                    skipped.append(candidate)
        else:
            # Parse comma-separated numbers (e.g., "1, 3, 5")
            try:
                indices = [int(x.strip()) - 1 for x in decision.split(",")]
                for i, candidate in enumerate(candidates):
                    if i in indices and candidate.relevance_score > 0:
                        result = await self._ingest_source(candidate)
                        if result:
                            ingested.append(candidate)
                        else:
                            skipped.append(candidate)
                    else:
                        skipped.append(candidate)
            except ValueError:
                # Invalid input, skip all
                skipped = candidates

        # Clear pending review for processed candidates
        processed_ids = {c.source_id for c in candidates}
        self.pending_review = [
            c for c in self.pending_review
            if c.source_id not in processed_ids
        ]

        return {
            "ingested_count": len(ingested),
            "skipped_count": len(skipped),
            "ingested": [c.title for c in ingested],
            "skipped": [c.title for c in skipped],
        }

    async def _ingest_source(self, candidate: IngestionCandidate) -> bool:
        """
        Ingest a source into the knowledge base.

        Args:
            candidate: The candidate to ingest

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.embedder is None:
                logger.warning("Embedder not configured; cannot ingest source")
                return False

            metadata = candidate.metadata.copy()
            metadata["ingested_at"] = self._get_timestamp()
            metadata["quality_score"] = candidate.quality_score.value
            metadata["relevance_score"] = candidate.relevance_score

            content = self._build_content(candidate)
            chunks = [content]
            embeddings = self._embed_texts(chunks)
            if not embeddings:
                return False

            # Add to vector store
            if candidate.source_type == "web":
                self.vector_store.add_web_source(
                    source_id=candidate.source_id,
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata=metadata,
                )
            else:
                self.vector_store.add_paper(
                    paper_id=candidate.source_id,
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata=metadata,
                )
            return True
        except Exception:
            return False

    def _embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for ingestion."""
        if not self.embedder or not texts:
            return None

        if hasattr(self.embedder, "embed_documents"):
            return self.embedder.embed_documents(texts, batch_size=16, show_progress=False)
        if hasattr(self.embedder, "embed_batch"):
            return self.embedder.embed_batch(texts, batch_size=16, show_progress=False)
        if hasattr(self.embedder, "embed"):
            embeddings = [self.embedder.embed(text) for text in texts]
            return [e.tolist() if hasattr(e, "tolist") else e for e in embeddings]
        if hasattr(self.embedder, "encode"):
            embeddings = [self.embedder.encode(text) for text in texts]
            return [e.tolist() if hasattr(e, "tolist") else e for e in embeddings]

        return None

    def _build_content(self, candidate: IngestionCandidate) -> str:
        """Build searchable content from candidate metadata."""
        parts = []
        meta = candidate.metadata

        if meta.get("title"):
            parts.append(f"Title: {meta['title']}")
        if meta.get("abstract"):
            parts.append(f"Abstract: {meta['abstract']}")
        if meta.get("authors"):
            authors = meta["authors"]
            if isinstance(authors, list):
                authors = ", ".join(authors)
            parts.append(f"Authors: {authors}")

        return "\n\n".join(parts)

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
