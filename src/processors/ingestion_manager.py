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
        auto_ingest: bool = False,
        auto_threshold: float = 0.85
    ):
        self.vector_store = vector_store
        self.llm = llm
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
        # TODO: Implement in Phase 6
        raise NotImplementedError("Implement in Phase 6")
    
    def _assess_quality(self, source: Dict) -> SourceQuality:
        """Heuristic quality assessment."""
        source_type = source.get("source", "")
        citations = source.get("citations", 0)
        
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
        # TODO: Implement in Phase 6
        raise NotImplementedError("Implement in Phase 6")
    
    async def check_duplicate(self, source: Dict) -> bool:
        """Check if source already exists in knowledge base."""
        # TODO: Implement in Phase 6
        raise NotImplementedError("Implement in Phase 6")
    
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
        # TODO: Implement in Phase 6
        raise NotImplementedError("Implement in Phase 6")
