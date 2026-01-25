"""
Citation Explorer

Follow citation chains to discover related work:
- Find papers that cite a given paper
- Find papers cited by a given paper
- Identify highly-connected foundational works
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CitationLink:
    """A citation relationship between papers."""
    paper_id: str
    title: str
    year: Optional[int]
    direction: str  # "citing" or "cited"


class CitationExplorer:
    """
    Explore citation networks to discover related research.
    
    Example:
        explorer = CitationExplorer(academic_search)
        
        # Get papers citing/cited by a paper
        citations = await explorer.get_citations("paper_id", direction="both")
        
        # Find foundational works in your knowledge base
        foundational = await explorer.find_highly_connected(paper_ids)
    """
    
    def __init__(self, academic_search):
        """
        Args:
            academic_search: AcademicSearchTools instance
        """
        self.search = academic_search
    
    async def get_citations(
        self,
        paper_id: str,
        direction: str = "both"
    ) -> Dict[str, List[CitationLink]]:
        """
        Get citation relationships for a paper.
        
        Args:
            paper_id: Semantic Scholar paper ID
            direction: "citing", "cited", or "both"
            
        Returns:
            Dict with "citing" and/or "cited" lists
        """
        # TODO: Implement in Phase 6
        raise NotImplementedError("Implement in Phase 6")
    
    async def find_highly_connected(
        self,
        paper_ids: List[str],
        min_connections: int = 2
    ) -> List[Dict]:
        """
        Find papers frequently cited by papers in the knowledge base.
        
        Useful for discovering foundational works that should be
        added to the knowledge base.
        
        Args:
            paper_ids: List of paper IDs already in knowledge base
            min_connections: Minimum number of KB papers that must cite it
            
        Returns:
            List of papers with kb_citations count
        """
        # TODO: Implement in Phase 6
        raise NotImplementedError("Implement in Phase 6")
    
    async def suggest_related(
        self,
        paper_id: str,
        max_suggestions: int = 10
    ) -> List[Dict]:
        """
        Suggest related papers based on citation overlap.
        
        Finds papers that share many citations with the given paper.
        """
        # TODO: Implement in Phase 6
        raise NotImplementedError("Implement in Phase 6")
