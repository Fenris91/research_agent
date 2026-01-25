"""
Academic Search Tools

Unified interface for searching academic databases:
- Semantic Scholar (free, 100 req/5min)
- OpenAlex (fully open, great for social sciences)
- Unpaywall (finds open access PDFs)
- CrossRef (DOI metadata)
"""

from typing import List, Dict, Optional
import asyncio
from dataclasses import dataclass

# TODO: Implement in Phase 3
# from semanticscholar import SemanticScholar
# from pyalex import Works
# import httpx


@dataclass
class Paper:
    """Standardized paper representation across sources."""
    id: str
    title: str
    abstract: Optional[str]
    year: Optional[int]
    authors: List[str]
    citations: Optional[int]
    doi: Optional[str]
    open_access_url: Optional[str]
    source: str  # "semantic_scholar", "openalex", etc.
    fields: Optional[List[str]] = None


class AcademicSearchTools:
    """
    Unified interface for academic search APIs.
    
    Example:
        search = AcademicSearchTools()
        papers = await search.search_all("urban gentrification theory")
        
        # Or search specific sources
        papers = await search.search_semantic_scholar(
            "participatory mapping indigenous",
            year_range=(2015, 2024)
        )
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        # TODO: Initialize API clients
        # self.s2 = SemanticScholar()
        # self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def search_semantic_scholar(
        self,
        query: str,
        limit: int = 20,
        year_range: Optional[tuple] = None,
        fields_of_study: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Search Semantic Scholar.
        
        Free tier: 100 requests per 5 minutes
        Great for: citation data, paper relationships
        """
        # TODO: Implement in Phase 3
        raise NotImplementedError("Implement in Phase 3")
    
    async def search_openalex(
        self,
        query: str,
        limit: int = 20,
        from_year: Optional[int] = None
    ) -> List[Paper]:
        """
        Search OpenAlex.
        
        Fully open, no API key needed.
        Great for: social sciences, broad coverage
        """
        # TODO: Implement in Phase 3
        raise NotImplementedError("Implement in Phase 3")
    
    async def get_open_access_pdf(self, doi: str) -> Optional[str]:
        """
        Find open access PDF via Unpaywall.
        
        Requires email for polite pool access.
        """
        # TODO: Implement in Phase 3
        raise NotImplementedError("Implement in Phase 3")
    
    async def search_all(
        self,
        query: str,
        limit_per_source: int = 10
    ) -> List[Paper]:
        """
        Search all sources and deduplicate results.
        
        Results are sorted by citation count.
        """
        # TODO: Implement in Phase 3
        raise NotImplementedError("Implement in Phase 3")
    
    def _deduplicate(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers based on title similarity."""
        seen_titles = set()
        unique = []
        
        for paper in papers:
            title_key = paper.title.lower()[:50] if paper.title else ""
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                unique.append(paper)
        
        return unique
