"""
Web Search Tool

Search the web for grey literature, reports, news, and other non-academic sources.
Supports multiple providers:
- Tavily (recommended, optimized for AI)
- Serper (Google search results)
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

# TODO: Implement in Phase 3
# import httpx


@dataclass 
class WebResult:
    """Standardized web search result."""
    title: str
    url: str
    content: str
    raw_content: Optional[str] = None
    score: Optional[float] = None


class WebSearchTool:
    """
    Web search for grey literature and reports.
    
    Example:
        search = WebSearchTool(api_key="...", provider="tavily")
        results = await search.search("UNESCO indigenous rights report 2023")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "tavily"
    ):
        self.api_key = api_key
        self.provider = provider
        # TODO: Initialize HTTP client
        # self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search(
        self,
        query: str,
        max_results: int = 10
    ) -> List[WebResult]:
        """
        Search the web.
        
        Automatically uses configured provider.
        """
        if self.provider == "tavily":
            return await self._search_tavily(query, max_results)
        elif self.provider == "serper":
            return await self._search_serper(query, max_results)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    async def _search_tavily(
        self,
        query: str,
        max_results: int
    ) -> List[WebResult]:
        """
        Search using Tavily API.
        
        Tavily is optimized for AI/LLM use cases.
        Good for: reports, news, organizational content
        Pricing: $5/month hobby tier
        """
        # TODO: Implement in Phase 3
        raise NotImplementedError("Implement in Phase 3")
    
    async def _search_serper(
        self,
        query: str,
        max_results: int
    ) -> List[WebResult]:
        """
        Search using Serper API (Google results).
        
        Good for: general web search
        Pricing: Pay per use
        """
        # TODO: Implement in Phase 3
        raise NotImplementedError("Implement in Phase 3")
