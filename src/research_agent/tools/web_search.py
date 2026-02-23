"""
Web Search Tool

Search the web for grey literature, reports, news, and other non-academic sources.
Supports multiple providers:
- DuckDuckGo (free, no API key required)
- Tavily (optimized for AI, requires API key)
- Serper (Google search results, requires API key)
- Perplexity (web-grounded answers with citations, requires API key)
"""

import asyncio
import logging
from typing import List, Optional
from dataclasses import dataclass

import httpx

from research_agent.utils.observability import timed
from research_agent.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)


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
        # Free search with DuckDuckGo (no API key needed)
        search = WebSearchTool(provider="duckduckgo")
        results = await search.search("UNESCO indigenous rights report 2023")

        # Paid search with Tavily
        search = WebSearchTool(api_key="...", provider="tavily")
        results = await search.search("climate policy analysis")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "duckduckgo"
    ):
        self.api_key = api_key
        self.provider = provider
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @timed
    async def search(
        self,
        query: str,
        max_results: int = 10
    ) -> List[WebResult]:
        """
        Search the web.

        Automatically uses configured provider.
        """
        if self.provider == "duckduckgo":
            return await self._search_duckduckgo(query, max_results)
        elif self.provider == "tavily":
            return await self._search_tavily(query, max_results)
        elif self.provider == "serper":
            return await self._search_serper(query, max_results)
        elif self.provider == "perplexity":
            return await self._search_perplexity(query, max_results)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int
    ) -> List[WebResult]:
        """
        Search using DuckDuckGo.

        Free, no API key required.
        Uses the duckduckgo-search package.
        """
        # Try importing the package (prefer new 'ddgs' name)
        DDGS = None
        try:
            from ddgs import DDGS
        except ImportError:
            pass

        if DDGS is None:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                raise ImportError(
                    "ddgs package required. "
                    "Install with: pip install ddgs"
                )

        try:
            # DDGS is synchronous, run in executor
            loop = asyncio.get_running_loop()

            def do_search():
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ddgs = DDGS()
                    return list(ddgs.text(query, max_results=max_results))

            for attempt in range(2):
                try:
                    raw_results = await loop.run_in_executor(None, do_search)
                    break
                except Exception as e:
                    if attempt == 0:
                        logger.debug(f"DuckDuckGo attempt 1 failed: {e}, retrying...")
                        await asyncio.sleep(1.0)
                    else:
                        raise

            results = []
            for r in raw_results:
                results.append(WebResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("link", "")),
                    content=r.get("body", r.get("snippet", "")),
                    raw_content=None,
                    score=None
                ))

            logger.info(f"DuckDuckGo found {len(results)} results for: {query}")
            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    async def _search_tavily(
        self,
        query: str,
        max_results: int
    ) -> List[WebResult]:
        """
        Search using Tavily API.

        Tavily is optimized for AI/LLM use cases.
        Good for: reports, news, organizational content
        Requires API key.
        """
        if not self.api_key:
            raise ValueError("Tavily API key required. Set api_key parameter.")

        client = await self._get_client()

        try:
            response = await retry_with_backoff(
                lambda: client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max_results,
                        "include_raw_content": False
                    }
                ),
                max_retries=2, base_delay=1.0, retry_on=(429, 503, 504),
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for r in data.get("results", []):
                results.append(WebResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    raw_content=r.get("raw_content"),
                    score=r.get("score")
                ))

            logger.info(f"Tavily found {len(results)} results for: {query}")
            return results

        except httpx.HTTPError as e:
            logger.error(f"Tavily API error: {e}")
            return []

    async def _search_serper(
        self,
        query: str,
        max_results: int
    ) -> List[WebResult]:
        """
        Search using Serper API (Google results).

        Good for: general web search
        Requires API key.
        """
        if not self.api_key:
            raise ValueError("Serper API key required. Set api_key parameter.")

        client = await self._get_client()

        try:
            response = await retry_with_backoff(
                lambda: client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": self.api_key},
                    json={
                        "q": query,
                        "num": max_results
                    }
                ),
                max_retries=2, base_delay=1.0, retry_on=(429, 503, 504),
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for r in data.get("organic", []):
                results.append(WebResult(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    content=r.get("snippet", ""),
                    raw_content=None,
                    score=None
                ))

            logger.info(f"Serper found {len(results)} results for: {query}")
            return results

        except httpx.HTTPError as e:
            logger.error(f"Serper API error: {e}")
            return []

    async def _search_perplexity(
        self,
        query: str,
        max_results: int
    ) -> List[WebResult]:
        """
        Search using Perplexity's sonar model for web-grounded answers.

        Returns a synthesized answer with cited sources.
        Each citation URL becomes a WebResult.
        Requires PERPLEXITY_API_KEY.
        """
        if not self.api_key:
            raise ValueError(
                "Perplexity API key required. Set PERPLEXITY_API_KEY in your environment."
            )

        client = await self._get_client()

        try:
            response = await retry_with_backoff(
                lambda: client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "sonar",
                        "messages": [
                            {"role": "user", "content": f"Search: {query}"}
                        ],
                    },
                ),
                max_retries=2, base_delay=1.0, retry_on=(429, 503, 504),
            )
            response.raise_for_status()
            data = response.json()

            # Extract the synthesized answer
            content = ""
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content", "")

            # Extract citations (list of URLs at the top level of the response)
            citations = data.get("citations", [])

            results = []
            if citations:
                # Build one WebResult per citation, linking snippet context
                for i, url in enumerate(citations[:max_results]):
                    # Try to extract the snippet around the citation reference [i+1]
                    marker = f"[{i + 1}]"
                    snippet = _extract_citation_snippet(content, marker)
                    title = _title_from_url(url)

                    results.append(WebResult(
                        title=title,
                        url=url,
                        content=snippet or content,
                        raw_content=content if i == 0 else None,
                        score=None,
                    ))
            elif content:
                # No citations returned — wrap the whole answer as a single result
                results.append(WebResult(
                    title=f"Perplexity answer: {query[:80]}",
                    url="",
                    content=content,
                    raw_content=content,
                    score=None,
                ))

            logger.info(
                f"Perplexity found {len(results)} results for: {query}"
            )
            return results

        except httpx.HTTPError as e:
            logger.error(f"Perplexity API error: {e}")
            return []


def _extract_citation_snippet(content: str, marker: str, context_chars: int = 200) -> str:
    """Extract text surrounding a citation marker like [1] from the response."""
    import re
    # Escape the marker for regex (brackets are special)
    escaped = re.escape(marker)
    # Find the sentence(s) containing this marker
    # Look for text around the marker
    match = re.search(escaped, content)
    if not match:
        return ""
    start = max(0, match.start() - context_chars)
    end = min(len(content), match.end() + context_chars)
    snippet = content[start:end].strip()
    # Clean up partial words at boundaries
    if start > 0:
        snippet = "..." + snippet.split(" ", 1)[-1] if " " in snippet else "..." + snippet
    if end < len(content):
        snippet = snippet.rsplit(" ", 1)[0] + "..." if " " in snippet else snippet + "..."
    return snippet


def _title_from_url(url: str) -> str:
    """Derive a readable title from a URL."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        # Use the last meaningful path segment as a hint
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        if path_parts:
            slug = path_parts[-1].replace("-", " ").replace("_", " ")
            # Truncate very long slugs
            if len(slug) > 60:
                slug = slug[:57] + "..."
            return f"{slug} ({domain})"
        return domain
    except Exception:
        return url[:80]
