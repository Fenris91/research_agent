"""
Academic Search Tools

Unified interface for searching academic databases:
- Semantic Scholar (free, 100 req/5min)
- OpenAlex (fully open, great for social sciences)
- Unpaywall (finds open access PDFs)
- CrossRef (DOI metadata)

Features:
- Rate limiting with exponential backoff
- Response caching to reduce API calls
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable
from functools import wraps

import httpx

from research_agent.utils.cache import TTLCache, PersistentCache, make_cache_key
from research_agent.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

# ── Cache TTL tiers (seconds) ──────────────────────────────────
CACHE_TTL_SEARCH = 3600        # 1 hour — search results change frequently
CACHE_TTL_PAPER = 21600        # 6 hours — paper details are stable
CACHE_TTL_OA = 86400           # 24 hours — OA status rarely changes
CACHE_TTL_EMBEDDINGS = 604800  # 7 days — SPECTER2 embeddings are immutable
CACHE_TTL_CROSSREF = 86400     # 24 hours — reference lists are stable


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Tracks calls within a sliding window and enforces rate limits.
    """

    def __init__(self, max_calls: int = 100, window_seconds: int = 300):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed in the window (default: 100)
            window_seconds: Time window in seconds (default: 300 = 5 minutes)
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """
        Acquire permission to make a call.

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        async with self._lock:
            now = time.time()

            # Remove calls outside the window
            self._calls = [t for t in self._calls if now - t < self.window_seconds]

            if len(self._calls) >= self.max_calls:
                # Calculate wait time until oldest call expires
                oldest = min(self._calls)
                wait_time = self.window_seconds - (now - oldest) + 0.1
                return max(wait_time, 0)

            # Record this call
            self._calls.append(now)
            return 0

    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        wait_time = await self.acquire()
        if wait_time > 0:
            logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            # Re-acquire after waiting
            await self.acquire()

    @property
    def calls_remaining(self) -> int:
        """Get number of calls remaining in current window."""
        now = time.time()
        recent_calls = [t for t in self._calls if now - t < self.window_seconds]
        return max(0, self.max_calls - len(recent_calls))



@dataclass
class Paper:
    """Standardized paper representation across sources."""
    paper_id: str
    title: str
    abstract: Optional[str]
    year: Optional[int]
    authors: List[str]
    citations: Optional[int]
    doi: Optional[str]
    open_access_url: Optional[str]
    source: str  # "semantic_scholar", "openalex", etc.
    fields: Optional[List[str]] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    oa_status: Optional[str] = None  # gold/green/hybrid/bronze from Unpaywall
    tldr: Optional[str] = None  # S2 TLDR summary
    specter_embedding: Optional[List[float]] = None  # S2 SPECTER2 768d vector

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.title} ({self.year}) - {self.citations or 0} citations"


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

    # API endpoints
    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
    OPENALEX_API = "https://api.openalex.org"
    UNPAYWALL_API = "https://api.unpaywall.org/v2"

    def __init__(
        self,
        config: Optional[dict] = None,
        email: Optional[str] = None,
        request_delay: float = 0.3,
        s2_rate_limit: int = 95,  # Slightly under 100 to be safe
        s2_rate_window: int = 300,  # 5 minutes
        cache_ttl: int = 3600,  # 1 hour default cache TTL
        cache_enabled: bool = True,
        persistent_cache_dir: Optional[str] = None,
    ):
        """
        Initialize academic search tools.

        Args:
            config: Optional configuration dict
            email: Email for Unpaywall/OpenAlex polite pool
            request_delay: Delay between requests in seconds
            s2_rate_limit: Semantic Scholar rate limit (calls per window)
            s2_rate_window: Semantic Scholar rate window in seconds
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
            cache_enabled: Enable response caching (default: True)
            persistent_cache_dir: Directory for disk-backed cache (survives restarts)
        """
        self.config = config or {}
        self.email = email
        self.request_delay = request_delay
        self._client: Optional[httpx.AsyncClient] = None

        # Rate limiter for Semantic Scholar (100 req/5min free tier)
        self._s2_rate_limiter = RateLimiter(
            max_calls=s2_rate_limit,
            window_seconds=s2_rate_window
        )

        # Response cache — disk-backed if persistent_cache_dir provided
        self.cache_enabled = cache_enabled
        if cache_enabled and persistent_cache_dir:
            self._cache = PersistentCache(
                cache_dir=persistent_cache_dir,
                name="academic_search",
                default_ttl=cache_ttl,
                max_size=5000,
            )
            logger.info(f"Using persistent cache at {persistent_cache_dir}")
        elif cache_enabled:
            self._cache = TTLCache(default_ttl=cache_ttl, max_size=2000)
        else:
            self._cache = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {
                "User-Agent": "ResearchAgent/1.0 (Academic Research Tool)"
            }
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers=headers
            )
        return self._client

    async def close(self):
        """Close HTTP client and persistent cache."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._cache and hasattr(self._cache, "close"):
            self._cache.close()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if self._cache:
            return self._cache.stats
        return {"enabled": False}

    def clear_cache(self):
        """Clear the response cache."""
        if self._cache:
            self._cache.clear()
            logger.info("API response cache cleared")

    async def search_semantic_scholar(
        self,
        query: str,
        limit: int = 20,
        year_range: Optional[tuple] = None,
        fields_of_study: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Search Semantic Scholar for papers.

        Free tier: 100 requests per 5 minutes
        Great for: citation data, paper relationships

        Args:
            query: Search query
            limit: Maximum number of results
            year_range: Optional (start_year, end_year) tuple
            fields_of_study: Optional list of fields to filter by

        Returns:
            List of Paper objects
        """
        # Check cache first
        cache_key = make_cache_key(
            "s2_search",
            query,
            limit=limit,
            year_range=year_range,
            fields=fields_of_study
        )

        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info(f"Cache hit for Semantic Scholar search: {query}")
                return cached

        # Wait for rate limit if needed
        await self._s2_rate_limiter.wait_if_needed()

        client = await self._get_client()

        # Build query parameters
        params = {
            "query": query,
            "limit": min(limit, 100),  # API max is 100
            "fields": "paperId,title,abstract,year,citationCount,authors,fieldsOfStudy,venue,openAccessPdf,externalIds"
        }

        # Add year filter if specified
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        # Add fields of study filter
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)

        try:
            # Use retry with exponential backoff
            response = await retry_with_backoff(
                lambda: client.get(
                    f"{self.SEMANTIC_SCHOLAR_API}/paper/search",
                    params=params
                ),
                max_retries=3,
                base_delay=2.0,
                retry_on=(429, 503, 504)
            )
            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data.get("data", []):
                # Extract DOI from external IDs
                doi = None
                external_ids = item.get("externalIds") or {}
                if "DOI" in external_ids:
                    doi = external_ids["DOI"]

                # Extract open access URL
                oa_url = None
                if item.get("openAccessPdf"):
                    oa_url = item["openAccessPdf"].get("url")

                paper = Paper(
                    paper_id=item.get("paperId", ""),
                    title=item.get("title", ""),
                    abstract=item.get("abstract"),
                    year=item.get("year"),
                    authors=[a.get("name", "") for a in (item.get("authors") or [])],
                    citations=item.get("citationCount"),
                    doi=doi,
                    open_access_url=oa_url,
                    source="semantic_scholar",
                    fields=item.get("fieldsOfStudy"),
                    venue=item.get("venue"),
                    url=f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}"
                )
                papers.append(paper)

            # Cache the results
            if self._cache:
                self._cache.set(cache_key, papers, ttl=CACHE_TTL_SEARCH)

            logger.info(f"Semantic Scholar found {len(papers)} papers for: {query} ({self._s2_rate_limiter.calls_remaining} calls remaining)")
            return papers

        except httpx.HTTPError as e:
            logger.error(f"Semantic Scholar API error: {e}")
            return []

    async def search_openalex(
        self,
        query: str,
        limit: int = 20,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None
    ) -> List[Paper]:
        """
        Search OpenAlex for papers.

        Fully open, no API key needed.
        Great for: social sciences, broad coverage

        Args:
            query: Search query
            limit: Maximum number of results
            from_year: Optional start year filter
            to_year: Optional end year filter

        Returns:
            List of Paper objects
        """
        # Check cache first
        cache_key = make_cache_key(
            "openalex_search",
            query,
            limit=limit,
            from_year=from_year,
            to_year=to_year
        )

        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info(f"Cache hit for OpenAlex search: {query}")
                return cached

        client = await self._get_client()

        # Build query parameters
        params = {
            "search": query,
            "per_page": min(limit, 200),  # API max is 200
            "sort": "cited_by_count:desc"
        }

        # Add email for polite pool
        if self.email:
            params["mailto"] = self.email

        # Build filter string
        filters = []
        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")

        if filters:
            params["filter"] = ",".join(filters)

        try:
            response = await retry_with_backoff(
                lambda: client.get(
                    f"{self.OPENALEX_API}/works",
                    params=params
                ),
                max_retries=2,
                base_delay=1.0,
                retry_on=(429, 503, 504),
            )
            response.raise_for_status()
            data = response.json()

            papers = []
            for item in data.get("results", []):
                # Reconstruct abstract from inverted index
                abstract = self._reconstruct_abstract(item.get("abstract_inverted_index"))

                # Extract authors
                authors = []
                for authorship in item.get("authorships", []):
                    author = authorship.get("author", {})
                    if author.get("display_name"):
                        authors.append(author["display_name"])

                # Extract fields/concepts
                fields = []
                for concept in item.get("concepts", [])[:5]:  # Top 5 concepts
                    if concept.get("display_name"):
                        fields.append(concept["display_name"])

                # Get open access URL
                oa_url = None
                oa_info = item.get("open_access", {})
                if oa_info.get("oa_url"):
                    oa_url = oa_info["oa_url"]

                # Get venue/journal
                venue = None
                primary_location = item.get("primary_location", {})
                if primary_location:
                    source = primary_location.get("source", {})
                    if source:
                        venue = source.get("display_name")

                paper = Paper(
                    paper_id=item.get("id", "").replace("https://openalex.org/", ""),
                    title=item.get("title", ""),
                    abstract=abstract,
                    year=item.get("publication_year"),
                    authors=authors,
                    citations=item.get("cited_by_count"),
                    doi=item.get("doi", "").replace("https://doi.org/", "") if item.get("doi") else None,
                    open_access_url=oa_url,
                    source="openalex",
                    fields=fields,
                    venue=venue,
                    url=item.get("id")
                )
                papers.append(paper)

            # Cache the results
            if self._cache:
                self._cache.set(cache_key, papers, ttl=CACHE_TTL_SEARCH)

            logger.info(f"OpenAlex found {len(papers)} papers for: {query}")
            return papers

        except httpx.HTTPError as e:
            logger.error(f"OpenAlex API error: {e}")
            return []

    def _reconstruct_abstract(self, inverted_index: Optional[Dict]) -> Optional[str]:
        """
        Reconstruct abstract from OpenAlex inverted index format.

        OpenAlex stores abstracts as {word: [positions]} for compression.
        """
        if not inverted_index:
            return None

        try:
            word_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))

            word_positions.sort()
            return " ".join([word for _, word in word_positions])
        except Exception:
            return None

    async def get_open_access_pdf(self, doi: str) -> Optional[str]:
        """
        Find open access PDF via Unpaywall.

        Args:
            doi: DOI of the paper

        Returns:
            URL to open access PDF if available
        """
        if not doi:
            return None

        if not self.email:
            logger.warning("Email required for Unpaywall API")
            return None

        # Clean DOI
        doi = doi.replace("https://doi.org/", "")

        # Check cache first (longer TTL for OA URLs - 24 hours)
        cache_key = make_cache_key("unpaywall", doi)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for Unpaywall: {doi}")
                return cached if cached != "__none__" else None

        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.UNPAYWALL_API}/{doi}",
                params={"email": self.email}
            )

            if response.status_code == 200:
                data = response.json()
                best_oa = data.get("best_oa_location")
                if best_oa:
                    url = best_oa.get("url_for_pdf") or best_oa.get("url")
                    if self._cache:
                        self._cache.set(cache_key, url, ttl=CACHE_TTL_OA)
                    return url

            # Cache negative result too
            if self._cache:
                self._cache.set(cache_key, "__none__", ttl=CACHE_TTL_OA)
            return None

        except httpx.HTTPError as e:
            logger.error(f"Unpaywall API error: {e}")
            return None

    async def get_oa_status(self, doi: str) -> Dict:
        """
        Get detailed OA status from Unpaywall.

        Returns:
            Dict with is_oa, oa_status (gold/green/hybrid/bronze), best_oa_url
        """
        result = {"is_oa": False, "oa_status": "closed", "best_oa_url": None}
        if not doi:
            return result
        if not self.email:
            return result

        doi = doi.replace("https://doi.org/", "")
        cache_key = make_cache_key("oa_status", doi)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.UNPAYWALL_API}/{doi}",
                params={"email": self.email}
            )
            if response.status_code == 200:
                data = response.json()
                result["is_oa"] = data.get("is_oa", False)
                result["oa_status"] = data.get("oa_status", "closed") or "closed"
                best_oa = data.get("best_oa_location")
                if best_oa:
                    result["best_oa_url"] = best_oa.get("url_for_pdf") or best_oa.get("url")
        except httpx.HTTPError as e:
            logger.debug(f"Unpaywall OA status error for {doi}: {e}")

        if self._cache:
            self._cache.set(cache_key, result, ttl=CACHE_TTL_OA)
        return result

    async def enrich_papers_oa_status(self, papers: List[Paper]) -> None:
        """Enrich a list of papers with OA status from Unpaywall (in-place)."""
        if not self.email:
            return
        for paper in papers:
            if paper.doi and not paper.oa_status:
                oa = await self.get_oa_status(paper.doi)
                if oa["oa_status"] != "closed":
                    paper.oa_status = oa["oa_status"]
                    if not paper.open_access_url and oa["best_oa_url"]:
                        paper.open_access_url = oa["best_oa_url"]

    async def get_paper_details(self, paper_id: str, source: str = "semantic_scholar") -> Optional[Paper]:
        """
        Get detailed information about a specific paper.

        Args:
            paper_id: Paper ID
            source: Source API ("semantic_scholar" or "openalex")

        Returns:
            Paper object with full details
        """
        # Check cache first (longer TTL for paper details - 6 hours)
        cache_key = make_cache_key("paper_details", paper_id, source=source)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for paper details: {paper_id}")
                return cached

        client = await self._get_client()
        paper = None

        if source == "semantic_scholar":
            # Wait for rate limit if needed
            await self._s2_rate_limiter.wait_if_needed()

            try:
                response = await retry_with_backoff(
                    lambda: client.get(
                        f"{self.SEMANTIC_SCHOLAR_API}/paper/{paper_id}",
                        params={
                            "fields": "paperId,title,abstract,year,citationCount,authors,fieldsOfStudy,venue,openAccessPdf,externalIds,references,citations,tldr,embedding"
                        }
                    ),
                    max_retries=3,
                    base_delay=2.0,
                    retry_on=(429, 503, 504)
                )
                response.raise_for_status()
                item = response.json()

                doi = None
                external_ids = item.get("externalIds") or {}
                if "DOI" in external_ids:
                    doi = external_ids["DOI"]

                oa_url = None
                if item.get("openAccessPdf"):
                    oa_url = item["openAccessPdf"].get("url")

                # Extract TLDR and embedding
                tldr_text = None
                tldr_data = item.get("tldr")
                if tldr_data and isinstance(tldr_data, dict):
                    tldr_text = tldr_data.get("text")

                embedding_vec = None
                embedding_data = item.get("embedding")
                if embedding_data and isinstance(embedding_data, dict):
                    embedding_vec = embedding_data.get("vector")

                paper = Paper(
                    paper_id=item.get("paperId", ""),
                    title=item.get("title", ""),
                    abstract=item.get("abstract"),
                    year=item.get("year"),
                    authors=[a.get("name", "") for a in (item.get("authors") or [])],
                    citations=item.get("citationCount"),
                    doi=doi,
                    open_access_url=oa_url,
                    source="semantic_scholar",
                    fields=item.get("fieldsOfStudy"),
                    venue=item.get("venue"),
                    url=f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                    tldr=tldr_text,
                    specter_embedding=embedding_vec,
                )

            except httpx.HTTPError as e:
                logger.error(f"Error fetching paper details: {e}")
                return None

        elif source == "openalex":
            try:
                response = await retry_with_backoff(
                    lambda: client.get(
                        f"{self.OPENALEX_API}/works/{paper_id}"
                    ),
                    max_retries=2,
                    base_delay=1.0,
                    retry_on=(429, 503, 504),
                )
                response.raise_for_status()
                item = response.json()

                abstract = self._reconstruct_abstract(item.get("abstract_inverted_index"))
                authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])]
                fields = [c.get("display_name", "") for c in item.get("concepts", [])[:5]]

                oa_url = None
                oa_info = item.get("open_access", {})
                if oa_info.get("oa_url"):
                    oa_url = oa_info["oa_url"]

                paper = Paper(
                    paper_id=item.get("id", "").replace("https://openalex.org/", ""),
                    title=item.get("title", ""),
                    abstract=abstract,
                    year=item.get("publication_year"),
                    authors=authors,
                    citations=item.get("cited_by_count"),
                    doi=item.get("doi", "").replace("https://doi.org/", "") if item.get("doi") else None,
                    open_access_url=oa_url,
                    source="openalex",
                    fields=fields,
                    venue=item.get("primary_location", {}).get("source", {}).get("display_name"),
                    url=item.get("id")
                )

            except httpx.HTTPError as e:
                logger.error(f"Error fetching paper details: {e}")
                return None

        # Cache the result
        if paper and self._cache:
            self._cache.set(cache_key, paper, ttl=CACHE_TTL_PAPER)

        return paper

    async def get_paper_embeddings(self, paper_ids: List[str]) -> Dict[str, List[float]]:
        """
        Batch-fetch SPECTER2 embeddings from Semantic Scholar.

        Uses POST /paper/batch endpoint for efficiency.
        Caches each embedding individually (7-day TTL — embeddings are immutable).

        Args:
            paper_ids: List of S2 paper IDs

        Returns:
            Dict mapping paper_id → embedding vector
        """
        if not paper_ids:
            return {}

        result = {}
        uncached_ids = []

        # Check cache first
        for pid in paper_ids:
            cache_key = make_cache_key("specter_emb", pid)
            if self._cache:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    result[pid] = cached
                    continue
            uncached_ids.append(pid)

        if not uncached_ids:
            logger.debug(f"All {len(paper_ids)} embeddings from cache")
            return result

        # Batch fetch from S2
        await self._s2_rate_limiter.wait_if_needed()
        client = await self._get_client()

        try:
            # S2 batch endpoint: POST with up to 500 IDs
            for i in range(0, len(uncached_ids), 100):
                batch = uncached_ids[i:i+100]
                if i > 0:
                    await self._s2_rate_limiter.wait_if_needed()

                response = await retry_with_backoff(
                    lambda b=batch: client.post(
                        f"{self.SEMANTIC_SCHOLAR_API}/paper/batch",
                        params={"fields": "paperId,embedding,tldr"},
                        json={"ids": b},
                    ),
                    max_retries=2,
                    base_delay=3.0,
                    retry_on=(429, 503, 504),
                )
                response.raise_for_status()
                items = response.json()

                for item in items:
                    if not item:
                        continue
                    pid = item.get("paperId")
                    if not pid:
                        continue

                    emb_data = item.get("embedding")
                    if emb_data and isinstance(emb_data, dict):
                        vec = emb_data.get("vector")
                        if vec:
                            result[pid] = vec
                            if self._cache:
                                self._cache.set(
                                    make_cache_key("specter_emb", pid),
                                    vec, ttl=CACHE_TTL_EMBEDDINGS,
                                )

            logger.info(f"Fetched {len(result)} SPECTER2 embeddings ({len(uncached_ids)} from API)")
        except httpx.HTTPError as e:
            logger.error(f"S2 batch embedding error: {e}")

        return result

    async def get_crossref_references(self, doi: str) -> List[str]:
        """
        Get reference DOIs for a paper from CrossRef.

        Args:
            doi: DOI of the paper

        Returns:
            List of referenced DOIs
        """
        if not doi:
            return []

        doi = doi.replace("https://doi.org/", "")
        cache_key = make_cache_key("crossref_refs", doi)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        client = await self._get_client()
        try:
            headers = {}
            if self.email:
                headers["User-Agent"] = f"ResearchAgent/1.0 (mailto:{self.email})"

            response = await client.get(
                f"https://api.crossref.org/works/{doi}",
                headers=headers,
            )
            if response.status_code != 200:
                return []

            data = response.json()
            refs = data.get("message", {}).get("reference", [])
            ref_dois = []
            for ref in refs:
                ref_doi = ref.get("DOI")
                if ref_doi:
                    ref_dois.append(ref_doi.lower())

            if self._cache:
                self._cache.set(cache_key, ref_dois, ttl=CACHE_TTL_CROSSREF)

            logger.info(f"CrossRef found {len(ref_dois)} references for DOI {doi}")
            return ref_dois

        except httpx.HTTPError as e:
            logger.debug(f"CrossRef error for {doi}: {e}")
            return []

    async def search_all(
        self,
        query: str,
        limit_per_source: int = 10,
        year_range: Optional[tuple] = None
    ) -> List[Paper]:
        """
        Search all sources and deduplicate results.

        Results are sorted by citation count.

        Args:
            query: Search query
            limit_per_source: Maximum results per source
            year_range: Optional (start_year, end_year) tuple

        Returns:
            List of deduplicated Paper objects sorted by citations
        """
        # Run searches concurrently
        tasks = [
            self.search_semantic_scholar(query, limit_per_source, year_range=year_range),
            self.search_openalex(
                query,
                limit_per_source,
                from_year=year_range[0] if year_range else None,
                to_year=year_range[1] if year_range else None
            )
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all papers
        all_papers = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Search error: {result}")
                continue
            all_papers.extend(result)

        # Deduplicate
        unique_papers = self._deduplicate(all_papers)

        # Sort by citation count (descending)
        unique_papers.sort(key=lambda p: p.citations or 0, reverse=True)

        logger.info(f"Combined search found {len(unique_papers)} unique papers for: {query}")
        return unique_papers

    def _deduplicate(self, papers: List[Paper]) -> List[Paper]:
        """
        Remove duplicate papers based on DOI and title similarity.

        Prefers papers with more metadata (abstract, DOI, etc.)
        """
        seen_dois = set()
        seen_titles = set()
        unique = []

        for paper in papers:
            # Check DOI first (most reliable)
            if paper.doi:
                doi_key = paper.doi.lower()
                if doi_key in seen_dois:
                    continue
                seen_dois.add(doi_key)

            # Fall back to title matching
            title_key = paper.title.lower()[:60] if paper.title else ""
            if title_key:
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

            unique.append(paper)

        return unique


# Convenience function for quick searches
async def search_papers(
    query: str,
    limit: int = 20,
    year_range: Optional[tuple] = None
) -> List[Paper]:
    """
    Quick search across all academic sources.

    Args:
        query: Search query
        limit: Maximum total results
        year_range: Optional (start_year, end_year) tuple

    Returns:
        List of Paper objects
    """
    search = AcademicSearchTools()
    try:
        papers = await search.search_all(query, limit_per_source=limit // 2, year_range=year_range)
        return papers[:limit]
    finally:
        await search.close()
