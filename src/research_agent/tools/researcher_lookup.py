"""
Researcher Lookup Tool

Fetch researcher profiles from multiple free/open APIs:
- OpenAlex Authors API (fully free, no key)
- Semantic Scholar Author API (free tier, 100 req/5min)
- DuckDuckGo Web Search (free, no key)
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

import httpx

from research_agent.tools.academic_search import RateLimiter, retry_with_backoff
from research_agent.utils.cache import TTLCache, PersistentCache, make_cache_key

logger = logging.getLogger(__name__)


@dataclass
class AuthorPaper:
    """A paper authored by a researcher."""

    paper_id: str
    title: str
    year: Optional[int] = None
    citation_count: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    fields: Optional[List[str]] = None
    source: str = "unknown"  # "openalex" or "semantic_scholar"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "citation_count": self.citation_count,
            "venue": self.venue,
            "doi": self.doi,
            "abstract": self.abstract,
            "fields": self.fields,
            "source": self.source,
        }


@dataclass
class ResearcherProfile:
    """Standardized researcher profile from multiple sources."""

    name: str
    normalized_name: str = ""
    openalex_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    affiliations: List[str] = field(default_factory=list)
    works_count: int = 0
    citations_count: int = 0
    h_index: Optional[int] = None
    fields: List[str] = field(default_factory=list)
    recent_works: List[dict] = field(default_factory=list)
    top_papers: List[AuthorPaper] = field(default_factory=list)
    web_results: List[dict] = field(default_factory=list)
    lookup_timestamp: str = ""

    # Raw API responses for debugging
    _openalex_data: Optional[dict] = field(default=None, repr=False)
    _semantic_scholar_data: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.normalized_name:
            self.normalized_name = self.name.strip().lower()
        if not self.lookup_timestamp:
            self.lookup_timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding private fields."""
        data = asdict(self)
        # Remove private fields
        data.pop("_openalex_data", None)
        data.pop("_semantic_scholar_data", None)
        # Convert AuthorPaper objects to dicts
        data["top_papers"] = [
            p.to_dict() if hasattr(p, "to_dict") else p for p in self.top_papers
        ]
        return data

    def get_paper_ids(self) -> List[str]:
        """Get list of paper IDs for citation exploration."""
        return [p.paper_id for p in self.top_papers if p.paper_id]

    def to_summary_row(self) -> dict:
        """Convert to summary row for CSV export."""
        return {
            "name": self.name,
            "affiliations": "; ".join(self.affiliations),
            "works_count": self.works_count,
            "citations_count": self.citations_count,
            "h_index": self.h_index or "",
            "fields": "; ".join(self.fields[:5]),  # Top 5 fields
            "openalex_id": self.openalex_id or "",
            "semantic_scholar_id": self.semantic_scholar_id or "",
            "lookup_date": self.lookup_timestamp[:10],
        }


class ResearcherLookup:
    """
    Lookup researcher profiles from free academic APIs.

    Example:
        lookup = ResearcherLookup()
        profile = await lookup.lookup_researcher("David Harvey")
        print(f"Citations: {profile.citations_count}")
    """

    # API endpoints
    OPENALEX_AUTHORS_URL = "https://api.openalex.org/authors"
    SEMANTIC_SCHOLAR_AUTHOR_URL = (
        "https://api.semanticscholar.org/graph/v1/author/search"
    )

    def __init__(
        self,
        email: Optional[str] = None,
        request_delay: float = 1.0,
        use_openalex: bool = True,
        use_semantic_scholar: bool = True,
        use_web_search: bool = True,
        s2_rate_limiter: Optional[RateLimiter] = None,
        persistent_cache_dir: Optional[str] = None,
    ):
        """
        Initialize researcher lookup.

        Args:
            email: Optional email for OpenAlex polite pool
            request_delay: Delay between requests in seconds
            use_openalex: Enable OpenAlex lookup
            use_semantic_scholar: Enable Semantic Scholar lookup
            use_web_search: Enable web search
            s2_rate_limiter: Optional shared rate limiter for Semantic Scholar
            persistent_cache_dir: Directory for disk-backed cache (survives restarts)
        """
        self.email = email
        self.request_delay = request_delay
        self.use_openalex = use_openalex
        self.use_semantic_scholar = use_semantic_scholar
        self.use_web_search = use_web_search
        self._persistent_cache_dir = persistent_cache_dir

        # HTTP client (initialized lazily)
        self._client: Optional[httpx.AsyncClient] = None

        # Rate limiter for Semantic Scholar (100 req/5min free tier)
        # Can be shared with AcademicSearchTools if passed in
        self._s2_rate_limiter = s2_rate_limiter or RateLimiter(
            max_calls=95,  # Slightly under 100 to be safe
            window_seconds=300,
        )

        # Response cache — disk-backed if persistent_cache_dir provided
        if persistent_cache_dir:
            self._cache = PersistentCache(
                cache_dir=persistent_cache_dir,
                name="researcher_lookup",
                default_ttl=86400,
                max_size=500,
            )
        else:
            self._cache = TTLCache(default_ttl=86400, max_size=500)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {"User-Agent": "ResearchAgent/1.0 (Academic Research Tool)"}
            if self.email:
                headers["From"] = self.email

            self._client = httpx.AsyncClient(timeout=30.0, headers=headers)
        return self._client

    async def close(self):
        """Close HTTP client and persistent cache."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._cache and hasattr(self._cache, "close"):
            self._cache.close()

    async def search_openalex_author(self, name: str) -> Optional[dict]:
        """
        Search OpenAlex Authors API.

        Args:
            name: Author name to search

        Returns:
            Best matching author data or None
        """
        # Check cache first
        cache_key = make_cache_key("openalex_author", name.lower().strip())
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for OpenAlex author: {name}")
            return cached if cached != "__none__" else None

        client = await self._get_client()

        params = {
            "search": name,
            "per_page": 5,  # Get top 5 matches
        }

        if self.email:
            params["mailto"] = self.email

        try:
            response = await retry_with_backoff(
                lambda: client.get(self.OPENALEX_AUTHORS_URL, params=params),
                max_retries=2,
                base_delay=1.0,
                retry_on=(429, 503, 504),
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                logger.info(f"No OpenAlex results for: {name}")
                self._cache.set(cache_key, "__none__")
                return None

            # Pick best match by citation count (assume more cited = more likely correct person)
            best = max(results, key=lambda x: x.get("cited_by_count", 0))
            logger.info(
                f"OpenAlex found: {best.get('display_name')} ({best.get('cited_by_count', 0)} citations)"
            )

            # Cache the result
            self._cache.set(cache_key, best)
            return best

        except httpx.HTTPError as e:
            logger.error(f"OpenAlex API error for {name}: {e}")
            return None

    async def search_semantic_scholar_author(self, name: str) -> Optional[dict]:
        """
        Search Semantic Scholar Author API.

        Note: Free tier has 100 requests per 5 minutes.

        Args:
            name: Author name to search

        Returns:
            Best matching author data or None
        """
        # Check cache first
        cache_key = make_cache_key("s2_author", name.lower().strip())
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for S2 author: {name}")
            return cached if cached != "__none__" else None

        # Wait for rate limit if needed
        await self._s2_rate_limiter.wait_if_needed()

        client = await self._get_client()

        params = {
            "query": name,
            "fields": "authorId,name,affiliations,paperCount,citationCount,hIndex",
            "limit": 5,
        }

        try:
            response = await retry_with_backoff(
                lambda: client.get(self.SEMANTIC_SCHOLAR_AUTHOR_URL, params=params),
                max_retries=3,
                base_delay=2.0,
                retry_on=(429, 503, 504),
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("data", [])
            if not results:
                logger.info(f"No Semantic Scholar results for: {name}")
                self._cache.set(cache_key, "__none__")
                return None

            # Pick best match by citation count
            best = max(results, key=lambda x: x.get("citationCount", 0) or 0)
            logger.info(
                f"S2 found: {best.get('name')} ({best.get('citationCount', 0)} citations, {self._s2_rate_limiter.calls_remaining} calls remaining)"
            )

            # Cache the result
            self._cache.set(cache_key, best)
            return best

        except httpx.HTTPError as e:
            logger.error(f"Semantic Scholar API error for {name}: {e}")
            return None

    async def fetch_author_papers_openalex(
        self, openalex_id: str, limit: int = 10
    ) -> List[AuthorPaper]:
        """
        Fetch author's papers from OpenAlex.

        Args:
            openalex_id: OpenAlex author ID (e.g., "A1234567890")
            limit: Maximum number of papers to fetch

        Returns:
            List of AuthorPaper objects
        """
        # Check cache first
        cache_key = make_cache_key("openalex_papers", openalex_id, str(limit))
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for OpenAlex papers: {openalex_id}")
            return cached if cached != "__none__" else []

        client = await self._get_client()

        # OpenAlex works endpoint filtered by author
        # Sort by citation count to get most impactful papers
        params = {
            "filter": f"author.id:{openalex_id}",
            "sort": "cited_by_count:desc",
            "per_page": limit,
        }

        if self.email:
            params["mailto"] = self.email

        try:
            response = await retry_with_backoff(
                lambda: client.get("https://api.openalex.org/works", params=params),
                max_retries=2,
                base_delay=1.0,
                retry_on=(429, 503, 504),
            )
            response.raise_for_status()
            data = response.json()

            papers = []
            for work in data.get("results", []):
                # Safely extract venue from nested structure
                venue = None
                primary_loc = work.get("primary_location")
                if primary_loc and isinstance(primary_loc, dict):
                    source = primary_loc.get("source")
                    if source and isinstance(source, dict):
                        venue = source.get("display_name")

                # Safely extract DOI
                doi_raw = work.get("doi")
                doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

                paper = AuthorPaper(
                    paper_id=work.get("id", "").replace("https://openalex.org/", ""),
                    title=work.get("title") or "Unknown",
                    year=work.get("publication_year"),
                    citation_count=work.get("cited_by_count", 0),
                    venue=venue,
                    doi=doi,
                    abstract=self._reconstruct_abstract(
                        work.get("abstract_inverted_index")
                    ),
                    fields=self._extract_openalex_fields(work),
                    source="openalex",
                )
                papers.append(paper)

            logger.info(f"OpenAlex found {len(papers)} papers for author {openalex_id}")
            self._cache.set(cache_key, papers)
            return papers

        except httpx.HTTPError as e:
            logger.error(f"OpenAlex papers API error for {openalex_id}: {e}")
            return []

    def _reconstruct_abstract(self, inverted_index: Optional[dict]) -> Optional[str]:
        """Reconstruct abstract from OpenAlex inverted index format."""
        if not inverted_index:
            return None

        try:
            # Create list of (position, word) tuples
            words = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    words.append((pos, word))

            # Sort by position and join
            words.sort(key=lambda x: x[0])
            return " ".join(word for _, word in words)
        except Exception:
            return None

    def _extract_openalex_fields(self, work: dict, limit: int = 5) -> List[str]:
        """Extract high-signal fields from OpenAlex work concepts."""
        concepts = work.get("concepts", []) or []
        scored = []
        for concept in concepts:
            if not concept or not concept.get("display_name"):
                continue
            score = concept.get("score", 0) or concept.get("relevance_score", 0) or 0
            level = concept.get("level")
            scored.append((concept["display_name"], score, level))

        filtered = [
            item
            for item in scored
            if item[1] >= 0.3 and (item[2] is None or item[2] <= 2)
        ]
        ranked = sorted(filtered or scored, key=lambda x: x[1], reverse=True)

        fields = []
        for name, _score, _level in ranked[:limit]:
            if name not in fields:
                fields.append(name)

        return fields

    async def fetch_author_papers_semantic_scholar(
        self, author_id: str, limit: int = 10
    ) -> List[AuthorPaper]:
        """
        Fetch author's papers from Semantic Scholar.

        Args:
            author_id: Semantic Scholar author ID
            limit: Maximum number of papers to fetch

        Returns:
            List of AuthorPaper objects
        """
        # Check cache first
        cache_key = make_cache_key("s2_papers", author_id, str(limit))
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for S2 papers: {author_id}")
            return cached if cached != "__none__" else []

        # Wait for rate limit
        await self._s2_rate_limiter.wait_if_needed()

        client = await self._get_client()

        try:
            response = await retry_with_backoff(
                lambda: client.get(
                    f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers",
                    params={
                        "fields": "paperId,title,year,citationCount,venue,externalIds,abstract,fieldsOfStudy",
                        "limit": limit,
                    },
                ),
                max_retries=3,
                base_delay=2.0,
                retry_on=(429, 503, 504),
            )
            response.raise_for_status()
            data = response.json()

            papers = []
            for paper_data in data.get("data", []):
                external_ids = paper_data.get("externalIds") or {}
                paper = AuthorPaper(
                    paper_id=paper_data.get("paperId", ""),
                    title=paper_data.get("title", "Unknown"),
                    year=paper_data.get("year"),
                    citation_count=paper_data.get("citationCount", 0),
                    venue=paper_data.get("venue"),
                    doi=external_ids.get("DOI"),
                    abstract=paper_data.get("abstract"),
                    fields=paper_data.get("fieldsOfStudy"),
                    source="semantic_scholar",
                )
                papers.append(paper)

            # Sort by citation count
            papers.sort(key=lambda p: p.citation_count or 0, reverse=True)

            logger.info(
                f"S2 found {len(papers)} papers for author {author_id} "
                f"({self._s2_rate_limiter.calls_remaining} calls remaining)"
            )
            self._cache.set(cache_key, papers)
            return papers

        except httpx.HTTPError as e:
            logger.error(f"Semantic Scholar papers API error for {author_id}: {e}")
            return []

    async def search_web(self, name: str, max_results: int = 5) -> List[dict]:
        """
        Search web for researcher using DuckDuckGo.

        Args:
            name: Researcher name
            max_results: Maximum number of results

        Returns:
            List of web search results
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
                logger.warning(
                    "ddgs/duckduckgo-search not installed. Skipping web search."
                )
                return []

        try:
            # Add "researcher" to help disambiguate common names
            query = f"{name} researcher academic"

            # DDGS is synchronous, run in executor
            loop = asyncio.get_running_loop()

            def do_search():
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ddgs = DDGS()
                    return list(ddgs.text(query, max_results=max_results))

            results = await loop.run_in_executor(None, do_search)

            # Normalize results
            web_results = []
            for r in results:
                web_results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", r.get("link", "")),
                        "snippet": r.get("body", r.get("snippet", "")),
                    }
                )

            logger.info(f"Web search found {len(web_results)} results for: {name}")
            return web_results

        except Exception as e:
            logger.error(f"Web search error for {name}: {e}")
            return []

    async def lookup_researcher(
        self, name: str, fetch_papers: bool = True, papers_limit: int = 10
    ) -> ResearcherProfile:
        """
        Lookup researcher from all configured sources.

        Args:
            name: Researcher name to lookup
            fetch_papers: Whether to fetch the researcher's papers
            papers_limit: Maximum number of papers to fetch per source

        Returns:
            ResearcherProfile with combined data
        """
        profile = ResearcherProfile(name=name)
        name_variants = self._get_name_variants(name)
        primary_name = name_variants[0]

        # Run enabled lookups concurrently
        tasks = []

        if self.use_openalex:
            tasks.append(("openalex", self.search_openalex_author(primary_name)))
        if self.use_semantic_scholar:
            tasks.append(
                ("semantic_scholar", self.search_semantic_scholar_author(primary_name))
            )
        if self.use_web_search:
            tasks.append(("web", self.search_web(primary_name)))

        if not tasks:
            return profile

        # Execute all tasks
        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        # Process results
        for (source, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Error in {source} lookup: {result}")
                continue

            if source == "openalex" and result:
                self._merge_openalex_data(profile, result)
            elif source == "semantic_scholar" and result:
                self._merge_semantic_scholar_data(profile, result)
            elif source == "web" and result:
                profile.web_results = result

        # Retry with alternative capitalization if nothing found
        if (
            (self.use_openalex and not profile.openalex_id)
            and (self.use_semantic_scholar and not profile.semantic_scholar_id)
            and len(name_variants) > 1
        ):
            fallback_name = name_variants[1]
            retry_tasks = []
            if self.use_openalex:
                retry_tasks.append(
                    ("openalex", self.search_openalex_author(fallback_name))
                )
            if self.use_semantic_scholar:
                retry_tasks.append(
                    (
                        "semantic_scholar",
                        self.search_semantic_scholar_author(fallback_name),
                    )
                )

            if retry_tasks:
                retry_results = await asyncio.gather(
                    *[t[1] for t in retry_tasks], return_exceptions=True
                )
                for (source, _), result in zip(retry_tasks, retry_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error in {source} lookup: {result}")
                        continue
                    if source == "openalex" and result:
                        self._merge_openalex_data(profile, result)
                    elif source == "semantic_scholar" and result:
                        self._merge_semantic_scholar_data(profile, result)

        # Fetch papers if requested and we have author IDs
        if fetch_papers:
            await self._fetch_and_merge_papers(profile, papers_limit)

        return profile

    def _get_name_variants(self, name: str) -> List[str]:
        """Return name variants for better matching."""
        cleaned = " ".join(name.strip().split())
        if not cleaned:
            return [name]

        parts = cleaned.split()
        # Title-case basic variant
        title_variant = " ".join([p.capitalize() for p in parts])

        variants = [cleaned]
        if title_variant.lower() != cleaned.lower():
            variants.append(title_variant)
        elif cleaned.islower() and title_variant != cleaned:
            variants.append(title_variant)

        # Ensure uniqueness while preserving order
        seen = set()
        ordered = []
        for v in variants:
            key = v.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(v)
        return ordered

    async def _fetch_and_merge_papers(
        self, profile: ResearcherProfile, limit: int = 10
    ) -> None:
        """
        Fetch papers for a researcher and merge into profile.

        Prefers Semantic Scholar (has paper IDs usable in citation explorer),
        falls back to OpenAlex.
        """
        papers = []

        # Try Semantic Scholar first (better for citation exploration)
        if profile.semantic_scholar_id:
            papers = await self.fetch_author_papers_semantic_scholar(
                profile.semantic_scholar_id, limit=limit
            )

        # Supplement with OpenAlex if needed
        if len(papers) < limit and profile.openalex_id:
            oa_papers = await self.fetch_author_papers_openalex(
                profile.openalex_id, limit=limit
            )

            # Deduplicate by DOI or title
            existing_dois = {p.doi for p in papers if p.doi}
            existing_titles = {(p.title or "").lower() for p in papers}

            for paper in oa_papers:
                if paper.doi and paper.doi in existing_dois:
                    continue
                if (paper.title or "").lower() in existing_titles:
                    continue
                papers.append(paper)

        # Sort by citation count and limit
        papers.sort(key=lambda p: p.citation_count or 0, reverse=True)
        profile.top_papers = papers[:limit]
        self._update_fields_from_papers(profile)
        logger.info(f"Fetched {len(profile.top_papers)} papers for {profile.name}")

    def _merge_openalex_data(self, profile: ResearcherProfile, data: dict):
        """Merge OpenAlex author data into profile."""
        profile._openalex_data = data
        profile.openalex_id = data.get("id", "").replace("https://openalex.org/", "")

        # Works and citations
        profile.works_count = max(profile.works_count, data.get("works_count", 0))
        profile.citations_count = max(
            profile.citations_count, data.get("cited_by_count", 0)
        )

        # Affiliations
        institutions = data.get("last_known_institutions", []) or []
        for inst in institutions:
            if inst and inst.get("display_name"):
                aff = inst["display_name"]
                if aff not in profile.affiliations:
                    profile.affiliations.append(aff)

        # Fields/concepts
        if not profile.fields:
            concepts = data.get("x_concepts", []) or []
            if concepts:
                scored = []
                for concept in concepts:
                    if not concept or not concept.get("display_name"):
                        continue
                    score = concept.get("score")
                    level = concept.get("level")
                    if score is None:
                        score = concept.get("relevance_score", 0)
                    scored.append((concept["display_name"], score or 0, level))

                # Prefer high-confidence, higher-level concepts
                filtered = [
                    item
                    for item in scored
                    if item[1] >= 0.5 and (item[2] is None or item[2] <= 1)
                ]
                ranked = sorted(filtered or scored, key=lambda x: x[1], reverse=True)

                for field_name, _score, _level in ranked[:10]:
                    if field_name not in profile.fields:
                        profile.fields.append(field_name)

        # Recent works
        works = data.get("works_api_url")
        # Note: Would need separate API call to get works details

    def _update_fields_from_papers(self, profile: ResearcherProfile, limit: int = 5):
        """Update profile fields using aggregated paper fields."""
        counts: Dict[str, int] = {}
        for paper in profile.top_papers:
            for field_name in paper.fields or []:
                counts[field_name] = counts.get(field_name, 0) + 1

        if not counts:
            return

        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        profile.fields = [name for name, _count in ranked[:limit]]

    def _merge_semantic_scholar_data(self, profile: ResearcherProfile, data: dict):
        """Merge Semantic Scholar author data into profile."""
        profile._semantic_scholar_data = data
        profile.semantic_scholar_id = data.get("authorId")

        # H-index (only available from S2)
        if data.get("hIndex"):
            profile.h_index = data["hIndex"]

        # Works and citations (use higher value from either source)
        if data.get("paperCount"):
            profile.works_count = max(profile.works_count, data["paperCount"])
        if data.get("citationCount"):
            profile.citations_count = max(
                profile.citations_count, data["citationCount"]
            )

        # Affiliations
        affiliations = data.get("affiliations") or []
        for aff in affiliations:
            if aff and aff not in profile.affiliations:
                profile.affiliations.append(aff)

    async def lookup_batch(
        self,
        names: List[str],
        output_dir: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
    ) -> List[ResearcherProfile]:
        """
        Lookup multiple researchers with rate limiting.

        Args:
            names: List of researcher names
            output_dir: Optional directory to save individual JSON files
            progress_callback: Optional callback(current, total, name) for progress

        Returns:
            List of ResearcherProfile objects
        """
        profiles = []
        total = len(names)

        for i, name in enumerate(names):
            if progress_callback:
                progress_callback(i, total, name)

            logger.info(f"Looking up {i + 1}/{total}: {name}")

            profile = await self.lookup_researcher(name)
            profiles.append(profile)

            # Save individual JSON if output_dir specified
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Safe filename
                safe_name = "".join(
                    c if c.isalnum() or c in " -_" else "_" for c in name
                )
                safe_name = safe_name.strip().replace(" ", "_")

                json_path = output_dir / f"{safe_name}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)

            # Rate limiting delay
            if i < total - 1:
                await asyncio.sleep(self.request_delay)

        if progress_callback:
            progress_callback(total, total, "Complete")

        return profiles

    @staticmethod
    def save_summary_csv(profiles: List[ResearcherProfile], path: Path):
        """
        Save summary of all profiles to CSV.

        Args:
            profiles: List of profiles
            path: Output CSV path
        """
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not profiles:
            return

        rows = [p.to_summary_row() for p in profiles]
        fieldnames = list(rows[0].keys())

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Saved summary CSV to: {path}")
