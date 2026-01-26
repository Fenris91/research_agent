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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

import httpx

logger = logging.getLogger(__name__)


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
        data.pop('_openalex_data', None)
        data.pop('_semantic_scholar_data', None)
        return data

    def to_summary_row(self) -> dict:
        """Convert to summary row for CSV export."""
        return {
            'name': self.name,
            'affiliations': '; '.join(self.affiliations),
            'works_count': self.works_count,
            'citations_count': self.citations_count,
            'h_index': self.h_index or '',
            'fields': '; '.join(self.fields[:5]),  # Top 5 fields
            'openalex_id': self.openalex_id or '',
            'semantic_scholar_id': self.semantic_scholar_id or '',
            'lookup_date': self.lookup_timestamp[:10]
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
    SEMANTIC_SCHOLAR_AUTHOR_URL = "https://api.semanticscholar.org/graph/v1/author/search"

    def __init__(
        self,
        email: Optional[str] = None,
        request_delay: float = 1.0,
        use_openalex: bool = True,
        use_semantic_scholar: bool = True,
        use_web_search: bool = True
    ):
        """
        Initialize researcher lookup.

        Args:
            email: Optional email for OpenAlex polite pool
            request_delay: Delay between requests in seconds
            use_openalex: Enable OpenAlex lookup
            use_semantic_scholar: Enable Semantic Scholar lookup
            use_web_search: Enable web search
        """
        self.email = email
        self.request_delay = request_delay
        self.use_openalex = use_openalex
        self.use_semantic_scholar = use_semantic_scholar
        self.use_web_search = use_web_search

        # HTTP client (initialized lazily)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {
                "User-Agent": "ResearchAgent/1.0 (Academic Research Tool)"
            }
            if self.email:
                headers["From"] = self.email

            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers=headers
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def search_openalex_author(self, name: str) -> Optional[dict]:
        """
        Search OpenAlex Authors API.

        Args:
            name: Author name to search

        Returns:
            Best matching author data or None
        """
        client = await self._get_client()

        params = {
            "search": name,
            "per_page": 5  # Get top 5 matches
        }

        if self.email:
            params["mailto"] = self.email

        try:
            response = await client.get(self.OPENALEX_AUTHORS_URL, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                logger.info(f"No OpenAlex results for: {name}")
                return None

            # Pick best match by citation count (assume more cited = more likely correct person)
            best = max(results, key=lambda x: x.get("cited_by_count", 0))
            logger.info(f"OpenAlex found: {best.get('display_name')} ({best.get('cited_by_count', 0)} citations)")

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
        client = await self._get_client()

        params = {
            "query": name,
            "fields": "authorId,name,affiliations,paperCount,citationCount,hIndex",
            "limit": 5
        }

        try:
            response = await client.get(self.SEMANTIC_SCHOLAR_AUTHOR_URL, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get("data", [])
            if not results:
                logger.info(f"No Semantic Scholar results for: {name}")
                return None

            # Pick best match by citation count
            best = max(results, key=lambda x: x.get("citationCount", 0) or 0)
            logger.info(f"S2 found: {best.get('name')} ({best.get('citationCount', 0)} citations)")

            return best

        except httpx.HTTPError as e:
            logger.error(f"Semantic Scholar API error for {name}: {e}")
            return None

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
                logger.warning("ddgs/duckduckgo-search not installed. Skipping web search.")
                return []

        try:
            # Add "researcher" to help disambiguate common names
            query = f"{name} researcher academic"

            # DDGS is synchronous, run in executor
            loop = asyncio.get_event_loop()

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
                web_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", ""))
                })

            logger.info(f"Web search found {len(web_results)} results for: {name}")
            return web_results

        except Exception as e:
            logger.error(f"Web search error for {name}: {e}")
            return []

    async def lookup_researcher(self, name: str) -> ResearcherProfile:
        """
        Lookup researcher from all configured sources.

        Args:
            name: Researcher name to lookup

        Returns:
            ResearcherProfile with combined data
        """
        profile = ResearcherProfile(name=name)

        # Run enabled lookups concurrently
        tasks = []

        if self.use_openalex:
            tasks.append(("openalex", self.search_openalex_author(name)))
        if self.use_semantic_scholar:
            tasks.append(("semantic_scholar", self.search_semantic_scholar_author(name)))
        if self.use_web_search:
            tasks.append(("web", self.search_web(name)))

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

        return profile

    def _merge_openalex_data(self, profile: ResearcherProfile, data: dict):
        """Merge OpenAlex author data into profile."""
        profile._openalex_data = data
        profile.openalex_id = data.get("id", "").replace("https://openalex.org/", "")

        # Works and citations
        profile.works_count = max(profile.works_count, data.get("works_count", 0))
        profile.citations_count = max(profile.citations_count, data.get("cited_by_count", 0))

        # Affiliations
        institutions = data.get("last_known_institutions", []) or []
        for inst in institutions:
            if inst and inst.get("display_name"):
                aff = inst["display_name"]
                if aff not in profile.affiliations:
                    profile.affiliations.append(aff)

        # Fields/concepts
        concepts = data.get("x_concepts", []) or []
        for concept in concepts[:10]:  # Top 10
            if concept and concept.get("display_name"):
                field_name = concept["display_name"]
                if field_name not in profile.fields:
                    profile.fields.append(field_name)

        # Recent works
        works = data.get("works_api_url")
        # Note: Would need separate API call to get works details

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
            profile.citations_count = max(profile.citations_count, data["citationCount"])

        # Affiliations
        affiliations = data.get("affiliations") or []
        for aff in affiliations:
            if aff and aff not in profile.affiliations:
                profile.affiliations.append(aff)

    async def lookup_batch(
        self,
        names: List[str],
        output_dir: Optional[Path] = None,
        progress_callback: Optional[callable] = None
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

            logger.info(f"Looking up {i+1}/{total}: {name}")

            profile = await self.lookup_researcher(name)
            profiles.append(profile)

            # Save individual JSON if output_dir specified
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Safe filename
                safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
                safe_name = safe_name.strip().replace(" ", "_")

                json_path = output_dir / f"{safe_name}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
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

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Saved summary CSV to: {path}")
