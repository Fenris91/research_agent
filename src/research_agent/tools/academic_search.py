"""
Academic Search Tools

Unified interface for searching academic databases:
- Semantic Scholar (free, 100 req/5min)
- OpenAlex (fully open, great for social sciences)
- Unpaywall (finds open access PDFs)
- CrossRef (DOI metadata)
"""

import asyncio
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


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
    venue: Optional[str] = None
    url: Optional[str] = None

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
        request_delay: float = 0.3
    ):
        """
        Initialize academic search tools.

        Args:
            config: Optional configuration dict
            email: Email for Unpaywall/OpenAlex polite pool
            request_delay: Delay between requests in seconds
        """
        self.config = config or {}
        self.email = email
        self.request_delay = request_delay
        self._client: Optional[httpx.AsyncClient] = None

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
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

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
            response = await client.get(
                f"{self.SEMANTIC_SCHOLAR_API}/paper/search",
                params=params
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
                    id=item.get("paperId", ""),
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

            logger.info(f"Semantic Scholar found {len(papers)} papers for: {query}")
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
            response = await client.get(
                f"{self.OPENALEX_API}/works",
                params=params
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
                    id=item.get("id", "").replace("https://openalex.org/", ""),
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

        client = await self._get_client()

        # Clean DOI
        doi = doi.replace("https://doi.org/", "")

        try:
            response = await client.get(
                f"{self.UNPAYWALL_API}/{doi}",
                params={"email": self.email}
            )

            if response.status_code == 200:
                data = response.json()
                best_oa = data.get("best_oa_location")
                if best_oa:
                    return best_oa.get("url_for_pdf") or best_oa.get("url")

            return None

        except httpx.HTTPError as e:
            logger.error(f"Unpaywall API error: {e}")
            return None

    async def get_paper_details(self, paper_id: str, source: str = "semantic_scholar") -> Optional[Paper]:
        """
        Get detailed information about a specific paper.

        Args:
            paper_id: Paper ID
            source: Source API ("semantic_scholar" or "openalex")

        Returns:
            Paper object with full details
        """
        client = await self._get_client()

        if source == "semantic_scholar":
            try:
                response = await client.get(
                    f"{self.SEMANTIC_SCHOLAR_API}/paper/{paper_id}",
                    params={
                        "fields": "paperId,title,abstract,year,citationCount,authors,fieldsOfStudy,venue,openAccessPdf,externalIds,references,citations"
                    }
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

                return Paper(
                    id=item.get("paperId", ""),
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

            except httpx.HTTPError as e:
                logger.error(f"Error fetching paper details: {e}")
                return None

        elif source == "openalex":
            try:
                response = await client.get(
                    f"{self.OPENALEX_API}/works/{paper_id}"
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

                return Paper(
                    id=item.get("id", "").replace("https://openalex.org/", ""),
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

        return None

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
