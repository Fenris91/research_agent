"""
Citation Explorer for analyzing academic citation networks.

This module provides functionality to:
- Get papers that cite a given paper
- Get papers cited by a given paper
- Find highly connected papers in a citation network
- Build citation graphs for visualization
- Suggest related papers based on citation overlap
- Explore citation networks for researchers (across their papers)
"""

from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import logging

import httpx

from research_agent.tools.academic_search import AcademicSearchTools, retry_with_backoff

if TYPE_CHECKING:
    from research_agent.tools.researcher_lookup import ResearcherProfile, AuthorPaper

logger = logging.getLogger(__name__)


@dataclass
class CitationPaper:
    """Represents a paper in a citation network."""

    paper_id: str
    title: str
    year: Optional[int] = None
    authors: Optional[List[str]] = None
    citation_count: Optional[int] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None


@dataclass
class CitationNetwork:
    """Represents a citation network."""

    seed_paper: CitationPaper
    citing_papers: List[CitationPaper]
    cited_papers: List[CitationPaper]
    highly_connected: List[CitationPaper]


@dataclass
class AuthorNetwork:
    """Represents an author's citation network across their papers."""

    author_name: str
    author_papers: List[CitationPaper]
    papers_citing_author: List[CitationPaper]  # Papers that cite any of author's work
    papers_cited_by_author: List[CitationPaper]  # Papers cited by author's work
    highly_connected: List[CitationPaper]  # Papers that appear multiple times
    collaborators: List[str] = field(default_factory=list)  # Co-authors

    @property
    def total_citations_received(self) -> int:
        """Total citations across all author papers."""
        return sum(p.citation_count or 0 for p in self.author_papers)

    @property
    def unique_citing_papers(self) -> int:
        """Number of unique papers citing author's work."""
        return len(self.papers_citing_author)

    @property
    def unique_references(self) -> int:
        """Number of unique papers cited by author."""
        return len(self.papers_cited_by_author)


class CitationExplorer:
    """
    Citation Explorer for analyzing academic citation networks.

    Allows exploration of citation relationships to discover:
    - Foundational works (highly cited papers)
    - Recent developments (papers citing the seed)
    - Related work (shared citations)
    """

    def __init__(self, academic_search: AcademicSearchTools):
        """
        Initialize citation explorer.

        Args:
            academic_search: AcademicSearchTools instance
        """
        self.search = academic_search
        self.rate_limited = False

    async def get_citations(
        self, paper_id: str, direction: str = "both", limit: int = 20
    ) -> CitationNetwork:
        """
        Get citation relationships for a paper.

        Args:
            paper_id: Paper ID (Semantic Scholar or OpenAlex)
            direction: 'citing', 'cited', or 'both'
            limit: Maximum number of papers to fetch per direction

        Returns:
            CitationNetwork with citation relationships
        """
        resolved_id = await self._resolve_to_s2_id(paper_id)
        if resolved_id:
            paper_id = resolved_id

        # Get seed paper details
        seed_paper = await self._get_paper_details(paper_id)

        citing_papers = []
        cited_papers = []

        if direction in ["citing", "both"]:
            citing_papers = await self._get_citing_papers(paper_id, limit)

        if direction in ["cited", "both"]:
            cited_papers = await self._get_cited_papers(paper_id, limit)

        # Find highly connected papers
        highly_connected = await self.find_highly_connected(
            [p.paper_id for p in citing_papers + cited_papers]
        )

        return CitationNetwork(
            seed_paper=seed_paper,
            citing_papers=citing_papers,
            cited_papers=cited_papers,
            highly_connected=highly_connected,
        )

    async def find_highly_connected(
        self, paper_ids: List[str], min_connections: int = 2
    ) -> List[CitationPaper]:
        """
        Find papers frequently cited by the given papers.

        Args:
            paper_ids: List of paper IDs to analyze
            min_connections: Minimum number of connections to include

        Returns:
            List of highly connected papers sorted by connection count
        """
        citation_counts = {}

        for pid in paper_ids:
            try:
                # Get references for this paper
                refs = await self._get_cited_papers(pid, limit=50)

                for ref in refs:
                    ref_id = ref.paper_id
                    if ref_id not in citation_counts:
                        citation_counts[ref_id] = {"paper": ref, "count": 0}
                    citation_counts[ref_id]["count"] += 1
            except Exception as e:
                print(f"Error processing paper {pid}: {e}")
                continue

        # Filter and sort by connection count
        connected_papers = [
            {**item["paper"].__dict__, "connection_count": item["count"]}
            for item in citation_counts.values()
            if item["count"] >= min_connections
        ]

        connected_papers.sort(key=lambda x: x["connection_count"], reverse=True)

        # Convert back to CitationPaper objects
        result = []
        for item in connected_papers[:10]:  # Top 10
            paper_dict = item.copy()
            paper_dict.pop("connection_count", None)
            result.append(CitationPaper(**paper_dict))

        return result

    async def suggest_related(
        self, paper_id: str, limit: int = 10
    ) -> List[CitationPaper]:
        """
        Suggest related papers based on citation overlap.

        Args:
            paper_id: Paper ID to find related papers for
            limit: Maximum number of related papers to return

        Returns:
            List of related papers with overlap scores
        """
        # Get citations for the target paper
        network = await self.get_citations(paper_id, direction="both", limit=50)

        # Find papers that cite many of the same references
        overlap_scores = {}
        target_refs = {p.paper_id for p in network.cited_papers}

        for citing_paper in network.citing_papers:
            try:
                # Get what this citing paper references
                citing_refs = await self._get_cited_papers(
                    citing_paper.paper_id, limit=50
                )
                citing_ref_ids = {p.paper_id for p in citing_refs}

                # Calculate overlap
                overlap = len(target_refs & citing_ref_ids)
                if overlap > 0:
                    overlap_scores[citing_paper.paper_id] = {
                        "paper": citing_paper,
                        "overlap_score": overlap,
                        "overlap_percentage": overlap / len(target_refs),
                    }
            except Exception as e:
                continue

        # Sort by overlap score
        sorted_papers = sorted(
            overlap_scores.items(), key=lambda x: x[1]["overlap_score"], reverse=True
        )

        result = []
        for paper_id, data in sorted_papers[:limit]:
            paper = data["paper"]
            result.append(paper)

        return result

    def build_network_data(self, network: CitationNetwork) -> Dict[str, Any]:
        """
        Build network data for visualization.

        Args:
            network: CitationNetwork to visualize

        Returns:
            Dictionary with nodes and edges for network visualization
        """
        nodes = []
        edges = []

        # Add seed paper
        seed_id = f"seed_{network.seed_paper.paper_id}"
        nodes.append(
            {
                "id": seed_id,
                "label": network.seed_paper.title[:50] + "...",
                "type": "seed",
                "year": network.seed_paper.year,
                "citation_count": network.seed_paper.citation_count,
            }
        )

        # Add citing papers (papers that cite the seed)
        for i, paper in enumerate(network.citing_papers):
            citing_id = f"citing_{i}"
            nodes.append(
                {
                    "id": citing_id,
                    "label": paper.title[:50] + "...",
                    "type": "citing",
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                }
            )
            edges.append({"from": citing_id, "to": seed_id, "type": "cites"})

        # Add cited papers (papers cited by the seed)
        for i, paper in enumerate(network.cited_papers):
            cited_id = f"cited_{i}"
            nodes.append(
                {
                    "id": cited_id,
                    "label": paper.title[:50] + "...",
                    "type": "cited",
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                }
            )
            edges.append({"from": seed_id, "to": cited_id, "type": "cites"})

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_papers": len(nodes),
                "citing_count": len(network.citing_papers),
                "cited_count": len(network.cited_papers),
                "highly_connected_count": len(network.highly_connected),
            },
        }

    async def explore_author_network(
        self,
        profile: "ResearcherProfile",
        papers_limit: int = 5,
        citations_per_paper: int = 10,
    ) -> AuthorNetwork:
        """
        Explore citation network for a researcher across their papers.

        Args:
            profile: ResearcherProfile with top_papers
            papers_limit: Maximum number of author papers to analyze
            citations_per_paper: Maximum citations to fetch per paper

        Returns:
            AuthorNetwork with aggregated citation data
        """
        from research_agent.tools.researcher_lookup import AuthorPaper

        # Convert AuthorPaper to CitationPaper for author's own papers
        author_papers = []
        for paper in profile.top_papers[:papers_limit]:
            author_papers.append(
                CitationPaper(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    year=paper.year,
                    authors=None,
                    citation_count=paper.citation_count,
                    abstract=paper.abstract,
                    venue=paper.venue,
                    url=None,
                )
            )

        # Collect citing and cited papers across all author papers
        all_citing: Dict[str, CitationPaper] = {}
        all_cited: Dict[str, CitationPaper] = {}
        citation_counts: Dict[str, int] = {}  # Track how often papers appear

        for paper in author_papers:
            if not paper.paper_id:
                continue

            try:
                # Get citations for this paper
                citing = await self._get_citing_papers(
                    paper.paper_id, limit=citations_per_paper
                )
                for p in citing:
                    if p.paper_id not in all_citing:
                        all_citing[p.paper_id] = p
                    citation_counts[p.paper_id] = citation_counts.get(p.paper_id, 0) + 1

                # Get references for this paper
                cited = await self._get_cited_papers(
                    paper.paper_id, limit=citations_per_paper
                )
                for p in cited:
                    if p.paper_id not in all_cited:
                        all_cited[p.paper_id] = p
                    citation_counts[p.paper_id] = citation_counts.get(p.paper_id, 0) + 1

            except Exception as e:
                logger.warning(f"Error fetching citations for {paper.paper_id}: {e}")
                continue

        # Find highly connected papers (appear in multiple contexts)
        highly_connected = []
        for paper_id, count in citation_counts.items():
            if count >= 2:
                paper = all_citing.get(paper_id) or all_cited.get(paper_id)
                if paper:
                    highly_connected.append(paper)

        # Sort by connection count (papers that appear most often)
        highly_connected.sort(
            key=lambda p: citation_counts.get(p.paper_id, 0), reverse=True
        )

        return AuthorNetwork(
            author_name=profile.name,
            author_papers=author_papers,
            papers_citing_author=list(all_citing.values()),
            papers_cited_by_author=list(all_cited.values()),
            highly_connected=highly_connected[:20],
            collaborators=[],  # Could be populated from author data
        )

    def build_author_network_data(self, network: AuthorNetwork) -> Dict[str, Any]:
        """
        Build network data for author visualization.

        Args:
            network: AuthorNetwork to visualize

        Returns:
            Dictionary with nodes and edges for network visualization
        """
        nodes = []
        edges = []

        # Add author's papers as central nodes
        for i, paper in enumerate(network.author_papers[:10]):
            node_id = f"author_{i}"
            nodes.append(
                {
                    "id": node_id,
                    "label": paper.title[:40] + "..."
                    if len(paper.title) > 40
                    else paper.title,
                    "type": "author_paper",
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                }
            )

        # Add citing papers
        for i, paper in enumerate(network.papers_citing_author[:15]):
            citing_id = f"citing_{i}"
            nodes.append(
                {
                    "id": citing_id,
                    "label": paper.title[:30] + "..."
                    if len(paper.title) > 30
                    else paper.title,
                    "type": "citing",
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                }
            )
            # Connect to first author paper (simplified)
            if network.author_papers:
                edges.append({"from": citing_id, "to": "author_0", "type": "cites"})

        # Add cited papers
        for i, paper in enumerate(network.papers_cited_by_author[:15]):
            cited_id = f"cited_{i}"
            nodes.append(
                {
                    "id": cited_id,
                    "label": paper.title[:30] + "..."
                    if len(paper.title) > 30
                    else paper.title,
                    "type": "cited",
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                }
            )
            # Connect from first author paper (simplified)
            if network.author_papers:
                edges.append({"from": "author_0", "to": cited_id, "type": "cites"})

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "author_papers": len(network.author_papers),
                "total_citing": len(network.papers_citing_author),
                "total_cited": len(network.papers_cited_by_author),
                "highly_connected": len(network.highly_connected),
                "total_citations": network.total_citations_received,
            },
        }

    async def _get_paper_details(self, paper_id: str) -> CitationPaper:
        """Get detailed information about a paper."""
        try:
            # Wait for rate limit if needed
            await self.search._s2_rate_limiter.wait_if_needed()

            # Use semantic scholar API endpoint with retry
            client = await self.search._get_client()
            response = await retry_with_backoff(
                lambda: client.get(
                    f"{self.search.SEMANTIC_SCHOLAR_API}/paper/{paper_id}",
                    params={
                        "fields": "paperId,title,year,authors,citationCount,abstract,venue,externalIds"
                    },
                ),
                max_retries=3,
                base_delay=2.0,
                retry_on=(429, 503, 504),
            )
            response.raise_for_status()
            s2_data = response.json()

            return CitationPaper(
                paper_id=s2_data.get("paperId", paper_id),
                title=s2_data.get("title", "Unknown Title"),
                year=s2_data.get("year"),
                authors=[a.get("name", "") for a in s2_data.get("authors", [])],
                citation_count=s2_data.get("citationCount"),
                abstract=s2_data.get("abstract"),
                venue=s2_data.get("venue"),
                url=None,
            )
        except Exception as e:
            logger.warning(f"Error getting paper details for {paper_id}: {e}")
            # Return basic info
            return CitationPaper(
                paper_id=paper_id,
                title=f"Paper {paper_id}",
                year=None,
                authors=[],
                citation_count=0,
                abstract=None,
                venue=None,
                url=None,
            )

    async def _get_citing_papers(
        self, paper_id: str, limit: int
    ) -> List[CitationPaper]:
        """Get papers that cite the given paper."""
        try:
            resolved_id = await self._resolve_to_s2_id(paper_id)
            if resolved_id:
                paper_id = resolved_id

            # Wait for rate limit if needed
            await self.search._s2_rate_limiter.wait_if_needed()

            client = await self.search._get_client()
            response = await retry_with_backoff(
                lambda: client.get(
                    f"{self.search.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/citations",
                    params={
                        "fields": "citingPaper.paperId,citingPaper.title,citingPaper.year,citingPaper.citationCount",
                        "limit": limit,
                    },
                ),
                max_retries=3,
                base_delay=2.0,
                retry_on=(429, 503, 504),
            )
            response.raise_for_status()
            data = response.json() or {}

            citing_papers = []
            items = data.get("data") or []
            if not isinstance(items, list):
                items = []
            for citation_data in items:
                citing_paper = citation_data.get("citingPaper", {})
                if citing_paper and citing_paper.get("paperId"):
                    paper = CitationPaper(
                        paper_id=citing_paper.get("paperId"),
                        title=citing_paper.get("title", "Unknown"),
                        year=citing_paper.get("year"),
                        authors=[],
                        citation_count=citing_paper.get("citationCount"),
                        abstract=None,
                        venue=None,
                        url=None,
                    )
                    citing_papers.append(paper)

            return citing_papers
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 429:
                self.rate_limited = True
            logger.warning(f"Error getting citing papers for {paper_id}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error getting citing papers for {paper_id}: {e}")
            return []

    async def _get_cited_papers(self, paper_id: str, limit: int) -> List[CitationPaper]:
        """Get papers cited by the given paper."""
        try:
            resolved_id = await self._resolve_to_s2_id(paper_id)
            if resolved_id:
                paper_id = resolved_id

            # Wait for rate limit if needed
            await self.search._s2_rate_limiter.wait_if_needed()

            client = await self.search._get_client()
            response = await retry_with_backoff(
                lambda: client.get(
                    f"{self.search.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/references",
                    params={
                        "fields": "citedPaper.paperId,citedPaper.title,citedPaper.year,citedPaper.citationCount",
                        "limit": limit,
                    },
                ),
                max_retries=3,
                base_delay=2.0,
                retry_on=(429, 503, 504),
            )
            response.raise_for_status()
            data = response.json() or {}

            cited_papers = []
            items = data.get("data") or []
            if not isinstance(items, list):
                items = []
            for ref_data in items:
                cited_paper = ref_data.get("citedPaper", {})
                if cited_paper and cited_paper.get("paperId"):
                    paper = CitationPaper(
                        paper_id=cited_paper.get("paperId"),
                        title=cited_paper.get("title", "Unknown"),
                        year=cited_paper.get("year"),
                        authors=[],
                        citation_count=cited_paper.get("citationCount"),
                        abstract=None,
                        venue=None,
                        url=None,
                    )
                    cited_papers.append(paper)

            return cited_papers
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 429:
                self.rate_limited = True
            logger.warning(f"Error getting cited papers for {paper_id}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error getting cited papers for {paper_id}: {e}")
            return []

    def _normalize_openalex_id(self, paper_id: str) -> str:
        if paper_id.startswith("https://openalex.org/"):
            return paper_id.replace("https://openalex.org/", "")
        return paper_id

    def _is_openalex_id(self, paper_id: str) -> bool:
        normalized = self._normalize_openalex_id(paper_id)
        return normalized.startswith("W")

    async def _resolve_to_s2_id(self, paper_id: str) -> Optional[str]:
        """Resolve OpenAlex IDs to Semantic Scholar IDs when possible."""
        if not paper_id:
            return None

        normalized = self._normalize_openalex_id(paper_id)
        if not self._is_openalex_id(normalized):
            return normalized

        try:
            paper = await self.search.get_paper_details(normalized, source="openalex")
            if not paper:
                return None

            if paper.doi:
                results = await self.search.search_semantic_scholar(paper.doi, limit=1)
                if results:
                    return results[0].id

            if paper.title:
                results = await self.search.search_semantic_scholar(
                    paper.title, limit=3
                )
                if results:
                    return results[0].id
        except Exception as e:
            logger.warning("Failed to resolve OpenAlex ID %s: %s", paper_id, e)

        return None
