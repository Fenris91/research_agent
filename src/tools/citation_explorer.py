"""
Citation Explorer

Follow citation chains to discover related work:
- Find papers that cite a given paper
- Find papers cited by a given paper
- Identify highly-connected foundational works
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import Counter

import httpx

logger = logging.getLogger(__name__)


@dataclass
class CitationLink:
    """A citation relationship between papers."""
    paper_id: str
    title: str
    year: Optional[int]
    direction: str  # "citing" or "cited"
    authors: Optional[List[str]] = None
    citation_count: Optional[int] = None


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

    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, academic_search=None):
        """
        Args:
            academic_search: AcademicSearchTools instance (optional, for convenience methods)
        """
        self.search = academic_search
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

    async def get_citations(
        self,
        paper_id: str,
        direction: str = "both",
        limit: int = 50
    ) -> Dict[str, List[CitationLink]]:
        """
        Get citation relationships for a paper.

        Args:
            paper_id: Semantic Scholar paper ID
            direction: "citing", "cited", or "both"
            limit: Max citations to return per direction

        Returns:
            Dict with "citing" and/or "cited" lists
        """
        result = {"citing": [], "cited": []}
        client = await self._get_client()

        # Get papers that cite this paper
        if direction in ("citing", "both"):
            try:
                response = await client.get(
                    f"{self.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/citations",
                    params={
                        "fields": "paperId,title,year,authors,citationCount",
                        "limit": limit
                    }
                )
                response.raise_for_status()
                data = response.json() or {}

                for item in (data.get("data") or []):
                    citing_paper = item.get("citingPaper", {})
                    if citing_paper.get("paperId"):
                        result["citing"].append(CitationLink(
                            paper_id=citing_paper.get("paperId", ""),
                            title=citing_paper.get("title", "Unknown"),
                            year=citing_paper.get("year"),
                            direction="citing",
                            authors=[a.get("name", "") for a in (citing_paper.get("authors") or [])[:3]],
                            citation_count=citing_paper.get("citationCount")
                        ))

                logger.info(f"Found {len(result['citing'])} papers citing {paper_id}")

            except httpx.HTTPError as e:
                logger.error(f"Error fetching citations: {e}")

        # Get papers cited by this paper (references)
        if direction in ("cited", "both"):
            try:
                response = await client.get(
                    f"{self.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/references",
                    params={
                        "fields": "paperId,title,year,authors,citationCount",
                        "limit": limit
                    }
                )
                response.raise_for_status()
                data = response.json() or {}

                for item in (data.get("data") or []):
                    cited_paper = item.get("citedPaper", {})
                    if cited_paper.get("paperId"):
                        result["cited"].append(CitationLink(
                            paper_id=cited_paper.get("paperId", ""),
                            title=cited_paper.get("title", "Unknown"),
                            year=cited_paper.get("year"),
                            direction="cited",
                            authors=[a.get("name", "") for a in (cited_paper.get("authors") or [])[:3]],
                            citation_count=cited_paper.get("citationCount")
                        ))

                logger.info(f"Found {len(result['cited'])} papers cited by {paper_id}")

            except httpx.HTTPError as e:
                logger.error(f"Error fetching references: {e}")

        return result

    async def find_highly_connected(
        self,
        paper_ids: List[str],
        min_connections: int = 2,
        limit_per_paper: int = 30
    ) -> List[Dict]:
        """
        Find papers frequently cited by papers in the knowledge base.

        Useful for discovering foundational works that should be
        added to the knowledge base.

        Args:
            paper_ids: List of paper IDs already in knowledge base
            min_connections: Minimum number of KB papers that must cite it
            limit_per_paper: Max references to fetch per paper

        Returns:
            List of papers with kb_citations count, sorted by connections
        """
        # Count how often each paper is cited
        citation_counts = Counter()
        paper_info = {}

        for pid in paper_ids:
            try:
                citations = await self.get_citations(pid, direction="cited", limit=limit_per_paper)

                for ref in citations["cited"]:
                    citation_counts[ref.paper_id] += 1
                    # Store paper info (update with latest)
                    paper_info[ref.paper_id] = {
                        "paper_id": ref.paper_id,
                        "title": ref.title,
                        "year": ref.year,
                        "authors": ref.authors,
                        "citation_count": ref.citation_count
                    }

            except Exception as e:
                logger.warning(f"Error processing paper {pid}: {e}")
                continue

        # Filter by minimum connections and sort
        highly_connected = []
        for paper_id, count in citation_counts.most_common():
            if count >= min_connections:
                info = paper_info.get(paper_id, {})
                highly_connected.append({
                    **info,
                    "kb_citations": count  # Number of KB papers that cite this
                })

        logger.info(f"Found {len(highly_connected)} highly connected papers (min {min_connections} connections)")

        return highly_connected

    async def suggest_related(
        self,
        paper_id: str,
        max_suggestions: int = 10
    ) -> List[Dict]:
        """
        Suggest related papers based on citation overlap.

        Finds papers that share many citations with the given paper,
        indicating similar research topics.

        Args:
            paper_id: Source paper ID
            max_suggestions: Maximum suggestions to return

        Returns:
            List of related papers with overlap score
        """
        # Get references of the source paper
        source_citations = await self.get_citations(paper_id, direction="cited", limit=50)
        source_refs = {c.paper_id for c in source_citations["cited"]}

        if not source_refs:
            logger.warning(f"No references found for {paper_id}")
            return []

        # Get papers citing the source paper
        citing_papers = await self.get_citations(paper_id, direction="citing", limit=30)

        # For each citing paper, calculate citation overlap
        candidates = []
        for citing in citing_papers["citing"]:
            try:
                their_citations = await self.get_citations(citing.paper_id, direction="cited", limit=50)
                their_refs = {c.paper_id for c in their_citations["cited"]}

                # Calculate Jaccard similarity
                overlap = len(source_refs & their_refs)
                union = len(source_refs | their_refs)
                similarity = overlap / union if union > 0 else 0

                if similarity > 0.1:  # Minimum threshold
                    candidates.append({
                        "paper_id": citing.paper_id,
                        "title": citing.title,
                        "year": citing.year,
                        "authors": citing.authors,
                        "citation_count": citing.citation_count,
                        "overlap_score": round(similarity, 3),
                        "shared_references": overlap
                    })

            except Exception as e:
                logger.debug(f"Error processing related paper {citing.paper_id}: {e}")
                continue

        # Sort by overlap score
        candidates.sort(key=lambda x: x["overlap_score"], reverse=True)

        return candidates[:max_suggestions]

    async def build_citation_graph(
        self,
        seed_paper_id: str,
        depth: int = 1,
        direction: str = "both",
        limit_per_level: int = 10
    ) -> Dict:
        """
        Build a citation graph starting from a seed paper.

        Args:
            seed_paper_id: Starting paper ID
            depth: How many levels deep to explore
            direction: "citing", "cited", or "both"
            limit_per_level: Max papers to follow at each level

        Returns:
            Dict with nodes and edges for visualization
        """
        nodes = {}
        edges = []
        visited = set()

        async def explore(paper_id: str, current_depth: int):
            if current_depth > depth or paper_id in visited:
                return

            visited.add(paper_id)
            citations = await self.get_citations(paper_id, direction=direction, limit=limit_per_level)

            # Add citing papers
            for citing in citations.get("citing", [])[:limit_per_level]:
                if citing.paper_id not in nodes:
                    nodes[citing.paper_id] = {
                        "id": citing.paper_id,
                        "title": citing.title,
                        "year": citing.year,
                        "level": current_depth + 1
                    }
                edges.append({
                    "source": citing.paper_id,
                    "target": paper_id,
                    "type": "cites"
                })

                if current_depth < depth:
                    await explore(citing.paper_id, current_depth + 1)

            # Add cited papers
            for cited in citations.get("cited", [])[:limit_per_level]:
                if cited.paper_id not in nodes:
                    nodes[cited.paper_id] = {
                        "id": cited.paper_id,
                        "title": cited.title,
                        "year": cited.year,
                        "level": current_depth + 1
                    }
                edges.append({
                    "source": paper_id,
                    "target": cited.paper_id,
                    "type": "cites"
                })

                if current_depth < depth:
                    await explore(cited.paper_id, current_depth + 1)

        # Start exploration
        await explore(seed_paper_id, 0)

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "seed_paper": seed_paper_id
        }
