"""
Graph builder for D3-compatible knowledge graph data.

Accumulates nodes and edges from research data (Papers, Researchers, Queries)
into a JSON structure ready for D3.js force-directed rendering.
"""

import json
import math
from dataclasses import asdict
from typing import Optional


# Color palette for researchers (up to 8 distinct)
RESEARCHER_COLORS = [
    "#4a90d9",  # blue
    "#7c5cbf",  # purple
    "#d9534f",  # red
    "#5cb85c",  # green
    "#f0ad4e",  # amber
    "#5bc0de",  # cyan
    "#d9534f",  # coral
    "#8a6d3b",  # brown
]


def _paper_size(citations: int) -> float:
    """Calculate paper node size based on citation count."""
    return max(7, min(19, 7 + math.sqrt(citations / 80) * 4))


def _oa_status(open_access_url: Optional[str]) -> str:
    """Derive OA status from open_access_url field."""
    if not open_access_url:
        return "closed"
    if "preprint" in open_access_url.lower() or "arxiv" in open_access_url.lower():
        return "preprint"
    return "open"


class GraphBuilder:
    """Builds D3-compatible graph data from research entities."""

    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._edges: list[dict] = []
        self._researcher_color_idx = 0

    def _next_researcher_color(self) -> str:
        color = RESEARCHER_COLORS[self._researcher_color_idx % len(RESEARCHER_COLORS)]
        self._researcher_color_idx += 1
        return color

    def add_researcher(self, profile) -> str:
        """Add a researcher node from a ResearcherProfile or dict."""
        if hasattr(profile, "to_dict"):
            d = profile.to_dict()
        elif isinstance(profile, dict):
            d = profile
        else:
            d = asdict(profile)

        rid = f"researcher:{d.get('openalex_id') or d.get('semantic_scholar_id') or d.get('name', 'unknown')}"
        if rid in self._nodes:
            return rid

        color = self._next_researcher_color()
        self._nodes[rid] = {
            "id": rid,
            "type": "researcher",
            "label": d.get("name", "Unknown"),
            "size": 24,
            "color": color,
            "metadata": {
                "h_index": d.get("h_index"),
                "citations": d.get("citations_count", 0),
                "affiliations": d.get("affiliations", []),
                "fields": d.get("fields", []),
                "works_count": d.get("works_count", 0),
            },
        }
        return rid

    def add_paper(self, paper) -> str:
        """Add a paper node from a Paper, CitationPaper, or dict."""
        if hasattr(paper, "to_dict"):
            d = paper.to_dict()
        elif isinstance(paper, dict):
            d = paper
        else:
            d = asdict(paper)

        pid = f"paper:{d.get('id') or d.get('paper_id', 'unknown')}"
        if pid in self._nodes:
            return pid

        citations = d.get("citations") or d.get("citation_count") or 0
        oa_url = d.get("open_access_url")
        oa = _oa_status(oa_url)

        self._nodes[pid] = {
            "id": pid,
            "type": "paper",
            "label": d.get("title", "Untitled"),
            "size": _paper_size(citations),
            "color": "#666",
            "metadata": {
                "year": d.get("year"),
                "citations": citations,
                "oa_status": oa,
                "abstract": d.get("abstract"),
                "authors": d.get("authors", []),
                "fields": d.get("fields", []),
                "venue": d.get("venue"),
                "doi": d.get("doi"),
            },
        }
        return pid

    def add_citation_edge(self, source_id: str, target_id: str, intent: Optional[str] = None):
        """Add a citation edge (source cites target)."""
        self._edges.append({
            "source": source_id,
            "target": target_id,
            "type": "citation",
            "metadata": {"intent": intent},
        })

    def add_semantic_edge(self, source_id: str, target_id: str, score: float):
        """Add a semantic similarity edge."""
        self._edges.append({
            "source": source_id,
            "target": target_id,
            "type": "semantic",
            "metadata": {"score": score},
        })

    def add_authorship_edge(self, researcher_id: str, paper_id: str):
        """Add an authorship edge (researcher authored paper)."""
        self._edges.append({
            "source": researcher_id,
            "target": paper_id,
            "type": "authorship",
            "metadata": {},
        })

    def add_query(self, query_text: str, matched_ids: list[str]) -> str:
        """Add a query node and semantic edges to matched nodes."""
        qid = f"query:{len([n for n in self._nodes if n.startswith('query:')])}"
        self._nodes[qid] = {
            "id": qid,
            "type": "query",
            "label": query_text[:60] + ("..." if len(query_text) > 60 else ""),
            "size": 18,
            "color": "#c45c4a",
            "metadata": {"full_query": query_text},
        }
        for mid in matched_ids:
            if mid in self._nodes:
                self.add_semantic_edge(qid, mid, score=0.8)
        return qid

    def to_dict(self) -> dict:
        """Return D3-compatible dict with nodes and links arrays."""
        return {
            "nodes": list(self._nodes.values()),
            "links": self._edges,
        }

    def to_json(self) -> str:
        """Return JSON string of graph data."""
        return json.dumps(self.to_dict(), indent=2)
