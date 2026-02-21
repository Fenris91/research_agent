"""
Graph builder for D3-compatible knowledge graph data.

Accumulates nodes and edges from research data (Papers, Researchers, Queries)
into a JSON structure ready for D3.js force-directed rendering.
"""

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to the editable field→domain mapping
_MAPPING_PATH = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "field_domain_mapping.json"

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

# Structural context colors
FIELD_COLOR = "#2e5a88"      # muted blue for academic fields
DOMAIN_COLORS = {
    "Economy":      "#d4a04a",
    "Governance":   "#7c5cbf",
    "Culture":      "#c45c7a",
    "Environment":  "#4a9c6a",
    "Technology":   "#5098ab",
    "Society":      "#8a6d3b",
    "Health":       "#d9534f",
    "Education":    "#5bc0de",
}

# ── Dynamic field→domain mapping (loaded from JSON) ───────────────

_mapping_cache: dict | None = None


def _load_mapping() -> dict[str, str]:
    """Load field→domain mapping from JSON config, with in-memory cache."""
    global _mapping_cache
    if _mapping_cache is not None:
        return _mapping_cache
    try:
        with open(_MAPPING_PATH) as f:
            data = json.load(f)
        _mapping_cache = {k.lower().strip(): v for k, v in data.get("mapping", {}).items()}
        logger.debug(f"Loaded {len(_mapping_cache)} field→domain mappings from {_MAPPING_PATH}")
    except FileNotFoundError:
        logger.warning(f"Mapping file not found: {_MAPPING_PATH}, using empty mapping")
        _mapping_cache = {}
    except Exception:
        logger.exception("Failed to load field→domain mapping")
        _mapping_cache = {}
    return _mapping_cache


def _save_mapping(mapping: dict[str, str]) -> None:
    """Persist updated mapping back to JSON config."""
    global _mapping_cache
    try:
        # Load full file to preserve structure
        try:
            with open(_MAPPING_PATH) as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"domains": list(DOMAIN_COLORS.keys()), "mapping": {}}

        data["mapping"] = {k: v for k, v in sorted(mapping.items())}
        with open(_MAPPING_PATH, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        _mapping_cache = mapping
        logger.info(f"Saved {len(mapping)} field→domain mappings to {_MAPPING_PATH}")
    except Exception:
        logger.exception("Failed to save field→domain mapping")


def _classify_field_llm(field_name: str, domains: list[str]) -> str | None:
    """Use LLM to classify an unknown field into a domain.

    Returns the domain name, or None on failure.
    """
    try:
        import os
        domain_list = ", ".join(domains)
        prompt = (
            f"Classify the academic field \"{field_name}\" into exactly one "
            f"of these societal domains: {domain_list}.\n"
            f"Reply with ONLY the domain name, nothing else."
        )

        # Try Groq first (default provider, free tier)
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            import httpx
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "temperature": 0,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"].strip()
                # Validate it's actually one of our domains
                for d in domains:
                    if d.lower() == answer.lower():
                        return d
                logger.warning(f"LLM returned unexpected domain '{answer}' for field '{field_name}'")
                return None

        # Try OpenAI as fallback
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            import httpx
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "temperature": 0,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"].strip()
                for d in domains:
                    if d.lower() == answer.lower():
                        return d

    except Exception:
        logger.debug(f"LLM classification failed for field '{field_name}'", exc_info=True)
    return None


def resolve_field_domain(field_name: str) -> str:
    """Resolve a field to its domain — from JSON, then LLM, then fallback.

    If the LLM classifies a new field, the result is cached to the JSON file.
    """
    mapping = _load_mapping()
    key = field_name.lower().strip()

    # 1. Exact match
    if key in mapping:
        return mapping[key]

    # 2. Substring match
    for map_key, domain in mapping.items():
        if map_key in key or key in map_key:
            return domain

    # 3. LLM classification
    domains = list(DOMAIN_COLORS.keys())
    domain = _classify_field_llm(field_name, domains)
    if domain:
        mapping[key] = domain
        _save_mapping(mapping)
        logger.info(f"LLM classified '{field_name}' → '{domain}' (saved to config)")
        return domain

    # 4. Default fallback
    return "Society"


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

    def add_field(self, field_name: str) -> str:
        """Add a field/discipline node."""
        fid = f"field:{field_name.lower().replace(' ', '_')}"
        if fid in self._nodes:
            return fid
        self._nodes[fid] = {
            "id": fid,
            "type": "field",
            "label": field_name,
            "size": 16,
            "color": FIELD_COLOR,
            "metadata": {"field_name": field_name},
        }
        return fid

    def add_domain(self, domain_name: str) -> str:
        """Add a societal domain node (large background scaffold)."""
        did = f"domain:{domain_name.lower().replace(' ', '_')}"
        if did in self._nodes:
            return did
        self._nodes[did] = {
            "id": did,
            "type": "domain",
            "label": domain_name,
            "size": 40,
            "color": DOMAIN_COLORS.get(domain_name, "#555"),
            "metadata": {"domain_name": domain_name},
        }
        return did

    def add_field_membership_edge(self, entity_id: str, field_id: str):
        """Connect a researcher or paper to a field."""
        self._edges.append({
            "source": entity_id,
            "target": field_id,
            "type": "field_membership",
            "metadata": {},
        })

    def add_domain_mapping_edge(self, field_id: str, domain_id: str):
        """Connect a field to its societal domain."""
        self._edges.append({
            "source": field_id,
            "target": domain_id,
            "type": "domain_mapping",
            "metadata": {},
        })

    def build_structural_context(self) -> None:
        """Auto-build field and domain scaffold from existing nodes.

        Extracts unique fields from all paper/researcher metadata,
        creates field nodes, maps them to domains via FIELD_TO_DOMAIN,
        and wires all edges.  Call AFTER adding content nodes.
        """
        field_sources: dict[str, list[str]] = {}
        for nid, node in self._nodes.items():
            if node["type"] not in ("paper", "researcher"):
                continue
            fields = (node.get("metadata") or {}).get("fields") or []
            for f in fields:
                key = f.lower().strip()
                if key:
                    field_sources.setdefault(key, []).append(nid)

        domain_fields: dict[str, set[str]] = {}
        for field_lower, entity_ids in field_sources.items():
            display = field_lower.title()
            field_id = self.add_field(display)

            for eid in entity_ids:
                self.add_field_membership_edge(eid, field_id)

            # Map to domain (JSON → LLM → fallback)
            domain_name = resolve_field_domain(field_lower)

            domain_fields.setdefault(domain_name, set()).add(field_id)

        for domain_name, field_ids in domain_fields.items():
            domain_id = self.add_domain(domain_name)
            for fid in field_ids:
                self.add_domain_mapping_edge(fid, domain_id)

    def to_dict(self, active_layer: str = "soc", highlight_terms: list | None = None) -> dict:
        """Return D3-compatible dict with nodes, links, and layer metadata."""
        result = {
            "nodes": list(self._nodes.values()),
            "links": self._edges,
            "active_layer": active_layer,
        }
        if highlight_terms:
            result["highlight_terms"] = highlight_terms
        return result

    def to_json(self, active_layer: str = "soc", highlight_terms: list | None = None) -> str:
        """Return JSON string of graph data."""
        return json.dumps(self.to_dict(active_layer=active_layer, highlight_terms=highlight_terms), indent=2)
