"""
Graph builder for D3-compatible knowledge graph data.

Accumulates nodes and edges from research data (Papers, Researchers, Queries)
into a JSON structure ready for D3.js force-directed rendering.
"""

import json
import logging
import math
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

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
    "#e8843a",  # coral/orange
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
_mapping_lock = threading.Lock()


def _load_mapping() -> dict[str, str]:
    """Load field→domain mapping from JSON config, with in-memory cache."""
    global _mapping_cache
    with _mapping_lock:
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
    with _mapping_lock:
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
    Tries Groq (free tier), then OpenAI as fallback.
    """
    import os
    import httpx

    domain_list = ", ".join(domains)
    prompt = (
        f"Classify the academic field \"{field_name}\" into exactly one "
        f"of these societal domains: {domain_list}.\n"
        f"Reply with ONLY the domain name, nothing else."
    )

    def _try_provider(url: str, api_key: str, model: str) -> str | None:
        try:
            resp = httpx.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
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
                logger.warning(f"LLM returned unexpected domain '{answer}' for field '{field_name}'")
        except Exception:
            logger.debug(f"LLM classification failed for field '{field_name}'", exc_info=True)
        return None

    providers = [
        ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "llama-3.1-8b-instant"),
        ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY", "gpt-4o-mini"),
    ]
    for url, key_env, model in providers:
        api_key = os.environ.get(key_env)
        if api_key:
            result = _try_provider(url, api_key, model)
            if result:
                return result
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


_OA_STATUS_MAP = {
    "gold": "gold", "green": "green", "hybrid": "hybrid", "bronze": "bronze",
}


def _oa_status(open_access_url: Optional[str], oa_status_field: Optional[str] = None) -> str:
    """Derive OA status — prefer explicit Unpaywall status, fall back to URL heuristic."""
    if oa_status_field:
        return _OA_STATUS_MAP.get(oa_status_field.lower(), "closed")
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
        self._query_counter = 0

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

        citations = d.get("citation_count") or d.get("citations") or 0
        oa_url = d.get("open_access_url")
        oa = _oa_status(oa_url, d.get("oa_status"))

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
        qid = f"query:{self._query_counter}"
        self._query_counter += 1
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

    def get_structural_items(self) -> list[dict]:
        """Return field and domain labels for SOC context pills."""
        items = []
        for node in self._nodes.values():
            if node["type"] == "field":
                items.append({"label": node["label"], "type": "field", "auto": True, "enabled": True})
            elif node["type"] == "domain":
                items.append({"label": node["label"], "type": "domain", "auto": True, "enabled": True})
        return items

    def inject_embeddings(self, embeddings: Dict[str, List[float]]) -> int:
        """Inject SPECTER2 embeddings into paper nodes (by S2 paper ID).

        Returns count of nodes enriched.
        """
        count = 0
        for nid, node in self._nodes.items():
            if node["type"] != "paper":
                continue
            # Extract S2 paper ID from node id "paper:<id>"
            paper_id = nid.split(":", 1)[1] if ":" in nid else nid
            if paper_id in embeddings:
                node["_specter_embedding"] = embeddings[paper_id]
                count += 1
        return count

    def inject_tldrs(self, tldrs: Dict[str, str]) -> int:
        """Inject TLDR summaries into paper node metadata.

        Args:
            tldrs: Dict mapping paper_id → tldr text

        Returns count of nodes enriched.
        """
        count = 0
        for nid, node in self._nodes.items():
            if node["type"] != "paper":
                continue
            paper_id = nid.split(":", 1)[1] if ":" in nid else nid
            if paper_id in tldrs:
                node["metadata"]["tldr"] = tldrs[paper_id]
                count += 1
        return count

    def compute_semantic_edges(self, threshold: float = 0.65) -> int:
        """Compute pairwise cosine similarity between papers with embeddings.

        Adds semantic edges for pairs above threshold. Uses pure-Python math
        (no numpy) — fine for n < 50 papers.

        Returns count of edges added.
        """
        # Collect paper nodes with embeddings
        papers = []
        for nid, node in self._nodes.items():
            if node["type"] == "paper" and "_specter_embedding" in node:
                papers.append((nid, node["_specter_embedding"]))

        if len(papers) < 2:
            return 0

        def _cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        # Existing semantic edge pairs (avoid duplicates)
        existing = set()
        for e in self._edges:
            if e["type"] == "semantic":
                s = e["source"]
                t = e["target"]
                existing.add((s, t))
                existing.add((t, s))

        count = 0
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                nid_a, emb_a = papers[i]
                nid_b, emb_b = papers[j]
                if (nid_a, nid_b) in existing:
                    continue
                score = _cosine(emb_a, emb_b)
                if score >= threshold:
                    self.add_semantic_edge(nid_a, nid_b, round(score, 3))
                    count += 1

        logger.info(f"Computed {count} semantic edges from SPECTER2 (threshold={threshold})")
        return count

    def fill_citation_gaps(self, reference_map: Dict[str, List[str]]) -> int:
        """Add citation edges from CrossRef reference data.

        Only adds edges between papers that already exist as nodes.

        Args:
            reference_map: {doi: [referenced_dois]} for papers in the graph

        Returns count of edges added.
        """
        # Build DOI → node ID lookup
        doi_to_nid: Dict[str, str] = {}
        for nid, node in self._nodes.items():
            if node["type"] == "paper":
                doi = (node.get("metadata") or {}).get("doi")
                if doi:
                    doi_to_nid[doi.lower()] = nid

        # Existing citation edges
        existing = set()
        for e in self._edges:
            if e["type"] == "citation":
                existing.add((e["source"], e["target"]))

        count = 0
        for doi, ref_dois in reference_map.items():
            src_nid = doi_to_nid.get(doi.lower())
            if not src_nid:
                continue
            for ref_doi in ref_dois:
                tgt_nid = doi_to_nid.get(ref_doi.lower())
                if tgt_nid and tgt_nid != src_nid and (src_nid, tgt_nid) not in existing:
                    self.add_citation_edge(src_nid, tgt_nid)
                    existing.add((src_nid, tgt_nid))
                    count += 1

        if count:
            logger.info(f"CrossRef added {count} citation edges")
        return count

    @staticmethod
    def diff(old_data: dict, new_data: dict) -> dict:
        """Compute node/edge delta between two graph states.

        Returns dict with addNodes, removeNodes, addLinks, removeLinks.
        """
        old_node_ids = {n["id"] for n in old_data.get("nodes", [])}
        new_node_ids = {n["id"] for n in new_data.get("nodes", [])}
        new_node_map = {n["id"]: n for n in new_data.get("nodes", [])}

        add_nodes = [new_node_map[nid] for nid in (new_node_ids - old_node_ids)]
        remove_nodes = list(old_node_ids - new_node_ids)

        def _link_key(l):
            s = l["source"]["id"] if isinstance(l["source"], dict) else l["source"]
            t = l["target"]["id"] if isinstance(l["target"], dict) else l["target"]
            return (s, t, l.get("type", ""))

        old_links = {_link_key(l) for l in old_data.get("links", [])}
        new_links_list = new_data.get("links", [])
        new_links_set = {_link_key(l) for l in new_links_list}
        new_links_map = {_link_key(l): l for l in new_links_list}

        add_links = [new_links_map[k] for k in (new_links_set - old_links)]
        remove_links = [{"source": k[0], "target": k[1], "type": k[2]} for k in (old_links - new_links_set)]

        return {
            "addNodes": add_nodes,
            "removeNodes": remove_nodes,
            "addLinks": add_links,
            "removeLinks": remove_links,
        }

    def to_dict(
        self,
        active_layer: str = "structure",
        highlight_terms: list | None = None,
        context_items: dict | None = None,
    ) -> dict:
        """Return D3-compatible dict with nodes, links, and layer metadata.

        Strips internal-only fields (_specter_embedding) from node data.
        """
        # Deep-copy nodes and strip large internal fields
        clean_nodes = []
        for node in self._nodes.values():
            n = dict(node)
            n.pop("_specter_embedding", None)
            clean_nodes.append(n)

        result = {
            "nodes": clean_nodes,
            "links": self._edges,
            "active_layer": active_layer,
        }
        if highlight_terms:
            result["highlight_terms"] = highlight_terms
        if context_items:
            result["context_items"] = context_items
        return result

    def to_json(self, active_layer: str = "structure", highlight_terms: list | None = None) -> str:
        """Return JSON string of graph data."""
        return json.dumps(self.to_dict(active_layer=active_layer, highlight_terms=highlight_terms), indent=2)

    # ------------------------------------------------------------------
    # Convenience: build graph from KB papers
    # ------------------------------------------------------------------

    def build_from_kb_papers(self, papers: list[dict]) -> "GraphBuilder":
        """Populate the graph from a list of KB paper dicts.

        Groups papers by ``researcher`` metadata, creates researcher nodes
        with authorship edges, then builds structural context (fields/domains).
        Returns self for chaining.
        """
        import json as _json
        from collections import defaultdict

        by_researcher: dict[str, list[dict]] = defaultdict(list)
        orphans: list[dict] = []

        for p in papers:
            # Ensure fields is a list (SQLite stores as string)
            raw_fields = p.get("fields", [])
            if isinstance(raw_fields, str) and raw_fields:
                try:
                    parsed = _json.loads(raw_fields)
                    if isinstance(parsed, list):
                        p["fields"] = parsed
                    else:
                        p["fields"] = [s.strip() for s in raw_fields.split(",") if s.strip()]
                except (ValueError, TypeError):
                    p["fields"] = [s.strip() for s in raw_fields.split(",") if s.strip()]
            elif not isinstance(raw_fields, list):
                p["fields"] = []

            researcher = p.get("researcher") or ""
            if researcher:
                by_researcher[researcher].append(p)
            else:
                orphans.append(p)

        # Create researcher nodes + paper nodes with authorship edges
        for name, rpapers in by_researcher.items():
            total_cites = sum(
                (pp.get("citation_count") or pp.get("citations") or 0) for pp in rpapers
            )
            # Collect all fields from this researcher's papers
            all_fields: list[str] = []
            for pp in rpapers:
                all_fields.extend(pp.get("fields") or [])
            rid = self.add_researcher({
                "name": name,
                "works_count": len(rpapers),
                "citations_count": total_cites,
                "fields": list(dict.fromkeys(all_fields)),  # deduplicated, order-preserving
            })
            for pp in rpapers:
                pid = self.add_paper(pp)
                self.add_authorship_edge(rid, pid)

        # Orphan papers (no researcher tag) — add standalone
        for pp in orphans:
            self.add_paper(pp)

        # Build field → domain scaffold
        self.build_structural_context()
        return self
