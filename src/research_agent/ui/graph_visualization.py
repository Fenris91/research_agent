"""
Context Map Graph Visualization

Provides 4 interactive graph views for the research agent:
- KB Graph: Knowledge base papers/notes as a network (shared-author edges)
- Researcher Graph: Researcher network by shared fields
- Query Graph: Retrieved chunks around a query node
- Citation Graph: Citation network from citation store

Uses NetworkX for graph computation and Plotly for interactive rendering.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Plotly renderer
# ---------------------------------------------------------------------------


def _make_plotly_figure(
    G,
    pos: Dict,
    node_labels: Dict,
    node_colors: Dict,
    node_sizes: Dict,
    hover_texts: Dict,
    title: str = "",
    directed: bool = False,
) -> Any:
    """Convert a NetworkX graph + layout positions to an interactive Plotly figure."""
    import plotly.graph_objects as go

    # Edge traces
    edge_x: List[float] = []
    edge_y: List[float] = []
    for u, v in G.edges():
        x0, y0 = pos.get(u, (0, 0))
        x1, y1 = pos.get(v, (0, 0))
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.9, color="rgba(150,150,160,0.45)"),
        hoverinfo="none",
        showlegend=False,
    )

    # Node trace
    nodes = list(G.nodes())
    node_x = [pos.get(n, (0, 0))[0] for n in nodes]
    node_y = [pos.get(n, (0, 0))[1] for n in nodes]
    labels = [node_labels.get(n, str(n)) for n in nodes]
    colors = [node_colors.get(n, "#4C72B0") for n in nodes]
    sizes = [node_sizes.get(n, 12) for n in nodes]
    texts = [hover_texts.get(n, str(n)) for n in nodes]
    short_labels = [
        (lbl[:22] + "…" if len(lbl) > 22 else lbl) for lbl in labels
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hovertext=texts,
        hoverinfo="text",
        text=short_labels,
        textposition="top center",
        textfont=dict(size=9, color="#333333", family="Arial"),
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=1.5, color="rgba(255,255,255,0.85)"),
            opacity=0.92,
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, family="Arial", color="#222"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
        margin=dict(l=10, r=10, t=50, b=10),
        height=460,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        legend=dict(
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ddd",
            borderwidth=1,
        ),
    )
    return fig


def _require_networkx():
    """Import networkx or raise a clear error."""
    try:
        import networkx as nx
        return nx
    except ImportError:
        raise ImportError(
            "networkx is required for graph visualization. "
            "Install it with: pip install networkx"
        )


# ---------------------------------------------------------------------------
# 1. KB Graph
# ---------------------------------------------------------------------------


def build_kb_graph(vector_store) -> Tuple[Any, str]:
    """
    Build an interactive knowledge-base graph.

    Nodes  = papers, notes, web sources in the vector store.
    Edges  = shared author surname(s) between papers.
    Size   = proportional to citation count.
    Color  = by collection type (papers / notes / web_sources).
    """
    try:
        nx = _require_networkx()
    except ImportError as e:
        return None, str(e)

    import plotly.graph_objects as go

    try:
        papers = vector_store.list_papers_detailed(limit=300)
    except Exception as e:
        logger.warning("KB graph: could not list papers — %s", e)
        papers = []

    if not papers:
        return (
            None,
            "No papers in knowledge base. Upload some papers first to generate the KB graph.",
        )

    color_map = {
        "papers": "#4C72B0",
        "notes": "#DD8452",
        "web_sources": "#55A868",
    }

    G = nx.Graph()
    node_labels: Dict = {}
    node_colors: Dict = {}
    node_sizes: Dict = {}
    hover_texts: Dict = {}

    valid_papers = []
    for paper in papers:
        pid = paper.get("paper_id", "")
        if not pid:
            continue
        title = paper.get("title", "Unknown")
        year = paper.get("year", "")
        authors = paper.get("authors", "") or ""
        source = paper.get("source", "papers")
        citations = paper.get("citation_count") or paper.get("citations") or 0

        G.add_node(pid)
        node_labels[pid] = title
        node_colors[pid] = color_map.get(source, "#7F7F7F")
        node_sizes[pid] = max(10, min(32, 10 + int(citations or 0) // 40))
        hover_texts[pid] = (
            f"<b>{title}</b><br>"
            f"Year: {year or 'Unknown'}<br>"
            f"Authors: {(authors[:70] + '…') if len(authors) > 70 else authors}<br>"
            f"Citations: {citations or 0}<br>"
            f"Type: {source}"
        )
        valid_papers.append(paper)

    if G.number_of_nodes() == 0:
        return None, "No papers with valid IDs found in the knowledge base."

    # Edges: shared author surnames
    for i, p1 in enumerate(valid_papers):
        surnames1 = {
            w.strip().lower()
            for a in (p1.get("authors") or "").split(",")
            for w in a.strip().split()
            if len(w.strip()) > 2
        }
        for p2 in valid_papers[i + 1 :]:
            pid1, pid2 = p1["paper_id"], p2["paper_id"]
            surnames2 = {
                w.strip().lower()
                for a in (p2.get("authors") or "").split(",")
                for w in a.strip().split()
                if len(w.strip()) > 2
            }
            if surnames1 & surnames2:
                G.add_edge(pid1, pid2)

    # Layout
    n = G.number_of_nodes()
    if G.number_of_edges() > 0:
        k = max(0.5, 2.5 / (n ** 0.5))
        pos = nx.spring_layout(G, seed=42, k=k, iterations=60)
    else:
        pos = nx.circular_layout(G)

    fig = _make_plotly_figure(
        G,
        pos,
        node_labels,
        node_colors,
        node_sizes,
        hover_texts,
        title=f"KB Graph — {n} items · {G.number_of_edges()} shared-author edges",
    )

    # Legend
    for label, color in [
        ("Papers", "#4C72B0"),
        ("Notes", "#DD8452"),
        ("Web Sources", "#55A868"),
    ]:
        import plotly.graph_objects as go
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=label,
                showlegend=True,
            )
        )

    status = (
        f"KB Graph: {n} nodes, {G.number_of_edges()} connections. "
        "Node size reflects citation count. Hover for details."
    )
    return fig, status


# ---------------------------------------------------------------------------
# 2. Researcher Graph
# ---------------------------------------------------------------------------


def build_researcher_graph(vector_store=None) -> Tuple[Any, str]:
    """
    Build a researcher network graph.

    Nodes  = researchers from the persisted registry.
    Edges  = shared research field(s).
    Size   = h-index.
    Color  = by h-index tier (low / mid / high).
    """
    try:
        nx = _require_networkx()
    except ImportError as e:
        return None, str(e)

    import plotly.graph_objects as go

    try:
        from research_agent.tools.researcher_registry import get_researcher_registry

        registry = get_researcher_registry()
        researchers = registry.list_all()
    except Exception as e:
        logger.warning("Researcher graph: could not load registry — %s", e)
        return None, f"Could not load researcher registry: {e}"

    if not researchers:
        return (
            None,
            "No researchers found. Use the Researcher tab to look up researchers first.",
        )

    G = nx.Graph()
    node_labels: Dict = {}
    node_colors: Dict = {}
    node_sizes: Dict = {}
    hover_texts: Dict = {}

    def _tier_color(h: int) -> str:
        if h >= 30:
            return "#C0392B"  # high — red
        if h >= 10:
            return "#E67E22"  # mid — orange
        return "#2980B9"      # low — blue

    valid_researchers = []
    for r in researchers:
        name = getattr(r, "name", None)
        if not name:
            continue
        h_index = int(getattr(r, "h_index", 0) or 0)
        citations = int(getattr(r, "citation_count", 0) or 0)
        works = int(getattr(r, "works_count", 0) or 0)
        fields = list(getattr(r, "fields", []) or [])
        affil = str(getattr(r, "affiliation", "") or "")

        G.add_node(name)
        node_labels[name] = name
        node_colors[name] = _tier_color(h_index)
        node_sizes[name] = max(12, min(38, 12 + h_index))
        fields_str = ", ".join(fields[:4]) if fields else "Unknown"
        hover_texts[name] = (
            f"<b>{name}</b><br>"
            f"H-index: {h_index}<br>"
            f"Citations: {citations:,}<br>"
            f"Works: {works}<br>"
            f"Fields: {fields_str}<br>"
            f"Affiliation: {affil[:60] or 'Unknown'}"
        )
        valid_researchers.append((name, set(fields)))

    if G.number_of_nodes() == 0:
        return None, "No valid researchers found in registry."

    # Edges: shared field(s)
    for i, (n1, f1) in enumerate(valid_researchers):
        for n2, f2 in valid_researchers[i + 1 :]:
            if f1 & f2:
                G.add_edge(n1, n2)

    n = G.number_of_nodes()
    k = max(0.8, 3.5 / (n ** 0.5))
    pos = nx.spring_layout(G, seed=42, k=k, iterations=80)

    fig = _make_plotly_figure(
        G,
        pos,
        node_labels,
        node_colors,
        node_sizes,
        hover_texts,
        title=f"Researcher Network — {n} researchers · {G.number_of_edges()} shared-field edges",
    )

    # Legend
    for label, color in [
        ("h-index ≥ 30", "#C0392B"),
        ("h-index 10–29", "#E67E22"),
        ("h-index < 10", "#2980B9"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=label,
                showlegend=True,
            )
        )

    status = (
        f"Researcher Network: {n} researchers, {G.number_of_edges()} shared-field connections. "
        "Node size = h-index. Hover for profile details."
    )
    return fig, status


# ---------------------------------------------------------------------------
# 3. Query Graph
# ---------------------------------------------------------------------------


def build_query_graph(
    query: Optional[str], chunks: Optional[List[Dict]]
) -> Tuple[Any, str]:
    """
    Build a query-centric context graph.

    Central node = the user's query.
    Surrounding nodes = sources of retrieved chunks.
    Edge weight (opacity) = retrieval score.
    """
    try:
        nx = _require_networkx()
    except ImportError as e:
        return None, str(e)

    if not query:
        return (
            None,
            "No recent query. Ask a question in the chat first — this view shows which sources were retrieved.",
        )

    chunks = chunks or []
    if not chunks:
        return (
            None,
            f"Query: '{query[:60]}' — no chunks retrieved. Try a research question to populate this view.",
        )

    G = nx.Graph()
    node_labels: Dict = {}
    node_colors: Dict = {}
    node_sizes: Dict = {}
    hover_texts: Dict = {}

    # Central query node
    QUERY_ID = "__QUERY__"
    G.add_node(QUERY_ID)
    q_short = query[:40] + ("…" if len(query) > 40 else "")
    node_labels[QUERY_ID] = q_short
    node_colors[QUERY_ID] = "#E74C3C"
    node_sizes[QUERY_ID] = 28
    hover_texts[QUERY_ID] = f"<b>Query</b><br>{query}"

    # Group chunks by source paper
    paper_groups: Dict[str, List[Dict]] = {}
    for chunk in chunks:
        pid = (
            chunk.get("paper_id")
            or chunk.get("title")
            or chunk.get("source", "Unknown")
        )
        paper_groups.setdefault(pid, []).append(chunk)

    for pid, pchunks in paper_groups.items():
        G.add_node(pid)
        G.add_edge(QUERY_ID, pid)
        title = (pchunks[0].get("title") or pid)[:45]
        score = pchunks[0].get("score", 0) or 0
        preview = (pchunks[0].get("text", "") or "")[:180]
        node_labels[pid] = title
        node_colors[pid] = "#3498DB"
        node_sizes[pid] = max(12, min(24, 12 + len(pchunks) * 3))
        hover_texts[pid] = (
            f"<b>{title}</b><br>"
            f"Chunks retrieved: {len(pchunks)}<br>"
            f"Top score: {score:.3f}<br>"
            f"<i>{preview}…</i>"
        )

    # Force query node to geometric center of its neighbors
    pos = nx.spring_layout(G, seed=42, k=2.2)
    if len(paper_groups) > 0:
        neighbor_positions = [pos[pid] for pid in paper_groups if pid in pos]
        if neighbor_positions:
            cx = float(np.mean([p[0] for p in neighbor_positions]))
            cy = float(np.mean([p[1] for p in neighbor_positions]))
            pos[QUERY_ID] = np.array([cx, cy])

    fig = _make_plotly_figure(
        G,
        pos,
        node_labels,
        node_colors,
        node_sizes,
        hover_texts,
        title=(
            f"Query Context — {len(chunks)} chunks from {len(paper_groups)} sources"
        ),
    )

    status = (
        f"Query: '{query[:60]}' — {len(chunks)} retrieved chunks from {len(paper_groups)} sources. "
        "Red = query node, blue = retrieved sources. Hover for details."
    )
    return fig, status


# ---------------------------------------------------------------------------
# 4. Citation Graph
# ---------------------------------------------------------------------------


def build_citation_graph(
    paper_id: Optional[str] = None, vector_store=None
) -> Tuple[Any, str]:
    """
    Build a citation network graph.

    If paper_id is provided, focuses on that paper.
    Falls back to all papers in the citation store (up to 40).

    Nodes  = papers.
    Edges  = directed citation relationships (→ cites →).
    Color  = green for seed papers, grey for referenced papers.
    """
    try:
        nx = _require_networkx()
    except ImportError as e:
        return None, str(e)

    import plotly.graph_objects as go

    # Try to load from citation store
    seed_papers = []
    try:
        from research_agent.db.citation_store import CitationStore

        store = CitationStore()
        if paper_id:
            p = store.get_paper(paper_id)
            if p:
                seed_papers = [p]
        if not seed_papers:
            seed_papers = store.list_papers(limit=40) or []
    except Exception as e:
        logger.debug("Citation store unavailable: %s", e)

    # Fallback: use KB papers if citation store is empty
    if not seed_papers and vector_store is not None:
        try:
            kb_papers = vector_store.list_papers_detailed(limit=40)
            seed_papers = [
                {
                    "paper_id": p.get("paper_id"),
                    "title": p.get("title", "Unknown"),
                    "year": p.get("year"),
                    "citations": p.get("citation_count") or p.get("citations") or 0,
                    "references": [],
                }
                for p in kb_papers
                if p.get("paper_id")
            ]
        except Exception as e:
            logger.debug("KB fallback for citation graph failed: %s", e)

    if not seed_papers:
        return (
            None,
            "No citation data available. Use the Citation Explorer to fetch citation networks, "
            "or upload papers to the knowledge base.",
        )

    G = nx.DiGraph()
    node_labels: Dict = {}
    node_colors: Dict = {}
    node_sizes: Dict = {}
    hover_texts: Dict = {}
    seed_ids = set()

    for paper in seed_papers[:40]:
        pid = paper.get("paper_id") or paper.get("id", "")
        if not pid:
            continue
        title = (paper.get("title", "Unknown") or "Unknown")[:45]
        year = paper.get("year", "")
        ccount = int(paper.get("citationCount") or paper.get("citation_count") or paper.get("citations") or 0)

        G.add_node(pid)
        seed_ids.add(pid)
        node_labels[pid] = title
        node_colors[pid] = "#27AE60"
        node_sizes[pid] = max(12, min(32, 12 + ccount // 80))
        hover_texts[pid] = (
            f"<b>{title}</b><br>"
            f"Year: {year or 'Unknown'}<br>"
            f"Citations: {ccount:,}"
        )

        # Add referenced papers as grey nodes
        refs = paper.get("references", []) or []
        for ref in refs[:12]:
            ref_id = ref.get("paperId") or ref.get("paper_id", "")
            ref_title = (ref.get("title", "Unknown") or "Unknown")[:45]
            if not ref_id or ref_id == pid:
                continue
            if ref_id not in G:
                G.add_node(ref_id)
                node_labels[ref_id] = ref_title
                node_colors[ref_id] = "#95A5A6"
                node_sizes[ref_id] = 8
                hover_texts[ref_id] = (
                    f"<b>{ref_title}</b><br>(Referenced by {title[:30]})"
                )
            G.add_edge(pid, ref_id)

    if G.number_of_nodes() == 0:
        return None, "No citation data with valid paper IDs found."

    n = G.number_of_nodes()
    k = max(0.5, 2.0 / (n ** 0.5))
    pos = nx.spring_layout(G, seed=42, k=k, iterations=80)

    fig = _make_plotly_figure(
        G,
        pos,
        node_labels,
        node_colors,
        node_sizes,
        hover_texts,
        title=f"Citation Network — {n} papers · {G.number_of_edges()} citation links",
        directed=True,
    )

    # Legend
    for label, color in [
        ("Seed papers", "#27AE60"),
        ("Referenced papers", "#95A5A6"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                name=label,
                showlegend=True,
            )
        )

    status = (
        f"Citation Network: {n} papers, {G.number_of_edges()} citation links. "
        "Green = seed papers, grey = referenced. Node size reflects citation count."
    )
    return fig, status
