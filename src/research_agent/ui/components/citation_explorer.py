"""Gradio components for the Citation Explorer tab."""

from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import networkx as nx

from research_agent.tools.academic_search import AcademicSearchTools
from research_agent.tools.citation_explorer import (
    CitationExplorer,
    CitationNetwork,
    AuthorNetwork,
)
from research_agent.tools.researcher_registry import get_researcher_registry


def render_citation_explorer():
    """Render citation explorer UI component and wire events."""
    with gr.Column():
        gr.Markdown("## 🧬 Citation Network Explorer")
        gr.Markdown(
            "Explore citation relationships for papers or researchers from your lookup history."
        )

        # Researcher selection section
        with gr.Accordion("👤 Explore by Researcher", open=True):
            gr.Markdown(
                "_Select a researcher from your lookup history to explore their citation network._"
            )
            with gr.Row():
                researcher_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Researcher",
                    interactive=True,
                    scale=3,
                )
                refresh_researchers_btn = gr.Button(
                    "🔄 Refresh", scale=1, min_width=100
                )

            researcher_papers_table = gr.Dataframe(
                headers=["Title", "Year", "Citations", "Paper ID"],
                datatype=["str", "number", "number", "str"],
                interactive=True,  # Allow selection/copying
                wrap=True,
                column_widths=["50%", "10%", "10%", "30%"],
                label="Researcher's Top Papers (click to copy Paper ID)",
                visible=True,
            )

            explore_researcher_btn = gr.Button(
                "🔍 Explore Researcher's Citation Network",
                variant="primary",
            )

        gr.Markdown("---")
        gr.Markdown("### Or explore a specific paper:")

        with gr.Row():
            with gr.Column(scale=2):
                paper_input = gr.Textbox(
                    label="Paper ID or Title",
                    placeholder="Enter Semantic Scholar paper ID or search by title",
                    value="",
                )

            with gr.Column(scale=1):
                direction = gr.Radio(
                    choices=["both", "citing", "cited"],
                    value="both",
                    label="Direction",
                    info="What citation relationships to explore",
                )

                depth = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Search Depth",
                    info="Maximum papers to fetch per direction",
                )

        with gr.Row():
            search_btn = gr.Button("🔍 Explore Citations", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

        # Results sections
        with gr.Accordion("📊 Citation Network Summary", open=True):
            summary_output = gr.Markdown(
                "Enter a paper ID to explore its citation network."
            )

        with gr.Tabs():
            with gr.TabItem("📄 Papers That Cite This"):
                citing_output = gr.DataFrame(
                    headers=["Save", "Title", "Year", "Citations", "Paper ID"],
                    datatype=["bool", "str", "number", "number", "str"],
                    interactive=True,
                    wrap=True,
                    column_widths=["5%", "45%", "10%", "10%", "30%"],
                )
                with gr.Row():
                    citing_save_btn = gr.Button("Save checked to KB")
                    citing_clear_btn = gr.Button("Clear checks")

            with gr.TabItem("📚 Papers Cited by This"):
                cited_output = gr.DataFrame(
                    headers=["Save", "Title", "Year", "Citations", "Paper ID"],
                    datatype=["bool", "str", "number", "number", "str"],
                    interactive=True,
                    wrap=True,
                    column_widths=["5%", "45%", "10%", "10%", "30%"],
                )
                with gr.Row():
                    cited_save_btn = gr.Button("Save checked to KB")
                    cited_clear_btn = gr.Button("Clear checks")

            with gr.TabItem("⭐ Highly Connected Papers"):
                connected_output = gr.DataFrame(
                    headers=["Save", "Title", "Year", "Citations", "Paper ID"],
                    datatype=["bool", "str", "number", "number", "str"],
                    interactive=True,
                    wrap=True,
                    column_widths=["5%", "45%", "10%", "10%", "30%"],
                )
                with gr.Row():
                    connected_save_btn = gr.Button("Save checked to KB")
                    connected_clear_btn = gr.Button("Clear checks")

            with gr.TabItem("🔗 Related Papers"):
                related_output = gr.DataFrame(
                    headers=["Save", "Title", "Year", "Citations", "Paper ID"],
                    datatype=["bool", "str", "number", "number", "str"],
                    interactive=True,
                    wrap=True,
                    column_widths=["5%", "45%", "10%", "10%", "30%"],
                )
                with gr.Row():
                    related_save_btn = gr.Button("Save checked to KB")
                    related_clear_btn = gr.Button("Clear checks")

        # Network visualization
        with gr.Accordion("🌐 Network Visualization", open=False):
            network_plot = gr.Plot(label="Citation Network Graph")

        kb_save_status = gr.Markdown("")

        # Event handlers
        search_btn.click(
            fn=explore_citations,
            inputs=[paper_input, direction, depth],
            outputs=[
                summary_output,
                citing_output,
                cited_output,
                connected_output,
                related_output,
                network_plot,
            ],
        )

        clear_btn.click(
            fn=lambda: ["", None, None, None, None, None],
            outputs=[
                summary_output,
                citing_output,
                cited_output,
                connected_output,
                related_output,
                network_plot,
            ],
        )

        citing_clear_btn.click(
            fn=_clear_checks, inputs=[citing_output], outputs=[citing_output]
        )
        cited_clear_btn.click(
            fn=_clear_checks, inputs=[cited_output], outputs=[cited_output]
        )
        connected_clear_btn.click(
            fn=_clear_checks, inputs=[connected_output], outputs=[connected_output]
        )
        related_clear_btn.click(
            fn=_clear_checks, inputs=[related_output], outputs=[related_output]
        )

        citing_save_btn.click(
            fn=save_selected_to_kb, inputs=[citing_output], outputs=[kb_save_status]
        )
        cited_save_btn.click(
            fn=save_selected_to_kb, inputs=[cited_output], outputs=[kb_save_status]
        )
        connected_save_btn.click(
            fn=save_selected_to_kb, inputs=[connected_output], outputs=[kb_save_status]
        )
        related_save_btn.click(
            fn=save_selected_to_kb, inputs=[related_output], outputs=[kb_save_status]
        )

        # Researcher exploration events
        refresh_researchers_btn.click(
            fn=refresh_researcher_dropdown,
            outputs=[researcher_dropdown],
        )

        researcher_dropdown.change(
            fn=on_researcher_selected,
            inputs=[researcher_dropdown],
            outputs=[researcher_papers_table],
        )

        explore_researcher_btn.click(
            fn=explore_researcher_network,
            inputs=[researcher_dropdown, depth],
            outputs=[
                summary_output,
                citing_output,
                cited_output,
                connected_output,
                related_output,
                network_plot,
            ],
        )

    return {
        "paper_input": paper_input,
        "direction": direction,
        "depth": depth,
        "search_btn": search_btn,
        "summary_output": summary_output,
        "citing_output": citing_output,
        "cited_output": cited_output,
        "connected_output": connected_output,
        "related_output": related_output,
        "network_plot": network_plot,
        "kb_save_status": kb_save_status,
        "researcher_dropdown": researcher_dropdown,
        "researcher_papers_table": researcher_papers_table,
        "refresh_researchers_btn": refresh_researchers_btn,
    }


async def explore_citations(paper_input: str, direction: str, depth: int):
    """Explore citation relationships for a paper."""
    if not paper_input.strip():
        return "Please enter a paper ID or title.", None, None, None, None, None

    try:
        search_tools = AcademicSearchTools()
        explorer = CitationExplorer(search_tools)

        paper_input = paper_input.strip()
        paper_id = None

        # Check if it looks like a known paper ID format
        is_doi = paper_input.startswith("10.")
        is_s2_id = (
            len(paper_input) == 40 and paper_input.isalnum()
        )  # S2 IDs are 40-char hex
        is_openalex_id = paper_input.startswith("W") and paper_input[1:].isdigit()

        if is_doi or is_s2_id:
            # Use directly as Semantic Scholar paper ID
            paper_id = paper_input
        elif is_openalex_id:
            # OpenAlex ID - need to search by it or convert
            # Try searching Semantic Scholar by the OpenAlex work
            papers = await search_tools.search_semantic_scholar(paper_input, limit=1)
            if papers:
                paper_id = papers[0].id
            else:
                # OpenAlex IDs don't work directly with S2, search by title would be needed
                await search_tools.close()
                return (
                    f"OpenAlex ID '{paper_input}' not found in Semantic Scholar. Try searching by paper title instead.",
                    None,
                    None,
                    None,
                    None,
                    None,
                )
        else:
            # Treat as title search
            papers = await search_tools.search_semantic_scholar(paper_input, limit=5)
            if papers:
                # Try to find best match
                paper_id = papers[0].id
                # Show what we found
                found_title = (
                    papers[0].title[:50] + "..."
                    if len(papers[0].title) > 50
                    else papers[0].title
                )
            else:
                await search_tools.close()
                return (
                    f"No papers found for: '{paper_input}'. Try a different search term.",
                    None,
                    None,
                    None,
                    None,
                    None,
                )

        network = await explorer.get_citations(paper_id, direction, depth)

        summary = f"""## 📊 Citation Network Summary

**Mode:** Paper

**Seed Paper:** {network.seed_paper.title}
- **Year:** {network.seed_paper.year or "Unknown"}
- **Citations:** {network.seed_paper.citation_count or 0}

**Network Statistics:**
- 📄 Papers that cite this: {len(network.citing_papers)}
- 📚 Papers cited by this: {len(network.cited_papers)}
- ⭐ Highly connected papers: {len(network.highly_connected)}
"""

        if explorer.rate_limited:
            summary = (
                "> ⚠️ Semantic Scholar rate limit reached. Results may be incomplete. "
                "Wait a few minutes and retry.\n\n" + summary
            )

        citing_df = _papers_to_dataframe(network.citing_papers)
        cited_df = _papers_to_dataframe(network.cited_papers)
        connected_df = _papers_to_dataframe(network.highly_connected)

        related_papers = await explorer.suggest_related(paper_id, limit=depth)
        related_df = _papers_to_dataframe(related_papers)

        # Generate network visualization
        network_fig = _render_network_graph(network)

        await search_tools.close()

        return summary, citing_df, cited_df, connected_df, related_df, network_fig

    except Exception as e:
        error_msg = f"Error exploring citations: {str(e)}"
        return error_msg, None, None, None, None, None


def _clear_checks(table_data):
    if table_data is None:
        return None

    if hasattr(table_data, "values"):
        rows = table_data.values.tolist()
    else:
        rows = list(table_data)

    updated = []
    for row in rows:
        row_list = list(row)
        if row_list:
            row_list[0] = False
        updated.append(row_list)

    return updated


def _load_config():
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        return {}

    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _get_kb_resources():
    from research_agent.db.embeddings import get_embedder
    from research_agent.db.vector_store import ResearchVectorStore

    cfg = _load_config()
    embed_cfg = cfg.get("embedding", {})
    vec_cfg = cfg.get("vector_store", {})

    embedder = get_embedder(
        model_name=embed_cfg.get("name", "BAAI/bge-base-en-v1.5"),
        device=embed_cfg.get("device"),
    )
    store = ResearchVectorStore(
        persist_dir=vec_cfg.get("persist_directory", "./data/chroma_db")
    )
    return store, embedder


async def save_selected_to_kb(table_data):
    if table_data is None:
        return "No table data available."

    if hasattr(table_data, "values"):
        rows = table_data.values.tolist()
    else:
        rows = list(table_data)

    def _is_checked(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return False

    selected = [row for row in rows if len(row) > 0 and _is_checked(row[0])]
    if not selected:
        return "Check one or more rows first."

    store, embedder = _get_kb_resources()
    search_tools = AcademicSearchTools()

    added = 0
    skipped = 0
    errors = []

    for row in selected:
        paper_id = row[4] if len(row) > 4 else None
        if not paper_id:
            skipped += 1
            continue

        if store.get_paper(paper_id):
            skipped += 1
            continue

        try:
            paper = await search_tools.get_paper_details(
                paper_id, source="semantic_scholar"
            )
            if paper:
                title = paper.title or "Unknown"
                abstract = paper.abstract or ""
                venue = paper.venue or ""
                authors = paper.authors or []
                fields = paper.fields or []
                citations = paper.citations
                doi = paper.doi
                source = paper.source
            else:
                title = row[1] or "Unknown"
                abstract = ""
                venue = ""
                authors = []
                fields = []
                citations = row[3] if len(row) > 3 else None
                doi = None
                source = "semantic_scholar"

            parts = [title]
            if abstract:
                parts.append(f"Abstract: {abstract}")
            if venue:
                parts.append(f"Venue: {venue}")
            if fields:
                parts.append(f"Fields: {', '.join(fields)}")
            if doi:
                parts.append(f"DOI: {doi}")

            content = "\n".join(parts).strip()
            if not content:
                skipped += 1
                continue

            embeddings = embedder.embed_documents(
                [content], batch_size=1, show_progress=False
            )

            metadata = {
                "title": title,
                "year": paper.year if paper else row[2],
                "venue": venue,
                "citations": citations,
                "doi": doi,
                "fields": fields,
                "authors": authors,
                "source": source,
                "ingest_source": "citation_explorer",
            }

            store.add_paper(paper_id, [content], embeddings, metadata)
            added += 1
        except Exception as e:
            errors.append(f"{paper_id}: {e}")

    await search_tools.close()

    status_parts = [f"Added {added} paper(s)"]
    if skipped:
        status_parts.append(f"Skipped {skipped} duplicate/invalid")
    if errors:
        status_parts.append("Errors: " + "; ".join(errors[:3]))
        if len(errors) > 3:
            status_parts.append(f"(+{len(errors) - 3} more)")

    return ". ".join(status_parts)


def _papers_to_dataframe(papers):
    """Convert list of CitationPaper objects to DataFrame-friendly rows."""
    if not papers:
        return None

    data = []
    for paper in papers:
        data.append(
            [
                False,
                paper.title or "Unknown Title",
                paper.year or "Unknown",
                paper.citation_count or 0,
                paper.paper_id,
            ]
        )

    return data


def _render_network_graph(network: CitationNetwork):
    """Render a citation network as a matplotlib figure."""
    if not network:
        return None

    G = nx.DiGraph()

    # Add seed node
    seed_label = _truncate_title(network.seed_paper.title, 30)
    G.add_node("seed", label=seed_label, node_type="seed")

    # Add citing papers (papers that cite the seed)
    for i, paper in enumerate(network.citing_papers[:15]):  # Limit for readability
        node_id = f"citing_{i}"
        label = _truncate_title(paper.title, 25)
        G.add_node(node_id, label=label, node_type="citing")
        G.add_edge(node_id, "seed")  # Arrow points to seed (they cite it)

    # Add cited papers (papers cited by the seed)
    for i, paper in enumerate(network.cited_papers[:15]):  # Limit for readability
        node_id = f"cited_{i}"
        label = _truncate_title(paper.title, 25)
        G.add_node(node_id, label=label, node_type="cited")
        G.add_edge("seed", node_id)  # Arrow from seed (it cites them)

    if len(G.nodes()) <= 1:
        # No connections found
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No citation relationships found",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Color map by node type
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        node_type = G.nodes[node].get("node_type", "other")
        if node_type == "seed":
            node_colors.append("#FF6B6B")  # Red for seed
            node_sizes.append(800)
        elif node_type == "citing":
            node_colors.append("#4ECDC4")  # Teal for citing
            node_sizes.append(400)
        else:  # cited
            node_colors.append("#1F4E79")  # Dark blue for cited
            node_sizes.append(400)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="#CCCCCC", arrows=True, arrowsize=15, alpha=0.7
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.9
    )

    # Draw labels
    labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

    # Legend
    legend_elements = [
        plt.scatter([], [], c="#FF6B6B", s=100, label="Seed Paper"),
        plt.scatter([], [], c="#4ECDC4", s=60, label="Papers Citing This"),
        plt.scatter([], [], c="#1F4E79", s=60, label="Papers Cited By This"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_title("Citation Network", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    return fig


def _truncate_title(title: str, max_len: int = 30) -> str:
    """Truncate a title to a maximum length."""
    if not title:
        return "Unknown"
    if len(title) <= max_len:
        return title
    return title[: max_len - 3] + "..."


def refresh_researcher_dropdown():
    """Refresh the researcher dropdown with current registry contents."""
    import gradio as gr

    registry = get_researcher_registry()
    researchers = registry.list_all()  # Get all researchers, not just those with papers

    if not researchers:
        return gr.update(choices=[], value=None)

    choices = []
    for r in researchers:
        papers_count = len(r.top_papers)
        if papers_count > 0:
            label = f"{r.name} ({papers_count} papers, {r.citations_count:,} citations)"
        else:
            label = f"{r.name} ({r.citations_count:,} citations) - no papers fetched"
        choices.append((label, r.name))

    return gr.update(
        choices=choices,
        value=choices[0][1] if choices else None,
    )


def on_researcher_selected(researcher_name: str):
    """Handle researcher selection - show their papers."""
    if not researcher_name:
        return None

    registry = get_researcher_registry()
    profile = registry.get(researcher_name)

    if not profile or not profile.top_papers:
        return None

    # Format papers for dataframe
    data = []
    for paper in profile.top_papers:
        data.append(
            [
                paper.title or "Unknown",
                paper.year or "Unknown",
                paper.citation_count or 0,
                paper.paper_id,
            ]
        )

    return data


async def explore_researcher_network(researcher_name: str, depth: int):
    """Explore citation network for a researcher."""
    import logging

    logger = logging.getLogger(__name__)

    if not researcher_name:
        return "Please select a researcher first.", None, None, None, None, None

    registry = get_researcher_registry()
    profile = registry.get(researcher_name)

    if not profile:
        return (
            f"Researcher '{researcher_name}' not found in registry.",
            None,
            None,
            None,
            None,
            None,
        )

    if not profile.top_papers:
        return (
            f"No papers found for {researcher_name}. Go back to Researcher Lookup and enable 'Fetch Papers' checkbox, then look them up again.",
            None,
            None,
            None,
            None,
            None,
        )

    try:
        logger.info(
            f"Exploring network for {researcher_name} with {len(profile.top_papers)} papers"
        )
        search_tools = AcademicSearchTools()
        explorer = CitationExplorer(search_tools)

        # Explore author network - limit to 3 papers for speed
        network = await explorer.explore_author_network(
            profile,
            papers_limit=min(3, len(profile.top_papers)),
            citations_per_paper=min(depth // 2, 10),  # Cap at 10 for speed
        )

        logger.info(
            f"Got network: {len(network.papers_citing_author)} citing, {len(network.papers_cited_by_author)} cited"
        )

        # Build summary
        summary = f"""## 👤 {network.author_name}'s Citation Network

**Author's Papers Analyzed:** {len(network.author_papers)}
**Total Citations Received:** {network.total_citations_received:,}

**Network Statistics:**
- 📄 Papers citing their work: {network.unique_citing_papers}
- 📚 Papers they cite: {network.unique_references}
- ⭐ Highly connected papers: {len(network.highly_connected)}

### Top Papers by {network.author_name}:
"""
        for i, paper in enumerate(network.author_papers[:5], 1):
            citations = paper.citation_count or 0
            summary += f"{i}. **{paper.title}** ({paper.year or 'N/A'}) - {citations:,} citations\n"

        # Format dataframes
        citing_df = _papers_to_dataframe(network.papers_citing_author[:depth])
        cited_df = _papers_to_dataframe(network.papers_cited_by_author[:depth])
        connected_df = _papers_to_dataframe(network.highly_connected)

        # Skip suggest_related for speed - it makes many additional API calls
        related_df = None

        # Generate network visualization
        network_fig = _render_author_network_graph(network)

        await search_tools.close()

        return summary, citing_df, cited_df, connected_df, related_df, network_fig

    except Exception as e:
        import traceback

        traceback.print_exc()
        error_msg = f"Error exploring researcher network: {str(e)}"
        return error_msg, None, None, None, None, None


def _render_author_network_graph(network: AuthorNetwork):
    """Render an author's citation network as a matplotlib figure."""
    if not network:
        return None

    G = nx.DiGraph()

    # Add author's papers as central cluster
    for i, paper in enumerate(network.author_papers[:8]):
        node_id = f"author_{i}"
        label = _truncate_title(paper.title, 25)
        G.add_node(node_id, label=label, node_type="author_paper")

    # Add citing papers
    for i, paper in enumerate(network.papers_citing_author[:12]):
        node_id = f"citing_{i}"
        label = _truncate_title(paper.title, 20)
        G.add_node(node_id, label=label, node_type="citing")
        # Connect to first author paper
        if network.author_papers:
            G.add_edge(node_id, "author_0")

    # Add cited papers
    for i, paper in enumerate(network.papers_cited_by_author[:12]):
        node_id = f"cited_{i}"
        label = _truncate_title(paper.title, 20)
        G.add_node(node_id, label=label, node_type="cited")
        # Connect from first author paper
        if network.author_papers:
            G.add_edge("author_0", node_id)

    if len(G.nodes()) <= 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            f"No citation data found for {network.author_name}",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.axis("off")
        return fig

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # Color map by node type
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        node_type = G.nodes[node].get("node_type", "other")
        if node_type == "author_paper":
            node_colors.append("#FF6B6B")  # Red for author's papers
            node_sizes.append(700)
        elif node_type == "citing":
            node_colors.append("#4ECDC4")  # Teal for citing
            node_sizes.append(350)
        else:  # cited
            node_colors.append("#45B7D1")  # Blue for cited
            node_sizes.append(350)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="#CCCCCC", arrows=True, arrowsize=12, alpha=0.6
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.9
    )

    # Draw labels
    labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=6)

    # Legend
    legend_elements = [
        plt.scatter(
            [], [], c="#FF6B6B", s=100, label=f"{network.author_name}'s Papers"
        ),
        plt.scatter([], [], c="#4ECDC4", s=60, label="Papers Citing Their Work"),
        plt.scatter([], [], c="#45B7D1", s=60, label="Papers They Cite"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_title(
        f"Citation Network for {network.author_name}", fontsize=14, fontweight="bold"
    )
    ax.axis("off")
    plt.tight_layout()

    return fig


__all__ = [
    "render_citation_explorer",
    "explore_citations",
    "explore_researcher_network",
    "refresh_researcher_dropdown",
    "on_researcher_selected",
    "_papers_to_dataframe",
    "_render_network_graph",
    "_render_author_network_graph",
]
