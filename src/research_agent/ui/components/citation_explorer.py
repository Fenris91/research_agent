"""Gradio components for the Citation Explorer tab."""

import gradio as gr
import matplotlib.pyplot as plt
import networkx as nx

from research_agent.tools.academic_search import AcademicSearchTools
from research_agent.tools.citation_explorer import CitationExplorer, CitationNetwork


def render_citation_explorer():
    """Render citation explorer UI component and wire events."""
    with gr.Column():
        gr.Markdown("## 🧬 Citation Network Explorer")
        gr.Markdown(
            "Explore citation relationships and discover influential papers in academic networks."
        )

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
                    headers=["Title", "Year", "Citations", "Paper ID"],
                    datatype=["str", "number", "number", "str"],
                    interactive=False,
                )

            with gr.TabItem("📚 Papers Cited by This"):
                cited_output = gr.DataFrame(
                    headers=["Title", "Year", "Citations", "Paper ID"],
                    datatype=["str", "number", "number", "str"],
                    interactive=False,
                )

            with gr.TabItem("⭐ Highly Connected Papers"):
                connected_output = gr.DataFrame(
                    headers=["Title", "Year", "Citations", "Paper ID"],
                    datatype=["str", "number", "number", "str"],
                    interactive=False,
                )

            with gr.TabItem("🔗 Related Papers"):
                related_output = gr.DataFrame(
                    headers=["Title", "Year", "Citations", "Paper ID"],
                    datatype=["str", "number", "number", "str"],
                    interactive=False,
                )

        # Network visualization
        with gr.Accordion("🌐 Network Visualization", open=False):
            network_plot = gr.Plot(label="Citation Network Graph")

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
    }


async def explore_citations(paper_input: str, direction: str, depth: int):
    """Explore citation relationships for a paper."""
    if not paper_input.strip():
        return "Please enter a paper ID or title.", None, None, None, None, None

    try:
        search_tools = AcademicSearchTools()
        explorer = CitationExplorer(search_tools)

        # If it's not a DOI/paper ID format, try to search by title
        if not paper_input.startswith("10.") and len(paper_input) < 20:
            papers = await search_tools.search_semantic_scholar(paper_input, limit=1)
            if papers:
                paper_id = papers[0].id
            else:
                await search_tools.close()
                return f"No papers found for: {paper_input}", None, None, None, None, None
        else:
            paper_id = paper_input

        network = await explorer.get_citations(paper_id, direction, depth)

        summary = f"""## 📊 Citation Network Summary

**Seed Paper:** {network.seed_paper.title}
- **Year:** {network.seed_paper.year or "Unknown"}
- **Citations:** {network.seed_paper.citation_count or 0}

**Network Statistics:**
- 📄 Papers that cite this: {len(network.citing_papers)}
- 📚 Papers cited by this: {len(network.cited_papers)}
- ⭐ Highly connected papers: {len(network.highly_connected)}
"""

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


def _papers_to_dataframe(papers):
    """Convert list of CitationPaper objects to DataFrame-friendly rows."""
    if not papers:
        return None

    data = []
    for paper in papers:
        data.append(
            [
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
            node_colors.append("#45B7D1")  # Blue for cited
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
        plt.scatter([], [], c="#45B7D1", s=60, label="Papers Cited By This"),
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


__all__ = [
    "render_citation_explorer",
    "explore_citations",
    "_papers_to_dataframe",
    "_render_network_graph",
]
