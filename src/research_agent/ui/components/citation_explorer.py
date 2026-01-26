"""Gradio components for the Citation Explorer tab."""

import gradio as gr

from research_agent.tools.academic_search import AcademicSearchTools
from research_agent.tools.citation_explorer import CitationExplorer


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

        # Network visualization (placeholder for now)
        with gr.Accordion("🌐 Network Visualization", open=False):
            gr.Plot(label="Citation Network Graph")
            gr.Markdown("*Network visualization coming soon!*")

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
            ],
        )

        clear_btn.click(
            fn=lambda: ["", None, None, None, None],
            outputs=[
                summary_output,
                citing_output,
                cited_output,
                connected_output,
                related_output,
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
    }


async def explore_citations(paper_input: str, direction: str, depth: int):
    """Explore citation relationships for a paper."""
    if not paper_input.strip():
        return "Please enter a paper ID or title.", None, None, None, None

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
                return f"No papers found for: {paper_input}", None, None, None, None
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

        await search_tools.close()

        return summary, citing_df, cited_df, connected_df, related_df

    except Exception as e:
        error_msg = f"Error exploring citations: {str(e)}"
        return error_msg, None, None, None, None


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


__all__ = ["render_citation_explorer", "explore_citations", "_papers_to_dataframe"]
