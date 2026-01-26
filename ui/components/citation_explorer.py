import gradio as gr
from research_agent.tools.citation_explorer import CitationExplorer
from research_agent.tools.academic_search import AcademicSearchTools


def render_citation_explorer():
    with gr.Tab("🧬 Citation Network"):
        gr.Markdown("## Explore Academic Citations")

        with gr.Row():
            paper_id = gr.Textbox(
                label="Paper ID", placeholder="Enter paper ID from OpenAlex/S2"
            )
            depth = gr.Slider(label="Search Depth", minimum=1, maximum=3, value=2)
            submit_btn = gr.Button("Generate Network")

        result = gr.Markdown(label="Citation Graph")
        network = gr.Plot(label="Citation Visualization")

        submit_btn.click(
            fn=generate_citation_network,
            inputs=[paper_id, depth],
            outputs=[result, network],
        )

        return paper_id, result, network


def generate_citation_network(paper_id, depth):
    explorer = CitationExplorer(academic_search=AcademicSearchTools())
    network = explorer.get_citations(paper_id, direction="both")

    # Simple textual representation first
    output = f"Citations for {paper_id}:"
