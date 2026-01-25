"""
Research Agent UI

Gradio-based interface for the research assistant.
"""

import asyncio
import csv
import io
import json
import logging
from pathlib import Path
from typing import List, Optional

import gradio as gr

logger = logging.getLogger(__name__)


def create_app(agent=None):
    """
    Create the Gradio application.

    Args:
        agent: ResearchAgent instance (optional for testing UI)

    Returns:
        Gradio Blocks app
    """

    with gr.Blocks(
        title="Research Assistant",
        theme=gr.themes.Soft()
    ) as app:

        gr.Markdown("""
        # Research Assistant

        Social sciences research helper with autonomous knowledge building.

        **Capabilities:**
        - Literature review and paper discovery
        - Paper summarization
        - Web search for grey literature
        - Researcher profile lookup
        - Data analysis
        """)

        with gr.Tab("Research Chat"):
            chatbot = gr.Chatbot(
                height=500,
                placeholder="Ask me about your research topic..."
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="What are the key theories in urban anthropology?",
                    label="Your question",
                    scale=4
                )
                submit = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                clear = gr.Button("Clear Chat")

            with gr.Accordion("Settings", open=False):
                search_depth = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Max external results per source"
                )
                auto_ingest = gr.Checkbox(
                    label="Automatically add high-quality sources to knowledge base",
                    value=False
                )

        with gr.Tab("Knowledge Base"):
            gr.Markdown("## Your Research Library")

            with gr.Row():
                kb_stats = gr.JSON(
                    label="Statistics",
                    value={
                        "total_papers": 0,
                        "total_notes": 0,
                        "total_web_sources": 0
                    }
                )
                refresh_btn = gr.Button("Refresh")

            gr.Markdown("### Add Papers")

            with gr.Row():
                upload_pdf = gr.File(
                    label="Upload PDFs",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                upload_btn = gr.Button("Process & Add", variant="primary")

            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Upload PDFs to add to your knowledge base"
            )

            gr.Markdown("### Browse Papers")
            papers_table = gr.Dataframe(
                headers=["Title", "Year", "Authors", "Added"],
                label="Papers in Knowledge Base"
            )

        with gr.Tab("Researcher Lookup"):
            gr.Markdown("""
            ## Researcher Profile Lookup

            Look up citation data, publications, and web presence for researchers.

            **Data Sources:**
            - OpenAlex (open academic database)
            - Semantic Scholar (citation data, h-index)
            - DuckDuckGo (web presence)
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    researcher_input = gr.Textbox(
                        label="Researcher Names",
                        placeholder="Enter names separated by commas or newlines:\n\nDavid Harvey\nDoreen Massey, Tim Ingold\nAnna Tsing",
                        lines=6
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Options")
                    use_openalex = gr.Checkbox(label="OpenAlex", value=True)
                    use_semantic_scholar = gr.Checkbox(label="Semantic Scholar", value=True)
                    use_web_search = gr.Checkbox(label="Web Search", value=True)

            with gr.Row():
                lookup_btn = gr.Button("Lookup Researchers", variant="primary", scale=2)
                clear_results_btn = gr.Button("Clear Results", scale=1)

            lookup_status = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Enter researcher names and click 'Lookup Researchers'"
            )

            gr.Markdown("### Results")

            results_table = gr.Dataframe(
                headers=["Name", "Affiliations", "Works", "Citations", "H-Index", "Fields"],
                label="Researcher Profiles",
                interactive=False
            )

            with gr.Accordion("Web Results", open=False):
                web_results_output = gr.JSON(label="Web Search Results")

            with gr.Row():
                export_csv_btn = gr.Button("Export CSV")
                export_json_btn = gr.Button("Export JSON")

            csv_download = gr.File(label="Download CSV", visible=False)
            json_download = gr.File(label="Download JSON", visible=False)

            # State to store full results
            researcher_results_state = gr.State([])

        with gr.Tab("Data Analysis"):
            gr.Markdown("## Analyze Your Data")

            data_input = gr.File(
                label="Upload CSV or Excel file",
                file_types=[".csv", ".xlsx", ".xls"]
            )

            analysis_type = gr.Radio(
                choices=[
                    "Descriptive Statistics",
                    "Correlation Analysis",
                    "Frequency Analysis",
                    "Custom Query"
                ],
                label="Analysis Type",
                value="Descriptive Statistics"
            )

            custom_query = gr.Textbox(
                label="Custom analysis request",
                placeholder="e.g., 'Show me the distribution of ages by region'",
                visible=True
            )

            analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Row():
                analysis_output = gr.Markdown(label="Results")

            analysis_plot = gr.Plot(label="Visualization")

        # Event handlers
        def respond(message, history):
            """Handle chat messages."""
            if agent is None:
                # Demo mode
                response = f"[Demo mode] You asked: {message}\n\nThe agent is not loaded. Run with a real agent to get responses."
            else:
                # Real mode - call the agent
                try:
                    result = agent.run(message)
                    response = result.get("answer", "No response generated")
                except Exception as e:
                    response = f"Error: {str(e)}"

            # Append in Gradio's message format
            history.append({
                "role": "user",
                "content": message
            })
            history.append({
                "role": "assistant",
                "content": response
            })
            return "", history

        def refresh_stats():
            """Refresh knowledge base statistics."""
            # TODO: Get real stats from vector store
            return {
                "total_papers": 0,
                "total_notes": 0,
                "total_web_sources": 0
            }

        def lookup_researchers(names_text, use_oa, use_s2, use_web):
            """Look up researcher profiles."""
            from src.tools.researcher_file_parser import parse_researchers_text
            from src.tools.researcher_lookup import ResearcherLookup

            names = parse_researchers_text(names_text)

            if not names:
                return (
                    "No valid names found",
                    [],
                    [],
                    None
                )

            # Create lookup instance
            lookup = ResearcherLookup(
                use_openalex=use_oa,
                use_semantic_scholar=use_s2,
                use_web_search=use_web,
                request_delay=0.5
            )

            # Run async lookup
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                profiles = loop.run_until_complete(lookup.lookup_batch(names))
                loop.run_until_complete(lookup.close())
            except Exception as e:
                logger.error(f"Lookup error: {e}")
                return (
                    f"Error during lookup: {str(e)}",
                    [],
                    [],
                    None
                )

            # Format results for table
            table_data = []
            web_results = {}

            for p in profiles:
                table_data.append([
                    p.name,
                    "; ".join(p.affiliations) if p.affiliations else "",
                    p.works_count,
                    f"{p.citations_count:,}",
                    p.h_index if p.h_index else "",
                    "; ".join(p.fields[:3]) if p.fields else ""
                ])

                if p.web_results:
                    web_results[p.name] = p.web_results

            status = f"Found {len(profiles)} researcher profiles"

            # Store full profiles for export
            full_results = [p.to_dict() for p in profiles]

            return (
                status,
                table_data,
                full_results,
                web_results if web_results else None
            )

        def clear_results():
            """Clear researcher lookup results."""
            return "", [], [], None

        def export_to_csv(results):
            """Export results to CSV."""
            if not results:
                return None

            output = io.StringIO()
            writer = csv.writer(output)

            # Header
            writer.writerow([
                "Name", "Affiliations", "Works", "Citations",
                "H-Index", "Fields", "OpenAlex ID", "S2 ID"
            ])

            # Data
            for r in results:
                writer.writerow([
                    r.get("name", ""),
                    "; ".join(r.get("affiliations", [])),
                    r.get("works_count", 0),
                    r.get("citations_count", 0),
                    r.get("h_index", ""),
                    "; ".join(r.get("fields", [])[:5]),
                    r.get("openalex_id", ""),
                    r.get("semantic_scholar_id", "")
                ])

            # Save to temp file
            csv_path = Path("/tmp/researchers.csv")
            csv_path.write_text(output.getvalue())

            return str(csv_path)

        def export_to_json(results):
            """Export results to JSON."""
            if not results:
                return None

            json_path = Path("/tmp/researchers.json")
            json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

            return str(json_path)

        # Wire up events
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], outputs=[chatbot])
        refresh_btn.click(refresh_stats, outputs=[kb_stats])

        # Researcher lookup events
        lookup_btn.click(
            lookup_researchers,
            inputs=[researcher_input, use_openalex, use_semantic_scholar, use_web_search],
            outputs=[lookup_status, results_table, researcher_results_state, web_results_output]
        )

        clear_results_btn.click(
            clear_results,
            outputs=[lookup_status, results_table, researcher_results_state, web_results_output]
        )

        export_csv_btn.click(
            export_to_csv,
            inputs=[researcher_results_state],
            outputs=[csv_download]
        )

        export_json_btn.click(
            export_to_json,
            inputs=[researcher_results_state],
            outputs=[json_download]
        )

    return app


def launch_app(
    agent=None,
    port: int = 7860,
    share: bool = False
):
    """
    Launch the Gradio app.

    Args:
        agent: ResearchAgent instance
        port: Port to run on
        share: Create public link
    """
    app = create_app(agent)
    app.launch(
        server_port=port,
        share=share,
        show_error=True
    )

if __name__ == "__main__":
    # Launch with agent
    import sys
    import os
    
    # Ensure we can import from src
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")
    
    # Check for Ollama preference via environment variable
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    ollama_model = os.getenv("OLLAMA_MODEL", "mistral")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    agent = None
    try:
        from src.agents.research_agent import ResearchAgent
        
        print("Initializing Research Agent...")
        if use_ollama:
            print(f"Using Ollama model: {ollama_model}")
            agent = ResearchAgent(use_ollama=True, ollama_model=ollama_model, ollama_base_url=ollama_url)
        else:
            print("Using HuggingFace models (set USE_OLLAMA=true to use Ollama)")
            agent = ResearchAgent()
        print("✓ Agent loaded successfully")
        launch_app(agent=agent)
    except Exception as e:
        print(f"⚠️ Failed to load agent: {e}")
        import traceback
        traceback.print_exc()
        print("\nLaunching in demo mode...")
        launch_app(agent=None)
