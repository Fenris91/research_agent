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

    with gr.Blocks(title="Research Assistant", theme=gr.themes.Soft()) as app:
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
                height=500, placeholder="Ask me about your research topic..."
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="What are the key theories in urban anthropology?",
                    label="Your question",
                    scale=4,
                )
                submit = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                clear = gr.Button("Clear Chat")

            with gr.Accordion("Settings", open=False):
                gr.Markdown("### LLM Model")
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=["Loading..."],
                        value="Loading...",
                        label="Select Model",
                        scale=3,
                        interactive=True,
                    )
                    refresh_models_btn = gr.Button("🔄", scale=1, min_width=50)

                current_model_display = gr.Textbox(
                    label="Current Model",
                    interactive=False,
                    placeholder="No model loaded",
                )

                gr.Markdown("### Search Settings")
                search_depth = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Max external results per source",
                )
                auto_ingest = gr.Checkbox(
                    label="Automatically add high-quality sources to knowledge base",
                    value=False,
                )

        with gr.Tab("Knowledge Base"):
            gr.Markdown("## Your Research Library")

            with gr.Row():
                kb_stats = gr.JSON(
                    label="Statistics",
                    value={"total_papers": 0, "total_notes": 0, "total_web_sources": 0},
                )
                refresh_btn = gr.Button("Refresh")

            gr.Markdown("### Add Papers")

            with gr.Row():
                upload_pdf = gr.File(
                    label="Upload Documents",
                    file_types=[".pdf", ".txt", ".md", ".docx"],
                    file_count="multiple",
                )
                upload_btn = gr.Button("Process & Add", variant="primary")

            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Upload PDFs to add to your knowledge base",
            )

            gr.Markdown("### Browse Papers")
            papers_table = gr.Dataframe(
                headers=["Title", "Year", "Authors", "Added", "Paper ID"],
                label="Papers in Knowledge Base",
            )

            with gr.Row():
                delete_paper_id = gr.Textbox(
                    label="Paper ID",
                    placeholder="Enter paper ID to delete",
                    scale=4,
                )
                delete_paper_btn = gr.Button("Delete", variant="stop", scale=1)

            delete_status = gr.Textbox(
                label="Delete Status",
                interactive=False,
                placeholder="Enter a paper ID to delete",
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
                        lines=6,
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Options")
                    use_openalex = gr.Checkbox(label="OpenAlex", value=True)
                    use_semantic_scholar = gr.Checkbox(
                        label="Semantic Scholar", value=True
                    )
                    use_web_search = gr.Checkbox(label="Web Search", value=True)

            with gr.Row():
                lookup_btn = gr.Button("Lookup Researchers", variant="primary", scale=2)
                clear_results_btn = gr.Button("Clear Results", scale=1)

            lookup_status = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Enter researcher names and click 'Lookup Researchers'",
            )

            gr.Markdown("### Results")

            results_table = gr.Dataframe(
                headers=[
                    "Name",
                    "Affiliations",
                    "Works",
                    "Citations",
                    "H-Index",
                    "Fields",
                ],
                label="Researcher Profiles",
                interactive=False,
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

        with gr.Tab("Citation Explorer"):
            from research_agent.ui.components import render_citation_explorer

            render_citation_explorer()

        with gr.Tab("Data Analysis"):
            gr.Markdown("## Analyze Your Data")

            data_input = gr.File(
                label="Upload CSV or Excel file", file_types=[".csv", ".xlsx", ".xls"]
            )

            analysis_type = gr.Radio(
                choices=[
                    "Descriptive Statistics",
                    "Correlation Analysis",
                    "Frequency Analysis",
                    "Custom Query",
                ],
                label="Analysis Type",
                value="Descriptive Statistics",
            )

            custom_query = gr.Textbox(
                label="Custom analysis request",
                placeholder="e.g., 'Show me the distribution of ages by region'",
                visible=True,
            )

            analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Row():
                analysis_output = gr.Markdown(label="Results")

            analysis_plot = gr.Plot(label="Visualization")

        # Event handlers

        def get_available_models():
            """Get list of available Ollama models."""
            if agent is None or not agent.use_ollama:
                return ["No Ollama models available"]
            try:
                models = agent.list_available_models()
                if models:
                    # Sort with preferred models first
                    preferred = [
                        "qwen3:32b",
                        "qwen2.5-coder:32b",
                        "mistral-small3.2:latest",
                    ]
                    sorted_models = []
                    for p in preferred:
                        if p in models:
                            sorted_models.append(p)
                    for m in models:
                        if m not in sorted_models:
                            sorted_models.append(m)
                    return sorted_models
                return ["No models found"]
            except Exception as e:
                logger.error(f"Error getting models: {e}")
                return ["Error loading models"]

        def get_current_model():
            """Get the currently active model name."""
            if agent is None:
                return "No agent loaded"
            return agent.get_current_model()

        def switch_model(model_name):
            """Switch to a different model."""
            if agent is None:
                return "No agent loaded"
            if agent.switch_model(model_name):
                return f"✓ Switched to: {model_name}"
            return f"✗ Failed to switch to: {model_name}"

        def refresh_model_list():
            """Refresh the model dropdown and current model display."""
            models = get_available_models()
            current = get_current_model()
            # Return: dropdown choices, dropdown value, current model display
            return gr.update(choices=models, value=current), current

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
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return "", history

        vector_store = None
        embedder = None
        processor = None
        reranker = None
        rerank_top_k = None
        _config_cache = None

        def _load_config():
            nonlocal _config_cache
            if _config_cache is not None:
                return _config_cache

            config_path = Path("configs/config.yaml")
            if not config_path.exists():
                _config_cache = {}
                return _config_cache

            try:
                import yaml

                with config_path.open("r", encoding="utf-8") as f:
                    _config_cache = yaml.safe_load(f) or {}
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Failed to load config: {e}")
                _config_cache = {}
            return _config_cache

        def _get_kb_resources():
            nonlocal vector_store, embedder, processor, reranker, rerank_top_k

            if vector_store is None:
                from research_agent.db.vector_store import ResearchVectorStore

                cfg = _load_config()

                if reranker is None:
                    try:
                        from research_agent.models.reranker import (
                            load_reranker_from_config,
                        )

                        reranker = load_reranker_from_config(cfg)
                        rerank_top_k = (
                            cfg.get("retrieval", {}).get("rerank_top_k")
                            if isinstance(cfg, dict)
                            else None
                        )
                    except Exception as e:  # pragma: no cover - optional
                        logger.warning(f"Reranker unavailable: {e}")
                        reranker = None
                        rerank_top_k = None

                vector_store = ResearchVectorStore(
                    reranker=reranker, rerank_top_k=rerank_top_k
                )

            if embedder is None:
                from research_agent.db.embeddings import get_embedder

                embedder = get_embedder()

            if processor is None:
                from research_agent.processors.document_processor import (
                    DocumentProcessor,
                )

                processor = DocumentProcessor()

            return vector_store, embedder, processor

        def _format_papers_table(papers):
            if not papers:
                return []
            table_data = []
            for paper in papers:
                table_data.append(
                    [
                        paper.get("title", "Unknown"),
                        paper.get("year", ""),
                        paper.get("authors", ""),
                        paper.get("added_at", ""),
                        paper.get("paper_id", ""),
                    ]
                )
            return table_data

        def refresh_stats_and_table():
            """Refresh knowledge base statistics and paper list."""
            store, _, _ = _get_kb_resources()
            stats = store.get_stats()
            papers = store.list_papers(limit=200)
            return stats, _format_papers_table(papers)

        def ingest_documents(files):
            """Process and add documents to the knowledge base."""
            if not files:
                stats, table = refresh_stats_and_table()
                return "No files selected.", stats, table

            store, embedder_model, doc_processor = _get_kb_resources()
            added = 0
            skipped = 0
            errors = []

            for file_obj in files:
                try:
                    if isinstance(file_obj, str):
                        file_path = file_obj
                    elif isinstance(file_obj, dict) and "name" in file_obj:
                        file_path = file_obj["name"]
                    else:
                        file_path = file_obj.name

                    doc = doc_processor.process_document(file_path)
                    paper_id = doc.metadata.get("doi") or doc.doc_id

                    if store.get_paper(paper_id):
                        skipped += 1
                        continue

                    chunk_texts = [chunk.text for chunk in doc.chunks]
                    embeddings = embedder_model.embed_documents(
                        chunk_texts, batch_size=32, show_progress=False
                    )
                    store.add_paper(paper_id, chunk_texts, embeddings, doc.metadata)
                    added += 1
                except Exception as e:
                    errors.append(f"{getattr(file_obj, 'name', file_obj)}: {e}")

            stats, table = refresh_stats_and_table()

            status_parts = [f"Added {added} document(s)"]
            if skipped:
                status_parts.append(f"Skipped {skipped} duplicate(s)")
            if errors:
                status_parts.append("Errors: " + "; ".join(errors[:3]))
                if len(errors) > 3:
                    status_parts.append(f"(+{len(errors) - 3} more)")

            return ". ".join(status_parts), stats, table

        def delete_paper(paper_id):
            """Delete a paper from the knowledge base."""
            if not paper_id:
                stats, table = refresh_stats_and_table()
                return "Enter a paper ID to delete.", stats, table

            store, _, _ = _get_kb_resources()
            deleted = store.delete_paper(paper_id)
            status = (
                f"Deleted paper {paper_id}."
                if deleted
                else f"Paper not found: {paper_id}."
            )
            stats, table = refresh_stats_and_table()
            return status, stats, table

        def lookup_researchers(names_text, use_oa, use_s2, use_web):
            """Look up researcher profiles."""
            from research_agent.tools.researcher_file_parser import (
                parse_researchers_text,
            )
            from research_agent.tools.researcher_lookup import ResearcherLookup

            names = parse_researchers_text(names_text)

            if not names:
                return ("No valid names found", [], [], None)

            # Create lookup instance
            lookup = ResearcherLookup(
                use_openalex=use_oa,
                use_semantic_scholar=use_s2,
                use_web_search=use_web,
                request_delay=0.5,
            )

            # Run async lookup
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                profiles = loop.run_until_complete(lookup.lookup_batch(names))
                loop.run_until_complete(lookup.close())
            except Exception as e:
                logger.error(f"Lookup error: {e}")
                return (f"Error during lookup: {str(e)}", [], [], None)

            # Format results for table
            table_data = []
            web_results = {}

            for p in profiles:
                table_data.append(
                    [
                        p.name,
                        "; ".join(p.affiliations) if p.affiliations else "",
                        p.works_count,
                        f"{p.citations_count:,}",
                        p.h_index if p.h_index else "",
                        "; ".join(p.fields[:3]) if p.fields else "",
                    ]
                )

                if p.web_results:
                    web_results[p.name] = p.web_results

            status = f"Found {len(profiles)} researcher profiles"

            # Store full profiles for export
            full_results = [p.to_dict() for p in profiles]

            return (
                status,
                table_data,
                full_results,
                web_results if web_results else None,
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
            writer.writerow(
                [
                    "Name",
                    "Affiliations",
                    "Works",
                    "Citations",
                    "H-Index",
                    "Fields",
                    "OpenAlex ID",
                    "S2 ID",
                ]
            )

            # Data
            for r in results:
                writer.writerow(
                    [
                        r.get("name", ""),
                        "; ".join(r.get("affiliations", [])),
                        r.get("works_count", 0),
                        r.get("citations_count", 0),
                        r.get("h_index", ""),
                        "; ".join(r.get("fields", [])[:5]),
                        r.get("openalex_id", ""),
                        r.get("semantic_scholar_id", ""),
                    ]
                )

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
        refresh_btn.click(refresh_stats_and_table, outputs=[kb_stats, papers_table])
        upload_btn.click(
            ingest_documents,
            inputs=[upload_pdf],
            outputs=[upload_status, kb_stats, papers_table],
        )
        delete_paper_btn.click(
            delete_paper,
            inputs=[delete_paper_id],
            outputs=[delete_status, kb_stats, papers_table],
        )

        # Model selector events
        refresh_models_btn.click(
            refresh_model_list, outputs=[model_dropdown, current_model_display]
        )
        model_dropdown.change(
            switch_model, inputs=[model_dropdown], outputs=[current_model_display]
        )

        # Initialize model list on load
        app.load(refresh_model_list, outputs=[model_dropdown, current_model_display])

        # Initialize knowledge base stats/table on load
        app.load(refresh_stats_and_table, outputs=[kb_stats, papers_table])

        # Researcher lookup events
        lookup_btn.click(
            lookup_researchers,
            inputs=[
                researcher_input,
                use_openalex,
                use_semantic_scholar,
                use_web_search,
            ],
            outputs=[
                lookup_status,
                results_table,
                researcher_results_state,
                web_results_output,
            ],
        )

        clear_results_btn.click(
            clear_results,
            outputs=[
                lookup_status,
                results_table,
                researcher_results_state,
                web_results_output,
            ],
        )

        export_csv_btn.click(
            export_to_csv, inputs=[researcher_results_state], outputs=[csv_download]
        )

        export_json_btn.click(
            export_to_json, inputs=[researcher_results_state], outputs=[json_download]
        )

    return app


def launch_app(agent=None, port: int = 7860, share: bool = False):
    """
    Launch the Gradio app.

    Args:
        agent: ResearchAgent instance
        port: Port to run on
        share: Create public link
    """
    app = create_app(agent)
    app.launch(server_port=port, share=share, show_error=True)


if __name__ == "__main__":
    # Launch with agent
    import os

    # Check for Ollama preference via environment variable
    # Default: use Ollama with qwen3:32b (most capable), falls back to mistral-small3.2
    use_ollama = os.getenv("USE_OLLAMA", "true").lower() == "true"
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:32b")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    agent = None
    try:
        from research_agent.agents.research_agent import ResearchAgent

        print("Initializing Research Agent...")
        if use_ollama:
            print(f"Using Ollama model: {ollama_model}")
            agent = ResearchAgent(
                use_ollama=True, ollama_model=ollama_model, ollama_base_url=ollama_url
            )
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
