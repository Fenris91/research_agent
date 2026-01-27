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

            gr.Markdown("### Search Filters")
            with gr.Row():
                year_from_chat = gr.Slider(
                    minimum=1900,
                    maximum=2030,
                    value=1900,
                    step=1,
                    label="From Year",
                    info="Use 1900 for no lower bound",
                )
                year_to_chat = gr.Slider(
                    minimum=1900,
                    maximum=2030,
                    value=2030,
                    step=1,
                    label="To Year",
                    info="Use 2030 for no upper bound",
                )
            min_citations_chat = gr.Slider(
                minimum=0,
                maximum=1000,
                value=0,
                step=10,
                label="Min citations",
                info="0 to disable",
            )

            gr.Markdown("### Reranker (Retrieval)")
            with gr.Row():
                reranker_enable_chat = gr.Checkbox(
                    label="Enable reranker (BGE)",
                    value=False,
                    info="Improves ranking of retrieved chunks",
                )
                rerank_topk_chat = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Rerank top-k",
                    info="How many results to rerank",
                )
            rerank_status_chat = gr.Textbox(
                label="Reranker Status",
                interactive=False,
                placeholder="Using config defaults",
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

            with gr.Row():
                year_from_kb = gr.Slider(
                    minimum=1900,
                    maximum=2030,
                    value=1900,
                    step=1,
                    label="From Year",
                    info="1900 = no lower bound",
                )
                year_to_kb = gr.Slider(
                    minimum=1900,
                    maximum=2030,
                    value=2030,
                    step=1,
                    label="To Year",
                    info="2030 = no upper bound",
                )
            min_citations_kb = gr.Slider(
                minimum=0,
                maximum=1000,
                value=0,
                step=10,
                label="Min citations",
                info="0 = no filter",
            )

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

            gr.Markdown("### Export")
            with gr.Row():
                export_bibtex_btn = gr.Button("Export BibTeX", variant="secondary")
                bibtex_download = gr.File(label="Download", visible=False)
            export_status = gr.Textbox(
                label="Export Status",
                interactive=False,
                placeholder="Click Export to generate BibTeX file",
            )

            with gr.Accordion("Retrieval Settings", open=False):
                gr.Markdown("Reranker settings (shared with Chat)")
                with gr.Row():
                    reranker_enable_kb = gr.Checkbox(
                        label="Enable reranker (BGE)", value=False
                    )
                    rerank_topk_kb = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Rerank top-k",
                    )
                rerank_status_kb = gr.Textbox(
                    label="Reranker Status",
                    interactive=False,
                    placeholder="Using config defaults",
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

            with gr.Row():
                data_input = gr.File(
                    label="Upload CSV or Excel file",
                    file_types=[".csv", ".xlsx", ".xls"],
                    scale=2,
                )
                with gr.Column(scale=1):
                    data_info = gr.Textbox(
                        label="Data Info",
                        interactive=False,
                        placeholder="Upload a file to see info",
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    analysis_type = gr.Radio(
                        choices=[
                            "Descriptive Statistics",
                            "Correlation Analysis",
                            "Frequency Analysis",
                            "Pivot Table",
                            "Time Series",
                            "Custom Query",
                        ],
                        label="Analysis Type",
                        value="Descriptive Statistics",
                    )

                with gr.Column(scale=1):
                    plot_type = gr.Radio(
                        choices=["Histogram", "Box Plot", "Bar Chart", "Line Chart", "Scatter"],
                        label="Plot Type",
                        value="Histogram",
                    )

            with gr.Row():
                column_select = gr.Dropdown(
                    choices=[],
                    label="Select Column(s)",
                    multiselect=True,
                    interactive=True,
                )
                group_by_col = gr.Dropdown(
                    choices=[],
                    label="Group By (for Pivot/Comparison)",
                    multiselect=False,
                    interactive=True,
                )

            custom_query = gr.Textbox(
                label="Custom analysis request",
                placeholder="e.g., 'Show me the distribution of ages by region'",
                visible=True,
            )

            with gr.Row():
                analyze_btn = gr.Button("Analyze", variant="primary", scale=2)
                download_plot_btn = gr.Button("Download Plot", scale=1)

            analysis_output = gr.Markdown(label="Results")

            analysis_plot = gr.Plot(label="Visualization")
            plot_download = gr.File(label="Download", visible=False)

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

        def respond(message, history, year_from, year_to, min_citations):
            """Handle chat messages."""
            if agent is None:
                # Demo mode
                response = f"[Demo mode] You asked: {message}\n\nThe agent is not loaded. Run with a real agent to get responses."
            else:
                # Real mode - call the agent
                try:
                    filters = {
                        "year_from": int(year_from) if year_from else None,
                        "year_to": int(year_to) if year_to else None,
                        "min_citations": int(min_citations) if min_citations else None,
                    }
                    result = agent.run(message, search_filters=filters)
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

        # Shared reranker settings
        reranker_enabled = None
        rerank_top_k = None

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
            nonlocal \
                vector_store, \
                embedder, \
                processor, \
                reranker, \
                rerank_top_k, \
                reranker_enabled

            if vector_store is None:
                from research_agent.db.vector_store import ResearchVectorStore

                cfg = _load_config()

                if reranker_enabled is None:
                    reranker_enabled_default = (
                        cfg.get("embedding", {})
                        .get("reranker", {})
                        .get("enabled", False)
                    )
                    reranker_enabled = bool(reranker_enabled_default)

                if rerank_top_k is None:
                    retrieval_cfg = cfg.get("retrieval", {}) if isinstance(cfg, dict) else {}
                    rerank_top_k_cfg = retrieval_cfg.get("rerank_top_k")
                    if rerank_top_k_cfg is None:
                        rerank_top_k_cfg = retrieval_cfg.get("top_k")
                    rerank_top_k = rerank_top_k_cfg

                if reranker is None:
                    try:
                        from research_agent.models.reranker import (
                            load_reranker_from_config,
                        )

                        candidate = load_reranker_from_config(cfg)
                        reranker = candidate if reranker_enabled else None
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

        def _set_reranker_settings(enabled: bool, top_k: int | None):
            """Update reranker settings and propagate to vector store."""
            nonlocal reranker_enabled, rerank_top_k, reranker, vector_store

            reranker_enabled = bool(enabled)
            rerank_top_k = int(top_k) if top_k else None

            if reranker_enabled and reranker is None:
                cfg = _load_config()
                try:
                    from research_agent.models.reranker import load_reranker_from_config

                    reranker = load_reranker_from_config(cfg)
                except Exception as e:  # pragma: no cover - optional
                    logger.warning(f"Reranker unavailable: {e}")
                    reranker = None
                    rerank_top_k = None

            if vector_store is not None:
                vector_store.reranker = reranker if reranker_enabled else None
                vector_store.rerank_top_k = rerank_top_k

            status = (
                f"Reranker enabled (top_k={rerank_top_k or 'all'})"
                if reranker_enabled and reranker is not None
                else "Reranker disabled"
            )

            return (
                gr.update(value=reranker_enabled),
                gr.update(value=rerank_top_k or 10),
                gr.update(value=reranker_enabled),
                gr.update(value=rerank_top_k or 10),
                status,
                status,
            )

        def _format_papers_table(papers, year_from=None, year_to=None, min_citations=0):
            if not papers:
                return []
            table_data = []
            for paper in papers:
                year = paper.get("year")
                citations = paper.get("citations") or paper.get("citation_count")
                if year_from and year and year < year_from:
                    continue
                if year_to and year and year > year_to:
                    continue
                if (
                    min_citations
                    and citations is not None
                    and citations < min_citations
                ):
                    continue
                table_data.append(
                    [
                        paper.get("title", "Unknown"),
                        year or "",
                        paper.get("authors", ""),
                        paper.get("added_at", ""),
                        paper.get("paper_id", ""),
                    ]
                )
            return table_data

        def refresh_stats_and_table(year_from=None, year_to=None, min_citations=0):
            """Refresh knowledge base statistics and paper list."""
            store, _, _ = _get_kb_resources()
            stats = store.get_stats()
            papers = store.list_papers(limit=500)
            return stats, _format_papers_table(
                papers,
                year_from=year_from if year_from and year_from > 1900 else None,
                year_to=year_to if year_to and year_to < 2030 else None,
                min_citations=min_citations or 0,
            )

        def ingest_documents(files, year_from=None, year_to=None, min_citations=0):
            """Process and add documents to the knowledge base."""
            if not files:
                stats, table = refresh_stats_and_table(
                    year_from, year_to, min_citations
                )
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

            stats, table = refresh_stats_and_table(year_from, year_to, min_citations)

            status_parts = [f"Added {added} document(s)"]
            if skipped:
                status_parts.append(f"Skipped {skipped} duplicate(s)")
            if errors:
                status_parts.append("Errors: " + "; ".join(errors[:3]))
                if len(errors) > 3:
                    status_parts.append(f"(+{len(errors) - 3} more)")

            return ". ".join(status_parts), stats, table

        def delete_paper(paper_id, year_from=None, year_to=None, min_citations=0):
            """Delete a paper from the knowledge base."""
            if not paper_id:
                stats, table = refresh_stats_and_table(
                    year_from, year_to, min_citations
                )
                return "Enter a paper ID to delete.", stats, table

            store, _, _ = _get_kb_resources()
            deleted = store.delete_paper(paper_id)
            status = (
                f"Deleted paper {paper_id}."
                if deleted
                else f"Paper not found: {paper_id}."
            )
            stats, table = refresh_stats_and_table(year_from, year_to, min_citations)
            return status, stats, table

        def export_bibtex():
            """Export all papers in knowledge base to BibTeX format."""
            store, _, _ = _get_kb_resources()
            papers = store.list_papers(limit=10000)

            if not papers:
                return "No papers in knowledge base to export.", None

            bibtex_entries = []
            for paper in papers:
                entry = _paper_to_bibtex(paper)
                if entry:
                    bibtex_entries.append(entry)

            if not bibtex_entries:
                return "No papers could be converted to BibTeX.", None

            bibtex_content = "\n\n".join(bibtex_entries)

            # Write to temp file
            bibtex_path = Path("/tmp/research_agent_export.bib")
            bibtex_path.write_text(bibtex_content, encoding="utf-8")

            return f"Exported {len(bibtex_entries)} papers to BibTeX.", str(bibtex_path)

        def _paper_to_bibtex(paper: dict) -> str:
            """Convert paper metadata to BibTeX entry."""
            paper_id = paper.get("paper_id", "unknown")
            title = paper.get("title", "Unknown Title")
            year = paper.get("year")
            authors = paper.get("authors", "")

            # Generate citation key from first author + year
            if authors:
                first_author = authors.split(",")[0].strip()
                first_author_key = "".join(
                    c for c in first_author.split()[-1] if c.isalnum()
                ).lower()
            else:
                first_author_key = "unknown"

            year_str = str(year) if year else "nd"
            cite_key = f"{first_author_key}{year_str}"

            # Build BibTeX entry
            lines = [f"@article{{{cite_key},"]
            lines.append(f'  title = {{{title}}},')

            if authors:
                # Convert "First Last, First Last" to "Last, First and Last, First"
                author_list = [a.strip() for a in authors.split(",")]
                bibtex_authors = " and ".join(author_list)
                lines.append(f'  author = {{{bibtex_authors}}},')

            if year:
                lines.append(f"  year = {{{year}}},")

            # Add DOI if it looks like the paper_id is a DOI
            if paper_id.startswith("10."):
                lines.append(f'  doi = {{{paper_id}}},')

            lines.append("}")

            return "\n".join(lines)

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

        # Store loaded dataframe in state
        _analysis_df = {"df": None, "path": None}
        _current_fig = {"fig": None}

        def _load_data_file(file_obj):
            """Load CSV/Excel file and return dataframe."""
            import pandas as pd

            if file_obj is None:
                return None, None

            if isinstance(file_obj, str):
                file_path = file_obj
            elif isinstance(file_obj, dict) and "name" in file_obj:
                file_path = file_obj["name"]
            else:
                file_path = file_obj.name

            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path, parse_dates=True, infer_datetime_format=True)
            else:
                df = pd.read_excel(file_path, parse_dates=True)

            # Try to parse date columns
            for col in df.columns:
                if df[col].dtype == "object":
                    try:
                        import pandas as pd
                        parsed = pd.to_datetime(df[col], errors="coerce")
                        if parsed.notna().sum() > len(df) * 0.5:
                            df[col] = parsed
                    except Exception:
                        pass

            return df, file_path

        def on_file_upload(file_obj):
            """Handle file upload - update dropdowns and info."""
            if file_obj is None:
                _analysis_df["df"] = None
                _analysis_df["path"] = None
                return (
                    "No file uploaded",
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=None),
                )

            df, path = _load_data_file(file_obj)
            if df is None or df.empty:
                return (
                    "Failed to load file or file is empty",
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=None),
                )

            _analysis_df["df"] = df
            _analysis_df["path"] = path

            # Build column info
            cols = list(df.columns)
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            info = f"**Rows:** {len(df)} | **Cols:** {len(cols)}\n"
            info += f"Numeric: {len(numeric_cols)} | Date: {len(date_cols)} | Text: {len(cat_cols)}"

            return (
                info,
                gr.update(choices=cols, value=numeric_cols[:2] if numeric_cols else cols[:2]),
                gr.update(choices=["(None)"] + cols, value="(None)"),
            )

        def analyze_data(file_obj, analysis_type, plot_type, columns, group_by, custom_query):
            """Analyze uploaded CSV/Excel data."""
            import pandas as pd
            import matplotlib.pyplot as plt

            if _analysis_df["df"] is None:
                if file_obj is None:
                    return "Please upload a CSV or Excel file.", None
                df, _ = _load_data_file(file_obj)
                if df is None:
                    return "Failed to load file.", None
            else:
                df = _analysis_df["df"]

            if df.empty:
                return "The uploaded file is empty.", None

            # Handle group_by
            grp = group_by if group_by and group_by != "(None)" else None

            # Handle columns selection
            cols = columns if columns else []

            try:
                if analysis_type == "Descriptive Statistics":
                    result, fig = _descriptive_stats(df, cols, plot_type)
                elif analysis_type == "Correlation Analysis":
                    result, fig = _correlation_analysis(df, cols)
                elif analysis_type == "Frequency Analysis":
                    result, fig = _frequency_analysis(df, cols, plot_type)
                elif analysis_type == "Pivot Table":
                    result, fig = _pivot_table(df, cols, grp, plot_type)
                elif analysis_type == "Time Series":
                    result, fig = _time_series(df, cols, plot_type)
                elif analysis_type == "Custom Query":
                    result, fig = _custom_analysis(df, custom_query)
                else:
                    result, fig = "Unknown analysis type.", None

                _current_fig["fig"] = fig
                return result, fig

            except Exception as e:
                logger.error(f"Data analysis error: {e}")
                return f"Error analyzing data: {str(e)}", None

        def download_current_plot():
            """Save current plot to file for download."""
            import matplotlib.pyplot as plt

            if _current_fig["fig"] is None:
                return None

            plot_path = Path("/tmp/analysis_plot.png")
            _current_fig["fig"].savefig(plot_path, dpi=150, bbox_inches="tight")
            return str(plot_path)

        def _descriptive_stats(df, columns, plot_type):
            """Generate descriptive statistics."""
            import matplotlib.pyplot as plt

            # Use selected columns or all numeric
            if columns:
                numeric_cols = [c for c in columns if c in df.select_dtypes(include=["number"]).columns]
            else:
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            if not numeric_cols:
                return "No numeric columns found in selection.", None

            numeric_df = df[numeric_cols]
            stats = numeric_df.describe().round(2)

            # Format as markdown table
            result = "### Descriptive Statistics\n\n"
            result += f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}\n\n"
            result += "| Statistic | " + " | ".join(stats.columns) + " |\n"
            result += "|---" * (len(stats.columns) + 1) + "|\n"

            for idx in stats.index:
                row_vals = [str(stats.loc[idx, col]) for col in stats.columns]
                result += f"| {idx} | " + " | ".join(row_vals) + " |\n"

            # Create plot based on type
            col = numeric_cols[0]
            fig, ax = plt.subplots(figsize=(10, 6))

            if plot_type == "Histogram":
                ax.hist(numeric_df[col].dropna(), bins=20, edgecolor="black", alpha=0.7)
                ax.set_ylabel("Frequency")
            elif plot_type == "Box Plot":
                numeric_df.boxplot(ax=ax)
                ax.set_ylabel("Value")
            elif plot_type == "Bar Chart":
                numeric_df.mean().plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
                ax.set_ylabel("Mean")
            elif plot_type == "Line Chart":
                numeric_df.plot(ax=ax)
            elif plot_type == "Scatter" and len(numeric_cols) >= 2:
                ax.scatter(numeric_df[numeric_cols[0]], numeric_df[numeric_cols[1]], alpha=0.6)
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel(numeric_cols[1])
            else:
                ax.hist(numeric_df[col].dropna(), bins=20, edgecolor="black", alpha=0.7)
                ax.set_ylabel("Frequency")

            ax.set_title(f"Distribution of {col}" if plot_type == "Histogram" else f"Analysis of {', '.join(numeric_cols[:3])}")
            plt.tight_layout()

            return result, fig

        def _correlation_analysis(df, columns):
            """Generate correlation matrix."""
            import matplotlib.pyplot as plt
            import numpy as np

            # Use selected columns or all numeric
            if columns:
                numeric_cols = [c for c in columns if c in df.select_dtypes(include=["number"]).columns]
                if len(numeric_cols) >= 2:
                    numeric_df = df[numeric_cols]
                else:
                    numeric_df = df.select_dtypes(include=["number"])
            else:
                numeric_df = df.select_dtypes(include=["number"])

            if len(numeric_df.columns) < 2:
                return "Need at least 2 numeric columns for correlation.", None

            # Calculate correlation
            corr = numeric_df.corr().round(3)

            # Format as markdown
            result = "### Correlation Matrix\n\n"
            result += "| | " + " | ".join(corr.columns) + " |\n"
            result += "|---" * (len(corr.columns) + 1) + "|\n"

            for idx in corr.index:
                row_vals = [str(corr.loc[idx, col]) for col in corr.columns]
                result += f"| {idx} | " + " | ".join(row_vals) + " |\n"

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

            ax.set_xticks(np.arange(len(corr.columns)))
            ax.set_yticks(np.arange(len(corr.index)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.index)

            # Add correlation values
            for i in range(len(corr.index)):
                for j in range(len(corr.columns)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

            plt.colorbar(im, ax=ax, label="Correlation")
            ax.set_title("Correlation Heatmap")
            plt.tight_layout()

            return result, fig

        def _frequency_analysis(df, columns, plot_type):
            """Generate frequency counts for categorical columns."""
            import matplotlib.pyplot as plt

            # Use selected columns or categorical
            if columns:
                cat_cols = [c for c in columns if c in df.columns]
            else:
                cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            if not cat_cols:
                cat_cols = [df.columns[0]]

            result = "### Frequency Analysis\n\n"

            for col in cat_cols[:3]:
                counts = df[col].value_counts().head(10)
                result += f"**{col}** (top 10):\n\n"
                result += "| Value | Count | % |\n|---|---|---|\n"
                total = len(df)
                for val, cnt in counts.items():
                    pct = (cnt / total) * 100
                    result += f"| {val} | {cnt} | {pct:.1f}% |\n"
                result += "\n"

            # Plot first column
            col = cat_cols[0]
            counts = df[col].value_counts().head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            if plot_type == "Bar Chart" or plot_type == "Histogram":
                counts.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
            elif plot_type == "Line Chart":
                counts.plot(kind="line", ax=ax, marker="o")
            else:
                counts.plot(kind="barh", ax=ax, edgecolor="black", alpha=0.7)

            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.set_title(f"Frequency of {col}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            return result, fig

        def _pivot_table(df, columns, group_by, plot_type):
            """Generate pivot table analysis."""
            import matplotlib.pyplot as plt

            if not group_by:
                return "Please select a 'Group By' column for pivot table.", None

            # Get value column (first numeric from selection or first numeric overall)
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if columns:
                value_cols = [c for c in columns if c in numeric_cols]
            else:
                value_cols = numeric_cols[:1]

            if not value_cols:
                return "Need at least one numeric column for pivot table values.", None

            value_col = value_cols[0]

            # Create pivot
            pivot = df.groupby(group_by)[value_col].agg(["mean", "sum", "count"]).round(2)
            pivot = pivot.head(15)  # Limit rows

            result = f"### Pivot Table: {value_col} by {group_by}\n\n"
            result += "| " + group_by + " | Mean | Sum | Count |\n"
            result += "|---|---|---|---|\n"

            for idx in pivot.index:
                result += f"| {idx} | {pivot.loc[idx, 'mean']} | {pivot.loc[idx, 'sum']} | {int(pivot.loc[idx, 'count'])} |\n"

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            if plot_type == "Bar Chart" or plot_type == "Histogram":
                pivot["mean"].plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
                ax.set_ylabel(f"Mean {value_col}")
            elif plot_type == "Line Chart":
                pivot["mean"].plot(kind="line", ax=ax, marker="o")
                ax.set_ylabel(f"Mean {value_col}")
            elif plot_type == "Box Plot":
                df.boxplot(column=value_col, by=group_by, ax=ax)
                ax.set_ylabel(value_col)
            else:
                pivot["mean"].plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
                ax.set_ylabel(f"Mean {value_col}")

            ax.set_title(f"{value_col} by {group_by}")
            ax.set_xlabel(group_by)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            return result, fig

        def _time_series(df, columns, plot_type):
            """Generate time series analysis."""
            import matplotlib.pyplot as plt

            # Find date column
            date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
            if not date_cols:
                return "No date/time columns detected. Upload data with dates or ensure date format is recognized.", None

            date_col = date_cols[0]

            # Get numeric columns to plot
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if columns:
                plot_cols = [c for c in columns if c in numeric_cols]
            else:
                plot_cols = numeric_cols[:3]

            if not plot_cols:
                return "No numeric columns to plot over time.", None

            # Sort by date
            df_sorted = df.sort_values(date_col)

            result = f"### Time Series Analysis\n\n"
            result += f"**Date column:** {date_col}\n"
            result += f"**Date range:** {df_sorted[date_col].min()} to {df_sorted[date_col].max()}\n"
            result += f"**Plotting:** {', '.join(plot_cols)}\n\n"

            # Stats per column
            for col in plot_cols:
                result += f"**{col}:** min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}\n"

            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))

            for col in plot_cols:
                if plot_type == "Line Chart" or plot_type == "Histogram":
                    ax.plot(df_sorted[date_col], df_sorted[col], label=col, marker="." if len(df) < 50 else "")
                elif plot_type == "Scatter":
                    ax.scatter(df_sorted[date_col], df_sorted[col], label=col, alpha=0.6)
                elif plot_type == "Bar Chart":
                    # Resample to fewer points for bar chart
                    ax.bar(range(len(df_sorted)), df_sorted[col].values, label=col, alpha=0.7)
                else:
                    ax.plot(df_sorted[date_col], df_sorted[col], label=col)

            ax.set_xlabel(date_col)
            ax.set_ylabel("Value")
            ax.set_title("Time Series")
            ax.legend()
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            return result, fig

        def _custom_analysis(df, query):
            """Handle custom analysis queries."""
            import matplotlib.pyplot as plt

            if not query or not query.strip():
                return "Please enter a custom query.", None

            query_lower = query.lower()

            # Simple query parsing
            result = f"### Custom Analysis: {query}\n\n"

            # Check for common patterns
            if "distribution" in query_lower or "histogram" in query_lower:
                # Find mentioned column
                col = _find_column_in_query(df, query)
                if col:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df[col].hist(ax=ax, bins=20, edgecolor="black", alpha=0.7)
                    ax.set_title(f"Distribution of {col}")
                    ax.set_xlabel(col)
                    result += f"Showing distribution of **{col}**\n\n"
                    result += df[col].describe().to_markdown()
                    return result, fig

            elif "compare" in query_lower or "by" in query_lower or "group" in query_lower:
                # Try to find two columns to compare
                cols = _find_columns_in_query(df, query, n=2)
                if len(cols) >= 2:
                    grouped = df.groupby(cols[0])[cols[1]].mean()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    grouped.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.7)
                    ax.set_title(f"Mean {cols[1]} by {cols[0]}")
                    result += grouped.to_markdown()
                    return result, fig

            elif "trend" in query_lower or "over time" in query_lower:
                # Look for date column
                date_cols = df.select_dtypes(include=["datetime64"]).columns
                if len(date_cols) > 0:
                    numeric_cols = df.select_dtypes(include=["number"]).columns
                    if len(numeric_cols) > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df.plot(x=date_cols[0], y=numeric_cols[0], ax=ax)
                        ax.set_title(f"{numeric_cols[0]} over time")
                        result += "Showing trend over time."
                        return result, fig

            # Default: show summary
            result += "**Data Summary:**\n\n"
            result += f"- Rows: {len(df)}\n"
            result += f"- Columns: {len(df.columns)}\n"
            result += f"- Column names: {', '.join(df.columns)}\n\n"
            result += "**Tip:** Try queries like:\n"
            result += "- 'Show distribution of [column]'\n"
            result += "- 'Compare [column1] by [column2]'\n"

            return result, None

        def _find_column_in_query(df, query):
            """Find a column name mentioned in the query."""
            query_lower = query.lower()
            for col in df.columns:
                if col.lower() in query_lower:
                    return col
            # Return first numeric column as fallback
            numeric = df.select_dtypes(include=["number"]).columns
            return numeric[0] if len(numeric) > 0 else df.columns[0]

        def _find_columns_in_query(df, query, n=2):
            """Find multiple column names in query."""
            query_lower = query.lower()
            found = []
            for col in df.columns:
                if col.lower() in query_lower:
                    found.append(col)
                    if len(found) >= n:
                        break
            return found

        # Wire up events
        msg.submit(
            respond,
            [msg, chatbot, year_from_chat, year_to_chat, min_citations_chat],
            [msg, chatbot],
        )
        submit.click(
            respond,
            [msg, chatbot, year_from_chat, year_to_chat, min_citations_chat],
            [msg, chatbot],
        )
        clear.click(lambda: [], outputs=[chatbot])
        refresh_btn.click(
            refresh_stats_and_table,
            inputs=[year_from_kb, year_to_kb, min_citations_kb],
            outputs=[kb_stats, papers_table],
        )
        year_from_kb.change(
            refresh_stats_and_table,
            inputs=[year_from_kb, year_to_kb, min_citations_kb],
            outputs=[kb_stats, papers_table],
        )
        year_to_kb.change(
            refresh_stats_and_table,
            inputs=[year_from_kb, year_to_kb, min_citations_kb],
            outputs=[kb_stats, papers_table],
        )
        min_citations_kb.change(
            refresh_stats_and_table,
            inputs=[year_from_kb, year_to_kb, min_citations_kb],
            outputs=[kb_stats, papers_table],
        )
        upload_btn.click(
            ingest_documents,
            inputs=[upload_pdf, year_from_kb, year_to_kb, min_citations_kb],
            outputs=[upload_status, kb_stats, papers_table],
        )
        delete_paper_btn.click(
            delete_paper,
            inputs=[delete_paper_id, year_from_kb, year_to_kb, min_citations_kb],
            outputs=[delete_status, kb_stats, papers_table],
        )
        export_bibtex_btn.click(
            export_bibtex,
            outputs=[export_status, bibtex_download],
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
        app.load(
            refresh_stats_and_table,
            inputs=[year_from_kb, year_to_kb, min_citations_kb],
            outputs=[kb_stats, papers_table],
        )

        # Reranker toggle syncing (Chat + KB share state)
        reranker_enable_chat.change(
            _set_reranker_settings,
            inputs=[reranker_enable_chat, rerank_topk_chat],
            outputs=[
                reranker_enable_chat,
                rerank_topk_chat,
                reranker_enable_kb,
                rerank_topk_kb,
                rerank_status_chat,
                rerank_status_kb,
            ],
        )
        rerank_topk_chat.change(
            _set_reranker_settings,
            inputs=[reranker_enable_chat, rerank_topk_chat],
            outputs=[
                reranker_enable_chat,
                rerank_topk_chat,
                reranker_enable_kb,
                rerank_topk_kb,
                rerank_status_chat,
                rerank_status_kb,
            ],
        )
        reranker_enable_kb.change(
            _set_reranker_settings,
            inputs=[reranker_enable_kb, rerank_topk_kb],
            outputs=[
                reranker_enable_chat,
                rerank_topk_chat,
                reranker_enable_kb,
                rerank_topk_kb,
                rerank_status_chat,
                rerank_status_kb,
            ],
        )
        rerank_topk_kb.change(
            _set_reranker_settings,
            inputs=[reranker_enable_kb, rerank_topk_kb],
            outputs=[
                reranker_enable_chat,
                rerank_topk_chat,
                reranker_enable_kb,
                rerank_topk_kb,
                rerank_status_chat,
                rerank_status_kb,
            ],
        )
        app.load(
            _set_reranker_settings,
            inputs=[reranker_enable_chat, rerank_topk_chat],
            outputs=[
                reranker_enable_chat,
                rerank_topk_chat,
                reranker_enable_kb,
                rerank_topk_kb,
                rerank_status_chat,
                rerank_status_kb,
            ],
        )

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

        # Data analysis events
        data_input.change(
            on_file_upload,
            inputs=[data_input],
            outputs=[data_info, column_select, group_by_col],
        )
        analyze_btn.click(
            analyze_data,
            inputs=[data_input, analysis_type, plot_type, column_select, group_by_col, custom_query],
            outputs=[analysis_output, analysis_plot],
        )
        download_plot_btn.click(
            download_current_plot,
            outputs=[plot_download],
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
