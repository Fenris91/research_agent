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

from research_agent.explorer import ExplorerRenderer, GraphBuilder, get_mock_graph_data
from research_agent.utils.config import load_config as _load_config
from research_agent.utils.openalex import SOURCE_LABELS_SHORT

logger = logging.getLogger(__name__)


def create_app(agent=None):
    """
    Create the Gradio application.

    Args:
        agent: ResearchAgent instance (optional for testing UI)

    Returns:
        Gradio Blocks app
    """

    # ── Dark theme matching the knowledge explorer ──────────────────
    explorer_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.red,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("IBM Plex Mono"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ).set(
        # Background
        body_background_fill="#0a0d13",
        body_background_fill_dark="#0a0d13",
        background_fill_primary="#0e1219",
        background_fill_primary_dark="#0e1219",
        background_fill_secondary="#0a0d13",
        background_fill_secondary_dark="#0a0d13",
        # Text
        body_text_color="#c8d0e0",
        body_text_color_dark="#c8d0e0",
        body_text_color_subdued="#5a6580",
        body_text_color_subdued_dark="#5a6580",
        # Blocks / panels
        block_background_fill="#0e1219",
        block_background_fill_dark="#0e1219",
        block_border_color="#1a1f2e",
        block_border_color_dark="#1a1f2e",
        block_label_background_fill="#0e1219",
        block_label_background_fill_dark="#0e1219",
        block_label_text_color="#5a6580",
        block_label_text_color_dark="#5a6580",
        block_title_text_color="#c8d0e0",
        block_title_text_color_dark="#c8d0e0",
        block_title_background_fill="#0e1219",
        block_title_background_fill_dark="#0e1219",
        block_shadow="none",
        block_shadow_dark="none",
        # Borders
        border_color_accent="#c45c4a",
        border_color_accent_dark="#c45c4a",
        border_color_primary="#1a1f2e",
        border_color_primary_dark="#1a1f2e",
        color_accent="#c45c4a",
        color_accent_soft="#c45c4a15",
        color_accent_soft_dark="#c45c4a15",
        # Input
        input_background_fill="#0a0e16",
        input_background_fill_dark="#0a0e16",
        input_border_color="#1a1f2e",
        input_border_color_dark="#1a1f2e",
        input_border_color_focus="#c45c4a40",
        input_border_color_focus_dark="#c45c4a40",
        input_placeholder_color="#3e4a64",
        input_placeholder_color_dark="#3e4a64",
        input_shadow="none",
        input_shadow_dark="none",
        input_shadow_focus="none",
        input_shadow_focus_dark="none",
        # Buttons
        button_primary_background_fill="#c45c4a",
        button_primary_background_fill_dark="#c45c4a",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        button_primary_background_fill_hover="#d4705f",
        button_primary_background_fill_hover_dark="#d4705f",
        button_primary_border_color="#c45c4a",
        button_primary_border_color_dark="#c45c4a",
        button_primary_shadow="none",
        button_primary_shadow_dark="none",
        button_secondary_background_fill="#1a1f2e",
        button_secondary_background_fill_dark="#1a1f2e",
        button_secondary_text_color="#c8d0e0",
        button_secondary_text_color_dark="#c8d0e0",
        button_secondary_border_color="#1a1f2e",
        button_secondary_border_color_dark="#1a1f2e",
        button_secondary_shadow="none",
        button_secondary_shadow_dark="none",
        # Shadows
        shadow_drop="none",
        shadow_drop_lg="none",
        shadow_spread="0px",
        shadow_spread_dark="0px",
        # Panel
        panel_background_fill="#0e1219",
        panel_background_fill_dark="#0e1219",
        panel_border_color="#1a1f2e",
        panel_border_color_dark="#1a1f2e",
        # Table
        table_border_color="#1a1f2e",
        table_border_color_dark="#1a1f2e",
        table_even_background_fill="#0e1219",
        table_even_background_fill_dark="#0e1219",
        table_odd_background_fill="#0a0d13",
        table_odd_background_fill_dark="#0a0d13",
        table_text_color="#c8d0e0",
        table_text_color_dark="#c8d0e0",
        # Accordion
        accordion_text_color="#5a6580",
        accordion_text_color_dark="#5a6580",
        # Checkbox
        checkbox_shadow="none",
        checkbox_label_shadow="none",
    )

    explorer_css = """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

    /*
     * Layout: pure flexbox propagation using :has() selectors to target
     * only the Gradio wrapper divs in the chain leading to #split-pane.
     * !important used only where Gradio's Svelte scoped CSS must yield.
     */

    /* ── Dark scrollbars ─────────────────────────── */
    * {
        scrollbar-width: thin;
        scrollbar-color: #1a1f2e #0a0d13;
    }
    *::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    *::-webkit-scrollbar-track {
        background: #0a0d13;
    }
    *::-webkit-scrollbar-thumb {
        background: #1a1f2e;
        border-radius: 3px;
    }
    *::-webkit-scrollbar-thumb:hover {
        background: #2a3048;
    }

    /* ── Viewport ────────────────────────────────── */
    html, body {
        height: 100dvh;
        height: 100vh;
        overflow: hidden;
    }
    .gradio-container {
        max-width: 100% !important;             /* theme sets 768/1280px */
        padding: 0 2px !important;              /* theme sets 16px+ */
        height: 100dvh;
        height: 100vh;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    /* Target only the wrapper divs that lead to #split-pane */
    div:has(> div > #split-pane),
    div:has(> #split-pane) {
        flex: 1 1 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    /* ── Status bar (top of left pane) ── */
    #status-bar, #status-bar > div {
        padding: 0; border: none; background: none; min-height: 0;
    }
    .status-bar {
        display: flex;
        align-items: center;
        gap: 0;
        padding: 3px 4px;
        background: #0a0d13;
        border-bottom: 1px solid #1a1f2e;
    }
    .si {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 8px;
        color: #5a6580;
        padding: 3px 7px;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
        gap: 2px;
        line-height: 1.4;
    }
    .si-star { color: #c45c4a; margin-right: 2px; }
    .si-val { color: #c8d0e0; margin-left: 4px; }
    .si-click {
        cursor: pointer;
        border-radius: 3px;
        transition: background 0.15s;
    }
    .si-click:hover { background: #1a1f2e; }
    .si-click.si-active { background: #1a1f2e; border-bottom: 1px solid #c45c4a; }

    /* Status bar expand panels */
    #sb-user-panel, #sb-ai-panel {
        padding: 0; border: none; background: none; min-height: 0;
    }
    #sb-user-panel > div, #sb-ai-panel > div {
        padding: 0; border: none; background: none; min-height: 0;
    }
    .sb-panel {
        background: #0e1219;
        border-bottom: 1px solid #1a1f2e;
        padding: 6px 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .sb-panel label, .sb-panel .label-wrap { display: none; }
    .sb-panel input, .sb-panel textarea {
        font-size: 11px;
        padding: 4px 8px;
        background: #141822;
        border: 1px solid #1a1f2e;
        color: #c8d0e0;
        border-radius: 3px;
    }
    .sb-panel .sb-hint {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 8px;
        color: #3e4a64;
    }
    .sb-panel button {
        font-size: 9px;
        padding: 3px 10px;
    }

    /* ── Chat layer buttons (left pane) ──────── */
    #chat-layers {
        flex: 0 0 auto;
        gap: 0;
        padding: 0 6px;
        border-bottom: 1px solid #1a1f2e;
        background: #0e1219;
        display: flex;
        align-items: stretch;
    }
    #chat-layers > div {
        padding: 0; border: none; background: none; min-height: 0;
    }
    .chat-layer-btn {
        font-size: 8px;
        letter-spacing: 0.6px;
        padding: 5px 10px;
        color: #3e4a64;
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        transition: all 0.2s;
        min-height: 0;
        min-width: 0;
        font-family: 'IBM Plex Mono', monospace;
        cursor: pointer;
    }
    .chat-layer-btn:hover {
        color: #5a6580;
    }
    .chat-layer-active {
        color: #c45c4a;
        border-bottom-color: #c45c4a;
    }


    /* ── Context pill strip ─────────────────────── */
    #ctx-pills-row {
        flex: 0 0 auto;
        min-height: 0;
    }
    #ctx-pills-row > div { padding: 0; border: none; background: none; min-height: 0; }
    #ctx-pills-container { padding: 0; min-height: 0; }
    #ctx-pills-container > div { min-height: 0; }
    .ctx-pills-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 3px;
        padding: 2px 6px;
        background: #0e1219;
        border: 1px solid #1a1f2e;
        border-radius: 4px;
    }
    .ctx-pill {
        display: inline-flex;
        align-items: center;
        gap: 3px;
        padding: 1px 6px;
        border-radius: 10px;
        font-size: 8px;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.3px;
        border: 1px solid;
        white-space: nowrap;
        max-width: 200px;
    }
    .ctx-pill-label { overflow: hidden; text-overflow: ellipsis; }
    .ctx-pill-keyword    { color: #c45c4a; border-color: rgba(196,92,74,0.3);  background: rgba(196,92,74,0.06); }
    .ctx-pill-paper_title { color: #5098ab; border-color: rgba(80,152,171,0.3); background: rgba(80,152,171,0.06); }
    .ctx-pill-field      { color: #2e5a88; border-color: rgba(46,90,136,0.3);  background: rgba(46,90,136,0.06); }
    .ctx-pill-domain     { color: #d4a04a; border-color: rgba(212,160,74,0.3); background: rgba(212,160,74,0.06); }
    .ctx-pill-researcher { color: #4a90d9; border-color: rgba(74,144,217,0.3); background: rgba(74,144,217,0.06); }
    .ctx-pill-pinned     { color: #7c5cbf; border-color: rgba(124,92,191,0.3); background: rgba(124,92,191,0.06); }
    .ctx-pill-metric     { color: #7a9a5a; border-color: rgba(122,154,90,0.3); background: rgba(122,154,90,0.06); }
    .ctx-pill-affiliation { color: #8a7ab5; border-color: rgba(138,122,181,0.3); background: rgba(138,122,181,0.06); }
    .ctx-pill-close, .ctx-pill-pin, .ctx-pill-lock {
        background: none; border: none; cursor: pointer;
        padding: 0 1px; font-size: 9px; line-height: 1;
        opacity: 0.5; color: inherit;
    }
    .ctx-pill-lock { cursor: default; font-size: 7px; }
    .ctx-pill-close:hover, .ctx-pill-pin:hover { opacity: 1; }
    .ctx-pill-disabled { opacity: 0.3; text-decoration: line-through; }
    .ctx-pill-disabled:hover { opacity: 0.5; }

    /* ── Split pane ──────────────────────────────── */
    #split-pane {
        flex: 1 1 0 !important;                 /* override Svelte row height */
        min-height: 0;
        gap: 4px;
        overflow: hidden;
    }
    #split-pane > div {
        border: 1px solid #1a1f2e;
        border-radius: 6px;
        overflow: hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    /* ── Chat pane ───────────────────────────────── */
    #chat-col {
        background: #0e1219;
        padding: 0 !important;                  /* Svelte column padding */
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    /* Kill all grey borders from Svelte block wrappers */
    #chat-col * {
        border-color: transparent !important;
    }
    #chat-col #chat-input {
        border-top: 1px solid #1a1f2e !important;
    }
    #chat-col #chat-input input {
        border: 1px solid #1a1f2e !important;
    }
    #chat-col #chat-input input:focus {
        border-color: #c45c4a40 !important;
    }
    /* Every wrapper div above the chatbot must propagate flex */
    #chat-col > div:has(.chatbot) {
        flex: 1 1 0 !important;
        min-height: 0;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        padding: 0;
    }
    /* Chatbot component and all its inner wrappers fill space */
    #chat-col .chatbot,
    #chat-col .chatbot > div {
        flex: 1 1 0 !important;
        height: auto !important;
        max-height: none !important;
        min-height: 0;
        border-radius: 0;
        background: #0e1219;
    }
    #chat-col .chatbot .bubble-wrap {
        height: 100%;
    }
    /* Chat input + clear pinned to bottom */
    #chat-input {
        padding: 6px 10px;
        border-top: 1px solid #1a1f2e;
        gap: 4px;
        background: #0e1219;
        flex: 0 0 auto !important;
    }
    #chat-input > div {
        border: none;
        background: none;
        padding: 0;
        min-height: 0;
    }
    #chat-input input {
        font-size: 10px;
        padding: 6px 10px;
        background: #0a0e16;
        border: 1px solid #1a1f2e;
        border-radius: 4px;
        color: #c8d0e0;
    }
    #chat-input input:focus {
        border-color: #c45c4a40;
    }
    #chat-input button {
        font-size: 9px;
        letter-spacing: 0.5px;
        padding: 6px 12px;
        border-radius: 4px;
        min-height: 0;
    }

    /* ── Explorer pane ───────────────────────────── */
    #explorer-col {
        padding: 0 !important;                  /* Svelte column padding */
        background: #0a0d13;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    /* Propagate flex through all Gradio wrapper divs (except layers bar) */
    #explorer-col > div,
    #explorer-col > div > div,
    #explorer-col > div > div > div {
        flex: 1 1 0 !important;
        min-height: 0;
        display: flex;
        flex-direction: column;
        border: none;
        padding: 0;
        overflow: hidden;
        position: relative;
    }

    /* ── Knowledge Management accordion ──────────── */
    #km-accordion {
        margin-top: 2px;
        border: 1px solid #1a1f2e;
        border-radius: 4px;
        flex: 0 0 auto;
        max-height: 45vh;
        overflow-y: auto;
    }
    #km-accordion .label-wrap {
        font-size: 8px;
        letter-spacing: 0.8px;
        padding: 3px 10px;
    }
    #km-accordion .tabitem {
        max-height: 40vh;
        overflow-y: auto;
    }
    """

    _pwa_head = """
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    <meta name="theme-color" content="#0a0d13">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <script>
    /* Shared helpers for iframe postMessage bridge and Gradio bus setters */
    function postToExplorer(type, data) {
      var iframe = document.querySelector("#explorer-col iframe");
      if (iframe && iframe.contentWindow) {
        iframe.contentWindow.postMessage(Object.assign({type: type}, data), "*");
      }
    }
    function setBusValue(busId, value) {
      var bus = document.querySelector(busId + " textarea, " + busId + " input");
      if (!bus) return;
      var setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value'
      ).set || Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value'
      ).set;
      setter.call(bus, value);
      bus.dispatchEvent(new Event('input', { bubbles: true }));
    }
    /* Public API — explorer iframe messaging */
    function sendLayerToExplorer(layer) { postToExplorer("set-layer", {layer: layer}); }
    function sendActionResultToExplorer(result) { postToExplorer("action-result", {result: result}); }
    function sendHighlightsToExplorer(terms) { postToExplorer("set-highlights", {terms: terms}); }
    function sendContextItemsToExplorer(items) { postToExplorer("set-context-items", {items: items}); }
    function sendGraphDelta(delta) { postToExplorer("update-graph", {delta: delta}); }
    /* Public API — Gradio bus setters */
    function ctxCommand(cmd) { setBusValue("#ctx-command-bus", cmd + ":" + Date.now()); }
    function toggleStatusDropdown(id) {
      var el = document.querySelector(id);
      if (!el) return;
      el.classList.toggle("si-reveal");
    }
    /* Listen for messages FROM explorer iframe — route to Python via hidden buses */
    window.addEventListener("message", function(e) {
      if (e.data && e.data.type === "explorer-layer") {
        setBusValue("#right-layer-bus", e.data.layer + ":" + Date.now());
      }
      if (e.data && e.data.type === "explorer-action") {
        setBusValue("#explorer-action-bus", JSON.stringify(e.data.payload) + ":" + Date.now());
      }
    });
    </script>
    """

    def _render_status_bar(state: dict) -> str:
        """Render all status indicators as a single HTML strip."""
        import html as html_mod
        state = state or {}
        # User
        researcher = state.get("researcher")
        is_anon = state.get("is_anon", True)
        email = state.get("email")
        if researcher and not is_anon:
            user_name = researcher
        elif email:
            user_name = email.split("@")[0]
        else:
            user_name = "anon"
        # Data
        data_val = "\u2014"
        try:
            from research_agent.db.vector_store import ResearchVectorStore
            store = ResearchVectorStore()
            stats = store.get_stats()
            papers = stats.get("total_papers", 0)
            notes = stats.get("total_notes", 0)
            parts = []
            if papers:
                parts.append(f"{papers} paper{'s' if papers != 1 else ''}")
            if notes:
                parts.append(f"{notes} note{'s' if notes != 1 else ''}")
            if parts:
                data_val = html_mod.escape(", ".join(parts))
        except Exception:
            pass
        # AI
        short = state.get("model_name") or "unknown"
        if short in ("unknown", "none"):
            short = "retrieval-only"
        if "/" in short:
            short = short.split("/")[-1]
        if len(short) > 20:
            short = short[:18] + "\u2026"
        short = html_mod.escape(short)

        return (
            f'<div class="status-bar">'
            f'<span class="si si-click" onclick="setBusValue(\'#sb-toggle-bus\',\'user:\'+Date.now())">'
            f'<span class="si-star">*</span>user<span class="si-val">{html_mod.escape(user_name)}</span></span>'
            f'<span class="si"><span class="si-star">*</span>data<span class="si-val">{data_val}</span></span>'
            f'<span class="si si-click" onclick="setBusValue(\'#sb-toggle-bus\',\'ai:\'+Date.now())">'
            f'<span class="si-star">*</span>ai<span class="si-val">{short}</span></span>'
            f'</div>'
        )

    with gr.Blocks(
        title="Research Assistant",
    ) as app:
        app._explorer_theme = explorer_theme
        app._explorer_css = explorer_css
        app._pwa_head = _pwa_head
        # Build initial soc_items from mock graph nodes
        _mock_data = get_mock_graph_data()
        _initial_soc = [
            {"label": n["label"], "type": n["type"], "auto": True, "enabled": True}
            for n in _mock_data.get("nodes", [])
            if n["type"] in ("field", "domain")
        ]
        # Shared state
        context_state = gr.State({
            "researcher": None, "paper_id": None,
            "active_layer_left": "context",      # Context / Author / Chat
            "active_layer_right": "structure",    # Structure / People / Topics
            "chat_context": None,
            "soc_items": _initial_soc, "auth_items": [], "chat_items": [],
            "is_anon": True,
            "model_name": "loading",
        })
        _query_state = gr.State({"query": None, "chunks": []})

        # ── Mapping left layers to item keys ──────────────────────────
        _LEFT_LAYER_ITEMS = {
            "context": "soc_items",
            "author": "auth_items",
            "chat": "chat_items",
        }

        # Pre-render initial SOC pills
        _has_initial_pills = bool(_initial_soc)
        if _has_initial_pills:
            import html as _html_mod
            _init_pills = []
            for _it in _initial_soc:
                _esc = _html_mod.escape(_it["label"])
                _jsl = _it["label"].replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
                _init_pills.append(
                    f'<span class="ctx-pill ctx-pill-{_it["type"]}" '
                    f'onclick="ctxCommand(\'toggle:soc:{_jsl}\')" style="cursor:pointer">'
                    f'<span class="ctx-pill-label">{_esc}</span></span>'
                )
            _init_pills_html = f'<div class="ctx-pills-wrap">{" ".join(_init_pills)}</div>'
        else:
            _init_pills_html = ""

        # Hidden buses (pill commands, explorer actions, right-layer sync, status bar)
        sb_toggle_bus = gr.Textbox(value="", visible=False, elem_id="sb-toggle-bus")
        ctx_command_bus = gr.Textbox(value="", visible=False, elem_id="ctx-command-bus")
        ctx_items_json = gr.Textbox(value="", visible=False, elem_id="ctx-items-json")
        explorer_action_bus = gr.Textbox(value="", visible=False, elem_id="explorer-action-bus")
        explorer_action_result = gr.Textbox(value="", visible=False, elem_id="explorer-action-result")
        right_layer_bus = gr.Textbox(value="", visible=False, elem_id="right-layer-bus")

        # Hidden components (needed by event handlers wired elsewhere)
        with gr.Group(visible=False):
            current_model_display = gr.Textbox(interactive=False, label="Current Model")
            kb_status_display = gr.Textbox()
            refresh_models_btn = gr.Button()
            year_from_chat = gr.Slider(minimum=1900, maximum=2030, value=1900)
            year_to_chat = gr.Slider(minimum=1900, maximum=2030, value=2030)
            min_citations_chat = gr.Slider(minimum=0, maximum=1000, value=0)
            reranker_enable_chat = gr.Checkbox(value=False)
            rerank_topk_chat = gr.Slider(minimum=1, maximum=50, value=10)
            rerank_status_chat = gr.Textbox()
            current_researcher = gr.Dropdown(choices=[], allow_custom_value=True)
            current_paper_id = gr.Textbox()
            refresh_context_btn = gr.Button()
            analyze_current_researcher_btn = gr.Button()
            analyze_current_paper_btn = gr.Button()
            ctx_kb_btn = gr.Button()
            ctx_researcher_btn = gr.Button()
            ctx_query_btn = gr.Button()
            ctx_citations_btn = gr.Button()
            context_map_plot = gr.Plot()
            context_map_status = gr.Textbox()
            # Status-bar dropdowns (toggled via JS)
            topbar_researcher = gr.Dropdown(
                choices=[], value=None, allow_custom_value=True,
                show_label=False, container=False,
                elem_id="topbar-researcher",
            )
            topbar_clear_btn = gr.Button(
                "\u00d7", variant="secondary", scale=0, min_width=20,
                elem_classes=["topbar-clear-btn"],
            )
            model_dropdown = gr.Dropdown(
                choices=["Loading..."], value="Loading...",
                show_label=False, container=False, interactive=True,
                min_width=140, elem_id="topbar-model-dropdown",
            )

        # ── SPLIT PANE: Chat (left) + Explorer (right) ───────────────────
        with gr.Row(equal_height=False, elem_id="split-pane"):
          # LEFT PANE: Chat
          with gr.Column(scale=2, min_width=340, elem_id="chat-col"):
            # Status indicators (single HTML to avoid Gradio wrapper fights)
            status_bar = gr.HTML(
                value=_render_status_bar({"is_anon": True, "model_name": "loading"}),
                elem_id="status-bar",
            )
            # ── Status bar expand panels ──
            with gr.Row(visible=False, elem_id="sb-user-panel") as sb_user_panel:
                gr.HTML('<div class="sb-panel"><span class="sb-hint">name:</span></div>', show_label=False)
                sb_user_input = gr.Textbox(
                    show_label=False, container=False, placeholder="Type your name...",
                    scale=3, elem_classes=["sb-panel"],
                )
                sb_user_set = gr.Button("Set", variant="primary", scale=0, min_width=40)

            with gr.Row(visible=False, elem_id="sb-ai-panel") as sb_ai_panel:
                gr.HTML(
                    '<div class="sb-panel">'
                    '<span class="sb-hint">provider:</span>'
                    '</div>',
                    show_label=False,
                )
                sb_ai_provider = gr.Dropdown(
                    choices=[
                        ("Groq (Free)", "groq"),
                        ("OpenAI", "openai"),
                        ("Anthropic (Claude)", "anthropic"),
                        ("OpenRouter (Free models)", "openrouter"),
                    ],
                    value="groq",
                    show_label=False, container=False, scale=2,
                )
                sb_ai_key = gr.Textbox(
                    show_label=False, container=False,
                    placeholder="API key",
                    type="password", scale=3,
                )
                sb_ai_connect = gr.Button("Connect", variant="primary", scale=0, min_width=60)

            # Layer tabs
            with gr.Row(elem_id="chat-layers"):
                layer_context_btn = gr.Button(
                    "Context", variant="secondary", scale=0, min_width=50,
                    elem_classes=["chat-layer-btn", "chat-layer-active"],
                )
                layer_author_btn = gr.Button(
                    "Author", variant="secondary", scale=0, min_width=50,
                    elem_classes=["chat-layer-btn"],
                )
                layer_chat_btn = gr.Button(
                    "Chat", variant="secondary", scale=0, min_width=50,
                    elem_classes=["chat-layer-btn"],
                )

            with gr.Row(visible=_has_initial_pills, elem_id="ctx-pills-row") as ctx_pills_row:
                ctx_pills_html = gr.HTML(value=_init_pills_html, elem_id="ctx-pills-container")

            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "What are you researching?"}],
            )

            with gr.Row(elem_id="chat-input"):
                msg = gr.Textbox(
                    show_label=False,
                    container=False,
                    scale=8,
                )
                submit = gr.Button("SEND", variant="primary", scale=0, min_width=50)
                clear = gr.Button("CLR", variant="secondary", scale=0, min_width=36)

          # RIGHT PANE: Knowledge Explorer
          with gr.Column(scale=3, elem_id="explorer-col"):
            _renderer = ExplorerRenderer()
            _mock_data["context_items"] = {"soc": _initial_soc, "auth": [], "chat": []}
            explorer_html = gr.HTML(
                value=_renderer.render(_mock_data)
            )

        # ── KNOWLEDGE MANAGEMENT (collapsed) ───────────────────────────
        with gr.Accordion("Knowledge Management", open=False, elem_id="km-accordion"):
            with gr.Tabs() as main_tabs:

                # ── Knowledge Base tab ────────────────────────────────────
                with gr.Tab("Knowledge Base"):
                    gr.Markdown("## Your Research Library")

                    with gr.Row():
                        kb_stats = gr.JSON(
                            label="Statistics",
                            value={
                                "total_papers": 0,
                                "total_notes": 0,
                                "total_web_sources": 0,
                            },
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

                    gr.Markdown("### Add Research Note")
                    gr.Markdown("*Save your own notes, annotations, or summaries to include in chat searches.*")

                    with gr.Row():
                        note_title = gr.Textbox(
                            label="Title",
                            placeholder="e.g., Notes on spatial theory",
                            scale=2,
                        )
                        note_tags = gr.Textbox(
                            label="Tags (comma-separated)",
                            placeholder="e.g., spatial, theory, urban",
                            scale=2,
                        )

                    note_content = gr.Textbox(
                        label="Note Content",
                        placeholder="Write your research notes here...",
                        lines=4,
                    )

                    with gr.Row():
                        add_note_btn = gr.Button("Add Note", variant="primary")
                        note_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            scale=3,
                        )

                    gr.Markdown("### Add Web Source")
                    gr.Markdown("*Save web content, reports, or grey literature to include in chat searches.*")

                    with gr.Row():
                        web_url = gr.Textbox(
                            label="URL (optional)",
                            placeholder="https://example.com/report.html",
                            scale=2,
                        )
                        web_title = gr.Textbox(
                            label="Title",
                            placeholder="e.g., City Planning Report 2024",
                            scale=2,
                        )

                    web_content = gr.Textbox(
                        label="Content",
                        placeholder="Paste the web content or report text here...",
                        lines=4,
                    )

                    with gr.Row():
                        add_web_btn = gr.Button("Add Web Source", variant="primary")
                        web_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            scale=3,
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
                        analyze_kb_btn = gr.Button(
                            "Analyze KB in Data Analysis",
                            variant="secondary",
                            scale=1,
                        )

                    with gr.Row():
                        delete_paper_id = gr.Textbox(
                            label="Paper ID",
                            placeholder="Enter paper ID to delete",
                            scale=4,
                        )
                        delete_paper_btn = gr.Button("Delete", variant="stop", scale=1)

                    with gr.Row():
                        reset_kb_btn = gr.Button("Reset KB", variant="stop", scale=1)

                    with gr.Row():
                        kb_selected_paper_id = gr.Textbox(
                            label="Selected Paper ID",
                            interactive=False,
                            placeholder="Click a row to select",
                            scale=4,
                        )
                        open_kb_citations_btn = gr.Button(
                            "Open in Citation Explorer",
                            variant="secondary",
                            scale=1,
                        )

                    delete_status = gr.Textbox(
                        label="Delete Status",
                        interactive=False,
                        placeholder="Enter a paper ID to delete",
                    )
                    reset_kb_status = gr.Textbox(
                        label="Reset Status",
                        interactive=False,
                        placeholder="Reset will remove all KB data",
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

                # ── Researcher Lookup tab ─────────────────────────────────
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
                            use_web_search = gr.Checkbox(label="Web Search", value=False)
                            fetch_papers = gr.Checkbox(
                                label="Fetch Papers (for Citation Explorer)",
                                value=False,
                                info="Slower but enables citation network exploration",
                            )
                            fetch_papers_limit = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=10,
                                step=1,
                                label="Max papers to fetch",
                            )

                    with gr.Row():
                        lookup_btn = gr.Button(
                            "Lookup Researchers", variant="primary", scale=2
                        )
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

                    gr.Markdown("### Explore Citations for Researcher")
                    with gr.Row():
                        researcher_select = gr.Dropdown(
                            choices=[],
                            label="Select researcher",
                            interactive=True,
                            value=None,
                        )
                        send_to_citations_btn = gr.Button(
                            "Explore Citations",
                            variant="secondary",
                        )

                    with gr.Row():
                        seed_paper_select = gr.Dropdown(
                            choices=[],
                            label="Seed paper (optional)",
                            interactive=True,
                            value=None,
                        )
                        load_papers_btn = gr.Button(
                            "Load Papers",
                            variant="secondary",
                        )

                    gr.Markdown("### Ingest Papers to Knowledge Base")
                    with gr.Row():
                        ingest_papers_limit = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Max papers to ingest",
                        )
                        ingest_researcher_btn = gr.Button(
                            "Ingest Top Papers",
                            variant="primary",
                        )
                    ingest_researcher_status = gr.Textbox(
                        label="Ingestion Status",
                        interactive=False,
                        placeholder="Select a researcher and click Ingest",
                    )

                    # State to store full results
                    researcher_results_state = gr.State([])
                    researcher_papers_state = gr.State({})

                # ── Citation Explorer tab ─────────────────────────────────
                with gr.Tab("Citation Explorer"):
                    from research_agent.ui.components import render_citation_explorer

                    citation_ui = render_citation_explorer()

                # ── Data Analysis tab ─────────────────────────────────────
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
                                choices=[
                                    "Histogram",
                                    "Box Plot",
                                    "Bar Chart",
                                    "Line Chart",
                                    "Scatter",
                                ],
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

                # ── Concepts tab (concept cluster map) ────────────────────
                with gr.Tab("Concepts"):
                    gr.Markdown("""
                    ## Concept Cluster Map

                    Visualize your knowledge base as an interactive 2D map.
                    Chunks that are **semantically similar** appear close together;
                    groups are automatically labelled with their dominant themes.

                    **Shapes:** ● Papers &nbsp; ◆ Notes &nbsp; ■ Web sources
                    """)

                    with gr.Row():
                        cm_max_chunks = gr.Slider(
                            minimum=100,
                            maximum=2000,
                            value=800,
                            step=100,
                            label="Max chunks per collection",
                            info="Higher = more detail, slower to compute",
                            scale=3,
                        )
                        cm_dim_method = gr.Radio(
                            choices=["UMAP", "t-SNE"],
                            value="UMAP",
                            label="Dimensionality reduction",
                            info="UMAP is faster and better at preserving global structure",
                            scale=2,
                        )
                        cm_n_clusters = gr.Slider(
                            minimum=0,
                            maximum=20,
                            value=0,
                            step=1,
                            label="Number of clusters (0 = auto)",
                            info="Auto uses sqrt(N/2) heuristic, capped at 15",
                            scale=2,
                        )

                    with gr.Row():
                        cm_generate_btn = gr.Button(
                            "Generate Concept Map", variant="primary", scale=2
                        )

                    cm_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        placeholder="Click Generate Concept Map to visualize your knowledge base",
                    )

                    cm_plot = gr.Plot(label="Concept Map", show_label=False)

        # Event handlers

        # ── Status bar panel toggle (via JS bus) ──
        def _on_sb_toggle(toggle_val, state):
            """Toggle user/ai panels from status bar clicks."""
            if not toggle_val:
                return gr.update(), gr.update(), state
            panel = toggle_val.split(":")[0]
            # Toggle: if same panel clicked again, hide it
            active = state.get("_sb_active")
            if active == panel:
                new_state = dict(state)
                new_state["_sb_active"] = None
                return gr.update(visible=False), gr.update(visible=False), new_state
            new_state = dict(state)
            new_state["_sb_active"] = panel
            return (
                gr.update(visible=(panel == "user")),
                gr.update(visible=(panel == "ai")),
                new_state,
            )

        sb_toggle_bus.change(
            _on_sb_toggle,
            inputs=[sb_toggle_bus, context_state],
            outputs=[sb_user_panel, sb_ai_panel, context_state],
        )

        # ── User name set ──
        def _on_user_set(name, state):
            """Set user name from status bar panel."""
            new_state = dict(state)
            name = (name or "").strip()
            if name:
                new_state["researcher"] = name
                new_state["is_anon"] = False
            else:
                new_state["researcher"] = None
                new_state["is_anon"] = True
            new_state["_sb_active"] = None
            return (
                _render_status_bar(new_state),
                new_state,
                gr.update(visible=False),
            )

        sb_user_set.click(
            _on_user_set,
            inputs=[sb_user_input, context_state],
            outputs=[status_bar, context_state, sb_user_panel],
        )
        sb_user_input.submit(
            _on_user_set,
            inputs=[sb_user_input, context_state],
            outputs=[status_bar, context_state, sb_user_panel],
        )

        # ── AI provider connect (BYOK) ──
        def _on_ai_connect(provider_key, api_key, state):
            """Connect to a cloud provider with a user-supplied API key."""
            if not api_key or not api_key.strip():
                return state, _render_status_bar(state), gr.update(visible=True)
            if agent is None:
                return state, _render_status_bar(state), gr.update(visible=True)

            success = agent.connect_provider(provider_key, api_key.strip())
            new_state = dict(state)
            if success:
                model_name = agent.get_current_model()
                new_state["model_name"] = model_name
                new_state["_sb_active"] = None
            return (
                new_state,
                _render_status_bar(new_state),
                gr.update(visible=not success),
            )

        sb_ai_connect.click(
            _on_ai_connect,
            inputs=[sb_ai_provider, sb_ai_key, context_state],
            outputs=[context_state, status_bar, sb_ai_panel],
        )

        def get_available_models():
            """Get list of available models for the current provider."""
            if agent is None:
                return ["No agent loaded"]

            # Check provider type
            provider = getattr(agent, "provider", None)
            if provider not in {"ollama", "openai", "openai_compatible"}:
                return ["Local model (no switching)"]

            try:
                models = agent.list_available_models()
                if models:
                    if provider == "ollama":
                        # Sort Ollama models with preferred first
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
                    else:
                        # OpenAI/compatible - return as-is
                        return models
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

        def add_user_message(message, history):
            """Immediately show the user message in chat."""
            history.append({"role": "user", "content": message})
            return "", history

        def _render_context_pills(state: dict) -> str:
            """Render HTML pill strip for the active left layer's context items."""
            import html as html_mod
            layer = state.get("active_layer_left", "context")
            items_key = _LEFT_LAYER_ITEMS.get(layer, "soc_items")
            items = state.get(items_key, [])
            if not items:
                # Show anon hint for author layer when no items
                if layer == "author":
                    return '<div class="ctx-pills-wrap"><span class="ctx-pill ctx-pill-field" style="opacity:0.5;cursor:default">Select a researcher to set your perspective</span></div>'
                return ""
            # Map left layer name → items prefix for toggle commands
            _layer_to_prefix = {"context": "soc", "author": "auth", "chat": "chat"}
            prefix = _layer_to_prefix.get(layer, layer)
            pills = []
            for item in items:
                label = item["label"]
                itype = item.get("type", "keyword")
                auto = item.get("auto", True)
                enabled = item.get("enabled", True)
                escaped = html_mod.escape(label)
                # Escape label for JS string (handle quotes/backslashes)
                js_label = label.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
                disabled_cls = "" if enabled else " ctx-pill-disabled"
                pill = f'<span class="ctx-pill ctx-pill-{itype}{disabled_cls}" '
                pill += f'onclick="ctxCommand(\'toggle:{prefix}:{js_label}\')" '
                pill += f'style="cursor:pointer">'
                pill += f'<span class="ctx-pill-label">{escaped}</span>'
                if layer == "chat":
                    pill += f'<button class="ctx-pill-pin" onclick="event.stopPropagation();ctxCommand(\'pin:{js_label}\')" title="Pin to Author">&#x1F4CC;</button>'
                if layer == "chat" or (layer == "author" and not auto):
                    pill += f'<button class="ctx-pill-close" onclick="event.stopPropagation();ctxCommand(\'remove:{prefix}:{js_label}\')">&#x00D7;</button>'
                if layer == "author" and auto and itype == "researcher":
                    pill += '<span class="ctx-pill-lock" title="Auto (researcher lookup)">&#x1F512;</span>'
                pill += '</span>'
                pills.append(pill)
            return f'<div class="ctx-pills-wrap">{" ".join(pills)}</div>'

        def _extract_chat_keywords(response_text, sources):
            """Extract keywords from agent response for CHAT layer highlighting."""
            import re
            keywords = []

            # Paper titles from sources
            for s in (sources or []):
                title = s.get("title", "")
                if title and title != "Unknown":
                    keywords.append(title)

            # Numbered/bulleted items (e.g. "1. **Key Theme**" or "- Something")
            items = re.findall(
                r'(?:^|\n)\s*(?:\d+[\.\)]\s*|[-*]\s+)\**(.+?)\**(?:\n|$|:)',
                response_text,
            )
            for item in items:
                cleaned = item.strip().rstrip(".:").strip("*")
                if 3 < len(cleaned) < 80:
                    keywords.append(cleaned)

            # Quoted phrases
            quoted = re.findall(r'"([^"]{4,60})"', response_text)
            keywords.extend(quoted)

            # Deduplicate (case-insensitive)
            seen = set()
            unique = []
            for kw in keywords:
                lower = kw.lower()
                if lower not in seen:
                    seen.add(lower)
                    unique.append(kw)

            return {"keywords": unique[:15], "paper_titles": [
                s.get("title", "") for s in (sources or []) if s.get("title")
            ]}

        def generate_response(history, year_from, year_to, min_citations, context_state):
            """Process the last user message and generate a response."""
            # Extract the last user message (Gradio 6 may use structured content)
            raw_content = history[-1]["content"] if history else ""
            if isinstance(raw_content, list):
                # Gradio 6 structured content: [{"type": "text", "text": "..."}]
                message = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in raw_content
                ).strip() or ""
            else:
                message = str(raw_content)

            new_state = dict(context_state or {})
            explorer_update = gr.update()
            layer_updates = (gr.update(), gr.update(), gr.update())
            sources = []

            if agent is None:
                response = f"[Demo mode] You asked: {message}\n\nThe agent is not loaded. Run with a real agent to get responses."
            else:
                try:
                    filters = {
                        "year_from": int(year_from) if year_from else None,
                        "year_to": int(year_to) if year_to else None,
                        "min_citations": int(min_citations) if min_citations else None,
                    }

                    # Extract context from state
                    context = context_state or {}
                    current_researcher = context.get("researcher")
                    current_paper_id = context.get("paper_id")

                    # Pass context to agent (include pinned/active items)
                    result = agent.run(
                        message,
                        search_filters=filters,
                        context={
                            "researcher": current_researcher,
                            "paper_id": current_paper_id,
                            "auth_items": [it["label"] for it in context.get("auth_items", [])],
                            "chat_items": [it["label"] for it in context.get("chat_items", [])],
                        }
                    )
                    response = result.get("answer", "No response generated")

                    # Append source attribution section if sources exist
                    sources = result.get("sources", [])
                    named_sources = [
                        s for s in sources
                        if s.get("title") and s.get("title") != "Unknown"
                    ]
                    if named_sources:
                        seen_titles = set()
                        source_parts = []
                        for s in named_sources:
                            title = s["title"]
                            if title in seen_titles:
                                continue
                            seen_titles.add(title)
                            tag = SOURCE_LABELS_SHORT.get(s.get("source", ""), "")
                            year = s.get("year")
                            part = f"{title}"
                            if year:
                                part += f" ({year})"
                            if tag:
                                part += f" [{tag}]"
                            source_parts.append(part)
                        response += "\n\n---\n*" + " · ".join(source_parts) + "*"

                    # Extract chat keywords for CHAT layer
                    chat_ctx = _extract_chat_keywords(response, sources)
                    new_state["chat_context"] = chat_ctx
                    # Build typed chat_items for pills
                    title_set = set(chat_ctx.get("paper_titles", []))
                    chat_items = []
                    for kw in chat_ctx.get("keywords", []):
                        chat_items.append({
                            "label": kw,
                            "type": "paper_title" if kw in title_set else "keyword",
                            "auto": True,
                            "enabled": True,
                        })
                    new_state["chat_items"] = chat_items

                    # Auto-switch left pane to Chat layer, right pane to Topics
                    if chat_ctx["keywords"]:
                        new_state["active_layer_left"] = "chat"
                        new_state["active_layer_right"] = "topics"
                        layer_updates = _left_layer_btn_classes("chat")
                        renderer = ExplorerRenderer()
                        researcher_name = new_state.get("researcher")
                        if researcher_name:
                            from research_agent.tools.researcher_registry import get_researcher_registry
                            registry = get_researcher_registry()
                            profile = registry.get(researcher_name)
                            if profile:
                                gb = GraphBuilder()
                                rid = gb.add_researcher(profile.to_dict())
                                for paper in (profile.top_papers or []):
                                    pid = gb.add_paper(paper)
                                    gb.add_authorship_edge(rid, pid)
                                gb.build_structural_context()
                                ctx_items = {
                                    "soc": new_state.get("soc_items", []),
                                    "auth": new_state.get("auth_items", []),
                                    "chat": chat_items,
                                }
                                explorer_update = renderer.render(
                                    gb.to_dict(active_layer="topics",
                                               highlight_terms=chat_ctx["keywords"],
                                               context_items=ctx_items)
                                )
                            else:
                                gd = get_mock_graph_data()
                                gd["active_layer"] = "topics"
                                gd["highlight_terms"] = chat_ctx["keywords"]
                                explorer_update = renderer.render(gd)
                        else:
                            gd = get_mock_graph_data()
                            gd["active_layer"] = "topics"
                            gd["highlight_terms"] = chat_ctx["keywords"]
                            explorer_update = renderer.render(gd)

                except Exception as e:
                    logger.error(f"[Chat] generate_response error: {e}", exc_info=True)
                    error_msg = str(e)
                    if "401" in error_msg:
                        response = f"**API Authentication Error:** Your API key may be invalid or expired. Please check your API key configuration in `configs/config.yaml`.\n\n*Details: {error_msg}*"
                    elif "429" in error_msg:
                        response = f"**Rate Limit Reached:** Too many API requests. Please wait a moment and try again.\n\n*Details: {error_msg}*"
                    else:
                        response = f"**Error:** {error_msg}"

            history.append({"role": "assistant", "content": response})
            pills_html = _render_context_pills(new_state)
            left_layer = new_state.get("active_layer_left", "context")
            items_key = _LEFT_LAYER_ITEMS.get(left_layer, "soc_items")
            has_items = bool(new_state.get(items_key, []))
            return history, new_state, explorer_update, *layer_updates, pills_html, gr.update(visible=has_items)

        vector_store = None
        embedder = None
        processor = None
        reranker = None
        rerank_top_k = None

        # Shared reranker settings
        reranker_enabled = None
        rerank_top_k = None

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
                    retrieval_cfg = (
                        cfg.get("retrieval", {}) if isinstance(cfg, dict) else {}
                    )
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
                citations = paper.get("citation_count") or paper.get("citations")
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

        def refresh_stats_and_table(
            year_from=None, year_to=None, min_citations=0, context_state=None
        ):
            """Refresh knowledge base statistics and paper list."""
            store, _, _ = _get_kb_resources()
            stats = store.get_stats()
            papers = store.list_papers_detailed(limit=5000)

            context = context_state or {}
            researcher_filter = context.get("researcher")
            paper_filter = context.get("paper_id")

            filtered = papers
            if paper_filter:
                filtered = [p for p in filtered if p.get("paper_id") == paper_filter]
            elif researcher_filter:
                filtered = [
                    p for p in filtered if p.get("researcher") == researcher_filter
                ]

            if filtered is not papers:
                stats = dict(stats)
                stats["filtered_papers"] = len(filtered)

            return stats, _format_papers_table(
                filtered,
                year_from=year_from if year_from and year_from > 1900 else None,
                year_to=year_to if year_to and year_to < 2030 else None,
                min_citations=min_citations or 0,
            )

        def ingest_documents(
            files, year_from=None, year_to=None, min_citations=0, context_state=None
        ):
            """Process and add documents to the knowledge base."""
            if not files:
                stats, table = refresh_stats_and_table(
                    year_from, year_to, min_citations, context_state
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
                    from research_agent.ui.kb_ingest import normalize_metadata

                    metadata = normalize_metadata(
                        doc.metadata, {"ingest_source": "upload"}
                    )
                    store.add_paper(paper_id, chunk_texts, embeddings, metadata)
                    added += 1
                except Exception as e:
                    errors.append(f"{getattr(file_obj, 'name', file_obj)}: {e}")

            stats, table = refresh_stats_and_table(
                year_from, year_to, min_citations, context_state
            )

            status_parts = [f"Added {added} document(s)"]
            if skipped:
                status_parts.append(f"Skipped {skipped} duplicate(s)")
            if errors:
                status_parts.append("Errors: " + "; ".join(errors[:3]))
                if len(errors) > 3:
                    status_parts.append(f"(+{len(errors) - 3} more)")

            return ". ".join(status_parts), stats, table

        def add_research_note(
            title, content, tags, year_from=None, year_to=None, min_citations=0, context_state=None
        ):
            """Add a research note to the knowledge base."""
            if not content or not content.strip():
                return "Please enter note content.", None, None

            store, embedder_model, _ = _get_kb_resources()

            # Generate note ID from title or timestamp
            import hashlib
            from datetime import datetime

            note_id = hashlib.md5(
                f"{title or 'note'}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]

            try:
                # Generate embedding for the note
                embedding = embedder_model.embed_query(content)
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()

                # Prepare metadata
                metadata = {
                    "title": title or "Untitled Note",
                    "tags": tags or "",
                }

                # Add to vector store
                store.add_note(note_id, content, embedding, metadata)

                stats, table = refresh_stats_and_table(
                    year_from, year_to, min_citations, context_state
                )

                return f"✓ Added note: {title or 'Untitled'}", stats, table

            except Exception as e:
                return f"Error adding note: {e}", None, None

        def add_web_source(
            url, title, content, year_from=None, year_to=None, min_citations=0, context_state=None
        ):
            """Add a web source to the knowledge base."""
            if not content or not content.strip():
                return "Please enter content for the web source.", None, None

            store, embedder_model, _ = _get_kb_resources()

            # Generate source ID from URL or title
            import hashlib
            from datetime import datetime

            source_base = url or title or "web"
            source_id = hashlib.md5(
                f"{source_base}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]

            try:
                # Chunk the content if it's long
                chunk_size = 512
                chunks = []
                words = content.split()
                current_chunk = []
                current_length = 0

                for word in words:
                    current_chunk.append(word)
                    current_length += len(word) + 1
                    if current_length >= chunk_size:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_length = 0

                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                if not chunks:
                    chunks = [content]

                # Generate embeddings
                embeddings = embedder_model.embed_documents(
                    chunks, batch_size=32, show_progress=False
                )

                # Prepare metadata
                metadata = {
                    "title": title or "Web Source",
                    "url": url or "",
                }

                # Add to vector store
                store.add_web_source(source_id, chunks, embeddings, metadata)

                stats, table = refresh_stats_and_table(
                    year_from, year_to, min_citations, context_state
                )

                return f"✓ Added web source: {title or url or 'Untitled'} ({len(chunks)} chunks)", stats, table

            except Exception as e:
                return f"Error adding web source: {e}", None, None

        def delete_paper(
            paper_id, year_from=None, year_to=None, min_citations=0, context_state=None
        ):
            """Delete a paper from the knowledge base."""
            if not paper_id:
                stats, table = refresh_stats_and_table(
                    year_from, year_to, min_citations, context_state
                )
                return "Enter a paper ID to delete.", stats, table

            store, _, _ = _get_kb_resources()
            deleted = store.delete_paper(paper_id)
            status = (
                f"Deleted paper {paper_id}."
                if deleted
                else f"Paper not found: {paper_id}."
            )
            stats, table = refresh_stats_and_table(
                year_from, year_to, min_citations, context_state
            )
            return status, stats, table

        def reset_kb(year_from=None, year_to=None, min_citations=0, context_state=None):
            """Reset the knowledge base."""
            store, _, _ = _get_kb_resources()
            store.reset()
            stats, table = refresh_stats_and_table(
                year_from, year_to, min_citations, context_state
            )
            return "Knowledge base reset.", stats, table

        def _get_researcher_choices():
            from research_agent.tools.researcher_registry import get_researcher_registry

            registry = get_researcher_registry()
            researchers = registry.list_all()
            return [r.name for r in researchers if r.name]

        def refresh_context_choices(state):
            """Refresh shared researcher dropdown choices.

            Also syncs the topbar dropdown + clear button so that on page
            refresh a previously-selected researcher is shown immediately.
            """
            choices = _get_researcher_choices()
            value = None
            if state and state.get("researcher") in choices:
                value = state.get("researcher")
            elif choices:
                value = choices[0]
            new_state = dict(state or {})
            new_state["researcher"] = value
            return (
                new_state,
                gr.update(choices=choices, value=value),
                gr.update(choices=choices, value=value),   # topbar dropdown
                gr.update(visible=bool(value)),             # clear btn
            )

        def sync_researcher_context(name: str, state):
            """Update shared context when a researcher is selected."""
            choices = _get_researcher_choices()
            if name and name not in choices:
                choices.insert(0, name)

            new_state = dict(state or {})
            new_state["researcher"] = name
            update = gr.update(choices=choices, value=name)
            return new_state, update, update, update

        def sync_researcher_context_with_table(
            name: str, state, year_from=None, year_to=None, min_citations=0
        ):
            new_state, current_update, select_update, citation_update = (
                sync_researcher_context(name, state)
            )
            stats, table = refresh_stats_and_table(
                year_from, year_to, min_citations, new_state
            )
            return (
                new_state,
                current_update,
                select_update,
                citation_update,
                stats,
                table,
            )

        def sync_paper_context(paper_id: str, state):
            new_state = dict(state or {})
            new_state["paper_id"] = paper_id
            update = gr.update(value=paper_id)
            return new_state, update, update

        def sync_paper_context_with_table(
            paper_id: str, state, year_from=None, year_to=None, min_citations=0
        ):
            new_state, current_update, citation_update = sync_paper_context(
                paper_id, state
            )
            stats, table = refresh_stats_and_table(
                year_from, year_to, min_citations, new_state
            )
            return (
                new_state,
                current_update,
                citation_update,
                stats,
                table,
            )

        def load_kb_into_analysis(year_from=None, year_to=None, min_citations=0):
            """Load KB papers into the data analysis tab."""
            import pandas as pd

            store, _, _ = _get_kb_resources()
            papers = store.list_papers_detailed(limit=5000)
            if not papers:
                _analysis_df["df"] = None
                _analysis_df["path"] = None
                return (
                    "No papers found in the knowledge base.",
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=None),
                    gr.update(value="Data Analysis"),
                )

            df = pd.DataFrame(papers)

            def _to_int(value):
                try:
                    return int(value)
                except Exception:
                    return None

            if "year" in df.columns:
                df["year"] = df["year"].apply(_to_int)

            if "citation_count" in df.columns:
                df["citation_count"] = df["citation_count"].apply(_to_int)

            if year_from and year_from > 1900:
                df = df[df["year"].isna() | (df["year"] >= year_from)]
            if year_to and year_to < 2030:
                df = df[df["year"].isna() | (df["year"] <= year_to)]
            if min_citations:
                df = df[df["citation_count"].isna() | (df["citation_count"] >= min_citations)]

            if df.empty:
                _analysis_df["df"] = None
                _analysis_df["path"] = None
                return (
                    "No papers match the current filters.",
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=None),
                    gr.update(value="Data Analysis"),
                )

            _analysis_df["df"] = df
            _analysis_df["path"] = None

            cols = list(df.columns)
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            info = f"**Rows:** {len(df)} | **Cols:** {len(cols)}\n"
            info += f"Numeric: {len(numeric_cols)} | Date: {len(date_cols)} | Text: {len(cat_cols)}"

            return (
                info,
                gr.update(
                    choices=cols, value=numeric_cols[:2] if numeric_cols else cols[:2]
                ),
                gr.update(choices=["(None)"] + cols, value="(None)"),
                gr.update(value="Data Analysis"),
            )

        def refresh_context_from_lookup(
            lookup_results, state, year_from=None, year_to=None, min_citations=0
        ):
            """Update shared researcher dropdown after lookup."""
            names = []
            for r in lookup_results or []:
                name = r.get("name") if isinstance(r, dict) else None
                if name:
                    names.append(name)

            unique = []
            seen = set()
            for name in names:
                if name not in seen:
                    seen.add(name)
                    unique.append(name)

            new_state = dict(state or {})
            new_state["researcher"] = unique[0] if unique else None
            stats, table = refresh_stats_and_table(
                year_from, year_to, min_citations, new_state
            )
            return (
                new_state,
                gr.update(choices=unique, value=unique[0] if unique else None),
                stats,
                table,
            )

        def _load_df_into_analysis(df):
            """Shared loader for Data Analysis from dataframe."""

            def _to_int(value):
                try:
                    return int(value)
                except Exception:
                    return None

            if "year" in df.columns:
                df["year"] = df["year"].apply(_to_int)
            if "citation_count" in df.columns:
                df["citation_count"] = df["citation_count"].apply(_to_int)

            _analysis_df["df"] = df
            _analysis_df["path"] = None

            cols = list(df.columns)
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            info = f"**Rows:** {len(df)} | **Cols:** {len(cols)}\n"
            info += f"Numeric: {len(numeric_cols)} | Date: {len(date_cols)} | Text: {len(cat_cols)}"

            return (
                info,
                gr.update(
                    choices=cols, value=numeric_cols[:2] if numeric_cols else cols[:2]
                ),
                gr.update(choices=["(None)"] + cols, value="(None)"),
                gr.update(value="Data Analysis"),
            )

        def load_kb_for_researcher(
            researcher_name, year_from=None, year_to=None, min_citations=0
        ):
            """Load KB papers for a specific researcher into analysis."""
            import pandas as pd

            if not researcher_name:
                return (
                    "Select a researcher first.",
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=None),
                    gr.update(value="Data Analysis"),
                )

            store, _, _ = _get_kb_resources()
            papers = store.list_papers_detailed(limit=5000)
            df = pd.DataFrame(papers)
            if df.empty or "researcher" not in df.columns:
                return (
                    "No researcher-tagged papers found in KB.",
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=None),
                    gr.update(value="Data Analysis"),
                )

            df = df[df["researcher"] == researcher_name]
            if df.empty:
                return (
                    "No papers found for this researcher in KB.",
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=None),
                    gr.update(value="Data Analysis"),
                )

            return _load_df_into_analysis(df)

        def load_kb_for_paper(paper_id):
            """Load a single KB paper into analysis."""
            import pandas as pd

            if not paper_id:
                return (
                    "Select a paper first.",
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=None),
                    gr.update(value="Data Analysis"),
                )

            store, _, _ = _get_kb_resources()
            papers = store.list_papers_detailed(limit=5000)
            df = pd.DataFrame(papers)
            if df.empty:
                return (
                    "No papers found in KB.",
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=None),
                    gr.update(value="Data Analysis"),
                )

            df = df[df["paper_id"] == paper_id]
            if df.empty:
                return (
                    "Selected paper not found in KB.",
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=None),
                    gr.update(value="Data Analysis"),
                )

            return _load_df_into_analysis(df)

        def select_kb_paper(table_data, evt: gr.SelectData):
            """Select a paper ID from the KB table."""
            if table_data is None:
                return ""
            rows = table_data if isinstance(table_data, list) else table_data.values
            row_index = (
                evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            )
            if not isinstance(row_index, int) or row_index < 0:
                return ""
            if row_index >= len(rows):
                return ""
            row = rows[row_index]
            return row[4] if len(row) > 4 else ""

        def select_kb_paper_with_context(
            table_data,
            evt: gr.SelectData,
            state,
            year_from=None,
            year_to=None,
            min_citations=0,
        ):
            paper_id = select_kb_paper(table_data, evt)
            new_state = dict(state or {})
            new_state["paper_id"] = paper_id
            stats, table = refresh_stats_and_table(
                year_from, year_to, min_citations, new_state
            )
            return new_state, paper_id, stats, table

        async def open_kb_citations(paper_id, direction, depth):
            """Open a KB paper in the citation explorer."""
            if not paper_id:
                return (
                    "",
                    "Select a paper from the KB table first.",
                    None,
                    None,
                    None,
                    None,
                    None,
                    gr.update(),
                )

            from research_agent.ui.components.citation_explorer import explore_citations

            (
                summary,
                citing_df,
                cited_df,
                connected_df,
                related_df,
                network_fig,
            ) = await explore_citations(paper_id, direction, depth)

            return (
                paper_id,
                summary,
                citing_df,
                cited_df,
                connected_df,
                related_df,
                network_fig,
                gr.update(value="Citation Explorer"),
            )

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
            lines.append(f"  title = {{{title}}},")

            if authors:
                # Convert "First Last, First Last" to "Last, First and Last, First"
                author_list = [a.strip() for a in authors.split(",")]
                bibtex_authors = " and ".join(author_list)
                lines.append(f"  author = {{{bibtex_authors}}},")

            if year:
                lines.append(f"  year = {{{year}}},")

            # Add DOI if it looks like the paper_id is a DOI
            if paper_id.startswith("10."):
                lines.append(f"  doi = {{{paper_id}}},")

            lines.append("}")

            return "\n".join(lines)

        def lookup_researchers(
            names_text,
            use_oa,
            use_s2,
            use_web,
            should_fetch_papers,
            papers_limit,
            existing_results,
        ):
            """Look up researcher profiles, optionally with papers."""
            from research_agent.tools.researcher_file_parser import (
                parse_researchers_text,
            )
            from research_agent.tools.researcher_lookup import ResearcherLookup
            from research_agent.tools.researcher_registry import get_researcher_registry

            names = parse_researchers_text(names_text)

            if not names:
                return (
                    "No valid names found",
                    [],
                    existing_results or [],
                    None,
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    {},
                )

            # Create lookup instance
            lookup = ResearcherLookup(
                use_openalex=use_oa,
                use_semantic_scholar=use_s2,
                use_web_search=use_web,
                request_delay=0.5,
            )

            # Run async lookup
            try:
                async def lookup_all():
                    profiles = []
                    for name in names:
                        profile = await lookup.lookup_researcher(
                            name,
                            fetch_papers=should_fetch_papers,
                            papers_limit=int(papers_limit)
                            if should_fetch_papers
                            else 0,
                        )
                        profiles.append(profile)
                    return profiles

                async def run_lookup():
                    try:
                        return await lookup_all()
                    finally:
                        await lookup.close()

                # Use a new event loop in a thread to avoid conflicts with Gradio's loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    profiles = pool.submit(asyncio.run, run_lookup()).result()
            except Exception as e:
                logger.error(f"Lookup error: {e}")
                import traceback

                traceback.print_exc()
                return (
                    f"Error during lookup: {str(e)}",
                    [],
                    existing_results or [],
                    None,
                    gr.update(choices=[], value=None),
                    gr.update(choices=[], value=None),
                    {},
                )

            # Store in registry for cross-tab access (Citation Explorer)
            registry = get_researcher_registry()
            registry.add_batch(profiles)

            incoming_results = [p.to_dict() for p in profiles]
            combined = {r.get("name"): r for r in (existing_results or [])}
            for r in incoming_results:
                combined[r.get("name")] = r

            merged_results = list(combined.values())

            merged_table = []
            merged_web_results = {}
            total_papers = 0
            for r in merged_results:
                merged_table.append(
                    [
                        r.get("name", ""),
                        "; ".join(r.get("affiliations", [])),
                        r.get("works_count", 0),
                        f"{r.get('citations_count', 0):,}",
                        r.get("h_index", ""),
                        "; ".join((r.get("fields") or [])[:3]),
                    ]
                )
                total_papers += len(r.get("top_papers", []) or [])
                if r.get("web_results"):
                    merged_web_results[r["name"]] = r["web_results"]

            if should_fetch_papers and total_papers > 0:
                status = (
                    f"Found {len(merged_results)} profiles with {total_papers} papers. "
                    "Select a researcher to explore citation networks."
                )
            else:
                status = (
                    f"Found {len(merged_results)} researcher profiles. "
                    "Enable 'Fetch Papers' for better citation exploration."
                )

            researcher_names = [
                r.get("name", "") for r in merged_results if r.get("name")
            ]
            dropdown_update = gr.update(
                choices=researcher_names,
                value=researcher_names[0] if researcher_names else None,
            )

            return (
                status,
                merged_table,
                merged_results,
                merged_web_results or None,
                dropdown_update,
                gr.update(choices=[], value=None),
                {},
            )

        def clear_results():
            """Clear researcher lookup results."""
            return (
                "",
                [],
                [],
                None,
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                {},
            )

        def _rank_papers(papers):
            def _sort_key(paper):
                return (paper.get("year") or 0, paper.get("citation_count") or 0)

            return sorted(papers, key=_sort_key, reverse=True)

        async def load_researcher_papers(researcher_name):
            """Load candidate papers for a researcher."""
            if not researcher_name:
                return gr.update(choices=[], value=None), {}

            from research_agent.tools.researcher_registry import get_researcher_registry
            from research_agent.tools.academic_search import AcademicSearchTools

            registry = get_researcher_registry()
            profile = registry.get(researcher_name)
            papers = []

            if profile and profile.top_papers:
                papers = [
                    p.to_dict() if hasattr(p, "to_dict") else p
                    for p in profile.top_papers
                ]
            else:
                search_tools = AcademicSearchTools()
                try:
                    results = await search_tools.search_semantic_scholar(
                        researcher_name, limit=20
                    )
                    papers = [
                        {
                            "paper_id": p.paper_id,
                            "title": p.title,
                            "year": p.year,
                            "citation_count": p.citation_count,
                        }
                        for p in results
                    ]
                finally:
                    await search_tools.close()

            ranked = _rank_papers(papers)

            choices = []
            mapping = {}
            for paper in ranked:
                label = (
                    f"{paper.get('title', 'Unknown')} ({paper.get('year') or 'n.d.'})"
                    f" — {paper.get('citation_count') or 0} cites"
                )
                choices.append(label)
                paper_id = paper.get("paper_id") or paper.get("id")
                if paper_id:
                    mapping[label] = paper_id

            return (
                gr.update(choices=choices, value=choices[0] if choices else None),
                mapping,
            )

        async def explore_researcher_citations(
            researcher_name, direction, depth, seed_label, seed_map
        ):
            """Resolve a researcher to a seed paper and explore citations."""
            if not researcher_name:
                return (
                    "",
                    "Select a researcher to explore.",
                    None,
                    None,
                    None,
                    None,
                    None,
                )

            from research_agent.tools.researcher_registry import get_researcher_registry
            from research_agent.tools.academic_search import AcademicSearchTools
            from research_agent.ui.components.citation_explorer import explore_citations

            registry = get_researcher_registry()
            profile = registry.get(researcher_name)

            seed_paper_id = None
            if seed_label and seed_map and seed_label in seed_map:
                seed_paper_id = seed_map[seed_label]
            elif profile and profile.top_papers:
                papers = [
                    p.to_dict() if hasattr(p, "to_dict") else p
                    for p in profile.top_papers
                ]
                ranked = _rank_papers(papers)
                if ranked:
                    seed_paper_id = ranked[0].get("paper_id")

            if not seed_paper_id:
                search_tools = AcademicSearchTools()
                try:
                    papers = await search_tools.search_semantic_scholar(
                        researcher_name, limit=20
                    )
                    if not papers:
                        return (
                            "",
                            f"No papers found for: {researcher_name}",
                            None,
                            None,
                            None,
                            None,
                            None,
                        )

                    def _sort_key(paper):
                        return (paper.year or 0, paper.citation_count or 0)

                    candidates = sorted(papers, key=_sort_key, reverse=True)
                    seed_paper_id = candidates[0].paper_id
                finally:
                    await search_tools.close()

            (
                summary,
                citing_df,
                cited_df,
                connected_df,
                related_df,
                network_fig,
            ) = await explore_citations(seed_paper_id, direction, depth)

            summary = (
                summary.replace("**Mode:** Paper", "**Mode:** Researcher")
                + f"\n\n**Researcher:** {researcher_name}"
            )

            return (
                seed_paper_id,
                summary,
                citing_df,
                cited_df,
                connected_df,
                related_df,
                network_fig,
            )

        def ingest_researcher_papers(
            researcher_name, max_papers, year_from=None, year_to=None, min_citations=0
        ):
            """Ingest a researcher's top papers into the KB."""
            if not researcher_name:
                stats, table = refresh_stats_and_table(
                    year_from, year_to, min_citations
                )
                return "Select a researcher first.", stats, table

            from research_agent.tools.researcher_registry import get_researcher_registry

            registry = get_researcher_registry()
            profile = registry.get(researcher_name)

            if not profile or not profile.top_papers:
                stats, table = refresh_stats_and_table(
                    year_from, year_to, min_citations
                )
                return (
                    "No papers loaded. Enable 'Fetch Papers' and run lookup again.",
                    stats,
                    table,
                )

            store, embedder_model, _ = _get_kb_resources()
            from research_agent.ui.kb_ingest import ingest_paper_to_kb

            added = 0
            skipped = 0
            errors = []

            for paper in profile.top_papers[: int(max_papers)]:
                try:
                    paper_id = paper.doi or paper.paper_id
                    if not paper_id:
                        skipped += 1
                        continue

                    if store.get_paper(paper_id):
                        skipped += 1
                        continue

                    title = paper.title or "Unknown"
                    abstract = paper.abstract or ""
                    venue = paper.venue or ""
                    fields = paper.fields or []

                    added_flag, reason = ingest_paper_to_kb(
                        store=store,
                        embedder=embedder_model,
                        paper_id=paper_id,
                        title=title,
                        abstract=abstract,
                        venue=venue,
                        fields=fields,
                        doi=paper.doi,
                        year=paper.year,
                        citation_count=paper.citation_count,
                        authors=None,
                        source=paper.source,
                        extra_metadata={
                            "researcher": researcher_name,
                            "ingest_source": "researcher_lookup",
                        },
                    )
                    if added_flag:
                        added += 1
                    else:
                        skipped += 1
                except Exception as e:
                    errors.append(f"{paper.title}: {e}")

            stats, table = refresh_stats_and_table(year_from, year_to, min_citations)

            status_parts = [f"Added {added} paper(s)"]
            if skipped:
                status_parts.append(f"Skipped {skipped} duplicate/empty")
            if errors:
                status_parts.append("Errors: " + "; ".join(errors[:3]))
                if len(errors) > 3:
                    status_parts.append(f"(+{len(errors) - 3} more)")

            return ". ".join(status_parts), stats, table

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
                df = pd.read_csv(
                    file_path, parse_dates=True, infer_datetime_format=True
                )
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
                gr.update(
                    choices=cols, value=numeric_cols[:2] if numeric_cols else cols[:2]
                ),
                gr.update(choices=["(None)"] + cols, value="(None)"),
            )

        def analyze_data(
            file_obj, analysis_type, plot_type, columns, group_by, custom_query
        ):
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
                numeric_cols = [
                    c
                    for c in columns
                    if c in df.select_dtypes(include=["number"]).columns
                ]
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
                ax.scatter(
                    numeric_df[numeric_cols[0]], numeric_df[numeric_cols[1]], alpha=0.6
                )
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel(numeric_cols[1])
            else:
                ax.hist(numeric_df[col].dropna(), bins=20, edgecolor="black", alpha=0.7)
                ax.set_ylabel("Frequency")

            ax.set_title(
                f"Distribution of {col}"
                if plot_type == "Histogram"
                else f"Analysis of {', '.join(numeric_cols[:3])}"
            )
            plt.tight_layout()

            return result, fig

        def _correlation_analysis(df, columns):
            """Generate correlation matrix."""
            import matplotlib.pyplot as plt
            import numpy as np

            # Use selected columns or all numeric
            if columns:
                numeric_cols = [
                    c
                    for c in columns
                    if c in df.select_dtypes(include=["number"]).columns
                ]
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
                cat_cols = df.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

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
            pivot = (
                df.groupby(group_by)[value_col].agg(["mean", "sum", "count"]).round(2)
            )
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
                return (
                    "No date/time columns detected. Upload data with dates or ensure date format is recognized.",
                    None,
                )

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
                    ax.plot(
                        df_sorted[date_col],
                        df_sorted[col],
                        label=col,
                        marker="." if len(df) < 50 else "",
                    )
                elif plot_type == "Scatter":
                    ax.scatter(
                        df_sorted[date_col], df_sorted[col], label=col, alpha=0.6
                    )
                elif plot_type == "Bar Chart":
                    # Resample to fewer points for bar chart
                    ax.bar(
                        range(len(df_sorted)),
                        df_sorted[col].values,
                        label=col,
                        alpha=0.7,
                    )
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

            elif (
                "compare" in query_lower
                or "by" in query_lower
                or "group" in query_lower
            ):
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

        def generate_concept_map(max_chunks, dim_method, n_clusters_raw):
            """Generate an interactive 2D concept cluster map from the vector store."""
            from research_agent.ui.visualization import build_concept_map

            store, _, _ = _get_kb_resources()
            n_clusters = None if int(n_clusters_raw) == 0 else int(n_clusters_raw)
            dim_reduction = "umap" if dim_method == "UMAP" else "tsne"
            try:
                fig, status = build_concept_map(
                    store,
                    max_chunks=int(max_chunks),
                    n_clusters=n_clusters,
                    dim_reduction=dim_reduction,
                )
                return fig, status
            except Exception as e:
                logger.error(f"Concept map error: {e}")
                return None, f"Error generating concept map: {e}"

        # Wire up events
        _generate_outputs = [
            chatbot, context_state, explorer_html,
            layer_context_btn, layer_author_btn, layer_chat_btn,
            ctx_pills_html, ctx_pills_row,
        ]
        msg.submit(
            add_user_message,
            [msg, chatbot],
            [msg, chatbot],
        ).then(
            generate_response,
            [chatbot, year_from_chat, year_to_chat, min_citations_chat, context_state],
            _generate_outputs,
        )
        submit.click(
            add_user_message,
            [msg, chatbot],
            [msg, chatbot],
        ).then(
            generate_response,
            [chatbot, year_from_chat, year_to_chat, min_citations_chat, context_state],
            _generate_outputs,
        )
        clear.click(lambda: [], outputs=[chatbot])
        refresh_btn.click(
            refresh_stats_and_table,
            inputs=[year_from_kb, year_to_kb, min_citations_kb, context_state],
            outputs=[kb_stats, papers_table],
        )
        year_from_kb.change(
            refresh_stats_and_table,
            inputs=[year_from_kb, year_to_kb, min_citations_kb, context_state],
            outputs=[kb_stats, papers_table],
        )
        year_to_kb.change(
            refresh_stats_and_table,
            inputs=[year_from_kb, year_to_kb, min_citations_kb, context_state],
            outputs=[kb_stats, papers_table],
        )
        min_citations_kb.change(
            refresh_stats_and_table,
            inputs=[year_from_kb, year_to_kb, min_citations_kb, context_state],
            outputs=[kb_stats, papers_table],
        )
        upload_btn.click(
            ingest_documents,
            inputs=[
                upload_pdf,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
                context_state,
            ],
            outputs=[upload_status, kb_stats, papers_table],
        )
        add_note_btn.click(
            add_research_note,
            inputs=[
                note_title,
                note_content,
                note_tags,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
                context_state,
            ],
            outputs=[note_status, kb_stats, papers_table],
        )
        add_web_btn.click(
            add_web_source,
            inputs=[
                web_url,
                web_title,
                web_content,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
                context_state,
            ],
            outputs=[web_status, kb_stats, papers_table],
        )
        delete_paper_btn.click(
            delete_paper,
            inputs=[
                delete_paper_id,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
                context_state,
            ],
            outputs=[delete_status, kb_stats, papers_table],
        )

        reset_kb_btn.click(
            reset_kb,
            inputs=[year_from_kb, year_to_kb, min_citations_kb, context_state],
            outputs=[reset_kb_status, kb_stats, papers_table],
        )

        papers_table.select(
            select_kb_paper_with_context,
            inputs=[
                papers_table,
                context_state,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
            ],
            outputs=[context_state, kb_selected_paper_id, kb_stats, papers_table],
        )

        papers_table.select(
            select_kb_paper_with_context,
            inputs=[
                papers_table,
                context_state,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
            ],
            outputs=[context_state, current_paper_id, kb_stats, papers_table],
        )

        open_kb_citations_btn.click(
            open_kb_citations,
            inputs=[
                kb_selected_paper_id,
                citation_ui["direction"],
                citation_ui["depth"],
            ],
            outputs=[
                citation_ui["paper_input"],
                citation_ui["summary_output"],
                citation_ui["citing_output"],
                citation_ui["cited_output"],
                citation_ui["connected_output"],
                citation_ui["related_output"],
                citation_ui["network_plot"],
                main_tabs,
            ],
        )

        refresh_context_btn.click(
            refresh_context_choices,
            inputs=[context_state],
            outputs=[context_state, current_researcher,
                     topbar_researcher, topbar_clear_btn],
        )

        researcher_select.change(
            sync_researcher_context_with_table,
            inputs=[
                researcher_select,
                context_state,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
            ],
            outputs=[
                context_state,
                current_researcher,
                researcher_select,
                citation_ui["researcher_dropdown"],
                kb_stats,
                papers_table,
            ],
        )

        citation_ui["researcher_dropdown"].change(
            sync_researcher_context_with_table,
            inputs=[
                citation_ui["researcher_dropdown"],
                context_state,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
            ],
            outputs=[
                context_state,
                current_researcher,
                researcher_select,
                citation_ui["researcher_dropdown"],
                kb_stats,
                papers_table,
            ],
        )

        current_researcher.change(
            sync_researcher_context_with_table,
            inputs=[
                current_researcher,
                context_state,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
            ],
            outputs=[
                context_state,
                current_researcher,
                researcher_select,
                citation_ui["researcher_dropdown"],
                kb_stats,
                papers_table,
            ],
        )

        citation_ui["paper_input"].change(
            sync_paper_context_with_table,
            inputs=[
                citation_ui["paper_input"],
                context_state,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
            ],
            outputs=[
                context_state,
                current_paper_id,
                citation_ui["paper_input"],
                kb_stats,
                papers_table,
            ],
        )

        current_paper_id.change(
            sync_paper_context_with_table,
            inputs=[
                current_paper_id,
                context_state,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
            ],
            outputs=[
                context_state,
                current_paper_id,
                citation_ui["paper_input"],
                kb_stats,
                papers_table,
            ],
        )

        analyze_kb_btn.click(
            load_kb_into_analysis,
            inputs=[year_from_kb, year_to_kb, min_citations_kb],
            outputs=[data_info, column_select, group_by_col, main_tabs],
        )

        analyze_current_researcher_btn.click(
            load_kb_for_researcher,
            inputs=[current_researcher, year_from_kb, year_to_kb, min_citations_kb],
            outputs=[data_info, column_select, group_by_col, main_tabs],
        )

        analyze_current_paper_btn.click(
            load_kb_for_paper,
            inputs=[current_paper_id],
            outputs=[data_info, column_select, group_by_col, main_tabs],
        )
        export_bibtex_btn.click(
            export_bibtex,
            outputs=[export_status, bibtex_download],
        )

        # Model selector events
        def refresh_model_list_with_status(state):
            dd_update, current = refresh_model_list()
            new_state = dict(state or {})
            new_state["model_name"] = current
            return dd_update, current, new_state, _render_status_bar(new_state)

        def switch_model_with_status(model_name, state):
            result = switch_model(model_name)
            new_state = dict(state or {})
            new_state["model_name"] = model_name
            return result, new_state, _render_status_bar(new_state)

        refresh_models_btn.click(
            refresh_model_list_with_status,
            inputs=[context_state],
            outputs=[model_dropdown, current_model_display, context_state, status_bar],
        )
        model_dropdown.change(
            switch_model_with_status,
            inputs=[model_dropdown, context_state],
            outputs=[current_model_display, context_state, status_bar],
        )

        # Initialize model list on load
        app.load(
            refresh_model_list_with_status,
            inputs=[context_state],
            outputs=[model_dropdown, current_model_display, context_state, status_bar],
        )

        # Initialize knowledge base stats/table on load
        app.load(
            refresh_stats_and_table,
            inputs=[year_from_kb, year_to_kb, min_citations_kb],
            outputs=[kb_stats, papers_table],
        )

        # Initialize researcher dropdowns from persisted data on load
        app.load(
            refresh_context_choices,
            inputs=[context_state],
            outputs=[context_state, current_researcher,
                     topbar_researcher, topbar_clear_btn],
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
                fetch_papers,
                fetch_papers_limit,
                researcher_results_state,
            ],
            outputs=[
                lookup_status,
                results_table,
                researcher_results_state,
                web_results_output,
                researcher_select,
                seed_paper_select,
                researcher_papers_state,
            ],
        )

        lookup_btn.click(
            refresh_context_from_lookup,
            inputs=[
                researcher_results_state,
                context_state,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
            ],
            outputs=[context_state, current_researcher, kb_stats, papers_table],
        )

        from research_agent.ui.components.citation_explorer import (
            refresh_researcher_dropdown,
        )

        lookup_btn.click(
            refresh_researcher_dropdown,
            outputs=[citation_ui["researcher_dropdown"]],
        )

        lookup_btn.click(
            sync_researcher_context_with_table,
            inputs=[
                current_researcher,
                context_state,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
            ],
            outputs=[
                context_state,
                current_researcher,
                researcher_select,
                citation_ui["researcher_dropdown"],
                kb_stats,
                papers_table,
            ],
        )

        clear_results_btn.click(
            clear_results,
            outputs=[
                lookup_status,
                results_table,
                researcher_results_state,
                web_results_output,
                researcher_select,
                seed_paper_select,
                researcher_papers_state,
            ],
        )

        load_papers_btn.click(
            load_researcher_papers,
            inputs=[researcher_select],
            outputs=[seed_paper_select, researcher_papers_state],
        )

        send_to_citations_btn.click(
            explore_researcher_citations,
            inputs=[
                researcher_select,
                citation_ui["direction"],
                citation_ui["depth"],
                seed_paper_select,
                researcher_papers_state,
            ],
            outputs=[
                citation_ui["paper_input"],
                citation_ui["summary_output"],
                citation_ui["citing_output"],
                citation_ui["cited_output"],
                citation_ui["connected_output"],
                citation_ui["related_output"],
                citation_ui["network_plot"],
            ],
        )

        ingest_researcher_btn.click(
            ingest_researcher_papers,
            inputs=[
                researcher_select,
                ingest_papers_limit,
                year_from_kb,
                year_to_kb,
                min_citations_kb,
            ],
            outputs=[
                ingest_researcher_status,
                kb_stats,
                papers_table,
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
            inputs=[
                data_input,
                analysis_type,
                plot_type,
                column_select,
                group_by_col,
                custom_query,
            ],
            outputs=[analysis_output, analysis_plot],
        )
        download_plot_btn.click(
            download_current_plot,
            outputs=[plot_download],
        )

        # Concept map events
        cm_generate_btn.click(
            generate_concept_map,
            inputs=[cm_max_chunks, cm_dim_method, cm_n_clusters],
            outputs=[cm_plot, cm_status],
        )

        # Settings toggle (hidden, kept for compat)
        settings_is_open = gr.State(False)

        # ── KB status display (top bar) ────────────────────────────────────
        def _get_kb_status():
            try:
                store, _, _ = _get_kb_resources()
                stats = store.get_stats()
                papers = stats.get("total_papers", 0)
                notes = stats.get("total_notes", 0)
                web = stats.get("total_web_sources", 0)
                chunks = stats.get("total_chunks", papers + notes + web)
                return f"KB: {chunks:,} chunks  ● Ready"
            except Exception:
                return "KB: — chunks  ● Loading"

        app.load(_get_kb_status, outputs=[kb_status_display])

        # ── Context map handlers ────────────────────────────────────────────
        def _ctx_kb_graph():
            from research_agent.ui.graph_visualization import build_kb_graph
            try:
                store, _, _ = _get_kb_resources()
                return build_kb_graph(store)
            except Exception as e:
                return None, f"Error: {e}"

        def _ctx_researcher_graph():
            from research_agent.ui.graph_visualization import build_researcher_graph
            try:
                return build_researcher_graph()
            except Exception as e:
                return None, f"Error: {e}"

        def _ctx_query_graph(qstate):
            from research_agent.ui.graph_visualization import build_query_graph
            try:
                q = (qstate or {}).get("query")
                chunks = (qstate or {}).get("chunks", [])
                return build_query_graph(q, chunks)
            except Exception as e:
                return None, f"Error: {e}"

        def _ctx_citation_graph(ctx_state):
            from research_agent.ui.graph_visualization import build_citation_graph
            try:
                paper_id = (ctx_state or {}).get("paper_id")
                store, _, _ = _get_kb_resources()
                return build_citation_graph(paper_id=paper_id, vector_store=store)
            except Exception as e:
                return None, f"Error: {e}"

        ctx_kb_btn.click(
            _ctx_kb_graph,
            outputs=[context_map_plot, context_map_status],
        )
        ctx_researcher_btn.click(
            _ctx_researcher_graph,
            outputs=[context_map_plot, context_map_status],
        )
        ctx_query_btn.click(
            _ctx_query_graph,
            inputs=[_query_state],
            outputs=[context_map_plot, context_map_status],
        )
        ctx_citations_btn.click(
            _ctx_citation_graph,
            inputs=[context_state],
            outputs=[context_map_plot, context_map_status],
        )

        # ── Context pill command handler ───────────────────────────────
        def _handle_ctx_command(cmd_raw, state):
            """Handle pill strip commands: remove/pin/toggle."""
            import json as _json
            if not cmd_raw:
                return state, gr.update(), gr.update(), ""
            # Strip timestamp suffix
            parts = cmd_raw.rsplit(":", 1)
            cmd = parts[0] if len(parts) > 1 and parts[-1].isdigit() else cmd_raw

            new_state = dict(state or {})

            if cmd.startswith("toggle:"):
                # toggle:soc:Feminist Theory or toggle:auth:h-index: 14
                rest = cmd[len("toggle:"):]
                layer_prefix, label = rest.split(":", 1)
                items_key = f"{layer_prefix}_items"
                # Deep-copy items list to avoid mutating original state
                new_state[items_key] = [dict(it) for it in new_state.get(items_key, [])]
                for it in new_state[items_key]:
                    if it["label"] == label:
                        it["enabled"] = not it.get("enabled", True)
                        break
            elif cmd.startswith("remove:chat:"):
                label = cmd[len("remove:chat:"):]
                new_state["chat_items"] = [
                    it for it in new_state.get("chat_items", [])
                    if it["label"] != label
                ]
            elif cmd.startswith("remove:auth:"):
                label = cmd[len("remove:auth:"):]
                new_state["auth_items"] = [
                    it for it in new_state.get("auth_items", [])
                    if it["label"] != label or it.get("auto", False)
                ]
            elif cmd.startswith("pin:"):
                label = cmd[len("pin:"):]
                auth_items = list(new_state.get("auth_items", []))
                if not any(it["label"] == label for it in auth_items):
                    auth_items.append({"label": label, "type": "pinned", "auto": False, "enabled": True})
                new_state["auth_items"] = auth_items

            pills_html = _render_context_pills(new_state)
            left_layer = new_state.get("active_layer_left", "context")
            items_key = _LEFT_LAYER_ITEMS.get(left_layer, "soc_items")
            has_items = bool(new_state.get(items_key, []))
            items_json = _json.dumps({
                "soc": new_state.get("soc_items", []),
                "auth": new_state.get("auth_items", []),
                "chat": new_state.get("chat_items", []),
            })
            return new_state, pills_html, gr.update(visible=has_items), items_json

        ctx_command_bus.input(
            _handle_ctx_command,
            inputs=[ctx_command_bus, context_state],
            outputs=[context_state, ctx_pills_html, ctx_pills_row, ctx_items_json],
        ).then(
            fn=None, inputs=[ctx_items_json], outputs=[],
            js="(json) => { try { sendContextItemsToExplorer(JSON.parse(json)); } catch(e) {} }",
        )

        # ── Explorer action bus (detail-panel buttons) ──────────────
        async def _handle_explorer_action(action_raw, state):
            """Handle action requests from explorer iframe detail-panel buttons."""
            import json as _json

            if not action_raw:
                return "", state

            # Strip timestamp suffix (same pattern as ctx_command_bus)
            parts = action_raw.rsplit(":", 1)
            payload_str = parts[0] if len(parts) > 1 and parts[-1].isdigit() else action_raw

            try:
                payload = _json.loads(payload_str)
            except (ValueError, TypeError):
                logger.warning(f"Invalid explorer action payload: {action_raw}")
                return _json.dumps({"success": False, "error": "Invalid request"}), state

            action = payload.get("action", "")
            node_id = payload.get("nodeId", "")
            researcher_name = payload.get("researcherName", "")
            layer = payload.get("layer", "auth")

            new_state = dict(state or {})

            try:
                if action == "load-papers":
                    result = await _action_load_papers(node_id, researcher_name, new_state)
                elif action == "discover-related":
                    result = await _action_discover_related(node_id, researcher_name, new_state)
                else:
                    result = {"success": False, "error": f"Not yet implemented: {action}"}
            except Exception as e:
                logger.error(f"Explorer action '{action}' failed: {e}", exc_info=True)
                result = {"success": False, "error": str(e)[:60]}

            return _json.dumps(result), new_state

        async def _action_load_papers(node_id, researcher_name, state):
            """AUTH mode: Fetch all papers for a researcher, return graph delta."""
            from research_agent.tools.researcher_registry import get_researcher_registry
            from research_agent.tools.researcher_lookup import ResearcherLookup

            registry = get_researcher_registry()
            profile = registry.get(researcher_name)

            if not profile:
                return {"success": False, "error": f"'{researcher_name}' not in registry"}

            oa_id = profile.openalex_id
            s2_id = profile.semantic_scholar_id

            if not oa_id and not s2_id:
                return {"success": False, "error": "No API IDs for this researcher"}

            # Fetch papers from APIs
            lookup = ResearcherLookup()
            try:
                papers = []
                if s2_id:
                    papers = await lookup.fetch_author_papers_semantic_scholar(s2_id, limit=20)
                if len(papers) < 10 and oa_id:
                    oa_papers = await lookup.fetch_author_papers_openalex(oa_id, limit=20)
                    existing_dois = {p.doi for p in papers if p.doi}
                    existing_titles = {(p.title or "").lower() for p in papers}
                    for p in oa_papers:
                        if p.doi and p.doi in existing_dois:
                            continue
                        if (p.title or "").lower() in existing_titles:
                            continue
                        papers.append(p)
                papers.sort(key=lambda p: p.citation_count or 0, reverse=True)
                papers = papers[:20]
            finally:
                await lookup.close()

            if not papers:
                return {"success": False, "error": "No papers found via APIs"}

            return _build_paper_delta(node_id, papers, state, edge_type="authorship")

        async def _action_discover_related(node_id, researcher_name, state):
            """CHAT mode: Discover related papers via semantic search, return delta."""
            from research_agent.tools.academic_search import AcademicSearchTools

            # Build query from researcher name + chat context keywords
            chat_items = state.get("chat_items", [])
            keywords = [it.get("label", "") for it in chat_items if it.get("label")]
            query = f"{researcher_name} {' '.join(keywords)}".strip() if keywords else researcher_name

            search = AcademicSearchTools()
            try:
                papers = await search.search_all(query, limit_per_source=8)
            finally:
                await search.close()

            if not papers:
                return {"success": False, "error": "No related papers found"}

            return _build_paper_delta(node_id, papers[:12], state, edge_type="semantic")

        def _build_paper_delta(anchor_node_id, papers, state, edge_type="authorship"):
            """Build a graph delta from new papers connected to an anchor node.

            Args:
                anchor_node_id: Existing node to connect new papers to
                papers: List of AuthorPaper or Paper objects
                state: Current context_state (must contain _prev_graph)
                edge_type: "authorship" or "semantic"

            Returns:
                dict with success, delta, paperCount
            """
            prev_graph = state.get("_prev_graph")
            if not prev_graph:
                return {"success": False, "error": "No graph state (try reloading)"}

            existing_ids = {n["id"] for n in prev_graph.get("nodes", [])}
            existing_link_keys = set()
            for l in prev_graph.get("links", []):
                s = l["source"]["id"] if isinstance(l["source"], dict) else l["source"]
                t = l["target"]["id"] if isinstance(l["target"], dict) else l["target"]
                existing_link_keys.add((s, t, l.get("type", "")))

            # Build mini-graph for new papers only
            gb = GraphBuilder()
            new_paper_ids = []
            for paper in papers:
                pid = gb.add_paper(paper)
                if pid not in existing_ids:
                    new_paper_ids.append(pid)

            if not new_paper_ids:
                return {"success": True, "delta": {"addNodes": [], "addLinks": []}, "paperCount": 0}

            # Add connecting edges
            for pid in new_paper_ids:
                if edge_type == "authorship":
                    gb.add_authorship_edge(anchor_node_id, pid)
                else:
                    gb.add_semantic_edge(anchor_node_id, pid, score=0.7)

            # Build structural context for new papers' fields
            gb.build_structural_context()
            mini = gb.to_dict()

            # Filter to truly new nodes and links
            add_nodes = [n for n in mini["nodes"] if n["id"] not in existing_ids]
            add_links = []
            for l in mini["links"]:
                s = l["source"]["id"] if isinstance(l["source"], dict) else l["source"]
                t = l["target"]["id"] if isinstance(l["target"], dict) else l["target"]
                if (s, t, l.get("type", "")) not in existing_link_keys:
                    add_links.append(l)

            delta = {
                "addNodes": add_nodes,
                "removeNodes": [],
                "addLinks": add_links,
                "removeLinks": [],
            }

            # Update stored graph state
            prev_graph["nodes"].extend(add_nodes)
            prev_graph["links"].extend(add_links)

            paper_count = len([n for n in add_nodes if n.get("type") == "paper"])
            return {"success": True, "delta": delta, "paperCount": paper_count}

        explorer_action_bus.input(
            _handle_explorer_action,
            inputs=[explorer_action_bus, context_state],
            outputs=[explorer_action_result, context_state],
        ).then(
            fn=None,
            inputs=[explorer_action_result],
            outputs=[],
            js="(result) => { try { sendActionResultToExplorer(JSON.parse(result)); } catch(e) {} }",
        )

        # ── Left-pane layer switching ─────────────────────────────────
        def _left_layer_btn_classes(layer):
            """Return elem_classes updates for the 3 left-pane layer buttons."""
            return (
                gr.update(elem_classes=["chat-layer-btn", "chat-layer-active"] if layer == "context" else ["chat-layer-btn"]),
                gr.update(elem_classes=["chat-layer-btn", "chat-layer-active"] if layer == "author" else ["chat-layer-btn"]),
                gr.update(elem_classes=["chat-layer-btn", "chat-layer-active"] if layer == "chat" else ["chat-layer-btn"]),
            )

        def switch_left_layer(layer, state):
            """Handle layer switch from left-pane buttons."""
            new_state = dict(state or {})
            new_state["active_layer_left"] = layer
            ctx_btn, auth_btn, chat_btn = _left_layer_btn_classes(layer)
            pills_html = _render_context_pills(new_state)
            items_key = _LEFT_LAYER_ITEMS.get(layer, "soc_items")
            has_items = bool(new_state.get(items_key, []))
            return new_state, ctx_btn, auth_btn, chat_btn, pills_html, gr.update(visible=has_items)

        _left_layer_outputs = [
            context_state, layer_context_btn, layer_author_btn, layer_chat_btn,
            ctx_pills_html, ctx_pills_row,
        ]

        layer_context_btn.click(
            lambda s: switch_left_layer("context", s),
            inputs=[context_state],
            outputs=_left_layer_outputs,
        )
        layer_author_btn.click(
            lambda s: switch_left_layer("author", s),
            inputs=[context_state],
            outputs=_left_layer_outputs,
        )
        layer_chat_btn.click(
            lambda s: switch_left_layer("chat", s),
            inputs=[context_state],
            outputs=_left_layer_outputs,
        )

        # ── Right-pane layer switching ────────────────────────────────
        # Right-pane layer changes from explorer iframe — sync Python state
        def _handle_right_layer(layer_raw, state):
            if not layer_raw:
                return state
            parts = layer_raw.rsplit(":", 1)
            layer = parts[0] if len(parts) > 1 and parts[-1].isdigit() else layer_raw
            new_state = dict(state or {})
            new_state["active_layer_right"] = layer
            return new_state

        right_layer_bus.input(
            _handle_right_layer,
            inputs=[right_layer_bus, context_state],
            outputs=[context_state],
        )

        # ── Top bar: researcher lookup → explorer graph ──────────────────
        def lookup_and_build_explorer(
            researcher_name, state, year_from=None, year_to=None, min_citations=0
        ):
            """Look up researcher via APIs, build graph, sync all surfaces."""
            logger.info(f"[Explorer] lookup_and_build_explorer called with: '{researcher_name}'")
            if not researcher_name or not researcher_name.strip():
                # Cleared field → reset everything
                renderer = ExplorerRenderer()
                mock_data = get_mock_graph_data()
                mock_data["active_layer"] = "structure"
                mock_soc = [
                    {"label": n["label"], "type": n["type"], "auto": True, "enabled": True}
                    for n in mock_data.get("nodes", [])
                    if n["type"] in ("field", "domain")
                ]
                mock_data["context_items"] = {"soc": mock_soc, "auth": [], "chat": []}
                mock_html = renderer.render(mock_data)
                reset_state = {
                    "researcher": None, "paper_id": None,
                    "active_layer_left": "context", "active_layer_right": "structure",
                    "chat_context": None,
                    "soc_items": mock_soc, "auth_items": [], "chat_items": [],
                    "is_anon": True,
                }
                choices = _get_researcher_choices()
                dd = gr.update(choices=choices, value=None)
                stats, table = refresh_stats_and_table(
                    year_from, year_to, min_citations, reset_state
                )
                ctx_b, auth_b, chat_b = _left_layer_btn_classes("context")

                pills_html = _render_context_pills(reset_state)
                return (
                    mock_html, reset_state,
                    gr.update(value=None, choices=choices),  # topbar dropdown reset
                    gr.update(visible=False),                # hide clear btn
                    dd, dd, dd, stats, table,
                    ctx_b, auth_b, chat_b,
                    pills_html, gr.update(visible=bool(mock_soc)),
                    _render_status_bar(reset_state),

                )

            name = researcher_name.strip()

            try:
                from research_agent.tools.researcher_lookup import ResearcherLookup
                from research_agent.tools.researcher_registry import get_researcher_registry
                import concurrent.futures

                explorer_cfg = (_load_config() or {}).get("explorer", {})
                cache_dir = explorer_cfg.get("cache_dir", "./cache/explorer")
                explorer_email = explorer_cfg.get("email")

                lookup = ResearcherLookup(
                    email=explorer_email,
                    use_openalex=True,
                    use_semantic_scholar=True,
                    use_web_search=False,
                    request_delay=0.3,
                    persistent_cache_dir=cache_dir,
                )

                async def _do_lookup():
                    try:
                        return await lookup.lookup_researcher(
                            name, fetch_papers=True, papers_limit=10
                        )
                    finally:
                        await lookup.close()

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    profile = pool.submit(asyncio.run, _do_lookup()).result()

                if not profile:
                    logger.warning(f"[Explorer] No results for '{name}'")
                    gr.Warning(f"Researcher '{name}' not found — try a different name or spelling.")
                    return (gr.update(),) * 15

                # Store in registry
                registry = get_researcher_registry()
                registry.add(profile)

                # ── Enrich papers with live API data ──────────────
                from research_agent.tools.academic_search import AcademicSearchTools

                async def _enrich_papers(papers, cfg):
                    search = AcademicSearchTools(
                        email=cfg.get("email"),
                        persistent_cache_dir=cfg.get("cache_dir", "./cache/explorer"),
                    )
                    try:
                        # OA status from Unpaywall
                        await search.enrich_papers_oa_status(papers)

                        # SPECTER2 embeddings + TLDRs from S2
                        s2_ids = [p.paper_id for p in papers if p.paper_id]
                        embeddings = await search.get_paper_embeddings(s2_ids) if s2_ids else {}

                        # CrossRef citation gap-filling
                        dois = [p.doi for p in papers if p.doi]
                        ref_map = {}
                        for doi in dois:
                            refs = await search.get_crossref_references(doi)
                            if refs:
                                ref_map[doi] = refs

                        return embeddings, ref_map
                    finally:
                        await search.close()

                try:
                    embeddings, ref_map = pool.submit(
                        asyncio.run, _enrich_papers(profile.top_papers or [], explorer_cfg)
                    ).result()
                except Exception as enrich_err:
                    logger.warning(f"[Explorer] Enrichment partial failure: {enrich_err}")
                    embeddings, ref_map = {}, {}

                # Build graph
                gb = GraphBuilder()
                researcher_id = gb.add_researcher(profile.to_dict())
                for paper in (profile.top_papers or []):
                    paper_id = gb.add_paper(paper)
                    gb.add_authorship_edge(researcher_id, paper_id)
                gb.build_structural_context()

                # Inject embeddings + compute semantic edges
                if embeddings:
                    gb.inject_embeddings(embeddings)
                    # Also inject TLDRs from papers that have them
                    tldrs = {}
                    for p in (profile.top_papers or []):
                        pid = getattr(p, "paper_id", None)
                        tldr = p.tldr if hasattr(p, "tldr") else None
                        if pid and tldr:
                            tldrs[pid] = tldr
                    if tldrs:
                        gb.inject_tldrs(tldrs)
                    gb.compute_semantic_edges(threshold=0.65)

                # Fill citation gaps from CrossRef
                if ref_map:
                    gb.fill_citation_gaps(ref_map)

                # Sync all surfaces
                new_state = dict(state or {})
                new_state["researcher"] = profile.name
                new_state["active_layer_left"] = "author"
                new_state["active_layer_right"] = "people"
                new_state["is_anon"] = False

                new_state["auth_items"] = _build_auth_items(profile, state)

                # Build soc_items from graph field/domain nodes
                new_state["soc_items"] = gb.get_structural_items()

                renderer = ExplorerRenderer()
                ctx_items = {
                    "soc": new_state["soc_items"],
                    "auth": new_state["auth_items"],
                    "chat": new_state.get("chat_items", []),
                }
                graph_data = gb.to_dict(active_layer="people", context_items=ctx_items)

                # Store graph data for incremental updates
                new_state["_prev_graph"] = graph_data

                html = renderer.render(graph_data)

                choices = _get_researcher_choices()
                if profile.name not in choices:
                    choices.insert(0, profile.name)
                dd = gr.update(choices=choices, value=profile.name)
                stats, table = refresh_stats_and_table(
                    year_from, year_to, min_citations, new_state
                )

                ctx_b, auth_b, chat_b = _left_layer_btn_classes("author")

                pills_html = _render_context_pills(new_state)
                has_items = bool(new_state.get("auth_items", []))
                return (
                    html, new_state,
                    gr.update(value=profile.name, choices=choices),  # topbar dropdown
                    gr.update(visible=True),                         # show clear btn
                    dd, dd, dd, stats, table,
                    ctx_b, auth_b, chat_b,
                    pills_html, gr.update(visible=has_items),
                    _render_status_bar(new_state),

                )

            except Exception as e:
                logger.error(f"[Explorer] {type(e).__name__}: {e}", exc_info=True)
                gr.Warning(f"Lookup failed: {e}")
                return (gr.update(),) * 15

        _lookup_outputs = [
            explorer_html, context_state, topbar_researcher,
            topbar_clear_btn,
            current_researcher, researcher_select,
            citation_ui["researcher_dropdown"],
            kb_stats, papers_table,
            layer_context_btn, layer_author_btn, layer_chat_btn,
            ctx_pills_html, ctx_pills_row,
            status_bar,
        ]

        def _instant_clear(state):
            """Instantly reset explorer to mock data — no API calls."""
            renderer = ExplorerRenderer()
            mock = get_mock_graph_data()
            mock["active_layer"] = "structure"
            mock_soc_items = [
                {"label": n["label"], "type": n["type"], "auto": True, "enabled": True}
                for n in mock.get("nodes", [])
                if n["type"] in ("field", "domain")
            ]
            mock["context_items"] = {"soc": mock_soc_items, "auth": [], "chat": []}
            reset_state = {
                "researcher": None, "paper_id": None,
                "active_layer_left": "context", "active_layer_right": "structure",
                "chat_context": None,
                "soc_items": mock_soc_items, "auth_items": [], "chat_items": [],
                "is_anon": True,
                "_prev_graph": mock,
            }
            choices = _get_researcher_choices()
            dd = gr.update(choices=choices, value=None)
            ctx_b, auth_b, chat_b = _left_layer_btn_classes("context")

            pills_html = _render_context_pills(reset_state)
            has_items = bool(mock_soc_items)
            return (
                renderer.render(mock), reset_state,
                gr.update(value=None, choices=choices),  # topbar dropdown reset
                gr.update(visible=False),                # hide clear btn
                dd, dd, dd,
                gr.update(), gr.update(),  # stats/table unchanged
                ctx_b, auth_b, chat_b,
                pills_html, gr.update(visible=has_items),
                _render_status_bar(reset_state),

            )

        def _build_auth_items(profile, state):
            """Build auth_items list with researcher metadata pills."""
            items = [
                {"label": profile.name, "type": "researcher", "auto": True, "enabled": True},
            ]
            if profile.h_index:
                items.append({"label": f"h-index: {profile.h_index}", "type": "metric", "auto": True, "enabled": True})
            if profile.citations_count:
                c = profile.citations_count
                cite_str = f"{c:,}" if c < 100_000 else f"{c // 1000}k"
                items.append({"label": f"{cite_str} citations", "type": "metric", "auto": True, "enabled": True})
            if profile.top_papers:
                items.append({"label": f"{len(profile.top_papers)} papers", "type": "metric", "auto": True, "enabled": True})
            if profile.affiliations:
                items.append({"label": profile.affiliations[0], "type": "affiliation", "auto": True, "enabled": True})
            # Preserve manually pinned items
            for it in (state or {}).get("auth_items", []):
                if not it.get("auto", False) and it["label"] != profile.name:
                    items.append(it)
            return items

        def _build_from_registry_full(name, state):
            """Build explorer from registry, returning all _lookup_outputs.

            Instant — no API calls. Returns None if researcher not in registry.
            """
            from research_agent.tools.researcher_registry import get_researcher_registry

            registry = get_researcher_registry()
            profile = registry.get(name.strip())
            if not profile:
                return None

            gb = GraphBuilder()
            researcher_id = gb.add_researcher(profile.to_dict())

            for paper in (profile.top_papers or []):
                paper_id = gb.add_paper(paper)
                gb.add_authorship_edge(researcher_id, paper_id)

            gb.build_structural_context()

            new_state = dict(state or {})
            new_state["active_layer_left"] = "author"
            new_state["active_layer_right"] = "people"
            new_state["is_anon"] = False
            new_state["researcher"] = profile.name
            new_state["auth_items"] = _build_auth_items(profile, state)
            new_state["soc_items"] = gb.get_structural_items()

            renderer = ExplorerRenderer()
            ctx_items = {
                "soc": new_state["soc_items"],
                "auth": new_state["auth_items"],
                "chat": new_state.get("chat_items", []),
            }
            graph_data = gb.to_dict(active_layer="people", context_items=ctx_items)
            new_state["_prev_graph"] = graph_data
            html = renderer.render(graph_data)

            choices = _get_researcher_choices()
            dd = gr.update(choices=choices, value=profile.name)
            ctx_b, auth_b, chat_b = _left_layer_btn_classes("author")

            pills_html = _render_context_pills(new_state)
            has_items = bool(new_state.get("auth_items", []))

            return (
                html, new_state,
                gr.update(value=profile.name, choices=choices),
                gr.update(visible=True),
                dd, dd, dd,
                gr.update(), gr.update(),  # stats/table unchanged
                ctx_b, auth_b, chat_b,
                pills_html, gr.update(visible=has_items),
                _render_status_bar(new_state),

            )

        def _on_researcher_dropdown_select(evt: gr.SelectData, state, yf, yt, mc):
            """Handle picking a researcher from the combobox dropdown.

            Fast path: if already in registry, build graph instantly.
            Slow path: otherwise, do full API lookup.
            """
            name = evt.value
            if not name or not name.strip():
                return _instant_clear(state)

            # Try instant build from registry first
            result = _build_from_registry_full(name, state)
            if result is not None:
                return result

            # Not in registry — full API lookup
            return lookup_and_build_explorer(name, state, yf, yt, mc)

        topbar_researcher.select(
            _on_researcher_dropdown_select,
            inputs=[context_state,
                    year_from_kb, year_to_kb, min_citations_kb],
            outputs=_lookup_outputs,
        )

        def _on_researcher_input(value, state, yf, yt, mc):
            """Handle combobox value change (typing + Enter, or programmatic).

            Guards against double-triggering when .select() already handled it:
            1. Empty + no current researcher → init no-op
            2. Same as current researcher → already loaded, skip
            3. Known choice → .select() handled it, skip
            4. In registry → instant build (no API)
            5. Otherwise → custom value, full API lookup
            """
            _noop = tuple(gr.update() for _ in _lookup_outputs)
            if not value or not value.strip():
                if not (state or {}).get("researcher"):
                    return _noop
                return _instant_clear(state)
            name = value.strip()
            # Already showing this researcher → skip
            if name == (state or {}).get("researcher", ""):
                return _noop
            # Known dropdown choice → .select() already fired, skip
            choices = _get_researcher_choices()
            if name in choices:
                return _noop
            # Try instant build from registry (e.g. typed a known name)
            result = _build_from_registry_full(name, state)
            if result is not None:
                return result
            # Truly new name: full API lookup
            return lookup_and_build_explorer(value, state, yf, yt, mc)

        topbar_researcher.change(
            _on_researcher_input,
            inputs=[topbar_researcher, context_state,
                    year_from_kb, year_to_kb, min_citations_kb],
            outputs=_lookup_outputs,
        )

        topbar_clear_btn.click(
            _instant_clear,
            inputs=[context_state],
            outputs=_lookup_outputs,
        )

        def build_explorer_from_registry(name, state):
            """Build explorer graph from an already-registered researcher profile.

            Used when a researcher is selected in the KM accordion dropdown
            (profile was already fetched by lookup_researchers and stored in
            the singleton registry).
            """
            if not name or not name.strip():
                renderer = ExplorerRenderer()
                mock = get_mock_graph_data()
                new_s = dict(state or {})
                new_s["_prev_graph"] = mock
                left_layer = new_s.get("active_layer_left", "context")
                right_layer = new_s.get("active_layer_right", "structure")
                ctx_b, auth_b, chat_b = _left_layer_btn_classes(left_layer)
                pills_html = _render_context_pills(new_s)
                items_key = _LEFT_LAYER_ITEMS.get(left_layer, "soc_items")
                has_items = bool(new_s.get(items_key, []))
                return renderer.render(mock), new_s, pills_html, gr.update(visible=has_items), ctx_b, auth_b, chat_b

            from research_agent.tools.researcher_registry import get_researcher_registry

            registry = get_researcher_registry()
            profile = registry.get(name.strip())

            if not profile:
                renderer = ExplorerRenderer()
                mock = get_mock_graph_data()
                new_s = dict(state or {})
                new_s["_prev_graph"] = mock
                left_layer = new_s.get("active_layer_left", "context")
                right_layer = new_s.get("active_layer_right", "structure")
                ctx_b, auth_b, chat_b = _left_layer_btn_classes(left_layer)
                pills_html = _render_context_pills(new_s)
                items_key = _LEFT_LAYER_ITEMS.get(left_layer, "soc_items")
                has_items = bool(new_s.get(items_key, []))
                return renderer.render(mock), new_s, pills_html, gr.update(visible=has_items), ctx_b, auth_b, chat_b

            gb = GraphBuilder()
            researcher_id = gb.add_researcher(profile.to_dict())

            for paper in (profile.top_papers or []):
                paper_id = gb.add_paper(paper)
                gb.add_authorship_edge(researcher_id, paper_id)

            gb.build_structural_context()

            new_state = dict(state or {})
            new_state["active_layer_left"] = "author"
            new_state["active_layer_right"] = "people"
            new_state["is_anon"] = False
            new_state["researcher"] = profile.name
            new_state["auth_items"] = _build_auth_items(profile, state)
            new_state["soc_items"] = gb.get_structural_items()

            renderer = ExplorerRenderer()
            ctx_items = {
                "soc": new_state["soc_items"],
                "auth": new_state["auth_items"],
                "chat": new_state.get("chat_items", []),
            }
            graph_data = gb.to_dict(active_layer="people", context_items=ctx_items)
            new_state["_prev_graph"] = graph_data
            html = renderer.render(graph_data)

            pills_html = _render_context_pills(new_state)
            has_items = bool(new_state.get("auth_items", []))
            ctx_b, auth_b, chat_b = _left_layer_btn_classes("author")

            return html, new_state, pills_html, gr.update(visible=has_items), ctx_b, auth_b, chat_b

        researcher_select.change(
            build_explorer_from_registry,
            inputs=[researcher_select, context_state],
            outputs=[explorer_html, context_state,
                     ctx_pills_html, ctx_pills_row,
                     layer_context_btn, layer_author_btn, layer_chat_btn],
        )

        def build_kb_explorer(state):
            """Build KB graph with optional researcher overlay."""
            store, _, _ = _get_kb_resources()
            papers = store.list_papers_detailed(limit=200)

            gb = GraphBuilder()

            # Add all KB papers as nodes
            for p in papers:
                gb.add_paper({
                    "id": p["paper_id"],
                    "title": p["title"],
                    "year": p.get("year"),
                    "citations": p.get("citation_count") or p.get("citations") or 0,
                    "fields": p.get("fields", "").split(", ") if p.get("fields") else [],
                    "venue": p.get("venue", ""),
                    "doi": p.get("doi", ""),
                    "authors": p.get("authors", ""),
                })

            # Overlay researcher if one is selected
            researcher_name = (state or {}).get("researcher")
            if researcher_name:
                from research_agent.tools.researcher_registry import get_researcher_registry
                registry = get_researcher_registry()
                profile = registry.get(researcher_name)

                if profile:
                    rid = gb.add_researcher(profile.to_dict())
                    # Connect researcher to matching KB papers
                    for p in papers:
                        authors = (p.get("authors") or "").lower()
                        if profile.normalized_name in authors:
                            pid = f"paper:{p['paper_id']}"
                            gb.add_authorship_edge(rid, pid)

            gb.build_structural_context()
            renderer = ExplorerRenderer()
            new_state = dict(state or {})
            graph_data = gb.to_dict()
            new_state["_prev_graph"] = graph_data
            return renderer.render(graph_data), new_state

        # KB view can be triggered programmatically if needed;
        # button was removed from top bar in combobox redesign.

    return app


def _find_available_port(preferred: int, max_attempts: int = 10) -> int:
    """Find an available port, starting with preferred.

    If the preferred port is occupied by a stale research_agent process,
    kill it and reuse the port. Otherwise try successive ports.
    """
    import socket
    import subprocess

    for attempt in range(max_attempts):
        candidate = preferred + attempt
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", candidate))
            sock.close()
            return candidate
        except OSError:
            sock.close()
            if attempt == 0:
                # Try to kill stale research_agent on the preferred port
                try:
                    result = subprocess.run(
                        ["lsof", "-ti", f":{candidate}"],
                        capture_output=True, text=True, timeout=3,
                    )
                    pids = result.stdout.strip().split()
                    for pid in pids:
                        # Only kill our own processes
                        cmd_result = subprocess.run(
                            ["ps", "-p", pid, "-o", "args="],
                            capture_output=True, text=True, timeout=3,
                        )
                        if "research_agent" in cmd_result.stdout:
                            logger.info(f"Killing stale research_agent (PID {pid}) on port {candidate}")
                            subprocess.run(["kill", pid], timeout=3)
                            import time
                            time.sleep(1)
                            # Verify it freed up
                            sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            try:
                                sock2.bind(("127.0.0.1", candidate))
                                sock2.close()
                                return candidate
                            except OSError:
                                sock2.close()
                except Exception as e:
                    logger.debug(f"Could not check/kill stale process: {e}")

            logger.info(f"Port {candidate} busy, trying {candidate + 1}")

    raise RuntimeError(f"No available port found in range {preferred}-{preferred + max_attempts}")


def launch_app(agent=None, port: int = 7860, share: bool = False, host: str = None):
    """
    Launch the Gradio app.

    Args:
        agent: ResearchAgent instance
        port: Port to run on (auto-increments if busy)
        share: Create public link
        host: Server hostname/IP to bind to (default: 127.0.0.1, use 0.0.0.0 for Docker)
    """
    actual_port = _find_available_port(port)
    if actual_port != port:
        logger.info(f"Using port {actual_port} (preferred {port} was busy)")

    app = create_app(agent)
    kwargs = {
        "server_port": actual_port,
        "share": share,
        "show_error": True,
        "theme": getattr(app, "_explorer_theme", None),
        "css": getattr(app, "_explorer_css", None),
        "head": getattr(app, "_pwa_head", None),
        "footer_links": [],
    }
    if host:
        kwargs["server_name"] = host
    app.launch(**kwargs)


if __name__ == "__main__":
    from research_agent.main import build_agent_from_config, load_config

    agent = None
    try:
        config = load_config()
        agent = build_agent_from_config(config)
        print("Agent loaded successfully")
    except Exception as e:
        print(f"Failed to load agent: {e}")
        import traceback
        traceback.print_exc()
        print("\nLaunching in demo mode...")

    launch_app(agent=agent)
