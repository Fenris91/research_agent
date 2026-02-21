"""
Explorer renderer using Jinja2 templates.

Renders the D3.js knowledge graph HTML from graph data.
Uses custom delimiters {= =} to avoid conflicts with JS {{ }}.
"""

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


TEMPLATE_DIR = Path(__file__).parent / "templates"


class ExplorerRenderer:
    """Renders knowledge graph HTML from graph data."""

    def __init__(self):
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            variable_start_string="{=",
            variable_end_string="=}",
            block_start_string="{%",
            block_end_string="%}",
            comment_start_string="{#",
            comment_end_string="#}",
        )
        self._template = self._env.get_template("explorer.html")

    def render(self, graph_data: dict) -> str:
        """Render the explorer HTML with embedded graph data.

        Wraps the output in an <iframe srcdoc="..."> so that Gradio's
        dynamic HTML updates actually execute the <script> tags.
        (Browsers ignore <script> tags injected via innerHTML.)
        """
        graph_json = json.dumps(graph_data)
        raw_html = self._template.render(graph_json=graph_json)
        # Wrap in a full HTML document inside an iframe
        # Escape quotes for the srcdoc attribute
        escaped = (raw_html
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))
        # Wrapper div with position:relative so the iframe can
        # use position:absolute to fill the space reliably
        # (height:100% on iframe fails when parent lacks explicit height).
        return (
            f'<div style="position:relative;width:100%;height:100%;flex:1;min-height:0;">'
            f'<iframe srcdoc="{escaped}" '
            f'style="position:absolute;inset:0;width:100%;height:100%;border:none;background:#0a0d13;" '
            f'sandbox="allow-scripts allow-same-origin"></iframe>'
            f'</div>'
        )
