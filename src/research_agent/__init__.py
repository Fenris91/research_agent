"""Research Agent package.

This module intentionally avoids importing heavy optional dependencies at import
time (for example, torch/langgraph stacks used by the runtime agent). Symbols
are loaded lazily via ``__getattr__`` so lightweight modules can be imported in
minimal environments.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "ResearchAgent",
    "AgentConfig",
    "create_research_agent",
    "ResearchVectorStore",
    "EmbeddingModel",
    "get_embedder",
    "AcademicSearchTools",
    "WebSearchTool",
    "ResearcherLookup",
]

_EXPORT_MAP = {
    "ResearchAgent": ("research_agent.agents", "ResearchAgent"),
    "AgentConfig": ("research_agent.agents", "AgentConfig"),
    "create_research_agent": ("research_agent.agents", "create_research_agent"),
    "ResearchVectorStore": ("research_agent.db", "ResearchVectorStore"),
    "EmbeddingModel": ("research_agent.db", "EmbeddingModel"),
    "get_embedder": ("research_agent.db", "get_embedder"),
    "AcademicSearchTools": ("research_agent.tools", "AcademicSearchTools"),
    "WebSearchTool": ("research_agent.tools", "WebSearchTool"),
    "ResearcherLookup": ("research_agent.tools", "ResearcherLookup"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load exported package symbols on first access."""
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
