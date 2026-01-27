"""Search, lookup, and citation tools for the research agent."""

from .academic_search import AcademicSearchTools, Paper, search_papers
from .web_search import WebSearchTool, WebResult
from .researcher_lookup import ResearcherLookup, ResearcherProfile, AuthorPaper
from .researcher_registry import ResearcherRegistry, get_researcher_registry
from .researcher_file_parser import (
    parse_researchers_file,
    parse_researchers_text,
    validate_name,
)

__all__ = [
    "AcademicSearchTools",
    "Paper",
    "search_papers",
    "WebSearchTool",
    "WebResult",
    "ResearcherLookup",
    "ResearcherProfile",
    "AuthorPaper",
    "ResearcherRegistry",
    "get_researcher_registry",
    "parse_researchers_file",
    "parse_researchers_text",
    "validate_name",
]
