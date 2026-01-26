"""Search, lookup, and citation tools for the research agent."""

from .academic_search import AcademicSearchTools, Paper, search_papers
from .web_search import WebSearchTool, WebResult
from .researcher_lookup import ResearcherLookup, ResearcherProfile
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
    "parse_researchers_file",
    "parse_researchers_text",
    "validate_name",
]
