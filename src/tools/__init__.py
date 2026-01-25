# Search and analysis tools
from .academic_search import AcademicSearchTools, Paper, search_papers
from .web_search import WebSearchTool, WebResult
from .citation_explorer import CitationExplorer
from .researcher_lookup import ResearcherLookup, ResearcherProfile
from .researcher_file_parser import parse_researchers_file, parse_researchers_text

__all__ = [
    "AcademicSearchTools",
    "Paper",
    "search_papers",
    "WebSearchTool",
    "WebResult",
    "CitationExplorer",
    "ResearcherLookup",
    "ResearcherProfile",
    "parse_researchers_file",
    "parse_researchers_text",
]
