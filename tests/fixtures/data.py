"""Reusable test data factories for research agent tests.

Factory functions return plain dicts so callers can use them either as:
    Paper(**make_paper())          — dataclass construction
    make_paper(title="Custom")    — dict-based usage / direct assertions

Field names match the actual dataclasses exactly:
    Paper            — src/research_agent/tools/academic_search.py
    AuthorPaper      — src/research_agent/tools/researcher_lookup.py
    ResearcherProfile — src/research_agent/tools/researcher_lookup.py
    CitationPaper    — src/research_agent/tools/citation_explorer.py
"""

from typing import Any


def make_paper(**overrides: Any) -> dict:
    """Create a Paper-compatible dict with sensible defaults.

    Matches the Paper dataclass in academic_search.py:
        paper_id, title, abstract, year, authors, citation_count, doi,
        open_access_url, source, fields, venue, url, oa_status,
        tldr, specter_embedding
    """
    defaults = {
        "paper_id": "test_paper_001",
        "title": "Test Paper Title",
        "abstract": "This is a test abstract about research.",
        "year": 2024,
        "authors": ["Author One", "Author Two"],
        "citation_count": 42,
        "doi": "10.1234/test.2024",
        "open_access_url": None,
        "source": "semantic_scholar",
        "fields": ["Computer Science"],
        "venue": "Test Conference",
        "url": "https://example.com/paper",
        "oa_status": None,
        "tldr": None,
        "specter_embedding": None,
    }
    defaults.update(overrides)
    return defaults


def make_author_paper(**overrides: Any) -> dict:
    """Create an AuthorPaper-compatible dict with sensible defaults.

    Matches the AuthorPaper dataclass in researcher_lookup.py:
        paper_id, title, year, citation_count, venue, doi,
        abstract, fields, source
    """
    defaults = {
        "paper_id": "test_paper_001",
        "title": "Test Paper Title",
        "year": 2024,
        "citation_count": 42,
        "venue": "Test Conference",
        "doi": "10.1234/test.2024",
        "abstract": "This is a test abstract about research.",
        "fields": ["Computer Science"],
        "source": "semantic_scholar",
    }
    defaults.update(overrides)
    return defaults


def make_researcher(**overrides: Any) -> dict:
    """Create a ResearcherProfile-compatible dict with sensible defaults.

    Matches the ResearcherProfile dataclass in researcher_lookup.py:
        name, normalized_name, openalex_id, semantic_scholar_id,
        affiliations, works_count, citations_count, h_index, fields,
        recent_works, top_papers, web_results, lookup_timestamp
    """
    defaults = {
        "name": "Test Researcher",
        "normalized_name": "test researcher",
        "openalex_id": "A1234567890",
        "semantic_scholar_id": "12345678",
        "affiliations": ["Test University"],
        "works_count": 100,
        "citations_count": 5000,
        "h_index": 25,
        "fields": ["Computer Science", "AI"],
        "recent_works": [],
        "top_papers": [],
        "web_results": [],
        "lookup_timestamp": "2024-01-01T00:00:00+00:00",
    }
    defaults.update(overrides)
    return defaults


def make_citation_paper(**overrides: Any) -> dict:
    """Create a CitationPaper-compatible dict with sensible defaults.

    Matches the CitationPaper dataclass in citation_explorer.py:
        paper_id, title, year, authors, citation_count, abstract,
        venue, url
    """
    defaults = {
        "paper_id": "test_cite_001",
        "title": "Test Citation Paper",
        "year": 2023,
        "authors": ["Citation Author"],
        "citation_count": 10,
        "abstract": None,
        "venue": None,
        "url": None,
    }
    defaults.update(overrides)
    return defaults
