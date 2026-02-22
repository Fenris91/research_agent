"""Shared paper dataclasses for the research agent.

BasePaper provides the common fields shared across all paper representations.
Specialized subclasses add source-specific or context-specific fields.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional


@dataclass
class BasePaper:
    """Common fields shared by all paper representations.

    Every paper in the system has at least an ID, title, and optional
    bibliographic metadata. Subclasses add fields specific to their
    context (search results, citation networks, author profiles).
    """

    paper_id: str
    title: str
    year: Optional[int] = None
    authors: Optional[List[str]] = None
    citation_count: Optional[int] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    source: Optional[str] = None
    fields: Optional[List[str]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
