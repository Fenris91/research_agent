"""Auto-link KB papers to known researchers in the registry."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from research_agent.db.vector_store import ResearchVectorStore
    from research_agent.tools.researcher_registry import ResearcherRegistry

logger = logging.getLogger(__name__)


def link_papers_to_researchers(
    vector_store: ResearchVectorStore,
    registry: ResearcherRegistry,
) -> Tuple[int, int]:
    """Scan all unlinked KB papers and link them to known researchers.

    Args:
        vector_store: Vector store (with SQLite metadata index)
        registry: Researcher registry with known profiles

    Returns:
        (linked_count, scanned_count)
    """
    meta = vector_store._meta
    if meta is None:
        return 0, 0

    unlinked = meta.list_unlinked_papers()
    linked = 0

    for paper in unlinked:
        authors_str = paper.get("authors", "")
        if not authors_str:
            continue
        authors = [a.strip() for a in authors_str.split(",")]
        for author in authors:
            match = registry.match_author_name(author)
            if match:
                vector_store.update_paper_metadata(
                    paper["paper_id"], {"researcher": match}
                )
                linked += 1
                break

    if linked:
        logger.info(f"Auto-linked {linked}/{len(unlinked)} papers to researchers")
    return linked, len(unlinked)


def link_papers_for_researcher(
    vector_store: ResearchVectorStore,
    researcher_name: str,
) -> int:
    """Link unlinked KB papers that match a specific researcher name.

    Args:
        vector_store: Vector store (with SQLite metadata index)
        researcher_name: Display name to match against paper authors

    Returns:
        Number of papers linked
    """
    meta = vector_store._meta
    if meta is None:
        return 0

    unlinked = meta.list_unlinked_papers()
    linked = 0
    name_lower = researcher_name.lower().strip()

    for paper in unlinked:
        authors_str = paper.get("authors", "")
        if not authors_str:
            continue
        authors = [a.strip() for a in authors_str.split(",")]
        for author in authors:
            if author.lower().strip() == name_lower:
                vector_store.update_paper_metadata(
                    paper["paper_id"], {"researcher": researcher_name}
                )
                linked += 1
                break

    if linked:
        logger.info(
            f"Auto-linked {linked} papers to researcher '{researcher_name}'"
        )
    return linked
