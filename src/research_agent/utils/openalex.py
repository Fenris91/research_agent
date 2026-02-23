"""Shared OpenAlex utilities used across academic search and researcher lookup."""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def reconstruct_abstract(inverted_index: Optional[Dict]) -> Optional[str]:
    """Reconstruct abstract from OpenAlex inverted index format.

    OpenAlex stores abstracts as ``{word: [positions]}`` for compression.
    Returns None when the index is missing or malformed.
    """
    if not inverted_index:
        return None

    try:
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))

        word_positions.sort()
        return " ".join(word for _, word in word_positions)
    except Exception:
        logger.debug("Failed to reconstruct abstract from inverted index")
        return None


def normalize_openalex_id(paper_id: str) -> str:
    """Strip the OpenAlex URL prefix from an ID, returning just the key (e.g. 'W12345')."""
    if paper_id.startswith("https://openalex.org/"):
        return paper_id.replace("https://openalex.org/", "")
    return paper_id


def is_openalex_id(paper_id: str) -> bool:
    """Return True if *paper_id* looks like an OpenAlex work ID (starts with 'W')."""
    return normalize_openalex_id(paper_id).startswith("W")


def extract_openalex_fields(
    work: dict,
    limit: int = 5,
    min_score: float = 0.3,
    max_level: int = 2,
) -> List[str]:
    """Extract high-signal fields from an OpenAlex work's concepts/topics.

    Filters by relevance score (>=*min_score*) and hierarchy level (<=*max_level*),
    then returns the top *limit* display names.  Falls back to unfiltered top-N
    when no concepts pass the filter.
    """
    concepts = work.get("concepts", []) or []
    scored = []
    for concept in concepts:
        if not concept or not concept.get("display_name"):
            continue
        score = concept.get("score", 0) or concept.get("relevance_score", 0) or 0
        level = concept.get("level")
        scored.append((concept["display_name"], score, level))

    filtered = [
        item
        for item in scored
        if item[1] >= min_score and (item[2] is None or item[2] <= max_level)
    ]
    ranked = sorted(filtered or scored, key=lambda x: x[1], reverse=True)

    fields: List[str] = []
    for name, _score, _level in ranked[:limit]:
        if name not in fields:
            fields.append(name)
    return fields


# ---------------------------------------------------------------------------
# Source-type label mappings — used in prompt construction and UI display
# ---------------------------------------------------------------------------

SOURCE_LABELS = {
    "local_kb": "Knowledge Base",
    "local_note": "Research Note",
    "local_web": "Saved Web Source",
    "semantic_scholar": "Semantic Scholar",
    "openalex": "OpenAlex",
    "web": "Web Search",
}

SOURCE_LABELS_LONG = {
    "local_kb": "Knowledge Base Paper",
    "local_note": "User Research Note",
    "local_web": "Saved Web Source",
    "semantic_scholar": "Semantic Scholar",
    "openalex": "OpenAlex",
    "web": "Web Search",
}

SOURCE_LABELS_SHORT = {
    "local_kb": "KB",
    "local_note": "note",
    "local_web": "web",
    "semantic_scholar": "S2",
    "openalex": "OA",
    "web": "web",
}
