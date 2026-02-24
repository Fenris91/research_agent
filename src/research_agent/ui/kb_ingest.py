"""Unified KB ingestion helpers for UI flows."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from research_agent.utils.field_enrichment import enrich_fields

logger = logging.getLogger(__name__)


def _build_paper_content(
    title: str,
    abstract: str = "",
    venue: str = "",
    fields: Optional[List[str]] = None,
    doi: Optional[str] = None,
) -> str:
    parts = [title]
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if venue:
        parts.append(f"Venue: {venue}")
    if fields:
        parts.append(f"Fields: {', '.join(fields)}")
    if doi:
        parts.append(f"DOI: {doi}")
    return "\n".join(parts).strip()


def ingest_paper_to_kb(
    store: Any,
    embedder: Any,
    paper_id: str,
    title: str,
    abstract: str = "",
    venue: str = "",
    fields: Optional[List[str]] = None,
    doi: Optional[str] = None,
    year: Optional[int] = None,
    citation_count: Optional[int] = None,
    authors: Optional[List[str]] = None,
    source: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Embed and store a single paper summary in KB.

    Returns (added, reason).
    """
    if not paper_id:
        return False, "missing_id"

    if store.get_paper(paper_id):
        return False, "duplicate"

    # Enrich coarse/missing fields via OpenAlex + LLM fallback
    try:
        fields = enrich_fields(fields=fields, doi=doi, title=title, abstract=abstract)
    except Exception:
        logger.debug("Field enrichment failed, using original fields", exc_info=True)

    content = _build_paper_content(
        title=title,
        abstract=abstract,
        venue=venue,
        fields=fields,
        doi=doi,
    )
    if not content:
        return False, "empty"

    embeddings = embedder.embed_documents([content], batch_size=1, show_progress=False)

    metadata = {
        "title": title,
        "year": year,
        "venue": venue,
        "citation_count": citation_count,
        "doi": doi,
        "fields": fields or [],
        "authors": authors or [],
        "source": source or "unknown",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    store.add_paper(paper_id, [content], embeddings, metadata)

    # Auto-link to known researcher if not already set
    if not (extra_metadata or {}).get("researcher") and authors:
        try:
            from research_agent.tools.researcher_registry import get_researcher_registry

            registry = get_researcher_registry()
            for author in authors:
                match = registry.match_author_name(author)
                if match:
                    store.update_paper_metadata(paper_id, {"researcher": match})
                    break
        except Exception:
            pass  # Never fail ingestion due to auto-link

    return True, "added"


def normalize_metadata(
    base: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize metadata for KB ingestion."""
    metadata = dict(base or {})
    if extra:
        metadata.update(extra)

    if "authors" in metadata and isinstance(metadata["authors"], str):
        metadata["authors"] = [a.strip() for a in metadata["authors"].split(",")]

    standard_keys = [
        "title",
        "year",
        "authors",
        "doi",
        "fields",
        "venue",
        "source",
        "researcher",
        "ingest_source",
    ]
    for key in standard_keys:
        metadata.setdefault(key, None)

    return metadata
