"""Unified KB ingestion helpers for UI flows."""

from typing import Any, Dict, List, Optional, Tuple


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
    citations: Optional[int] = None,
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
        "citations": citations,
        "doi": doi,
        "fields": fields or [],
        "authors": authors or [],
        "source": source or "unknown",
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    store.add_paper(paper_id, [content], embeddings, metadata)
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
