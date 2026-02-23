#!/usr/bin/env python
"""Retroactively enrich fields for existing KB papers with poor/missing fields.

Usage:
    uv run python scripts/enrich_existing_fields.py              # enrich all
    uv run python scripts/enrich_existing_fields.py --dry-run    # preview only
    uv run python scripts/enrich_existing_fields.py --limit 10   # first 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from research_agent.db.kb_metadata_store import KBMetadataStore
from research_agent.db.vector_store import ResearchVectorStore
from research_agent.utils.field_enrichment import enrich_fields, fields_need_enrichment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich fields for existing KB papers")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to process (0=all)")
    args = parser.parse_args()

    meta_store = KBMetadataStore()
    vector_store = ResearchVectorStore()

    papers = meta_store.list_papers_detailed(limit=10_000)
    logger.info("Found %d papers in KB", len(papers))

    candidates = []
    for p in papers:
        raw_fields = p.get("fields", "")
        if isinstance(raw_fields, str):
            try:
                field_list = json.loads(raw_fields) if raw_fields else []
            except json.JSONDecodeError:
                field_list = [f.strip() for f in raw_fields.split(",") if f.strip()]
        else:
            field_list = raw_fields or []

        if fields_need_enrichment(field_list):
            candidates.append((p, field_list))

    logger.info("%d papers need field enrichment", len(candidates))

    if args.limit > 0:
        candidates = candidates[: args.limit]

    enriched_count = 0
    failed_count = 0

    for paper, old_fields in candidates:
        pid = paper["paper_id"]
        title = paper.get("title", "")
        doi = paper.get("doi", "")

        logger.info("Processing: %s (doi=%s, fields=%s)", title[:60], doi or "none", old_fields)

        if args.dry_run:
            enriched_count += 1
            continue

        new_fields = enrich_fields(
            fields=old_fields,
            doi=doi if doi else None,
            title=title if title else None,
            abstract=None,  # abstract not in list_papers_detailed
        )

        if new_fields and new_fields != old_fields:
            fields_str = json.dumps(new_fields)

            # Update SQLite
            meta_store.update_paper_fields(pid, fields_str)

            # Update ChromaDB metadata
            try:
                existing = vector_store.papers.get(
                    where={"paper_id": pid}, include=["metadatas"]
                )
                if existing and existing.get("ids"):
                    for chunk_id, meta in zip(existing["ids"], existing["metadatas"]):
                        meta["fields"] = fields_str
                        vector_store.papers.update(ids=[chunk_id], metadatas=[meta])
            except Exception:
                logger.warning("Failed to update ChromaDB for %s", pid, exc_info=True)

            logger.info("  -> Enriched: %s", new_fields)
            enriched_count += 1
        else:
            logger.info("  -> No enrichment found")
            failed_count += 1

    action = "Would enrich" if args.dry_run else "Enriched"
    logger.info(
        "Done. %s %d/%d papers. %d not enrichable.",
        action, enriched_count, len(candidates), failed_count,
    )


if __name__ == "__main__":
    main()
