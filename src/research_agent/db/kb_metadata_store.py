"""SQLite-backed metadata index for the Knowledge Base.

Sits alongside ChromaDB: SQLite handles listing / counting / filtering,
ChromaDB stays the source of truth for vectors and chunk content.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KBMetadataStore:
    """Fast metadata index for papers, notes, and web sources."""

    def __init__(self, path: str = "./data/kb_metadata.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Connection / schema
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    year INTEGER,
                    authors TEXT,
                    doi TEXT,
                    abstract TEXT,
                    venue TEXT,
                    url TEXT,
                    open_access_url TEXT,
                    citation_count INTEGER,
                    source TEXT,
                    ingest_source TEXT,
                    researcher TEXT,
                    fields TEXT,
                    added_at TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 1,
                    doc_id TEXT,
                    file_path TEXT,
                    file_name TEXT,
                    file_type TEXT,
                    page_count INTEGER,
                    word_count INTEGER
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_added_at ON papers(added_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_researcher ON papers(researcher)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    note_id TEXT PRIMARY KEY,
                    title TEXT,
                    tags TEXT,
                    content_preview TEXT,
                    added_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_sources (
                    source_id TEXT PRIMARY KEY,
                    title TEXT,
                    url TEXT,
                    added_at TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 1
                )
                """
            )

    # ------------------------------------------------------------------
    # Papers
    # ------------------------------------------------------------------

    def upsert_paper(
        self,
        paper_id: str,
        title: str,
        added_at: str,
        *,
        year: Optional[int] = None,
        authors: Optional[str] = None,
        doi: Optional[str] = None,
        abstract: Optional[str] = None,
        venue: Optional[str] = None,
        url: Optional[str] = None,
        open_access_url: Optional[str] = None,
        citation_count: Optional[int] = None,
        source: Optional[str] = None,
        ingest_source: Optional[str] = None,
        researcher: Optional[str] = None,
        fields: Optional[str] = None,
        chunk_count: int = 1,
        doc_id: Optional[str] = None,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        file_type: Optional[str] = None,
        page_count: Optional[int] = None,
        word_count: Optional[int] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO papers
                    (paper_id, title, year, authors, doi, abstract, venue, url,
                     open_access_url, citation_count, source, ingest_source,
                     researcher, fields, added_at, chunk_count,
                     doc_id, file_path, file_name, file_type,
                     page_count, word_count)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title=excluded.title,
                    year=excluded.year,
                    authors=excluded.authors,
                    doi=excluded.doi,
                    abstract=excluded.abstract,
                    venue=excluded.venue,
                    url=excluded.url,
                    open_access_url=excluded.open_access_url,
                    citation_count=excluded.citation_count,
                    source=excluded.source,
                    ingest_source=excluded.ingest_source,
                    researcher=excluded.researcher,
                    fields=excluded.fields,
                    added_at=excluded.added_at,
                    chunk_count=excluded.chunk_count,
                    doc_id=excluded.doc_id,
                    file_path=excluded.file_path,
                    file_name=excluded.file_name,
                    file_type=excluded.file_type,
                    page_count=excluded.page_count,
                    word_count=excluded.word_count
                """,
                (
                    paper_id, title, year, authors, doi, abstract, venue, url,
                    open_access_url, citation_count, source, ingest_source,
                    researcher, fields, added_at, chunk_count,
                    doc_id, file_path, file_name, file_type,
                    page_count, word_count,
                ),
            )

    def upsert_paper_from_metadata(
        self, paper_id: str, metadata: Dict[str, Any], chunk_count: int = 1
    ) -> None:
        """Convenience: extract fields from a ChromaDB metadata dict."""
        self.upsert_paper(
            paper_id=paper_id,
            title=metadata.get("title", "Unknown"),
            added_at=metadata.get("added_at", datetime.now(timezone.utc).isoformat()),
            year=_int_or_none(metadata.get("year")),
            authors=metadata.get("authors"),
            doi=metadata.get("doi"),
            abstract=metadata.get("abstract"),
            venue=metadata.get("venue"),
            url=metadata.get("url"),
            open_access_url=metadata.get("open_access_url"),
            citation_count=_int_or_none(
                metadata.get("citation_count") or metadata.get("citations")
            ),
            source=metadata.get("source"),
            ingest_source=metadata.get("ingest_source"),
            researcher=metadata.get("researcher"),
            fields=metadata.get("fields"),
            chunk_count=chunk_count,
            doc_id=metadata.get("doc_id"),
            file_path=metadata.get("file_path"),
            file_name=metadata.get("file_name"),
            file_type=metadata.get("file_type"),
            page_count=_int_or_none(metadata.get("page_count")),
            word_count=_int_or_none(metadata.get("word_count")),
        )

    def delete_paper(self, paper_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
            return cur.rowcount > 0

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
            return dict(row) if row else None

    def paper_exists(self, paper_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
            return row is not None

    def paper_exists_by_doi(self, doi: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE doi = ?", (doi,)
            ).fetchone()
            return row is not None

    def list_papers(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Return papers in the same format as ResearchVectorStore.list_papers()."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT paper_id, title, year, authors, added_at "
                "FROM papers ORDER BY added_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [
            {
                "paper_id": r["paper_id"],
                "title": r["title"],
                "year": r["year"],
                "authors": r["authors"] or "",
                "added_at": r["added_at"] or "",
            }
            for r in rows
        ]

    def list_papers_detailed(self, limit: int = 1000, offset: int = 0) -> List[Dict]:
        """Return papers in the same format as ResearchVectorStore.list_papers_detailed()."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT paper_id, title, year, authors, added_at, "
                "citation_count, venue, fields, source, researcher, "
                "ingest_source, doi "
                "FROM papers ORDER BY added_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [
            {
                "paper_id": r["paper_id"],
                "title": r["title"],
                "year": r["year"],
                "authors": r["authors"] or "",
                "added_at": r["added_at"] or "",
                "citation_count": r["citation_count"],
                "venue": r["venue"] or "",
                "fields": r["fields"] or "",
                "source": r["source"] or "",
                "researcher": r["researcher"] or "",
                "ingest_source": r["ingest_source"] or "",
                "doi": r["doi"] or "",
            }
            for r in rows
        ]

    def count_papers(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def list_researchers(self) -> List[str]:
        """Return distinct researcher names from papers table."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT researcher FROM papers "
                "WHERE researcher IS NOT NULL AND researcher != ''"
            ).fetchall()
            return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    def upsert_note(
        self,
        note_id: str,
        added_at: str,
        *,
        title: Optional[str] = None,
        tags: Optional[str] = None,
        content_preview: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notes (note_id, title, tags, content_preview, added_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(note_id) DO UPDATE SET
                    title=excluded.title,
                    tags=excluded.tags,
                    content_preview=excluded.content_preview,
                    added_at=excluded.added_at
                """,
                (note_id, title, tags, content_preview, added_at),
            )

    def upsert_note_from_metadata(
        self, note_id: str, metadata: Dict[str, Any], content: str = ""
    ) -> None:
        preview = content[:100] + "..." if len(content) > 100 else content
        self.upsert_note(
            note_id=note_id,
            added_at=metadata.get("added_at", datetime.now(timezone.utc).isoformat()),
            title=metadata.get("title", "Untitled"),
            tags=metadata.get("tags"),
            content_preview=preview,
        )

    def delete_note(self, note_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))
            return cur.rowcount > 0

    def list_notes(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Return notes in the same format as ResearchVectorStore.list_notes()."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT note_id, title, content_preview, added_at, tags "
                "FROM notes ORDER BY added_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [
            {
                "note_id": r["note_id"],
                "title": r["title"] or "Untitled",
                "preview": r["content_preview"] or "",
                "added_at": r["added_at"] or "",
                "tags": r["tags"] or "",
            }
            for r in rows
        ]

    def count_notes(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    # ------------------------------------------------------------------
    # Web sources
    # ------------------------------------------------------------------

    def upsert_web_source(
        self,
        source_id: str,
        added_at: str,
        *,
        title: Optional[str] = None,
        url: Optional[str] = None,
        chunk_count: int = 1,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO web_sources (source_id, title, url, added_at, chunk_count)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    title=excluded.title,
                    url=excluded.url,
                    added_at=excluded.added_at,
                    chunk_count=excluded.chunk_count
                """,
                (source_id, title, url, added_at, chunk_count),
            )

    def upsert_web_source_from_metadata(
        self, source_id: str, metadata: Dict[str, Any], chunk_count: int = 1
    ) -> None:
        self.upsert_web_source(
            source_id=source_id,
            added_at=metadata.get("added_at", datetime.now(timezone.utc).isoformat()),
            title=metadata.get("title"),
            url=metadata.get("url"),
            chunk_count=chunk_count,
        )

    def delete_web_source(self, source_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM web_sources WHERE source_id = ?", (source_id,)
            )
            return cur.rowcount > 0

    def list_web_sources(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Return web sources in a format matching list_notes() style."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source_id, title, url, added_at, chunk_count "
                "FROM web_sources ORDER BY added_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [
            {
                "source_id": r["source_id"],
                "title": r["title"] or "",
                "url": r["url"] or "",
                "added_at": r["added_at"] or "",
                "chunk_count": r["chunk_count"] or 1,
            }
            for r in rows
        ]

    def count_web_sources(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM web_sources").fetchone()[0]

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return stats dict with same keys as ResearchVectorStore.get_stats()."""
        with self._connect() as conn:
            papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            paper_chunks = conn.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) FROM papers"
            ).fetchone()[0]
            notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            web = conn.execute("SELECT COUNT(*) FROM web_sources").fetchone()[0]
            web_chunks = conn.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) FROM web_sources"
            ).fetchone()[0]
        return {
            "total_papers": papers,
            "total_paper_chunks": paper_chunks,
            "total_notes": notes,
            "total_web_sources": web,
            "total_web_chunks": web_chunks,
            "total_chunks": paper_chunks + notes + web_chunks,
        }

    # ------------------------------------------------------------------
    # Rebuild / clear
    # ------------------------------------------------------------------

    def rebuild_from_chromadb(self, vector_store: Any) -> None:
        """Full resync from ChromaDB collections into SQLite.

        Args:
            vector_store: A ResearchVectorStore instance.
        """
        logger.info("Rebuilding KB metadata index from ChromaDB...")

        # Papers
        all_paper_data = vector_store.papers.get(include=["metadatas"])
        all_metas = all_paper_data.get("metadatas") or []
        seen_papers: Dict[str, int] = {}  # paper_id → chunk_count
        paper_meta: Dict[str, Dict] = {}  # paper_id → first metadata seen

        for meta in all_metas:
            pid = meta.get("paper_id", "")
            if not pid:
                continue
            seen_papers[pid] = seen_papers.get(pid, 0) + 1
            if pid not in paper_meta:
                paper_meta[pid] = meta

        for pid, meta in paper_meta.items():
            self.upsert_paper_from_metadata(pid, meta, chunk_count=seen_papers[pid])

        # Notes
        note_data = vector_store.notes.get(include=["metadatas", "documents"])
        note_metas = note_data.get("metadatas") or []
        note_docs = note_data.get("documents") or []
        note_ids = note_data.get("ids") or []

        for i, meta in enumerate(note_metas):
            nid = meta.get("note_id", note_ids[i] if i < len(note_ids) else "")
            if not nid:
                continue
            content = note_docs[i] if i < len(note_docs) else ""
            self.upsert_note_from_metadata(nid, meta, content)

        # Web sources
        web_data = vector_store.web_sources.get(include=["metadatas"])
        web_metas = web_data.get("metadatas") or []
        seen_web: Dict[str, int] = {}
        web_meta: Dict[str, Dict] = {}

        for meta in web_metas:
            sid = meta.get("source_id", "")
            if not sid:
                continue
            seen_web[sid] = seen_web.get(sid, 0) + 1
            if sid not in web_meta:
                web_meta[sid] = meta

        for sid, meta in web_meta.items():
            self.upsert_web_source_from_metadata(sid, meta, chunk_count=seen_web[sid])

        logger.info(
            f"Rebuilt metadata index: {len(paper_meta)} papers, "
            f"{len(note_metas)} notes, {len(web_meta)} web sources"
        )

    def clear_papers(self) -> None:
        """Delete all paper metadata."""
        with self._connect() as conn:
            conn.execute("DELETE FROM papers")

    def clear_notes(self) -> None:
        """Delete all note metadata."""
        with self._connect() as conn:
            conn.execute("DELETE FROM notes")

    def clear_web_sources(self) -> None:
        """Delete all web source metadata."""
        with self._connect() as conn:
            conn.execute("DELETE FROM web_sources")

    def clear(self) -> None:
        """Delete all metadata."""
        with self._connect() as conn:
            conn.execute("DELETE FROM web_sources")
            conn.execute("DELETE FROM notes")
            conn.execute("DELETE FROM papers")


def _int_or_none(val: Any) -> Optional[int]:
    """Safely coerce a value to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
