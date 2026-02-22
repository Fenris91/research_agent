"""SQLite-backed citation network store."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class CitationPaperRecord:
    paper_id: str
    title: str
    year: Optional[int]
    citation_count: Optional[int]
    source: Optional[str] = None
    doi: Optional[str] = None


class CitationStore:
    def __init__(self, path: str = "./data/citations.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT,
                    year INTEGER,
                    citations INTEGER,
                    source TEXT,
                    doi TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    seed_id TEXT,
                    source_id TEXT,
                    target_id TEXT,
                    relation TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_seed ON edges(seed_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)"
            )

    def save_papers(self, papers: Iterable[CitationPaperRecord]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                p.paper_id,
                p.title,
                p.year,
                p.citation_count,
                p.source,
                p.doi,
                now,
            )
            for p in papers
            if p.paper_id
        ]
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO papers (paper_id, title, year, citations, source, doi, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title=excluded.title,
                    year=excluded.year,
                    citations=excluded.citations,
                    source=excluded.source,
                    doi=excluded.doi,
                    updated_at=excluded.updated_at
                """,
                rows,
            )

    def save_edges(
        self,
        seed_id: str,
        edges: Iterable[tuple[str, str, str]],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rows = [(seed_id, s, t, rel, now) for s, t, rel in edges if s and t]
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO edges (seed_id, source_id, target_id, relation, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
