"""SQLite-backed store for researcher profiles."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ResearcherStore:
    """Persist researcher profiles, their papers, and web results to SQLite."""

    def __init__(self, path: str = "./data/researchers.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS researchers (
                    normalized_name TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    openalex_id TEXT,
                    semantic_scholar_id TEXT,
                    affiliations TEXT,
                    works_count INTEGER DEFAULT 0,
                    citations_count INTEGER DEFAULT 0,
                    h_index INTEGER,
                    fields TEXT,
                    lookup_timestamp TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS researcher_papers (
                    researcher_name TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    title TEXT,
                    year INTEGER,
                    citation_count INTEGER,
                    venue TEXT,
                    doi TEXT,
                    abstract TEXT,
                    fields TEXT,
                    source TEXT,
                    PRIMARY KEY (researcher_name, paper_id),
                    FOREIGN KEY (researcher_name)
                        REFERENCES researchers(normalized_name) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS researcher_web_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    researcher_name TEXT NOT NULL,
                    title TEXT,
                    url TEXT,
                    snippet TEXT,
                    FOREIGN KEY (researcher_name)
                        REFERENCES researchers(normalized_name) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rp_researcher "
                "ON researcher_papers(researcher_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rw_researcher "
                "ON researcher_web_results(researcher_name)"
            )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_researcher(self, profile) -> None:
        """Save a single ResearcherProfile (upsert researcher, replace papers & web results)."""
        now = datetime.now(timezone.utc).isoformat()
        key = profile.normalized_name or profile.name.lower().strip()

        with self._connect() as conn:
            # Upsert researcher row
            conn.execute(
                """
                INSERT INTO researchers
                    (normalized_name, name, openalex_id, semantic_scholar_id,
                     affiliations, works_count, citations_count, h_index,
                     fields, lookup_timestamp, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(normalized_name) DO UPDATE SET
                    name=excluded.name,
                    openalex_id=excluded.openalex_id,
                    semantic_scholar_id=excluded.semantic_scholar_id,
                    affiliations=excluded.affiliations,
                    works_count=excluded.works_count,
                    citations_count=excluded.citations_count,
                    h_index=excluded.h_index,
                    fields=excluded.fields,
                    lookup_timestamp=excluded.lookup_timestamp,
                    updated_at=excluded.updated_at
                """,
                (
                    key,
                    profile.name,
                    profile.openalex_id,
                    profile.semantic_scholar_id,
                    json.dumps(profile.affiliations),
                    profile.works_count,
                    profile.citations_count,
                    profile.h_index,
                    json.dumps(profile.fields),
                    profile.lookup_timestamp,
                    now,
                ),
            )

            # Replace papers
            conn.execute(
                "DELETE FROM researcher_papers WHERE researcher_name = ?", (key,)
            )
            if profile.top_papers:
                conn.executemany(
                    """
                    INSERT INTO researcher_papers
                        (researcher_name, paper_id, title, year, citation_count,
                         venue, doi, abstract, fields, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            key,
                            p.paper_id,
                            p.title,
                            p.year,
                            p.citation_count,
                            p.venue,
                            p.doi,
                            p.abstract,
                            json.dumps(p.fields) if p.fields else None,
                            p.source,
                        )
                        for p in profile.top_papers
                    ],
                )

            # Replace web results
            conn.execute(
                "DELETE FROM researcher_web_results WHERE researcher_name = ?", (key,)
            )
            if profile.web_results:
                conn.executemany(
                    """
                    INSERT INTO researcher_web_results
                        (researcher_name, title, url, snippet)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            key,
                            wr.get("title"),
                            wr.get("url"),
                            wr.get("snippet"),
                        )
                        for wr in profile.web_results
                    ],
                )

    def save_researchers(self, profiles) -> None:
        """Batch-save multiple profiles in a single transaction."""
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            for profile in profiles:
                key = profile.normalized_name or profile.name.lower().strip()

                conn.execute(
                    """
                    INSERT INTO researchers
                        (normalized_name, name, openalex_id, semantic_scholar_id,
                         affiliations, works_count, citations_count, h_index,
                         fields, lookup_timestamp, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(normalized_name) DO UPDATE SET
                        name=excluded.name,
                        openalex_id=excluded.openalex_id,
                        semantic_scholar_id=excluded.semantic_scholar_id,
                        affiliations=excluded.affiliations,
                        works_count=excluded.works_count,
                        citations_count=excluded.citations_count,
                        h_index=excluded.h_index,
                        fields=excluded.fields,
                        lookup_timestamp=excluded.lookup_timestamp,
                        updated_at=excluded.updated_at
                    """,
                    (
                        key,
                        profile.name,
                        profile.openalex_id,
                        profile.semantic_scholar_id,
                        json.dumps(profile.affiliations),
                        profile.works_count,
                        profile.citations_count,
                        profile.h_index,
                        json.dumps(profile.fields),
                        profile.lookup_timestamp,
                        now,
                    ),
                )

                conn.execute(
                    "DELETE FROM researcher_papers WHERE researcher_name = ?", (key,)
                )
                if profile.top_papers:
                    conn.executemany(
                        """
                        INSERT INTO researcher_papers
                            (researcher_name, paper_id, title, year, citation_count,
                             venue, doi, abstract, fields, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                key,
                                p.paper_id,
                                p.title,
                                p.year,
                                p.citation_count,
                                p.venue,
                                p.doi,
                                p.abstract,
                                json.dumps(p.fields) if p.fields else None,
                                p.source,
                            )
                            for p in profile.top_papers
                        ],
                    )

                conn.execute(
                    "DELETE FROM researcher_web_results WHERE researcher_name = ?",
                    (key,),
                )
                if profile.web_results:
                    conn.executemany(
                        """
                        INSERT INTO researcher_web_results
                            (researcher_name, title, url, snippet)
                        VALUES (?, ?, ?, ?)
                        """,
                        [
                            (
                                key,
                                wr.get("title"),
                                wr.get("url"),
                                wr.get("snippet"),
                            )
                            for wr in profile.web_results
                        ],
                    )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_researcher(self, normalized_name: str):
        """Load a single ResearcherProfile by normalized name."""
        from research_agent.tools.researcher_lookup import (
            AuthorPaper,
            ResearcherProfile,
        )

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM researchers WHERE normalized_name = ?",
                (normalized_name,),
            ).fetchone()
            if row is None:
                return None

            papers = conn.execute(
                "SELECT * FROM researcher_papers WHERE researcher_name = ?",
                (normalized_name,),
            ).fetchall()

            web_results = conn.execute(
                "SELECT title, url, snippet FROM researcher_web_results "
                "WHERE researcher_name = ?",
                (normalized_name,),
            ).fetchall()

        return self._build_profile(row, papers, web_results)

    def load_all(self) -> List:
        """Load all researcher profiles (avoids N+1 by grouping in Python)."""
        from research_agent.tools.researcher_lookup import (
            AuthorPaper,
            ResearcherProfile,
        )

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row

            researchers = conn.execute("SELECT * FROM researchers").fetchall()
            if not researchers:
                return []

            all_papers = conn.execute("SELECT * FROM researcher_papers").fetchall()
            all_web = conn.execute(
                "SELECT researcher_name, title, url, snippet "
                "FROM researcher_web_results"
            ).fetchall()

        # Group papers and web results by researcher
        papers_by_name: Dict[str, list] = {}
        for p in all_papers:
            papers_by_name.setdefault(p["researcher_name"], []).append(p)

        web_by_name: Dict[str, list] = {}
        for w in all_web:
            web_by_name.setdefault(w["researcher_name"], []).append(w)

        profiles = []
        for row in researchers:
            key = row["normalized_name"]
            profile = self._build_profile(
                row,
                papers_by_name.get(key, []),
                web_by_name.get(key, []),
            )
            profiles.append(profile)

        return profiles

    @staticmethod
    def _build_profile(row, paper_rows, web_rows):
        """Reconstruct a ResearcherProfile from DB rows."""
        from research_agent.tools.researcher_lookup import (
            AuthorPaper,
            ResearcherProfile,
        )

        papers = [
            AuthorPaper(
                paper_id=p["paper_id"],
                title=p["title"],
                year=p["year"],
                citation_count=p["citation_count"],
                venue=p["venue"],
                doi=p["doi"],
                abstract=p["abstract"],
                fields=json.loads(p["fields"]) if p["fields"] else None,
                source=p["source"] or "unknown",
            )
            for p in paper_rows
        ]

        web_results = [
            {"title": w["title"], "url": w["url"], "snippet": w["snippet"]}
            for w in web_rows
        ]

        return ResearcherProfile(
            name=row["name"],
            normalized_name=row["normalized_name"],
            openalex_id=row["openalex_id"],
            semantic_scholar_id=row["semantic_scholar_id"],
            affiliations=json.loads(row["affiliations"]) if row["affiliations"] else [],
            works_count=row["works_count"] or 0,
            citations_count=row["citations_count"] or 0,
            h_index=row["h_index"],
            fields=json.loads(row["fields"]) if row["fields"] else [],
            top_papers=papers,
            web_results=web_results,
            lookup_timestamp=row["lookup_timestamp"] or "",
        )

    # ------------------------------------------------------------------
    # Delete / utility
    # ------------------------------------------------------------------

    def delete_researcher(self, normalized_name: str) -> bool:
        """Delete a researcher and cascade to papers/web results."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM researchers WHERE normalized_name = ?",
                (normalized_name,),
            )
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Delete all researcher data."""
        with self._connect() as conn:
            conn.execute("DELETE FROM researcher_web_results")
            conn.execute("DELETE FROM researcher_papers")
            conn.execute("DELETE FROM researchers")

    def count(self) -> int:
        """Return the number of stored researchers."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM researchers").fetchone()[0]
