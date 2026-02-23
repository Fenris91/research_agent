"""Tests for enrichment caching: AuthorPaper fields, ResearcherStore enrichment, and injection."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from research_agent.tools.researcher_lookup import AuthorPaper, ResearcherProfile
from research_agent.db.researcher_store import ResearcherStore
from research_agent.explorer.graph_builder import GraphBuilder


# ---------------------------------------------------------------------------
# AuthorPaper enrichment fields
# ---------------------------------------------------------------------------


class TestAuthorPaperEnrichmentFields:
    def test_fields_exist_and_default_to_none(self):
        p = AuthorPaper(paper_id="p1", title="Test Paper")
        assert p.oa_status is None
        assert p.open_access_url is None
        assert p.tldr is None

    def test_fields_survive_to_dict(self):
        p = AuthorPaper(
            paper_id="p1",
            title="Test Paper",
            oa_status="gold",
            open_access_url="https://example.com/pdf",
            tldr="A short summary of the paper.",
        )
        d = p.to_dict()
        assert d["oa_status"] == "gold"
        assert d["open_access_url"] == "https://example.com/pdf"
        assert d["tldr"] == "A short summary of the paper."

    def test_source_defaults_to_unknown(self):
        p = AuthorPaper(paper_id="p1", title="Test Paper")
        assert p.source == "unknown"


# ---------------------------------------------------------------------------
# ResearcherStore enrichment roundtrip
# ---------------------------------------------------------------------------


def _make_profile(name="Alice Smith", papers=None):
    return ResearcherProfile(
        name=name,
        normalized_name=name.lower().strip(),
        openalex_id="A111",
        semantic_scholar_id="S222",
        affiliations=["MIT"],
        works_count=50,
        citations_count=2000,
        h_index=25,
        fields=["Sociology"],
        top_papers=papers or [
            AuthorPaper(
                paper_id="p1",
                title="Paper One",
                doi="10.1000/one",
                oa_status="gold",
                open_access_url="https://example.com/p1.pdf",
                tldr="Summary of paper one.",
            ),
            AuthorPaper(
                paper_id="p2",
                title="Paper Two",
                doi="10.1000/two",
            ),
        ],
    )


class TestResearcherStoreEnrichmentRoundtrip:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = ResearcherStore(path=f"{self.tmpdir}/test.sqlite")

    def test_paper_enrichment_fields_persist(self):
        profile = _make_profile()
        self.store.save_researcher(profile)

        loaded = self.store.load_researcher("alice smith")
        assert loaded is not None
        p1 = next(p for p in loaded.top_papers if p.paper_id == "p1")
        assert p1.oa_status == "gold"
        assert p1.open_access_url == "https://example.com/p1.pdf"
        assert p1.tldr == "Summary of paper one."

        p2 = next(p for p in loaded.top_papers if p.paper_id == "p2")
        assert p2.oa_status is None
        assert p2.open_access_url is None
        assert p2.tldr is None

    def test_enrichment_artifacts_save_and_load(self):
        profile = _make_profile()
        self.store.save_researcher(profile)

        embeddings = {"p1": [0.1] * 768, "p2": [0.2] * 768}
        ref_map = {"10.1000/one": ["10.2000/ref1", "10.2000/ref2"]}
        tldrs = {"p1": "Summary of paper one.", "p2": "Summary of paper two."}

        self.store.save_researcher_enrichment(
            "alice smith", embeddings=embeddings, ref_map=ref_map, tldrs=tldrs,
        )

        loaded_emb, loaded_ref, loaded_tldrs = self.store.load_researcher_enrichment(
            "alice smith"
        )
        assert len(loaded_emb) == 2
        assert len(loaded_emb["p1"]) == 768
        assert loaded_ref == ref_map
        assert loaded_tldrs == tldrs

    def test_enrichment_empty_when_not_cached(self):
        profile = _make_profile()
        self.store.save_researcher(profile)

        emb, refs, tldrs = self.store.load_researcher_enrichment("alice smith")
        assert emb == {}
        assert refs == {}
        assert tldrs == {}

    def test_enrichment_upsert_overwrites(self):
        profile = _make_profile()
        self.store.save_researcher(profile)

        self.store.save_enrichment("alice smith", "tldrs", {"p1": "old"})
        self.store.save_enrichment("alice smith", "tldrs", {"p1": "new"})

        loaded = self.store.load_enrichment("alice smith", "tldrs")
        assert loaded == {"p1": "new"}

    def test_enrichment_cascades_on_delete(self):
        profile = _make_profile()
        self.store.save_researcher(profile)
        self.store.save_researcher_enrichment(
            "alice smith", embeddings={"p1": [0.1]},
        )

        self.store.delete_researcher("alice smith")

        emb, refs, tldrs = self.store.load_researcher_enrichment("alice smith")
        assert emb == {}


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestResearcherStoreMigration:
    def test_migration_adds_columns_to_existing_db(self):
        """Simulate an old DB without enrichment columns and verify migration."""
        tmpdir = tempfile.mkdtemp()
        db_path = f"{tmpdir}/old.sqlite"

        # Create old-style DB without enrichment columns
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("""
            CREATE TABLE researchers (
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
        """)
        conn.execute("""
            CREATE TABLE researcher_papers (
                researcher_name TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                title TEXT, year INTEGER, citation_count INTEGER,
                venue TEXT, doi TEXT, abstract TEXT, fields TEXT, source TEXT,
                PRIMARY KEY (researcher_name, paper_id),
                FOREIGN KEY (researcher_name) REFERENCES researchers(normalized_name) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE researcher_web_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                researcher_name TEXT NOT NULL,
                title TEXT, url TEXT, snippet TEXT,
                FOREIGN KEY (researcher_name) REFERENCES researchers(normalized_name) ON DELETE CASCADE
            )
        """)
        conn.commit()
        conn.close()

        # Now open with ResearcherStore — should migrate
        store = ResearcherStore(path=db_path)

        # Verify columns exist
        conn = sqlite3.connect(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(researcher_papers)").fetchall()}
        conn.close()
        assert "oa_status" in cols
        assert "open_access_url" in cols
        assert "tldr" in cols

        # Verify enrichment table exists
        conn = sqlite3.connect(db_path)
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "researcher_enrichment" in tables


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_graph_works_without_enrichment(self):
        """Graph builder produces a valid graph even with no cached enrichment."""
        gb = GraphBuilder()
        profile = _make_profile()
        gb.add_researcher(profile.to_dict())
        for p in profile.top_papers:
            pid = gb.add_paper(p)
            gb.add_authorship_edge("researcher:A111", pid)
        gb.build_structural_context()
        # No enrichment injection — should still produce valid graph
        data = gb.to_dict()
        assert len(data["nodes"]) > 0
        assert len(data["links"]) > 0

    def test_inject_with_empty_enrichment(self):
        """Injecting empty dicts is a no-op."""
        gb = GraphBuilder()
        profile = _make_profile()
        gb.add_researcher(profile.to_dict())
        for p in profile.top_papers:
            pid = gb.add_paper(p)
            gb.add_authorship_edge("researcher:A111", pid)
        gb.build_structural_context()

        # Empty dicts — should do nothing
        gb.inject_embeddings({})
        gb.inject_tldrs({})
        gb.fill_citation_gaps({})

        data = gb.to_dict()
        assert len(data["nodes"]) > 0
        # No semantic edges should exist
        semantic_edges = [e for e in data["links"] if e.get("type") == "semantic"]
        assert len(semantic_edges) == 0
