"""Tests for researcher persistence: KB paper linking and auto-link."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research_agent.db.kb_metadata_store import KBMetadataStore
from research_agent.tools.researcher_lookup import ResearcherProfile


# ============================================
# Fixtures
# ============================================


@pytest.fixture()
def meta_store(tmp_path):
    """Fresh KBMetadataStore in a temp directory."""
    return KBMetadataStore(path=str(tmp_path / "kb_meta.sqlite"))


@pytest.fixture()
def registry():
    """Fresh ResearcherRegistry (not singleton — mocked for isolation)."""
    from research_agent.tools.researcher_registry import ResearcherRegistry

    reg = object.__new__(ResearcherRegistry)
    reg._researchers = {}
    reg._registry_lock = __import__("threading").Lock()
    reg._store = MagicMock()
    profiles = [
        ResearcherProfile(name="David Harvey", normalized_name="david harvey"),
        ResearcherProfile(name="Donna Haraway", normalized_name="donna haraway"),
    ]
    for p in profiles:
        key = p.normalized_name or p.name.lower().strip()
        reg._researchers[key] = p
    return reg


@pytest.fixture()
def vector_store(tmp_path, meta_store):
    """Minimal ResearchVectorStore with tmp ChromaDB + shared meta_store."""
    from research_agent.db.vector_store import ResearchVectorStore

    store = ResearchVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        metadata_store_path=str(tmp_path / "kb_meta.sqlite"),
    )
    # Replace the auto-created meta with our shared one
    store._meta = meta_store
    return store


def _seed_papers(meta_store):
    """Insert test papers into SQLite."""
    meta_store.upsert_paper(
        "p1", "Spaces of Capital", "2025-01-01T00:00:00Z",
        authors="David Harvey, Neil Smith", researcher="",
    )
    meta_store.upsert_paper(
        "p2", "Simians Cyborgs Women", "2025-01-02T00:00:00Z",
        authors="Donna Haraway", researcher="",
    )
    meta_store.upsert_paper(
        "p3", "Random Paper", "2025-01-03T00:00:00Z",
        authors="John Doe", researcher="",
    )
    meta_store.upsert_paper(
        "p4", "Already Linked", "2025-01-04T00:00:00Z",
        authors="David Harvey", researcher="David Harvey",
    )
    meta_store.upsert_paper(
        "p5", "No Authors", "2025-01-05T00:00:00Z",
        authors="",
    )


# ============================================
# KBMetadataStore new methods
# ============================================


class TestUpdatePaperResearcher:
    def test_updates_existing(self, meta_store):
        meta_store.upsert_paper("p1", "Title", "2025-01-01T00:00:00Z")
        assert meta_store.update_paper_researcher("p1", "David Harvey") is True
        row = meta_store.get_paper("p1")
        assert row["researcher"] == "David Harvey"

    def test_returns_false_for_missing(self, meta_store):
        assert meta_store.update_paper_researcher("missing", "Nobody") is False


class TestCountPapersByResearcher:
    def test_counts_correctly(self, meta_store):
        meta_store.upsert_paper(
            "p1", "A", "2025-01-01T00:00:00Z", researcher="Harvey"
        )
        meta_store.upsert_paper(
            "p2", "B", "2025-01-02T00:00:00Z", researcher="Harvey"
        )
        meta_store.upsert_paper(
            "p3", "C", "2025-01-03T00:00:00Z", researcher="Other"
        )
        assert meta_store.count_papers_by_researcher("Harvey") == 2
        assert meta_store.count_papers_by_researcher("Other") == 1
        assert meta_store.count_papers_by_researcher("Nobody") == 0


class TestListUnlinkedPapers:
    def test_returns_only_unlinked_with_authors(self, meta_store):
        _seed_papers(meta_store)
        unlinked = meta_store.list_unlinked_papers()
        ids = {p["paper_id"] for p in unlinked}
        # p1, p2, p3 are unlinked with authors
        # p4 is linked, p5 has no authors
        assert ids == {"p1", "p2", "p3"}


# ============================================
# ResearcherRegistry.match_author_name
# ============================================


class TestMatchAuthorName:
    def test_exact_match(self, registry):
        assert registry.match_author_name("David Harvey") == "David Harvey"

    def test_case_insensitive(self, registry):
        assert registry.match_author_name("david harvey") == "David Harvey"
        assert registry.match_author_name("DONNA HARAWAY") == "Donna Haraway"

    def test_no_match(self, registry):
        assert registry.match_author_name("Unknown Person") is None

    def test_empty_string(self, registry):
        assert registry.match_author_name("") is None
        assert registry.match_author_name("  ") is None


# ============================================
# researcher_linker
# ============================================


class TestLinkPapersToResearchers:
    def test_full_scan_links_matching(self, vector_store, meta_store, registry):
        from research_agent.db.researcher_linker import link_papers_to_researchers

        _seed_papers(meta_store)
        linked, scanned = link_papers_to_researchers(vector_store, registry)
        assert scanned == 3  # p1, p2, p3 (unlinked with authors)
        assert linked == 2  # p1 (Harvey), p2 (Haraway)

        # Verify SQLite was updated
        row1 = meta_store.get_paper("p1")
        assert row1["researcher"] == "David Harvey"
        row2 = meta_store.get_paper("p2")
        assert row2["researcher"] == "Donna Haraway"

    def test_no_false_positives(self, vector_store, meta_store, registry):
        from research_agent.db.researcher_linker import link_papers_to_researchers

        _seed_papers(meta_store)
        link_papers_to_researchers(vector_store, registry)
        # p3 (John Doe) should remain unlinked
        row3 = meta_store.get_paper("p3")
        assert row3["researcher"] == ""

    def test_idempotent(self, vector_store, meta_store, registry):
        from research_agent.db.researcher_linker import link_papers_to_researchers

        _seed_papers(meta_store)
        linked1, _ = link_papers_to_researchers(vector_store, registry)
        linked2, scanned2 = link_papers_to_researchers(vector_store, registry)
        assert linked1 == 2
        assert linked2 == 0  # already linked, not in unlinked list
        assert scanned2 == 1  # only p3 remains unlinked


class TestLinkPapersForResearcher:
    def test_targeted_link(self, vector_store, meta_store, registry):
        from research_agent.db.researcher_linker import link_papers_for_researcher

        _seed_papers(meta_store)
        linked = link_papers_for_researcher(vector_store, "David Harvey")
        assert linked == 1  # p1

        row = meta_store.get_paper("p1")
        assert row["researcher"] == "David Harvey"

        # p2 should not be linked (Donna Haraway, not David Harvey)
        row2 = meta_store.get_paper("p2")
        assert row2["researcher"] == ""

    def test_no_match_returns_zero(self, vector_store, meta_store, registry):
        from research_agent.db.researcher_linker import link_papers_for_researcher

        _seed_papers(meta_store)
        linked = link_papers_for_researcher(vector_store, "Nobody Here")
        assert linked == 0
