"""Tests for KBMetadataStore — standalone SQLite metadata index."""

import tempfile
from pathlib import Path

import pytest

from research_agent.db.kb_metadata_store import KBMetadataStore


@pytest.fixture()
def store(tmp_path):
    """Fresh KBMetadataStore in a temp directory."""
    return KBMetadataStore(path=str(tmp_path / "kb_meta.sqlite"))


# ============================================
# Paper CRUD
# ============================================


class TestPapers:
    def test_upsert_and_get(self, store):
        store.upsert_paper("p1", "Title A", "2025-01-01T00:00:00Z", year=2023)
        row = store.get_paper("p1")
        assert row is not None
        assert row["title"] == "Title A"
        assert row["year"] == 2023

    def test_upsert_updates_existing(self, store):
        store.upsert_paper("p1", "Old", "2025-01-01T00:00:00Z")
        store.upsert_paper("p1", "New", "2025-01-02T00:00:00Z", year=2024)
        row = store.get_paper("p1")
        assert row["title"] == "New"
        assert row["year"] == 2024

    def test_delete(self, store):
        store.upsert_paper("p1", "Title", "2025-01-01T00:00:00Z")
        assert store.delete_paper("p1") is True
        assert store.get_paper("p1") is None
        assert store.delete_paper("p1") is False

    def test_paper_exists(self, store):
        assert store.paper_exists("p1") is False
        store.upsert_paper("p1", "T", "2025-01-01T00:00:00Z")
        assert store.paper_exists("p1") is True

    def test_paper_exists_by_doi(self, store):
        assert store.paper_exists_by_doi("10.1234/test") is False
        store.upsert_paper("p1", "T", "2025-01-01T00:00:00Z", doi="10.1234/test")
        assert store.paper_exists_by_doi("10.1234/test") is True

    def test_list_papers_format(self, store):
        store.upsert_paper("p1", "Alpha", "2025-01-01T00:00:00Z", authors="A, B")
        store.upsert_paper("p2", "Beta", "2025-01-02T00:00:00Z")
        papers = store.list_papers()
        assert len(papers) == 2
        # Newest first
        assert papers[0]["paper_id"] == "p2"
        assert papers[1]["paper_id"] == "p1"
        # Expected keys
        assert set(papers[0].keys()) == {"paper_id", "title", "year", "authors", "added_at"}

    def test_list_papers_pagination(self, store):
        for i in range(5):
            store.upsert_paper(f"p{i}", f"Paper {i}", f"2025-01-0{i+1}T00:00:00Z")
        assert len(store.list_papers(limit=2)) == 2
        assert len(store.list_papers(limit=10, offset=3)) == 2

    def test_list_papers_detailed_format(self, store):
        store.upsert_paper(
            "p1", "Title", "2025-01-01T00:00:00Z",
            year=2023, doi="10/x", venue="ICML", source="openalex",
            citation_count=42, researcher="smith",
        )
        detailed = store.list_papers_detailed()
        assert len(detailed) == 1
        d = detailed[0]
        assert d["citation_count"] == 42
        assert d["doi"] == "10/x"
        assert d["venue"] == "ICML"
        assert d["researcher"] == "smith"

    def test_count_papers(self, store):
        assert store.count_papers() == 0
        store.upsert_paper("p1", "T", "2025-01-01T00:00:00Z")
        store.upsert_paper("p2", "T2", "2025-01-02T00:00:00Z")
        assert store.count_papers() == 2

    def test_upsert_from_metadata(self, store):
        meta = {
            "title": "From Meta",
            "year": 2022,
            "authors": "X, Y",
            "added_at": "2025-06-01T00:00:00Z",
            "citation_count": 10,
        }
        store.upsert_paper_from_metadata("pm1", meta, chunk_count=3)
        row = store.get_paper("pm1")
        assert row["title"] == "From Meta"
        assert row["chunk_count"] == 3
        assert row["citation_count"] == 10


# ============================================
# Note CRUD
# ============================================


class TestNotes:
    def test_upsert_and_list(self, store):
        store.upsert_note("n1", "2025-01-01T00:00:00Z", title="My Note", tags="a, b")
        notes = store.list_notes()
        assert len(notes) == 1
        assert notes[0]["note_id"] == "n1"
        assert notes[0]["title"] == "My Note"
        assert notes[0]["tags"] == "a, b"
        assert set(notes[0].keys()) == {"note_id", "title", "preview", "added_at", "tags"}

    def test_delete(self, store):
        store.upsert_note("n1", "2025-01-01T00:00:00Z")
        assert store.delete_note("n1") is True
        assert store.delete_note("n1") is False

    def test_count(self, store):
        assert store.count_notes() == 0
        store.upsert_note("n1", "2025-01-01T00:00:00Z")
        assert store.count_notes() == 1

    def test_upsert_from_metadata(self, store):
        meta = {"title": "Note X", "tags": "tag1", "added_at": "2025-01-01T00:00:00Z"}
        store.upsert_note_from_metadata("n1", meta, content="Hello world content")
        notes = store.list_notes()
        assert notes[0]["title"] == "Note X"
        assert notes[0]["preview"] == "Hello world content"


# ============================================
# Web source CRUD
# ============================================


class TestWebSources:
    def test_upsert_and_count(self, store):
        store.upsert_web_source("w1", "2025-01-01T00:00:00Z", title="Site", url="https://x.com")
        assert store.count_web_sources() == 1

    def test_delete(self, store):
        store.upsert_web_source("w1", "2025-01-01T00:00:00Z")
        assert store.delete_web_source("w1") is True
        assert store.delete_web_source("w1") is False

    def test_upsert_from_metadata(self, store):
        meta = {"title": "Web", "url": "https://a.b", "added_at": "2025-01-01T00:00:00Z"}
        store.upsert_web_source_from_metadata("w1", meta, chunk_count=5)
        assert store.count_web_sources() == 1

    def test_web_source_exists(self, store):
        assert store.web_source_exists("w1") is False
        store.upsert_web_source("w1", "2025-01-01T00:00:00Z", title="Site")
        assert store.web_source_exists("w1") is True
        assert store.web_source_exists("w2") is False


# ============================================
# Aggregate stats
# ============================================


class TestStats:
    def test_empty(self, store):
        stats = store.get_stats()
        assert stats["total_papers"] == 0
        assert stats["total_notes"] == 0
        assert stats["total_web_sources"] == 0
        assert stats["total_chunks"] == 0

    def test_with_data(self, store):
        store.upsert_paper("p1", "T1", "2025-01-01T00:00:00Z", chunk_count=3)
        store.upsert_paper("p2", "T2", "2025-01-02T00:00:00Z", chunk_count=2)
        store.upsert_note("n1", "2025-01-01T00:00:00Z")
        store.upsert_web_source("w1", "2025-01-01T00:00:00Z", chunk_count=4)
        stats = store.get_stats()
        assert stats["total_papers"] == 2
        assert stats["total_paper_chunks"] == 5
        assert stats["total_notes"] == 1
        assert stats["total_web_sources"] == 1
        assert stats["total_web_chunks"] == 4
        assert stats["total_chunks"] == 5 + 1 + 4

    def test_stats_keys_match_vector_store(self, store):
        """Verify we return the same keys as ResearchVectorStore.get_stats()."""
        expected_keys = {
            "total_papers", "total_paper_chunks", "total_notes",
            "total_web_sources", "total_web_chunks", "total_chunks",
        }
        assert set(store.get_stats().keys()) == expected_keys


# ============================================
# Clear
# ============================================


class TestClear:
    def test_clear_all(self, store):
        store.upsert_paper("p1", "T", "2025-01-01T00:00:00Z")
        store.upsert_note("n1", "2025-01-01T00:00:00Z")
        store.upsert_web_source("w1", "2025-01-01T00:00:00Z")
        store.clear()
        assert store.count_papers() == 0
        assert store.count_notes() == 0
        assert store.count_web_sources() == 0
