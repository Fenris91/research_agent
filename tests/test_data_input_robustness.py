"""Tests for data input robustness — hardening against None, empty, and malformed inputs."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import asdict

from research_agent.db.vector_store import ResearchVectorStore
from research_agent.utils.openalex import (
    reconstruct_abstract,
    normalize_openalex_id,
    is_openalex_id,
    SOURCE_LABELS,
    SOURCE_LABELS_LONG,
    SOURCE_LABELS_SHORT,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class _FakeCollection:
    """Minimal stub for ChromaDB collection."""

    def __init__(self):
        self.added = []

    def add(self, **kwargs):
        self.added.append(kwargs)

    def get(self, **kwargs):
        return {"ids": [], "documents": [], "metadatas": []}

    def query(self, **kwargs):
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def delete(self, **kwargs):
        pass


@pytest.fixture
def fake_store():
    """Create a ResearchVectorStore with fake collections."""
    store = object.__new__(ResearchVectorStore)
    store.papers = _FakeCollection()
    store.notes = _FakeCollection()
    store.web_sources = _FakeCollection()
    store.reranker = None
    store._meta = None
    return store


# ---------------------------------------------------------------------------
# vector_store: add_paper validation
# ---------------------------------------------------------------------------


class TestAddPaperValidation:
    def test_empty_paper_id_raises(self, fake_store):
        with pytest.raises(ValueError, match="paper_id cannot be empty"):
            fake_store.add_paper("", ["chunk"], [[0.1] * 768], {"title": "T"})

    def test_whitespace_paper_id_raises(self, fake_store):
        with pytest.raises(ValueError, match="paper_id cannot be empty"):
            fake_store.add_paper("   ", ["chunk"], [[0.1] * 768], {"title": "T"})

    def test_mismatched_chunks_embeddings_raises(self, fake_store):
        with pytest.raises(ValueError, match="must match"):
            fake_store.add_paper("p1", ["a", "b"], [[0.1]], {"title": "T"})

    def test_empty_chunks_skips(self, fake_store):
        fake_store.add_paper("p1", [], [], {"title": "T"})
        assert len(fake_store.papers.added) == 0

    def test_none_metadata_values_sanitized(self, fake_store):
        """None values in metadata should be stripped by _sanitize_metadata."""
        fake_store.add_paper(
            "p1",
            ["text chunk"],
            [[0.1] * 768],
            {"title": "T", "year": None, "doi": None},
        )
        assert len(fake_store.papers.added) == 1
        meta = fake_store.papers.added[0]["metadatas"][0]
        assert "year" not in meta  # None values stripped
        assert "doi" not in meta
        assert meta["title"] == "T"

    def test_valid_paper_added(self, fake_store):
        fake_store.add_paper(
            "p1",
            ["chunk1", "chunk2"],
            [[0.1] * 768, [0.2] * 768],
            {"title": "My Paper", "year": 2024},
        )
        assert len(fake_store.papers.added) == 1
        call = fake_store.papers.added[0]
        assert call["ids"] == ["p1_chunk_0", "p1_chunk_1"]
        assert len(call["documents"]) == 2


# ---------------------------------------------------------------------------
# vector_store: add_note validation
# ---------------------------------------------------------------------------


class TestAddNoteValidation:
    def test_empty_note_id_raises(self, fake_store):
        with pytest.raises(ValueError, match="note_id cannot be empty"):
            fake_store.add_note("", "content", [0.1] * 768, {"title": "T"})

    def test_empty_content_skips(self, fake_store):
        fake_store.add_note("n1", "", [0.1] * 768, {"title": "T"})
        assert len(fake_store.notes.added) == 0

    def test_whitespace_content_skips(self, fake_store):
        fake_store.add_note("n1", "   ", [0.1] * 768, {"title": "T"})
        assert len(fake_store.notes.added) == 0

    def test_valid_note_added(self, fake_store):
        fake_store.add_note("n1", "my note", [0.1] * 768, {"title": "Note"})
        assert len(fake_store.notes.added) == 1


# ---------------------------------------------------------------------------
# vector_store: add_web_source validation
# ---------------------------------------------------------------------------


class TestAddWebSourceValidation:
    def test_empty_source_id_raises(self, fake_store):
        with pytest.raises(ValueError, match="source_id cannot be empty"):
            fake_store.add_web_source("", ["text"], [[0.1] * 768], {"url": "x"})

    def test_mismatched_chunks_embeddings_raises(self, fake_store):
        with pytest.raises(ValueError, match="must match"):
            fake_store.add_web_source("s1", ["a"], [[0.1], [0.2]], {"url": "x"})

    def test_empty_chunks_skips(self, fake_store):
        fake_store.add_web_source("s1", [], [], {"url": "x"})
        assert len(fake_store.web_sources.added) == 0

    def test_valid_web_source_added(self, fake_store):
        fake_store.add_web_source(
            "s1", ["content"], [[0.1] * 768], {"url": "https://example.com"}
        )
        assert len(fake_store.web_sources.added) == 1


# ---------------------------------------------------------------------------
# utils/openalex: shared utilities
# ---------------------------------------------------------------------------


class TestReconstructAbstract:
    def test_none_returns_none(self):
        assert reconstruct_abstract(None) is None

    def test_empty_dict_returns_none(self):
        assert reconstruct_abstract({}) is None

    def test_valid_inverted_index(self):
        idx = {"hello": [0], "world": [1]}
        assert reconstruct_abstract(idx) == "hello world"

    def test_multiple_positions(self):
        idx = {"the": [0, 3], "cat": [1], "sat": [2]}
        assert reconstruct_abstract(idx) == "the cat sat the"

    def test_malformed_index_returns_none(self):
        """Non-iterable positions should return None."""
        assert reconstruct_abstract({"word": 42}) is None


class TestNormalizeOpenalexId:
    def test_full_url(self):
        assert normalize_openalex_id("https://openalex.org/W12345") == "W12345"

    def test_bare_id(self):
        assert normalize_openalex_id("W12345") == "W12345"


class TestIsOpenalexId:
    def test_work_id(self):
        assert is_openalex_id("W12345") is True

    def test_full_url_work(self):
        assert is_openalex_id("https://openalex.org/W12345") is True

    def test_non_work_id(self):
        assert is_openalex_id("A12345") is False

    def test_doi(self):
        assert is_openalex_id("10.1234/test") is False


class TestSourceLabels:
    def test_all_variants_have_same_keys(self):
        """All three label dicts must cover the same source types."""
        assert set(SOURCE_LABELS.keys()) == set(SOURCE_LABELS_LONG.keys())
        assert set(SOURCE_LABELS.keys()) == set(SOURCE_LABELS_SHORT.keys())

    def test_web_key_present(self):
        assert "web" in SOURCE_LABELS

    def test_short_labels_are_short(self):
        for key, val in SOURCE_LABELS_SHORT.items():
            assert len(val) <= 4, f"Short label for {key} is too long: {val}"


# ---------------------------------------------------------------------------
# Metadata sanitization
# ---------------------------------------------------------------------------


class TestSanitizeMetadata:
    def test_none_values_stripped(self, fake_store):
        result = fake_store._sanitize_metadata({"a": None, "b": "val"})
        assert "a" not in result
        assert result["b"] == "val"

    def test_list_to_string(self, fake_store):
        result = fake_store._sanitize_metadata({"tags": ["a", "b", "c"]})
        assert result["tags"] == "a, b, c"

    def test_primitive_types_kept(self, fake_store):
        result = fake_store._sanitize_metadata({
            "s": "str", "i": 42, "f": 3.14, "b": True
        })
        assert result == {"s": "str", "i": 42, "f": 3.14, "b": True}

    def test_complex_type_stringified(self, fake_store):
        result = fake_store._sanitize_metadata({"nested": {"a": 1}})
        assert result["nested"] == "{'a': 1}"
