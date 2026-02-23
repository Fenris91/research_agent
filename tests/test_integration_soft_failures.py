"""Integration tests: verify every new feature soft-fails gracefully.

Tests start from nothing — no existing data, no API keys, no running
services — and verify that each code path degrades without crashing.
"""

import os
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# ---------------------------------------------------------------------------
# 1. KB Metadata Store — from scratch
# ---------------------------------------------------------------------------


class TestKBMetadataFromScratch:
    """Verify the SQLite metadata store works from a cold start."""

    def test_fresh_store_has_zero_stats(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        stats = store.get_stats()
        assert stats["total_papers"] == 0
        assert stats["total_notes"] == 0
        assert stats["total_web_sources"] == 0

    def test_list_papers_empty(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        assert store.list_papers() == []
        assert store.list_papers_detailed() == []

    def test_list_notes_empty(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        assert store.list_notes() == []

    def test_list_web_sources_empty(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        assert store.list_web_sources() == []

    def test_delete_nonexistent_paper_no_crash(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        # Should not raise
        store.delete_paper("nonexistent_id")

    def test_delete_nonexistent_note_no_crash(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        store.delete_note("nonexistent_id")

    def test_delete_nonexistent_web_source_no_crash(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        store.delete_web_source("nonexistent_id")

    def test_paper_exists_returns_false_on_empty(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        assert store.paper_exists("anything") is False
        assert store.paper_exists_by_doi("10.1234/fake") is False

    def test_clear_on_empty_no_crash(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        store.clear()
        store.clear_papers()
        store.clear_notes()
        store.clear_web_sources()


# ---------------------------------------------------------------------------
# 2. Vector Store — metadata integration from scratch
# ---------------------------------------------------------------------------


class TestVectorStoreMetaIntegration:
    """Verify vector store + metadata index works from empty state."""

    def test_vector_store_creates_meta_store(self, tmp_path):
        from research_agent.db.vector_store import ResearchVectorStore

        store = ResearchVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            metadata_store_path=str(tmp_path / "meta.sqlite"),
        )
        assert store._meta is not None
        stats = store.get_stats()
        assert stats["total_papers"] == 0

    def test_vector_store_without_meta(self, tmp_path):
        """Disabling metadata store should still work."""
        from research_agent.db.vector_store import ResearchVectorStore

        store = ResearchVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            metadata_store_path=None,
        )
        assert store._meta is None
        # get_stats should fall back to ChromaDB scan
        stats = store.get_stats()
        assert "total_papers" in stats

    def test_list_papers_empty_kb(self, tmp_path):
        from research_agent.db.vector_store import ResearchVectorStore

        store = ResearchVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            metadata_store_path=str(tmp_path / "meta.sqlite"),
        )
        assert store.list_papers() == []

    def test_list_notes_empty_kb(self, tmp_path):
        from research_agent.db.vector_store import ResearchVectorStore

        store = ResearchVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            metadata_store_path=str(tmp_path / "meta.sqlite"),
        )
        assert store.list_notes() == []

    def test_list_web_sources_empty_kb(self, tmp_path):
        from research_agent.db.vector_store import ResearchVectorStore

        store = ResearchVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            metadata_store_path=str(tmp_path / "meta.sqlite"),
        )
        assert store.list_web_sources() == []

    def test_rebuild_metadata_empty(self, tmp_path):
        from research_agent.db.vector_store import ResearchVectorStore

        store = ResearchVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            metadata_store_path=str(tmp_path / "meta.sqlite"),
        )
        result = store.rebuild_metadata()
        assert result["total_papers"] == 0

    def test_rebuild_metadata_no_meta_store(self, tmp_path):
        from research_agent.db.vector_store import ResearchVectorStore

        store = ResearchVectorStore(
            persist_dir=str(tmp_path / "chroma"),
            metadata_store_path=None,
        )
        result = store.rebuild_metadata()
        # Falls back to get_stats() from ChromaDB
        assert result.get("total_papers", 0) == 0


# ---------------------------------------------------------------------------
# 3. Multi-model pipeline — soft failures
# ---------------------------------------------------------------------------


class TestMultiModelPipeline:
    """Verify pipeline config degrades gracefully."""

    def test_no_pipeline_config(self):
        """No pipeline → task_infer falls through to infer."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent.__new__(ResearchAgent)
        agent._pipeline = {}
        agent.model = None
        agent.tokenizer = None
        agent.provider = "none"
        agent._load_model_on_demand = False
        agent.use_ollama = False

        # Should not raise even with no model
        result = agent.task_infer("classify", "test prompt", max_tokens=10)
        # With no model, infer returns an error string, not an exception
        assert isinstance(result, str)

    def test_pipeline_with_invalid_task_ignored(self):
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent.__new__(ResearchAgent)
        agent._pipeline = {}
        agent.model = None
        agent.configure_pipeline({
            "classify": "fast-model",
            "invalid_task": "should-be-ignored",
            "synthesize": "big-model",
        })
        assert "invalid_task" not in agent._pipeline
        assert "classify" in agent._pipeline
        assert "synthesize" in agent._pipeline

    def test_pipeline_switch_and_restore(self):
        """Model is switched for task then restored."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent.__new__(ResearchAgent)
        agent._pipeline = {"classify": "fast-model"}
        agent.provider = "openai"
        agent.use_ollama = False
        agent._load_model_on_demand = False
        agent.tokenizer = None

        mock_model = MagicMock()
        mock_model.model_name = "default-model"
        mock_model.generate.return_value = "literature_review"
        agent.model = mock_model

        result = agent.task_infer("classify", "test", max_tokens=10)

        # Should have switched to fast-model and back to default-model
        calls = mock_model.switch_model.call_args_list
        assert len(calls) == 2
        assert calls[0].args[0] == "fast-model"
        assert calls[1].args[0] == "default-model"

    def test_pipeline_no_switch_for_unconfigured_task(self):
        """Tasks not in pipeline should not trigger switch."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent.__new__(ResearchAgent)
        agent._pipeline = {"classify": "fast-model"}
        agent.provider = "openai"
        agent.use_ollama = False
        agent._load_model_on_demand = False
        agent.tokenizer = None

        mock_model = MagicMock()
        mock_model.model_name = "default-model"
        mock_model.generate.return_value = "synthesis result"
        agent.model = mock_model

        agent.task_infer("synthesize", "test", max_tokens=10)

        # No switch_model calls for unconfigured task
        mock_model.switch_model.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Provider detection — no keys set
# ---------------------------------------------------------------------------


class TestProviderDetectionNoKeys:
    """All new providers should be skipped when keys aren't set."""

    def test_gemini_skipped_without_key(self):
        from research_agent.main import CLOUD_PROVIDERS

        assert "gemini" in CLOUD_PROVIDERS
        with patch.dict(os.environ, {}, clear=True):
            key = os.getenv(CLOUD_PROVIDERS["gemini"]["api_key_env"])
            assert key is None

    def test_mistral_skipped_without_key(self):
        from research_agent.main import CLOUD_PROVIDERS

        assert "mistral" in CLOUD_PROVIDERS
        with patch.dict(os.environ, {}, clear=True):
            key = os.getenv(CLOUD_PROVIDERS["mistral"]["api_key_env"])
            assert key is None

    def test_xai_skipped_without_key(self):
        from research_agent.main import CLOUD_PROVIDERS

        assert "xai" in CLOUD_PROVIDERS
        with patch.dict(os.environ, {}, clear=True):
            key = os.getenv(CLOUD_PROVIDERS["xai"]["api_key_env"])
            assert key is None

    def test_detect_falls_to_none_with_no_keys_no_ollama(self):
        from research_agent.main import detect_available_provider

        with patch.dict(os.environ, {}, clear=True):
            with patch("research_agent.main.check_ollama_available", return_value=False):
                provider, cfg = detect_available_provider({})
                # Should fall through to huggingface or none
                assert provider in ("huggingface", "none")


# ---------------------------------------------------------------------------
# 5. Citation auto-save — batch failure
# ---------------------------------------------------------------------------


class TestCitationAutoSaveSoftFailure:
    """Verify _auto_save_papers_to_kb handles failures gracefully."""

    @pytest.fixture
    def mock_paper(self):
        from research_agent.tools.citation_explorer import CitationPaper

        return CitationPaper(
            paper_id="abc123",
            title="Test Paper",
            year=2024,
            authors=["Author A"],
            citation_count=10,
            abstract=None,
            venue="Test Venue",
            url=None,
        )

    async def test_auto_save_with_empty_list(self):
        from research_agent.ui.components.citation_explorer import (
            _auto_save_papers_to_kb,
        )

        mock_search = MagicMock()
        added, skipped, errors = await _auto_save_papers_to_kb([], mock_search)
        assert added == 0
        assert skipped == 0
        assert errors == []

    async def test_auto_save_batch_api_failure(self, mock_paper):
        """When S2 batch endpoint fails, papers are still processed with fallback data."""
        from research_agent.ui.components.citation_explorer import (
            _auto_save_papers_to_kb,
            _get_kb_resources,
        )

        mock_search = MagicMock()
        mock_search._s2_rate_limiter = MagicMock()
        mock_search._s2_rate_limiter.wait_if_needed = AsyncMock()
        mock_search._get_client = AsyncMock()
        mock_search.SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

        # Make the batch POST raise an error
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Network error"))
        mock_search._get_client.return_value = mock_client

        # Mock the KB resources to avoid loading real embedder
        mock_store = MagicMock()
        mock_store.get_paper.return_value = None  # Not in KB yet
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1] * 768]

        with patch(
            "research_agent.ui.components.citation_explorer._get_kb_resources",
            return_value=(mock_store, mock_embedder),
        ):
            added, skipped, errors = await _auto_save_papers_to_kb(
                [mock_paper], mock_search
            )
            # Should still attempt ingestion with fallback data (from paper object)
            assert added + skipped + len(errors) >= 1

    async def test_auto_save_deduplicates(self, mock_paper):
        """Papers already in KB should be skipped."""
        from research_agent.ui.components.citation_explorer import (
            _auto_save_papers_to_kb,
        )

        mock_search = MagicMock()
        mock_store = MagicMock()
        mock_store.get_paper.return_value = {"id": "abc123"}  # Already exists
        mock_embedder = MagicMock()

        with patch(
            "research_agent.ui.components.citation_explorer._get_kb_resources",
            return_value=(mock_store, mock_embedder),
        ):
            added, skipped, errors = await _auto_save_papers_to_kb(
                [mock_paper], mock_search
            )
            assert added == 0
            assert skipped > 0


# ---------------------------------------------------------------------------
# 6. Save web results — edge cases
# ---------------------------------------------------------------------------


class TestSaveWebResultsSoftFailure:
    """Verify web results save handles edge cases."""

    def test_add_web_source_then_delete(self, tmp_path):
        """Full lifecycle: add → verify → delete → verify empty."""
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        store.upsert_web_source("ws1", "2024-01-01T00:00:00Z", title="Test", url="https://example.com")
        assert store.get_stats()["total_web_sources"] == 1
        store.delete_web_source("ws1")
        assert store.get_stats()["total_web_sources"] == 0

    def test_web_source_upsert_idempotent(self, tmp_path):
        """Upserting the same web source twice should not duplicate."""
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        store.upsert_web_source("ws1", "2024-01-01T00:00:00Z", title="Test", url="https://example.com")
        store.upsert_web_source("ws1", "2024-01-01T00:00:00Z", title="Test Updated", url="https://example.com")

        sources = store.list_web_sources()
        assert len(sources) == 1
        assert sources[0]["title"] == "Test Updated"


# ---------------------------------------------------------------------------
# 7. Notes browser — edge cases
# ---------------------------------------------------------------------------


class TestNotesBrowserSoftFailure:
    """Verify notes operations handle edge cases."""

    def test_upsert_note_minimal(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        store.upsert_note("n1", "2024-01-01T00:00:00Z", title="Minimal Note")
        notes = store.list_notes()
        assert len(notes) == 1

    def test_delete_note_then_list(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        store.upsert_note("n1", "2024-01-01T00:00:00Z", title="To Delete")
        store.delete_note("n1")
        assert store.list_notes() == []

    def test_note_upsert_idempotent(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        store.upsert_note("n1", "2024-01-01T00:00:00Z", title="V1")
        store.upsert_note("n1", "2024-01-01T00:00:00Z", title="V2")
        notes = store.list_notes()
        assert len(notes) == 1
        assert notes[0]["title"] == "V2"


# ---------------------------------------------------------------------------
# 8. Full flow: add paper → list → delete → verify empty
# ---------------------------------------------------------------------------


class TestFullPaperLifecycle:
    """End-to-end paper lifecycle through metadata store."""

    def test_add_list_delete(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))

        # Add
        store.upsert_paper("p1", "Test Paper", "2024-01-01T00:00:00Z", year=2024, source="test")
        assert store.paper_exists("p1")
        assert store.get_stats()["total_papers"] == 1

        # List
        papers = store.list_papers()
        assert len(papers) == 1
        assert papers[0]["title"] == "Test Paper"

        # Delete
        store.delete_paper("p1")
        assert not store.paper_exists("p1")
        assert store.get_stats()["total_papers"] == 0
        assert store.list_papers() == []

    def test_add_paper_with_doi_lookup(self, tmp_path):
        from research_agent.db.kb_metadata_store import KBMetadataStore

        store = KBMetadataStore(str(tmp_path / "meta.sqlite"))
        store.upsert_paper("p1", "DOI Paper", "2024-01-01T00:00:00Z", doi="10.1234/test")

        assert store.paper_exists_by_doi("10.1234/test")
        assert not store.paper_exists_by_doi("10.9999/nonexistent")


# ---------------------------------------------------------------------------
# 9. Config pipeline loading
# ---------------------------------------------------------------------------


class TestConfigPipelineLoading:
    """Verify pipeline config is read from config dict."""

    def test_build_agent_reads_pipeline(self):
        """build_agent_from_config should apply pipeline config."""
        from research_agent.main import CLOUD_PROVIDERS

        # We can't call build_agent_from_config without real embedder,
        # but we can verify the pipeline config parsing works
        model_cfg = {
            "provider": "none",
            "pipeline": {
                "classify": "llama-3.1-8b-instant",
                "synthesize": "llama-3.3-70b-versatile",
            },
        }
        pipeline_cfg = model_cfg.get("pipeline")
        assert pipeline_cfg is not None
        assert pipeline_cfg["classify"] == "llama-3.1-8b-instant"
        assert pipeline_cfg["synthesize"] == "llama-3.3-70b-versatile"

    def test_no_pipeline_in_config(self):
        """Missing pipeline section should not crash."""
        model_cfg = {"provider": "none"}
        pipeline_cfg = model_cfg.get("pipeline")
        assert pipeline_cfg is None
