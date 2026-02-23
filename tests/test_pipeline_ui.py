"""Tests for pipeline model configuration logic."""

import pytest
from unittest.mock import MagicMock, patch


class TestConfigurePipelineFromUI:
    """Test the pipeline dict construction from UI dropdown values."""

    def _build_pipeline(self, classify, keywords, synthesize):
        """Replicate the UI handler logic for building pipeline dict."""
        pipeline = {}
        if classify and classify != "default":
            pipeline["classify"] = classify
        if keywords and keywords != "default":
            pipeline["extract_keywords"] = keywords
        if synthesize and synthesize != "default":
            pipeline["synthesize"] = synthesize
        return pipeline

    def test_all_default(self):
        """All 'default' → empty pipeline dict."""
        result = self._build_pipeline("default", "default", "default")
        assert result == {}

    def test_fast_classify(self):
        """Only classify overridden."""
        result = self._build_pipeline("llama-3.1-8b-instant", "default", "default")
        assert result == {"classify": "llama-3.1-8b-instant"}

    def test_all_overridden(self):
        """All three tasks overridden."""
        result = self._build_pipeline(
            "gpt-4o-mini", "gpt-4o-mini", "gpt-4o"
        )
        assert result == {
            "classify": "gpt-4o-mini",
            "extract_keywords": "gpt-4o-mini",
            "synthesize": "gpt-4o",
        }

    def test_empty_string_treated_as_default(self):
        """Empty string → not included in pipeline."""
        result = self._build_pipeline("", "default", "gpt-4o")
        assert result == {"synthesize": "gpt-4o"}

    def test_none_treated_as_default(self):
        """None → not included in pipeline."""
        result = self._build_pipeline(None, None, None)
        assert result == {}


class TestPipelineChoices:
    """Test pipeline dropdown choices construction."""

    def test_choices_include_default(self):
        """'default' is always first in choices."""
        models = ["gpt-4o-mini", "gpt-4o"]
        choices = ["default"] + models
        assert choices[0] == "default"
        assert len(choices) == 3

    def test_choices_with_no_models(self):
        """When no models available, only 'default'."""
        models = []
        choices = ["default"] + models if models else ["default"]
        assert choices == ["default"]


@pytest.fixture(autouse=True)
def mock_model_loading():
    """Prevent actual model loading."""
    from research_agent.models.llm_utils import VRAMConstraintError

    with patch("research_agent.agents.research_agent.get_qlora_pipeline") as mock_qlora, \
         patch("research_agent.agents.research_agent.get_ollama_pipeline") as mock_ollama:
        mock_qlora.side_effect = VRAMConstraintError("disabled in tests")
        mock_ollama.side_effect = Exception("disabled in tests")
        yield


class TestAgentConfigurePipeline:
    """Test that agent.configure_pipeline() works with UI-style dicts."""

    def _make_agent(self):
        from research_agent.agents.research_agent import ResearchAgent
        return ResearchAgent(
            vector_store=MagicMock(),
            embedder=MagicMock(),
            use_ollama=False,
        )

    def test_valid_tasks_accepted(self):
        """Valid task types are stored."""
        agent = self._make_agent()
        agent.configure_pipeline({
            "classify": "fast-model",
            "extract_keywords": "fast-model",
            "synthesize": "big-model",
        })
        assert agent._pipeline["classify"] == "fast-model"
        assert agent._pipeline["synthesize"] == "big-model"

    def test_empty_pipeline_clears(self):
        """Empty dict clears all overrides."""
        agent = self._make_agent()
        agent.configure_pipeline({"classify": "fast"})
        assert len(agent._pipeline) == 1
        agent.configure_pipeline({})
        assert len(agent._pipeline) == 0

    def test_unknown_tasks_ignored(self):
        """Unknown task types are silently filtered out."""
        agent = self._make_agent()
        agent.configure_pipeline({
            "classify": "fast-model",
            "unknown_task": "some-model",
        })
        assert "classify" in agent._pipeline
        assert "unknown_task" not in agent._pipeline
