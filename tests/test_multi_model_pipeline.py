"""
Tests for the multi-model pipeline feature.

Covers:
- configure_pipeline accepts valid task types
- configure_pipeline filters invalid task types
- task_infer with no pipeline (uses default model)
- task_infer with pipeline override (switches model)
- task_infer restores original model after override
- task_infer restores original model even on error
- Pipeline config loading from YAML
- resolve_pipeline_aliases with known and unknown providers
"""

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_model_loading():
    """Prevent actual model loading in tests."""
    from research_agent.models.llm_utils import VRAMConstraintError

    with patch("research_agent.agents.research_agent.get_qlora_pipeline") as mock_qlora, \
         patch("research_agent.agents.research_agent.get_ollama_pipeline") as mock_ollama:
        mock_qlora.side_effect = VRAMConstraintError("Model loading disabled in tests")
        mock_ollama.side_effect = Exception("Ollama disabled in tests")
        yield


def _make_agent(**kwargs):
    """Create a minimal ResearchAgent for testing."""
    from research_agent.agents.research_agent import ResearchAgent
    return ResearchAgent(use_ollama=False, **kwargs)


def _make_agent_with_model():
    """Create a ResearchAgent with a mocked OpenAI-compatible model."""
    from research_agent.agents.research_agent import ResearchAgent

    agent = ResearchAgent(use_ollama=False)

    # Inject a mock model that supports switch_model
    mock_model = MagicMock()
    mock_model.model_name = "default-model"
    mock_model.generate.return_value = "mock response"
    mock_model.switch_model = MagicMock()
    mock_model.list_available_models.return_value = [
        "default-model", "fast-model", "big-model",
    ]

    agent.model = mock_model
    agent.provider = "openai"
    return agent


# ---------------------------------------------------------------------------
# configure_pipeline
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConfigurePipeline:

    def test_accepts_valid_task_types(self):
        agent = _make_agent()
        agent.configure_pipeline({
            "classify": "fast-model",
            "extract_keywords": "fast-model",
            "synthesize": "big-model",
        })
        assert agent._pipeline == {
            "classify": "fast-model",
            "extract_keywords": "fast-model",
            "synthesize": "big-model",
        }

    def test_filters_invalid_task_types(self):
        agent = _make_agent()
        agent.configure_pipeline({
            "classify": "fast-model",
            "summarize": "some-model",       # invalid
            "translate": "some-model",       # invalid
        })
        assert "classify" in agent._pipeline
        assert "summarize" not in agent._pipeline
        assert "translate" not in agent._pipeline

    def test_empty_pipeline(self):
        agent = _make_agent()
        agent.configure_pipeline({})
        assert agent._pipeline == {}

    def test_partial_pipeline(self):
        agent = _make_agent()
        agent.configure_pipeline({"synthesize": "big-model"})
        assert agent._pipeline == {"synthesize": "big-model"}
        assert "classify" not in agent._pipeline

    def test_warns_for_unknown_model(self):
        """configure_pipeline should log a warning for unknown models."""
        agent = _make_agent_with_model()
        with patch("research_agent.agents.research_agent.logger") as mock_logger:
            agent.configure_pipeline({"classify": "nonexistent-model"})
            # Should have at least one warning about the unknown model
            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if "nonexistent-model" in str(c)
            ]
            assert len(warning_calls) >= 1

    def test_no_warning_for_known_model(self):
        """configure_pipeline should not warn for models in the known list."""
        agent = _make_agent_with_model()
        with patch("research_agent.agents.research_agent.logger") as mock_logger:
            agent.configure_pipeline({"classify": "fast-model"})
            # Should NOT have warnings about unknown models
            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if "not in known model" in str(c)
            ]
            assert len(warning_calls) == 0


# ---------------------------------------------------------------------------
# task_infer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTaskInfer:

    def test_no_pipeline_uses_default(self):
        """With no pipeline configured, task_infer uses the default model."""
        agent = _make_agent_with_model()
        # No pipeline set -> _pipeline is empty
        result = agent.task_infer("classify", "test prompt")
        assert result == "mock response"
        # switch_model should NOT be called
        agent.model.switch_model.assert_not_called()

    def test_pipeline_override_switches_model(self):
        """With a pipeline override, task_infer switches to the specified model."""
        agent = _make_agent_with_model()
        agent.configure_pipeline({"classify": "fast-model"})

        result = agent.task_infer("classify", "test prompt")
        assert result == "mock response"

        # Should have switched to fast-model, then back to default-model
        calls = agent.model.switch_model.call_args_list
        assert len(calls) == 2
        assert calls[0].args[0] == "fast-model"
        assert calls[1].args[0] == "default-model"

    def test_restores_original_model_after_override(self):
        """After task_infer with override, the model is restored to the original."""
        agent = _make_agent_with_model()
        agent.configure_pipeline({"synthesize": "big-model"})

        agent.task_infer("synthesize", "test prompt")

        # Last call should restore to default-model
        last_call = agent.model.switch_model.call_args_list[-1]
        assert last_call.args[0] == "default-model"

    def test_restores_model_on_error(self):
        """Even if generate() raises, the model must be restored."""
        agent = _make_agent_with_model()
        agent.configure_pipeline({"classify": "fast-model"})
        agent.model.generate.side_effect = RuntimeError("LLM failure")

        # task_infer calls infer() which catches general exceptions
        result = agent.task_infer("classify", "test prompt")
        assert "Error" in result or "error" in result.lower()

        # Model should still be restored to default
        last_call = agent.model.switch_model.call_args_list[-1]
        assert last_call.args[0] == "default-model"

    def test_unmatched_task_uses_default(self):
        """A task not in the pipeline should use the default model."""
        agent = _make_agent_with_model()
        agent.configure_pipeline({"classify": "fast-model"})

        # "synthesize" is NOT in the pipeline
        agent.task_infer("synthesize", "test prompt")
        agent.model.switch_model.assert_not_called()

    def test_no_model_returns_inference_result(self):
        """With no model at all, task_infer should still return something."""
        agent = _make_agent()
        # agent.model is None (no LLM loaded)
        result = agent.task_infer("classify", "test prompt")
        # Should hit the infer() error handling path
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# resolve_pipeline_aliases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolvePipelineAliases:

    def test_resolves_fast_alias_for_groq(self):
        from research_agent.main import resolve_pipeline_aliases
        result = resolve_pipeline_aliases(
            {"classify": "fast", "synthesize": "default"},
            "groq",
        )
        assert result["classify"] == "llama-3.1-8b-instant"
        assert result["synthesize"] == "llama-3.3-70b-versatile"

    def test_resolves_fast_alias_for_openai(self):
        from research_agent.main import resolve_pipeline_aliases
        result = resolve_pipeline_aliases(
            {"classify": "fast", "synthesize": "default"},
            "openai",
        )
        assert result["classify"] == "gpt-4o-mini"
        assert result["synthesize"] == "gpt-4o"

    def test_resolves_fast_alias_for_anthropic(self):
        from research_agent.main import resolve_pipeline_aliases
        result = resolve_pipeline_aliases(
            {"classify": "fast", "synthesize": "default"},
            "anthropic",
        )
        assert result["classify"] == "claude-haiku-4-5"
        assert result["synthesize"] == "claude-sonnet-4-6"

    def test_literal_model_name_passes_through(self):
        from research_agent.main import resolve_pipeline_aliases
        result = resolve_pipeline_aliases(
            {"classify": "my-custom-model", "synthesize": "another-model"},
            "groq",
        )
        assert result["classify"] == "my-custom-model"
        assert result["synthesize"] == "another-model"

    def test_unknown_provider_passes_through(self):
        from research_agent.main import resolve_pipeline_aliases
        result = resolve_pipeline_aliases(
            {"classify": "fast", "synthesize": "default"},
            "unknown_provider",
        )
        # No aliases known for unknown provider, so values pass through
        assert result["classify"] == "fast"
        assert result["synthesize"] == "default"

    def test_mixed_aliases_and_literals(self):
        from research_agent.main import resolve_pipeline_aliases
        result = resolve_pipeline_aliases(
            {"classify": "fast", "extract_keywords": "my-model", "synthesize": "default"},
            "groq",
        )
        assert result["classify"] == "llama-3.1-8b-instant"
        assert result["extract_keywords"] == "my-model"
        assert result["synthesize"] == "llama-3.3-70b-versatile"

    def test_empty_pipeline(self):
        from research_agent.main import resolve_pipeline_aliases
        result = resolve_pipeline_aliases({}, "groq")
        assert result == {}


# ---------------------------------------------------------------------------
# Pipeline config loading from YAML
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPipelineConfigLoading:

    def test_pipeline_loaded_from_yaml(self, tmp_path):
        """Pipeline section in YAML should be loadable and resolvable."""
        from research_agent.utils.config import load_config
        from research_agent.main import resolve_pipeline_aliases

        cfg_file = tmp_path / "test_pipeline.yaml"
        cfg_file.write_text(
            "model:\n"
            "  provider: groq\n"
            "  pipeline:\n"
            "    classify: fast\n"
            "    extract_keywords: fast\n"
            "    synthesize: default\n"
        )

        config = load_config(str(cfg_file))
        model_cfg = config.get("model", {})
        pipeline_cfg = model_cfg.get("pipeline")

        assert pipeline_cfg is not None
        assert isinstance(pipeline_cfg, dict)
        assert pipeline_cfg["classify"] == "fast"

        resolved = resolve_pipeline_aliases(pipeline_cfg, "groq")
        assert resolved["classify"] == "llama-3.1-8b-instant"
        assert resolved["synthesize"] == "llama-3.3-70b-versatile"

    def test_no_pipeline_section(self, tmp_path):
        """Config without pipeline section should not break anything."""
        from research_agent.utils.config import load_config

        cfg_file = tmp_path / "no_pipeline.yaml"
        cfg_file.write_text(
            "model:\n"
            "  provider: groq\n"
        )

        config = load_config(str(cfg_file))
        model_cfg = config.get("model", {})
        pipeline_cfg = model_cfg.get("pipeline")

        assert pipeline_cfg is None

    def test_pipeline_with_literal_model_names(self, tmp_path):
        """Pipeline with literal model names (no aliases) should pass through."""
        from research_agent.utils.config import load_config
        from research_agent.main import resolve_pipeline_aliases

        cfg_file = tmp_path / "literal_pipeline.yaml"
        cfg_file.write_text(
            "model:\n"
            "  provider: groq\n"
            "  pipeline:\n"
            "    classify: llama-3.1-8b-instant\n"
            "    synthesize: llama-3.3-70b-versatile\n"
        )

        config = load_config(str(cfg_file))
        pipeline_cfg = config["model"]["pipeline"]

        resolved = resolve_pipeline_aliases(pipeline_cfg, "groq")
        assert resolved["classify"] == "llama-3.1-8b-instant"
        assert resolved["synthesize"] == "llama-3.3-70b-versatile"
