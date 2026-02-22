"""Tests for research agent formatting methods.

Verifies that _format_results_without_llm and _build_synthesis_prompt
handle edge cases (None fields, empty results, missing keys) without crashing.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def agent():
    """Create a ResearchAgent with no LLM (formatting-only tests)."""
    with patch("research_agent.agents.research_agent.torch"):
        from research_agent.agents.research_agent import ResearchAgent, AgentConfig

        a = ResearchAgent.__new__(ResearchAgent)
        a.config = AgentConfig()
        a.model = None
        a.vector_store = None
        a.embedder = None
        a.academic_search = None
        a.web_search = None
        a.workflow = None
        return a


# ---------------------------------------------------------------------------
# _format_results_without_llm
# ---------------------------------------------------------------------------


class TestFormatResultsWithoutLLM:
    """Tests for the no-LLM fallback formatter."""

    def test_empty_results_returns_unavailable_message(self, agent):
        result = agent._format_results_without_llm("test query", [])
        assert "LLM synthesis is currently unavailable" in result

    def test_none_results_returns_unavailable_message(self, agent):
        result = agent._format_results_without_llm("test query", None)
        assert "unavailable" in result

    def test_basic_result_formatting(self, agent):
        results = [
            {
                "title": "Urban Theory",
                "authors": "David Harvey",
                "year": 2020,
                "source": "semantic_scholar",
                "content": "A study on urbanization patterns.",
                "tags": "",
                "url": "",
            }
        ]
        result = agent._format_results_without_llm("urbanization", results)
        assert "**[1] Urban Theory**" in result
        assert "David Harvey" in result
        assert "(2020)" in result
        assert "Semantic Scholar" in result
        assert "A study on urbanization" in result

    def test_none_content_does_not_crash(self, agent):
        """Content stored as None must not cause TypeError on slicing."""
        results = [
            {
                "title": "Paper with None content",
                "content": None,
                "source": "openalex",
            }
        ]
        result = agent._format_results_without_llm("query", results)
        assert "Paper with None content" in result

    def test_missing_content_key_does_not_crash(self, agent):
        """Result dict with no 'content' key at all."""
        results = [{"title": "Minimal Paper"}]
        result = agent._format_results_without_llm("query", results)
        assert "Minimal Paper" in result

    def test_none_fields_do_not_crash(self, agent):
        """All optional fields as None should not crash."""
        results = [
            {
                "title": None,
                "authors": None,
                "year": None,
                "source": None,
                "content": None,
                "tags": None,
                "url": None,
            }
        ]
        # Should not raise
        result = agent._format_results_without_llm("query", results)
        assert isinstance(result, str)

    def test_source_label_mapping(self, agent):
        """Verify known source types get human-readable labels."""
        for source, label in [
            ("local_kb", "Knowledge Base"),
            ("local_note", "Research Note"),
            ("semantic_scholar", "Semantic Scholar"),
            ("openalex", "OpenAlex"),
            ("web", "Web Search"),
        ]:
            results = [{"title": "Test", "source": source, "content": "x"}]
            result = agent._format_results_without_llm("q", results)
            assert label in result, f"Expected '{label}' for source '{source}'"

    def test_user_note_author_hidden(self, agent):
        """Author 'User Note' should not be displayed."""
        results = [
            {"title": "My Note", "authors": "User Note", "content": "text"}
        ]
        result = agent._format_results_without_llm("q", results)
        assert "User Note" not in result

    def test_limits_to_10_results(self, agent):
        results = [{"title": f"Paper {i}", "content": "x"} for i in range(15)]
        result = agent._format_results_without_llm("q", results)
        assert "**[10]" in result
        assert "**[11]" not in result

    def test_url_included_when_present(self, agent):
        results = [
            {
                "title": "Linked Paper",
                "url": "https://example.com/paper",
                "content": "x",
            }
        ]
        result = agent._format_results_without_llm("q", results)
        assert "https://example.com/paper" in result

    def test_tags_included_when_present(self, agent):
        results = [
            {"title": "Tagged Paper", "tags": "climate, policy", "content": "x"}
        ]
        result = agent._format_results_without_llm("q", results)
        assert "climate, policy" in result


# ---------------------------------------------------------------------------
# _build_synthesis_prompt
# ---------------------------------------------------------------------------


class TestBuildSynthesisPrompt:
    """Tests for the LLM synthesis prompt builder."""

    def _make_results(self, **overrides):
        """Create a result list with one item, accepting overrides."""
        base = {
            "title": "Test Paper",
            "authors": "Author One",
            "year": 2023,
            "source": "semantic_scholar",
            "content": "This is the paper content about research methods.",
            "tags": "",
            "url": "",
        }
        base.update(overrides)
        return [base]

    def test_all_query_types_produce_prompt(self, agent):
        """Every query type should produce a non-empty prompt string."""
        results = self._make_results()
        for qtype in ["literature_review", "factual", "analysis", "general"]:
            prompt = agent._build_synthesis_prompt("test", qtype, results)
            assert isinstance(prompt, str)
            assert len(prompt) > 100
            assert "Question: test" in prompt

    def test_none_content_in_synthesis_prompt(self, agent):
        """None content in results must not crash the prompt builder."""
        results = self._make_results(content=None)
        prompt = agent._build_synthesis_prompt("test", "general", results)
        assert "Question: test" in prompt

    def test_sources_text_includes_content(self, agent):
        results = self._make_results(content="Urbanization has accelerated globally.")
        prompt = agent._build_synthesis_prompt("cities", "factual", results)
        assert "Urbanization has accelerated" in prompt

    def test_source_citation_number(self, agent):
        results = self._make_results()
        prompt = agent._build_synthesis_prompt("q", "general", results)
        assert "[1]" in prompt

    def test_empty_results_shows_no_sources(self, agent):
        """Non-general query with no results should show 'No sources found'."""
        prompt = agent._build_synthesis_prompt("q", "factual", [])
        assert "(No sources found)" in prompt

    def test_general_empty_results_uses_casual_prompt(self, agent):
        """General query with no results should use casual conversational prompt."""
        prompt = agent._build_synthesis_prompt("q", "general", [])
        assert "casual" in prompt.lower() or "conversational" in prompt.lower()
        assert "(No sources found)" not in prompt

    def test_literature_review_instructions(self, agent):
        results = self._make_results()
        prompt = agent._build_synthesis_prompt("q", "literature_review", results)
        assert "literature review" in prompt.lower()
        assert "MANDATORY" in prompt

    def test_analysis_instructions(self, agent):
        results = self._make_results()
        prompt = agent._build_synthesis_prompt("q", "analysis", results)
        assert "critical analysis" in prompt.lower()

    def test_context_included_when_provided(self, agent):
        results = self._make_results()
        prompt = agent._build_synthesis_prompt(
            "q",
            "general",
            results,
            current_researcher="David Harvey",
        )
        assert "David Harvey" in prompt

    def test_content_truncated_to_800_chars(self, agent):
        long_content = "x" * 2000
        results = self._make_results(content=long_content)
        prompt = agent._build_synthesis_prompt("q", "general", results)
        # The content in the prompt should be truncated
        assert "x" * 801 not in prompt
