"""
Smoke test for the core query → retrieve → synthesize path.

This is the "if this breaks, nothing works" test. It mocks the vector store
and LLM to verify the agent can produce a response with source references.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture(autouse=True)
def mock_model_loading():
    """Prevent actual model loading."""
    from research_agent.models.llm_utils import VRAMConstraintError

    with patch("research_agent.models.llm_utils.get_qlora_pipeline") as mock_qlora, \
         patch("research_agent.agents.research_agent.get_ollama_pipeline") as mock_ollama:
        mock_qlora.side_effect = VRAMConstraintError("disabled in tests")
        mock_ollama.side_effect = Exception("disabled in tests")
        yield


@pytest.mark.unit
class TestSmokeQueryPath:
    """End-to-end smoke test with all externals mocked."""

    @pytest.mark.asyncio
    async def test_query_produces_answer_with_sources(self):
        """The full run path: query → classify → local search → synthesize."""
        from research_agent.agents.research_agent import ResearchAgent

        # Build agent with mocked deps
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "content": "Indigenous communities in Northern Norway have developed resilience strategies.",
                "metadata": {
                    "title": "Arctic Resilience",
                    "authors": "Kramvig, B.",
                    "year": 2020,
                    "source": "knowledge_base",
                },
                "score": 0.85,
            },
            {
                "content": "Sami reindeer herding practices adapt to climate variability.",
                "metadata": {
                    "title": "Reindeer and Climate",
                    "authors": "Kristoffersen, B.",
                    "year": 2019,
                    "source": "knowledge_base",
                },
                "score": 0.78,
            },
        ]

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768

        agent = ResearchAgent(
            vector_store=mock_store,
            embedder=mock_embedder,
            use_ollama=False,
        )

        # Mock the LLM generate to return a canned synthesis
        canned = (
            "Based on the research, indigenous communities in Northern Norway "
            "have developed resilience strategies including traditional reindeer "
            "herding practices that adapt to climate variability (Kramvig 2020, "
            "Kristoffersen 2019)."
        )
        agent.model = MagicMock()
        agent.model.generate.return_value = canned
        agent._load_model_on_demand = False

        # Mock external search to return nothing (smoke test = local only)
        agent.academic_search = MagicMock()
        agent.academic_search.search_semantic_scholar = AsyncMock(return_value=[])
        agent.academic_search.search_openalex = AsyncMock(return_value=[])
        agent.web_search = MagicMock()
        agent.web_search.search = AsyncMock(return_value=[])

        # Run the agent
        result = agent.run("How do indigenous communities adapt to climate change?")

        # Core assertions: we got an answer with sources
        assert result is not None
        assert "answer" in result
        assert len(result["answer"]) > 0
        assert "sources" in result
        assert result.get("query_type") in (
            "literature_review", "factual", "analysis", "general",
        )
