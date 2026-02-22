"""
Tests for the ResearchAgent LangGraph workflow.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# Patch model loading for all tests
@pytest.fixture(autouse=True)
def mock_model_loading():
    """Prevent actual model loading in tests."""
    from research_agent.models.llm_utils import VRAMConstraintError

    with patch("research_agent.agents.research_agent.get_qlora_pipeline") as mock_qlora, \
         patch("research_agent.agents.research_agent.get_ollama_pipeline") as mock_ollama:
        # Raise VRAMConstraintError which is properly caught
        mock_qlora.side_effect = VRAMConstraintError("Model loading disabled in tests")
        mock_ollama.side_effect = Exception("Ollama disabled in tests")
        yield


def make_state(query: str, **overrides):
    """Helper to create a test state."""
    state = {
        "messages": [],
        "current_query": query,
        "query_type": "",
        "search_results": [],
        "local_results": [],
        "external_results": [],
        "context": "",
        "should_search_external": False,
        "candidates_for_ingestion": [],
        "final_answer": "",
        "error": None,
    }
    state.update(overrides)
    return state


@pytest.mark.unit
class TestQueryClassification:
    """Tests for query understanding and classification."""

    @pytest.mark.asyncio
    async def test_classify_literature_review_query(self):
        """Test classification of literature review queries."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent(use_ollama=False)
        state = make_state("What is the state of research on climate change impacts?")

        result = await agent._understand_query(state)
        assert result["query_type"] == "literature_review"

    @pytest.mark.asyncio
    async def test_classify_factual_query(self):
        """Test classification of factual queries."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent(use_ollama=False)
        state = make_state("What is the definition of social capital?")

        result = await agent._understand_query(state)
        assert result["query_type"] == "factual"

    @pytest.mark.asyncio
    async def test_classify_analysis_query(self):
        """Test classification of analysis queries."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent(use_ollama=False)
        state = make_state("Compare Marxist and Weberian theories of class")

        result = await agent._understand_query(state)
        assert result["query_type"] == "analysis"

    @pytest.mark.asyncio
    async def test_classify_general_query(self):
        """Test classification of general queries."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent(use_ollama=False)
        state = make_state("Hello, how are you?")

        result = await agent._understand_query(state)
        assert result["query_type"] == "general"


@pytest.mark.unit
class TestLocalSearch:
    """Tests for local vector store search."""

    @pytest.mark.asyncio
    async def test_search_local_with_results(self):
        """Test local search returns formatted results."""
        from research_agent.agents.research_agent import ResearchAgent

        # Mock vector store — return results only for the papers collection
        papers_result = {
            "documents": ["Content about climate change..."],
            "metadatas": [{"title": "Climate Paper", "authors": "Smith", "year": 2023, "paper_id": "123"}],
            "distances": [0.2],
        }
        empty_result = {"documents": [], "metadatas": [], "distances": []}

        mock_store = MagicMock()
        mock_store.search.side_effect = lambda **kwargs: (
            papers_result if kwargs.get("collection") == "papers" else empty_result
        )

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)

        agent = ResearchAgent(vector_store=mock_store, embedder=mock_embedder, use_ollama=False)
        state = make_state(
            "climate change",
            query_type="literature_review",
            search_query="climate change",
            should_search_external=True,
        )

        result = await agent._search_local(state)

        assert len(result["local_results"]) == 1
        assert result["local_results"][0]["title"] == "Climate Paper"
        assert result["local_results"][0]["source"] == "local_kb"

    @pytest.mark.asyncio
    async def test_search_local_without_store(self):
        """Test local search gracefully handles missing store."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent(vector_store=None, embedder=None, use_ollama=False)
        state = make_state(
            "test query", query_type="factual", search_query="test query"
        )

        result = await agent._search_local(state)

        assert result["local_results"] == []
        assert result["should_search_external"] is True


@pytest.mark.unit
class TestSynthesis:
    """Tests for result synthesis."""

    def test_build_synthesis_prompt_literature_review(self):
        """Test prompt building for literature review."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent(use_ollama=False)

        results = [
            {"title": "Paper 1", "authors": "Smith", "year": 2020, "content": "Content 1", "source": "local_kb"},
            {"title": "Paper 2", "authors": "Jones", "year": 2021, "content": "Content 2", "source": "semantic_scholar"},
        ]

        prompt = agent._build_synthesis_prompt("climate change research", "literature_review", results)

        assert "literature review" in prompt.lower()
        assert "[1] Paper 1" in prompt
        assert "[2] Paper 2" in prompt
        assert "climate change research" in prompt

    def test_build_synthesis_prompt_factual(self):
        """Test prompt building for factual queries."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent(use_ollama=False)
        results = [{"title": "Source", "content": "Definition...", "source": "web"}]

        prompt = agent._build_synthesis_prompt("what is X", "factual", results)

        assert "precise" in prompt.lower() or "factual" in prompt.lower()

    def test_format_results_without_llm(self):
        """Test result formatting when no LLM is available."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent(use_ollama=False)

        results = [
            {"title": "Paper 1", "authors": "Smith", "year": 2020, "content": "Abstract...", "source": "local_kb"},
        ]

        formatted = agent._format_results_without_llm("test query", results)

        assert "Paper 1" in formatted
        assert "Smith" in formatted
        assert "2020" in formatted

    def test_format_results_empty(self):
        """Test formatting with no results."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent(use_ollama=False)
        formatted = agent._format_results_without_llm("test query", [])

        assert "No retrieved sources" in formatted


@pytest.mark.unit
class TestIngestionOffer:
    """Tests for ingestion offer logic."""

    @pytest.mark.asyncio
    async def test_offer_ingestion_with_candidates(self):
        """Test ingestion offer with available candidates."""
        from research_agent.agents.research_agent import ResearchAgent, AgentConfig

        config = AgentConfig(auto_ingest=True)
        agent = ResearchAgent(config=config, use_ollama=False)

        state = make_state(
            "test",
            candidates_for_ingestion=[
                {"title": "Great Paper", "authors": "Author", "year": 2023, "citation_count": 100},
            ],
            final_answer="Here is the answer.",
        )

        result = await agent._offer_ingestion(state)

        assert "Great Paper" in result["final_answer"]
        assert "knowledge base" in result["final_answer"].lower()

    @pytest.mark.asyncio
    async def test_no_offer_when_disabled(self):
        """Test no offer when auto_ingest is disabled."""
        from research_agent.agents.research_agent import ResearchAgent, AgentConfig

        config = AgentConfig(auto_ingest=False)
        agent = ResearchAgent(config=config, use_ollama=False)

        state = make_state(
            "test",
            candidates_for_ingestion=[{"title": "Paper"}],
            final_answer="Answer.",
        )

        result = await agent._offer_ingestion(state)

        assert result["final_answer"] == "Answer."


@pytest.mark.unit
class TestGraphExecution:
    """Tests for full graph execution."""

    def test_graph_compiles(self):
        """Test that the LangGraph workflow compiles."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent(use_ollama=False)

        assert agent.graph is not None

    @pytest.mark.asyncio
    async def test_run_async_initializes_state(self):
        """Test that run_async properly initializes state."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent(use_ollama=False)

        # Mock the graph to return test state
        async def mock_invoke(state):
            return {**state, "final_answer": "Test answer", "query_type": "general"}

        agent.graph = MagicMock()
        agent.graph.ainvoke = AsyncMock(side_effect=mock_invoke)

        result = await agent._run_async("test query")

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"
        assert result["query_type"] == "general"

    @pytest.mark.asyncio
    async def test_run_async_with_filters(self):
        """Test that search filters are applied."""
        from research_agent.agents.research_agent import ResearchAgent

        agent = ResearchAgent(use_ollama=False)

        async def mock_invoke(state):
            return {**state, "final_answer": "Filtered answer", "query_type": "factual"}

        agent.graph = MagicMock()
        agent.graph.ainvoke = AsyncMock(side_effect=mock_invoke)

        result = await agent._run_async("test", search_filters={"year_from": 2020, "year_to": 2024})

        assert agent.config.year_range == (2020, 2024)
        assert result["search_filters"]["year_from"] == 2020
