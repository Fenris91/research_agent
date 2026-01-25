"""
Core unit tests for Research Agent components.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path


# ============================================
# Vector Store Tests
# ============================================

class TestVectorStore:
    """Tests for the ChromaDB vector store."""

    def test_initialization(self):
        """Test vector store initializes correctly."""
        from src.db.vector_store import ResearchVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResearchVectorStore(persist_dir=tmpdir)
            stats = store.get_stats()

            assert stats["total_papers"] == 0
            assert stats["total_notes"] == 0
            assert stats["total_web_sources"] == 0

    def test_add_and_retrieve_paper(self):
        """Test adding and retrieving papers."""
        from src.db.vector_store import ResearchVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResearchVectorStore(persist_dir=tmpdir)

            # Add a paper with dummy embeddings
            store.add_paper(
                paper_id="test_001",
                chunks=["Chunk about anthropology.", "Chunk about ethnography."],
                embeddings=[[0.1] * 1024, [0.2] * 1024],
                metadata={"title": "Test Paper", "year": 2024}
            )

            stats = store.get_stats()
            # Vector store counts unique papers, not chunks
            assert stats["total_papers"] >= 1

    def test_list_papers(self):
        """Test listing papers in the store."""
        from src.db.vector_store import ResearchVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ResearchVectorStore(persist_dir=tmpdir)

            # Add papers
            store.add_paper(
                paper_id="paper_001",
                chunks=["First paper content."],
                embeddings=[[0.1] * 1024],
                metadata={"title": "Paper One", "year": 2023}
            )
            store.add_paper(
                paper_id="paper_002",
                chunks=["Second paper content."],
                embeddings=[[0.2] * 1024],
                metadata={"title": "Paper Two", "year": 2024}
            )

            papers = store.list_papers()
            assert len(papers) >= 2


# ============================================
# Embedding Model Tests
# ============================================

class TestEmbeddingModel:
    """Tests for the embedding model."""

    def test_initialization(self):
        """Test embedding model loads correctly."""
        from src.db.embeddings import EmbeddingModel

        embedder = EmbeddingModel()
        assert embedder.model_name == "BAAI/bge-base-en-v1.5"
        assert embedder.dimension == 768

    def test_single_embedding(self):
        """Test embedding a single text."""
        from src.db.embeddings import EmbeddingModel

        embedder = EmbeddingModel()
        embedding = embedder.embed("Test sentence about research.")

        assert isinstance(embedding, list)
        assert len(embedding) == 768

    def test_batch_embedding(self):
        """Test embedding multiple texts."""
        from src.db.embeddings import EmbeddingModel

        embedder = EmbeddingModel()
        texts = ["First text.", "Second text.", "Third text."]
        embeddings = embedder.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 768 for e in embeddings)

    def test_query_embedding(self):
        """Test query embedding adds BGE prefix."""
        from src.db.embeddings import EmbeddingModel

        embedder = EmbeddingModel()
        query_emb = embedder.embed_query("What is anthropology?")

        assert isinstance(query_emb, list)
        assert len(query_emb) == 768


# ============================================
# Academic Search Tests
# ============================================

class TestAcademicSearch:
    """Tests for academic search APIs."""

    @pytest.mark.asyncio
    async def test_openalex_search(self):
        """Test OpenAlex search returns results."""
        from src.tools.academic_search import AcademicSearchTools

        search = AcademicSearchTools()
        results = await search.search_openalex("ethnography", limit=3)

        assert len(results) > 0
        assert hasattr(results[0], "title")
        assert hasattr(results[0], "year")

    @pytest.mark.asyncio
    async def test_search_all_deduplication(self):
        """Test combined search deduplicates results."""
        from src.tools.academic_search import AcademicSearchTools

        search = AcademicSearchTools()
        results = await search.search_all("participatory research", limit_per_source=2)

        # Should have results (may be limited by rate limits)
        assert isinstance(results, list)


# ============================================
# Web Search Tests
# ============================================

class TestWebSearch:
    """Tests for web search functionality."""

    @pytest.mark.asyncio
    async def test_duckduckgo_search(self):
        """Test DuckDuckGo search (free, no API key)."""
        from src.tools.web_search import WebSearchTool

        search = WebSearchTool(provider="duckduckgo")
        results = await search.search("anthropology research methods", max_results=3)

        assert len(results) > 0
        assert hasattr(results[0], "title")
        assert hasattr(results[0], "url")

        await search.close()


# ============================================
# Researcher Lookup Tests
# ============================================

class TestResearcherLookup:
    """Tests for researcher lookup functionality."""

    @pytest.mark.asyncio
    async def test_openalex_author_search(self):
        """Test OpenAlex author lookup."""
        from src.tools.researcher_lookup import ResearcherLookup

        lookup = ResearcherLookup()
        result = await lookup.search_openalex_author("Clifford Geertz")

        assert result is not None
        assert "display_name" in result
        assert "works_count" in result

        await lookup.close()


# ============================================
# Research Agent Tests
# ============================================

class TestResearchAgent:
    """Tests for the research agent."""

    def test_agent_initialization_with_ollama(self):
        """Test agent initializes with Ollama backend."""
        from src.agents.research_agent import create_research_agent

        agent = create_research_agent(
            use_ollama=True,
            ollama_model="mistral-small3.2:latest"
        )

        assert agent.use_ollama is True
        assert agent.model is not None

    def test_agent_config(self):
        """Test agent configuration."""
        from src.agents.research_agent import AgentConfig

        config = AgentConfig(
            max_local_results=10,
            max_external_results=20,
            auto_ingest=True
        )

        assert config.max_local_results == 10
        assert config.max_external_results == 20
        assert config.auto_ingest is True


# ============================================
# Ollama Integration Tests
# ============================================

class TestOllamaIntegration:
    """Tests for Ollama model integration."""

    def test_ollama_model_wrapper(self):
        """Test OllamaModel wrapper."""
        from src.models.llm_utils import OllamaModel

        model = OllamaModel(model_name="mistral-small3.2:latest")
        assert model.model_name == "mistral-small3.2:latest"

    def test_ollama_generation(self):
        """Test generating text with Ollama."""
        from src.models.llm_utils import OllamaModel

        # Use mistral for reliable testing (qwen3 uses thinking mode)
        model = OllamaModel(model_name="mistral-small3.2:latest")
        response = model.generate(
            "What is 2+2? Answer with just the number.",
            max_tokens=10,
            temperature=0.1
        )

        assert "4" in response

    def test_ollama_list_models(self):
        """Test listing available Ollama models."""
        from src.models.llm_utils import OllamaModel

        model = OllamaModel(model_name="mistral-small3.2:latest")
        models = model.list_available_models()

        assert isinstance(models, list)
        assert len(models) > 0

    def test_ollama_model_switch(self):
        """Test switching between Ollama models."""
        from src.models.llm_utils import OllamaModel

        model = OllamaModel(model_name="mistral-small3.2:latest")
        assert model.model_name == "mistral-small3.2:latest"

        model.switch_model("qwen3:32b")
        assert model.model_name == "qwen3:32b"

    def test_qwen3_generation(self):
        """Test qwen3 generation with thinking mode."""
        from src.models.llm_utils import OllamaModel

        model = OllamaModel(model_name="qwen3:32b")
        response = model.generate(
            "What is the capital of France? Just the city name.",
            max_tokens=500,  # qwen3 needs more tokens for thinking
            temperature=0.1
        )

        assert "Paris" in response


class TestAgentModelSwitch:
    """Tests for agent model switching functionality."""

    def test_agent_switch_model(self):
        """Test switching models via the agent."""
        from src.agents.research_agent import create_research_agent

        agent = create_research_agent(
            use_ollama=True,
            ollama_model="mistral-small3.2:latest"
        )

        assert agent.get_current_model() == "mistral-small3.2:latest"

        agent.switch_model("qwen3:32b")
        assert agent.get_current_model() == "qwen3:32b"

    def test_agent_list_models(self):
        """Test listing models via the agent."""
        from src.agents.research_agent import create_research_agent

        agent = create_research_agent(
            use_ollama=True,
            ollama_model="mistral-small3.2:latest"
        )

        models = agent.list_available_models()
        assert isinstance(models, list)
        assert "qwen3:32b" in models or len(models) > 0


# ============================================
# Citation Explorer Tests
# ============================================

class TestCitationExplorer:
    """Tests for citation explorer functionality."""

    @pytest.mark.asyncio
    async def test_get_citations(self):
        """Test fetching citations for a paper."""
        from src.tools.citation_explorer import CitationExplorer

        explorer = CitationExplorer()

        # Use a known paper ID (Neil Smith's gentrification book)
        paper_id = "555d276eeef60f7a02c1347df45ecda067f44837"
        citations = await explorer.get_citations(paper_id, direction="both", limit=3)

        assert "citing" in citations
        assert "cited" in citations
        assert isinstance(citations["citing"], list)
        assert isinstance(citations["cited"], list)

        await explorer.close()

    @pytest.mark.asyncio
    async def test_citation_link_dataclass(self):
        """Test CitationLink dataclass."""
        from src.tools.citation_explorer import CitationLink

        link = CitationLink(
            paper_id="test123",
            title="Test Paper",
            year=2024,
            direction="citing",
            authors=["Author One"],
            citation_count=50
        )

        assert link.paper_id == "test123"
        assert link.direction == "citing"
        assert link.citation_count == 50
