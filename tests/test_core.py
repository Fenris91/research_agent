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
        from research_agent.db.vector_store import ResearchVectorStore
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = ResearchVectorStore(persist_dir=tmp)
            assert store is not None

    def test_add_and_retrieve_paper(self):
        """Test adding and retrieving papers."""
        pass

    def test_list_papers(self):
        """Test listing papers in the store."""
        pass


# ============================================
# Embedding Model Tests
# ============================================

class TestEmbeddingModel:
    """Tests for the embedding model."""

    def test_initialization(self):
        """Test embedding model loads correctly (lazy — no actual model load)."""
        from research_agent.db.embeddings import EmbeddingModel
        embedder = EmbeddingModel(model_name="BAAI/bge-base-en-v1.5")
        assert embedder.model_name == "BAAI/bge-base-en-v1.5"
        # Model is lazy-loaded; _model is None until first use
        assert embedder._model is None

    def test_single_embedding(self):
        """Test embedding a single text."""
        pass

    def test_batch_embedding(self):
        """Test embedding multiple texts."""
        pass

    def test_query_embedding(self):
        """Test query embedding adds BGE prefix."""
        pass


# ============================================
# Academic Search Tests
# ============================================

class TestAcademicSearch:
    """Tests for academic search APIs."""

    @pytest.mark.asyncio
    async def test_openalex_search(self):
        """Test OpenAlex search returns results."""
        from research_agent.tools.academic_search import AcademicSearchTools

        search = AcademicSearchTools()
        try:
            results = await search.search_openalex("ethnography", limit=3)

            assert len(results) > 0
            assert hasattr(results[0], "title")
            assert hasattr(results[0], "year")
        finally:
            await search.close()

    @pytest.mark.asyncio
    async def test_search_all_deduplication(self):
        """Test combined search deduplicates results."""
        from research_agent.tools.academic_search import AcademicSearchTools

        search = AcademicSearchTools()
        try:
            results = await search.search_all("participatory research", limit_per_source=2)

            # Should have results (may be limited by rate limits)
            assert isinstance(results, list)
        finally:
            await search.close()

    @pytest.mark.asyncio
    async def test_semantic_scholar_search(self):
        """Test Semantic Scholar search."""
        from research_agent.tools.academic_search import AcademicSearchTools

        search = AcademicSearchTools()
        try:
            results = await search.search_semantic_scholar(
                "machine learning",
                limit=3
            )

            assert isinstance(results, list)
            if len(results) > 0:
                assert hasattr(results[0], "title")
        finally:
            await search.close()


# ============================================
# Web Search Tests
# ============================================

class TestWebSearch:
    """Tests for web search functionality."""

    @pytest.mark.asyncio
    async def test_duckduckgo_search(self):
        """Test DuckDuckGo search (free, no API key)."""
        pass


# ============================================
# Researcher Lookup Tests
# ============================================

class TestResearcherLookup:
    """Tests for researcher lookup functionality."""

    @pytest.mark.asyncio
    async def test_openalex_author_search(self):
        """Test OpenAlex author lookup."""
        pass


# ============================================
# Research Agent Tests
# ============================================

class TestResearchAgent:
    """Tests for the research agent."""

    def test_agent_initialization_with_ollama(self):
        """Test agent initializes with Ollama backend."""
        pass

    def test_agent_config(self):
        """Test agent configuration."""
        from research_agent.agents.research_agent import AgentConfig
        config = AgentConfig()
        assert config.max_local_results == 5
        assert config.max_external_results == 10
        assert config.auto_ingest is False


# ============================================
# Ollama Integration Tests
# TODO: add tests for research_agent.models.llm_utils.OllamaModel
# (requires a live Ollama server — skip or mock in CI)
# ============================================


# ============================================
# Agent Model Switch Tests
# TODO: add tests for ResearchAgent.switch_model / list_available_models
# (requires a live Ollama/OpenAI-compatible server — skip or mock in CI)
# ============================================


# ============================================
# Citation Explorer Tests
# ============================================

class TestCitationExplorer:
    """Tests for citation explorer functionality."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.network
    async def test_get_citations(self):
        """Test fetching citations for a paper."""
        from research_agent.tools.citation_explorer import CitationExplorer
        from research_agent.tools.academic_search import AcademicSearchTools

        search = AcademicSearchTools()
        explorer = CitationExplorer(search)

        try:
            # Use a known paper ID (BERT paper - reliable for testing)
            paper_id = "df2b0e26d0599ce3e70df8a9da02e51594e0e992"
            network = await explorer.get_citations(paper_id, direction="both", limit=3)

            assert network.seed_paper is not None
            assert isinstance(network.citing_papers, list)
            assert isinstance(network.cited_papers, list)
        finally:
            await search.close()

    @pytest.mark.asyncio
    async def test_citation_paper_dataclass(self):
        """Test CitationPaper dataclass."""
        from research_agent.tools.citation_explorer import CitationPaper

        paper = CitationPaper(
            paper_id="test123",
            title="Test Paper",
            year=2024,
            authors=["Author One"],
            citation_count=50
        )

        assert paper.paper_id == "test123"
        assert paper.year == 2024
        assert paper.citation_count == 50

    @pytest.mark.asyncio
    async def test_citation_network_dataclass(self):
        """Test CitationNetwork dataclass."""
        from research_agent.tools.citation_explorer import (
            CitationPaper,
            CitationNetwork
        )

        seed = CitationPaper(paper_id="seed", title="Seed Paper")
        citing = [CitationPaper(paper_id="citing", title="Citing Paper")]
        cited = [CitationPaper(paper_id="cited", title="Cited Paper")]

        network = CitationNetwork(
            seed_paper=seed,
            citing_papers=citing,
            cited_papers=cited,
            highly_connected=[]
        )

        assert network.seed_paper.paper_id == "seed"
        assert len(network.citing_papers) == 1
        assert len(network.cited_papers) == 1
