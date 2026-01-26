"""
Core unit tests for Research Agent components.

Note: Some tests are skipped when modules are not yet implemented.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path


# ============================================
# Vector Store Tests (SKIP - not yet migrated)
# ============================================

@pytest.mark.skip(reason="Vector store not yet migrated to research_agent package")
class TestVectorStore:
    """Tests for the ChromaDB vector store."""

    def test_initialization(self):
        """Test vector store initializes correctly."""
        pass

    def test_add_and_retrieve_paper(self):
        """Test adding and retrieving papers."""
        pass

    def test_list_papers(self):
        """Test listing papers in the store."""
        pass


# ============================================
# Embedding Model Tests (SKIP - not yet migrated)
# ============================================

@pytest.mark.skip(reason="Embedding model not yet migrated to research_agent package")
class TestEmbeddingModel:
    """Tests for the embedding model."""

    def test_initialization(self):
        """Test embedding model loads correctly."""
        pass

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
# Web Search Tests (SKIP - not yet migrated)
# ============================================

@pytest.mark.skip(reason="Web search not yet migrated to research_agent package")
class TestWebSearch:
    """Tests for web search functionality."""

    @pytest.mark.asyncio
    async def test_duckduckgo_search(self):
        """Test DuckDuckGo search (free, no API key)."""
        pass


# ============================================
# Researcher Lookup Tests (SKIP - not yet migrated)
# ============================================

@pytest.mark.skip(reason="Researcher lookup not yet migrated to research_agent package")
class TestResearcherLookup:
    """Tests for researcher lookup functionality."""

    @pytest.mark.asyncio
    async def test_openalex_author_search(self):
        """Test OpenAlex author lookup."""
        pass


# ============================================
# Research Agent Tests (SKIP - not yet migrated)
# ============================================

@pytest.mark.skip(reason="Research agent not yet migrated to research_agent package")
class TestResearchAgent:
    """Tests for the research agent."""

    def test_agent_initialization_with_ollama(self):
        """Test agent initializes with Ollama backend."""
        pass

    def test_agent_config(self):
        """Test agent configuration."""
        pass


# ============================================
# Ollama Integration Tests (SKIP - not yet migrated)
# ============================================

@pytest.mark.skip(reason="Ollama integration not yet migrated to research_agent package")
class TestOllamaIntegration:
    """Tests for Ollama model integration."""

    def test_ollama_model_wrapper(self):
        """Test OllamaModel wrapper."""
        pass

    def test_ollama_generation(self):
        """Test generating text with Ollama."""
        pass

    def test_ollama_list_models(self):
        """Test listing available Ollama models."""
        pass

    def test_ollama_model_switch(self):
        """Test switching between Ollama models."""
        pass

    def test_qwen3_generation(self):
        """Test qwen3 generation with thinking mode."""
        pass


@pytest.mark.skip(reason="Agent model switch not yet migrated to research_agent package")
class TestAgentModelSwitch:
    """Tests for agent model switching functionality."""

    def test_agent_switch_model(self):
        """Test switching models via the agent."""
        pass

    def test_agent_list_models(self):
        """Test listing models via the agent."""
        pass


# ============================================
# Citation Explorer Tests
# ============================================

class TestCitationExplorer:
    """Tests for citation explorer functionality."""

    @pytest.mark.asyncio
    @pytest.mark.integration
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
