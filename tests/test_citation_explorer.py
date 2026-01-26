"""
Comprehensive tests for Citation Explorer.

Tests are organized into:
1. Unit tests (mocked) - Core logic validation
2. Integration tests (limited real APIs) - End-to-end workflows
3. Error handling tests - Edge cases and invalid inputs
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

from tests.test_config import Config, TestPaperIDs, ExpectedResults, Timeouts


# ============================================
# Unit Tests (Mocked)
# ============================================

class TestCitationExplorerUnit:
    """Unit tests using mocked API responses."""

    @pytest.fixture
    def explorer_with_mock(self, mock_academic_search):
        """Create CitationExplorer with mocked dependencies."""
        from research_agent.tools.citation_explorer import CitationExplorer

        mock_search, mock_client = mock_academic_search
        explorer = CitationExplorer(mock_search)
        return explorer, mock_client

    @pytest.mark.asyncio
    async def test_citation_paper_dataclass(self, mock_citation_paper):
        """Test CitationPaper dataclass fields."""
        assert mock_citation_paper.paper_id == "test_paper_001"
        assert mock_citation_paper.title == "Test Paper Title"
        assert mock_citation_paper.year == 2024
        assert mock_citation_paper.citation_count == 100

    @pytest.mark.asyncio
    async def test_citation_network_dataclass(self, mock_citation_network):
        """Test CitationNetwork dataclass structure."""
        assert mock_citation_network.seed_paper.paper_id == "seed_001"
        assert len(mock_citation_network.citing_papers) == 3
        assert len(mock_citation_network.cited_papers) == 3

    @pytest.mark.asyncio
    async def test_get_citing_papers_mocked(
        self, explorer_with_mock, mock_api_response_citations
    ):
        """Test _get_citing_papers with mocked API."""
        explorer, mock_client = explorer_with_mock

        # Setup mock response - use MagicMock for sync methods, AsyncMock for get
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response_citations
        mock_response.raise_for_status = MagicMock()

        # Make get() return a coroutine that returns mock_response
        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client.get = mock_get

        papers = await explorer._get_citing_papers("test_id", limit=5)

        assert len(papers) == 2
        assert papers[0].paper_id == "citing_001"
        assert papers[0].title == "A Paper That Cites"

    @pytest.mark.asyncio
    async def test_get_cited_papers_mocked(
        self, explorer_with_mock, mock_api_response_references
    ):
        """Test _get_cited_papers with mocked API."""
        explorer, mock_client = explorer_with_mock

        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response_references
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client.get = mock_get

        papers = await explorer._get_cited_papers("test_id", limit=5)

        assert len(papers) == 2
        assert papers[0].paper_id == "cited_001"
        assert papers[0].title == "A Referenced Paper"

    @pytest.mark.asyncio
    async def test_get_paper_details_mocked(
        self, explorer_with_mock, mock_api_response_paper_details
    ):
        """Test _get_paper_details with mocked API."""
        explorer, mock_client = explorer_with_mock

        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_api_response_paper_details
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client.get = mock_get

        paper = await explorer._get_paper_details("test_id")

        assert paper.paper_id == "test_paper_001"
        assert paper.title == "Test Paper Title"
        assert paper.year == 2024
        assert paper.citation_count == 100

    @pytest.mark.asyncio
    async def test_build_network_data(self, mock_citation_network):
        """Test network visualization data building."""
        from research_agent.tools.citation_explorer import CitationExplorer
        from research_agent.tools.academic_search import AcademicSearchTools

        # Create explorer (we'll use instance method without hitting API)
        search = AcademicSearchTools()
        explorer = CitationExplorer(search)

        network_data = explorer.build_network_data(mock_citation_network)

        assert "nodes" in network_data
        assert "edges" in network_data
        assert "stats" in network_data

        # Should have seed + citing + cited nodes
        assert len(network_data["nodes"]) == 7  # 1 seed + 3 citing + 3 cited

        # Should have edges from citing to seed and from seed to cited
        assert len(network_data["edges"]) == 6  # 3 + 3

        assert network_data["stats"]["citing_count"] == 3
        assert network_data["stats"]["cited_count"] == 3

        await search.close()

    @pytest.mark.asyncio
    async def test_find_highly_connected_empty(self, explorer_with_mock):
        """Test find_highly_connected with no papers."""
        explorer, mock_client = explorer_with_mock

        result = await explorer.find_highly_connected([])

        assert result == []


# ============================================
# Integration Tests (Real APIs, Limited)
# ============================================

@pytest.mark.integration
class TestCitationExplorerIntegration:
    """Integration tests with real API calls (limited)."""

    @pytest_asyncio.fixture
    async def explorer(self):
        """Create real CitationExplorer for integration tests."""
        from research_agent.tools.citation_explorer import CitationExplorer
        from research_agent.tools.academic_search import AcademicSearchTools

        search = AcademicSearchTools(request_delay=Config.API_DELAY_SECONDS)
        explorer = CitationExplorer(search)
        yield explorer
        await search.close()

    @pytest.mark.asyncio
    @pytest.mark.timeout(Timeouts.INTEGRATION_TEST)
    async def test_get_paper_details_real(self, explorer, skip_if_no_network):
        """Test fetching real paper details."""
        # Use BERT paper as reliable test case
        paper = await explorer._get_paper_details(TestPaperIDs.BERT_PAPER)

        assert paper.paper_id is not None
        assert paper.title is not None
        assert len(paper.title) > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(Timeouts.INTEGRATION_TEST)
    async def test_get_citing_papers_real(self, explorer, skip_if_no_network):
        """Test fetching real citing papers."""
        # Use BERT paper - known to have many citations
        papers = await explorer._get_citing_papers(
            TestPaperIDs.BERT_PAPER,
            limit=Config.MAX_RESULTS_PER_CALL
        )

        # Should get some citing papers (BERT is heavily cited)
        assert isinstance(papers, list)
        # Note: API may return fewer than requested
        if len(papers) > 0:
            assert papers[0].paper_id is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(Timeouts.INTEGRATION_TEST)
    async def test_get_cited_papers_real(self, explorer, skip_if_no_network):
        """Test fetching real cited papers (references)."""
        # Use BERT paper
        papers = await explorer._get_cited_papers(
            TestPaperIDs.BERT_PAPER,
            limit=Config.MAX_RESULTS_PER_CALL
        )

        assert isinstance(papers, list)
        if len(papers) > 0:
            assert papers[0].paper_id is not None

    @pytest.mark.asyncio
    @pytest.mark.timeout(Timeouts.NETWORK_TEST)
    async def test_get_citations_both_directions(self, explorer, skip_if_no_network):
        """Test getting citations in both directions."""
        network = await explorer.get_citations(
            TestPaperIDs.BERT_PAPER,
            direction="both",
            limit=Config.MAX_RESULTS_PER_CALL
        )

        assert network.seed_paper is not None
        assert isinstance(network.citing_papers, list)
        assert isinstance(network.cited_papers, list)


# ============================================
# Error Handling Tests
# ============================================

class TestCitationExplorerErrors:
    """Tests for error handling and edge cases."""

    @pytest.fixture
    def explorer_with_failing_mock(self, mock_academic_search):
        """Create explorer with mock that raises errors."""
        from research_agent.tools.citation_explorer import CitationExplorer

        mock_search, mock_client = mock_academic_search
        explorer = CitationExplorer(mock_search)
        return explorer, mock_client

    @pytest.mark.asyncio
    async def test_get_citing_papers_api_error(self, explorer_with_failing_mock):
        """Test handling of API errors in _get_citing_papers."""
        explorer, mock_client = explorer_with_failing_mock

        async def mock_get(*args, **kwargs):
            raise Exception("API Error")

        mock_client.get = mock_get

        papers = await explorer._get_citing_papers("test_id", limit=5)

        # Should return empty list on error
        assert papers == []

    @pytest.mark.asyncio
    async def test_get_cited_papers_api_error(self, explorer_with_failing_mock):
        """Test handling of API errors in _get_cited_papers."""
        explorer, mock_client = explorer_with_failing_mock

        async def mock_get(*args, **kwargs):
            raise Exception("API Error")

        mock_client.get = mock_get

        papers = await explorer._get_cited_papers("test_id", limit=5)

        assert papers == []

    @pytest.mark.asyncio
    async def test_get_paper_details_fallback(self, explorer_with_failing_mock):
        """Test fallback behavior when paper details fail."""
        explorer, mock_client = explorer_with_failing_mock

        async def mock_get(*args, **kwargs):
            raise Exception("Not Found")

        mock_client.get = mock_get

        paper = await explorer._get_paper_details("unknown_id")

        # Should return basic paper info as fallback
        assert paper.paper_id == "unknown_id"
        assert "Paper" in paper.title

    @pytest.mark.asyncio
    async def test_empty_api_response(self, explorer_with_failing_mock):
        """Test handling of empty API response."""
        explorer, mock_client = explorer_with_failing_mock

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client.get = mock_get

        papers = await explorer._get_citing_papers("test_id", limit=5)

        assert papers == []

    @pytest.mark.asyncio
    async def test_malformed_api_response(self, explorer_with_failing_mock):
        """Test handling of malformed API response."""
        explorer, mock_client = explorer_with_failing_mock

        mock_response = MagicMock()
        mock_response.json.return_value = {"unexpected": "format"}
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client.get = mock_get

        papers = await explorer._get_citing_papers("test_id", limit=5)

        # Should handle gracefully
        assert papers == []

    @pytest.mark.asyncio
    async def test_null_fields_in_response(self, explorer_with_failing_mock):
        """Test handling of null/missing fields in response."""
        explorer, mock_client = explorer_with_failing_mock

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "citingPaper": {
                        "paperId": "paper_with_nulls",
                        "title": None,
                        "year": None,
                        "citationCount": None
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client.get = mock_get

        papers = await explorer._get_citing_papers("test_id", limit=5)

        assert len(papers) == 1
        assert papers[0].paper_id == "paper_with_nulls"
        # When title is explicitly null in response, it remains None
        assert papers[0].title is None or papers[0].title == "Unknown"


# ============================================
# Suggest Related Tests
# ============================================

class TestSuggestRelated:
    """Tests for the suggest_related functionality."""

    @pytest.fixture
    def explorer_with_mock(self, mock_academic_search):
        """Create CitationExplorer with mocked dependencies."""
        from research_agent.tools.citation_explorer import CitationExplorer

        mock_search, mock_client = mock_academic_search
        explorer = CitationExplorer(mock_search)
        return explorer, mock_client

    @pytest.mark.asyncio
    async def test_suggest_related_returns_list(self, explorer_with_mock):
        """Test that suggest_related returns a list."""
        explorer, mock_client = explorer_with_mock

        # Setup mock for get_citations call
        mock_response = AsyncMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        # Mock the network data
        with patch.object(explorer, 'get_citations') as mock_get_citations:
            from research_agent.tools.citation_explorer import CitationNetwork, CitationPaper

            mock_network = CitationNetwork(
                seed_paper=CitationPaper(paper_id="seed", title="Seed"),
                citing_papers=[],
                cited_papers=[],
                highly_connected=[]
            )
            mock_get_citations.return_value = mock_network

            result = await explorer.suggest_related("test_paper", limit=5)

            assert isinstance(result, list)
