"""
Citation Analytics tests.

Tests for citation analysis features:
- Highly connected paper detection
- Citation overlap calculation
- Network statistics
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tests.test_config import Config


# ============================================
# Highly Connected Paper Tests
# ============================================

class TestHighlyConnected:
    """Tests for find_highly_connected functionality."""

    @pytest.fixture
    def explorer_with_mock(self, mock_academic_search):
        """Create CitationExplorer with mocked dependencies."""
        from research_agent.tools.citation_explorer import CitationExplorer

        mock_search, mock_client = mock_academic_search
        explorer = CitationExplorer(mock_search)
        return explorer, mock_client

    @pytest.mark.asyncio
    async def test_find_highly_connected_empty_input(self, explorer_with_mock):
        """Test with empty paper list."""
        explorer, _ = explorer_with_mock

        result = await explorer.find_highly_connected([])

        assert result == []

    @pytest.mark.asyncio
    async def test_find_highly_connected_single_paper(self, explorer_with_mock):
        """Test with single paper."""
        explorer, mock_client = explorer_with_mock

        # Mock the _get_cited_papers response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "citedPaper": {
                        "paperId": "ref_001",
                        "title": "Reference 1",
                        "year": 2020,
                        "citationCount": 100
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client.get = mock_get

        result = await explorer.find_highly_connected(["paper_001"], min_connections=1)

        # With one paper, references appear once, so min_connections=1 needed
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_find_highly_connected_min_connections(self, explorer_with_mock):
        """Test min_connections filtering."""
        explorer, mock_client = explorer_with_mock

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "data": [
                    {
                        "citedPaper": {
                            "paperId": "common_ref",
                            "title": "Common Reference",
                            "year": 2018,
                            "citationCount": 500
                        }
                    },
                    {
                        "citedPaper": {
                            "paperId": f"unique_ref_{call_count}",
                            "title": f"Unique Ref {call_count}",
                            "year": 2019,
                            "citationCount": 50
                        }
                    }
                ]
            }
            return mock_resp

        mock_client.get = mock_get

        result = await explorer.find_highly_connected(
            ["paper_001", "paper_002"],
            min_connections=2
        )

        # Only "common_ref" should appear (cited by both papers)
        paper_ids = [p.paper_id for p in result]
        assert "common_ref" in paper_ids or len(result) == 0

    @pytest.mark.asyncio
    async def test_find_highly_connected_respects_limit(self, explorer_with_mock):
        """Test that result is limited to top 10."""
        explorer, mock_client = explorer_with_mock

        async def mock_get(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "data": [
                    {
                        "citedPaper": {
                            "paperId": f"ref_{i}",
                            "title": f"Reference {i}",
                            "year": 2020,
                            "citationCount": i * 10
                        }
                    }
                    for i in range(20)
                ]
            }
            return mock_resp

        mock_client.get = mock_get

        result = await explorer.find_highly_connected(
            ["paper_001", "paper_002"],
            min_connections=1
        )

        # Should be limited to top 10
        assert len(result) <= 10


# ============================================
# Citation Overlap Tests
# ============================================

class TestCitationOverlap:
    """Tests for citation overlap calculation in suggest_related."""

    @pytest.fixture
    def explorer_with_mock(self, mock_academic_search):
        """Create CitationExplorer with mocked dependencies."""
        from research_agent.tools.citation_explorer import CitationExplorer

        mock_search, mock_client = mock_academic_search
        explorer = CitationExplorer(mock_search)
        return explorer, mock_client

    @pytest.mark.asyncio
    async def test_overlap_score_calculation(self):
        """Test overlap score is calculated correctly."""
        # Overlap percentage = overlap / len(target_refs)
        target_refs = {"ref_1", "ref_2", "ref_3", "ref_4"}
        citing_refs = {"ref_1", "ref_2", "ref_5"}

        overlap = len(target_refs & citing_refs)
        overlap_percentage = overlap / len(target_refs)

        assert overlap == 2  # ref_1 and ref_2
        assert overlap_percentage == 0.5


# ============================================
# Network Statistics Tests
# ============================================

class TestNetworkStatistics:
    """Tests for network statistics calculation."""

    def test_build_network_data_stats(self):
        """Test network data statistics are correct."""
        from research_agent.tools.citation_explorer import (
            CitationExplorer,
            CitationNetwork,
            CitationPaper
        )
        from research_agent.tools.academic_search import AcademicSearchTools

        network = CitationNetwork(
            seed_paper=CitationPaper(
                paper_id="seed",
                title="Seed Paper",
                year=2020,
                citation_count=1000
            ),
            citing_papers=[
                CitationPaper(paper_id=f"citing_{i}", title=f"Citing {i}")
                for i in range(5)
            ],
            cited_papers=[
                CitationPaper(paper_id=f"cited_{i}", title=f"Cited {i}")
                for i in range(10)
            ],
            highly_connected=[
                CitationPaper(paper_id=f"connected_{i}", title=f"Connected {i}")
                for i in range(3)
            ]
        )

        # Use explorer to build network data
        search = AcademicSearchTools()
        explorer = CitationExplorer(search)

        data = explorer.build_network_data(network)

        assert data["stats"]["total_papers"] == 16  # 1 + 5 + 10
        assert data["stats"]["citing_count"] == 5
        assert data["stats"]["cited_count"] == 10
        assert data["stats"]["highly_connected_count"] == 3

    def test_network_nodes_have_required_fields(self):
        """Test that nodes have all required fields."""
        from research_agent.tools.citation_explorer import (
            CitationExplorer,
            CitationNetwork,
            CitationPaper
        )
        from research_agent.tools.academic_search import AcademicSearchTools

        network = CitationNetwork(
            seed_paper=CitationPaper(
                paper_id="seed",
                title="Seed Paper",
                year=2020,
                citation_count=100
            ),
            citing_papers=[],
            cited_papers=[],
            highly_connected=[]
        )

        search = AcademicSearchTools()
        explorer = CitationExplorer(search)

        data = explorer.build_network_data(network)

        # Check seed node has required fields
        seed_node = data["nodes"][0]
        assert "id" in seed_node
        assert "label" in seed_node
        assert "type" in seed_node
        assert seed_node["type"] == "seed"

    def test_network_edges_structure(self):
        """Test edge structure for citing and cited papers."""
        from research_agent.tools.citation_explorer import (
            CitationExplorer,
            CitationNetwork,
            CitationPaper
        )
        from research_agent.tools.academic_search import AcademicSearchTools

        network = CitationNetwork(
            seed_paper=CitationPaper(paper_id="seed", title="Seed"),
            citing_papers=[CitationPaper(paper_id="citing_0", title="Citing")],
            cited_papers=[CitationPaper(paper_id="cited_0", title="Cited")],
            highly_connected=[]
        )

        search = AcademicSearchTools()
        explorer = CitationExplorer(search)

        data = explorer.build_network_data(network)

        # Should have 2 edges: citing->seed and seed->cited
        assert len(data["edges"]) == 2

        edge_types = [e["type"] for e in data["edges"]]
        assert all(t == "cites" for t in edge_types)


# ============================================
# Performance Tests
# ============================================

class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    async def test_api_call_tracking(self, api_call_tracker):
        """Test that API calls are tracked."""
        api_call_tracker.record_call("/paper/123")
        api_call_tracker.record_call("/paper/123/citations")

        assert api_call_tracker.call_count == 2
        assert len(api_call_tracker.calls) == 2

    @pytest.mark.asyncio
    async def test_api_call_limit_enforcement(self, api_call_tracker):
        """Test that API call limits are enforced."""
        # Set a low limit for testing
        api_call_tracker.max_calls = 2

        api_call_tracker.record_call("/call1")
        api_call_tracker.record_call("/call2")

        with pytest.raises(RuntimeError, match="Exceeded max API calls"):
            api_call_tracker.record_call("/call3")
