"""
Pytest configuration and fixtures for Research Agent tests.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional, List

# Test configuration
from tests.test_config import Config


# ============================================
# Event Loop Configuration
# ============================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================
# Test Paper IDs and Data
# ============================================

@dataclass
class TestPaperInfo:
    """Information about test papers."""
    paper_id: str
    title: str
    author: str
    year: int
    expected_citations_min: int = 0


# Known papers for testing - using Semantic Scholar IDs
TEST_PAPERS = {
    "geertz_interpretation": TestPaperInfo(
        paper_id="204947902",  # The Interpretation of Cultures
        title="The Interpretation of Cultures",
        author="Clifford Geertz",
        year=1973,
        expected_citations_min=100
    ),
    "lefebvre_production_space": TestPaperInfo(
        paper_id="141568395",  # The Production of Space
        title="The Production of Space",
        author="Henri Lefebvre",
        year=1991,
        expected_citations_min=50
    ),
    "attention_is_all_you_need": TestPaperInfo(
        paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",  # Attention Is All You Need
        title="Attention Is All You Need",
        author="Vaswani et al.",
        year=2017,
        expected_citations_min=1000
    ),
}


@pytest.fixture
def test_paper_geertz():
    """Get Geertz test paper info."""
    return TEST_PAPERS["geertz_interpretation"]


@pytest.fixture
def test_paper_lefebvre():
    """Get Lefebvre test paper info."""
    return TEST_PAPERS["lefebvre_production_space"]


@pytest.fixture
def test_paper_transformer():
    """Get transformer paper info (high citation count, good for testing)."""
    return TEST_PAPERS["attention_is_all_you_need"]


@pytest.fixture
def all_test_papers():
    """Get all test papers."""
    return TEST_PAPERS


# ============================================
# Mock Data Fixtures
# ============================================

@pytest.fixture
def mock_citation_paper():
    """Create a mock CitationPaper."""
    from research_agent.tools.citation_explorer import CitationPaper
    return CitationPaper(
        paper_id="test_paper_001",
        title="Test Paper Title",
        year=2024,
        authors=["Test Author"],
        citation_count=100,
        abstract="This is a test abstract.",
        venue="Test Journal",
        url="https://example.com/paper"
    )


@pytest.fixture
def mock_citation_network():
    """Create a mock CitationNetwork."""
    from research_agent.tools.citation_explorer import CitationPaper, CitationNetwork

    seed = CitationPaper(
        paper_id="seed_001",
        title="Seed Paper",
        year=2020,
        authors=["Seed Author"],
        citation_count=500
    )

    citing = [
        CitationPaper(
            paper_id=f"citing_{i}",
            title=f"Citing Paper {i}",
            year=2021 + i,
            citation_count=10 * i
        )
        for i in range(3)
    ]

    cited = [
        CitationPaper(
            paper_id=f"cited_{i}",
            title=f"Cited Paper {i}",
            year=2015 + i,
            citation_count=100 * i
        )
        for i in range(3)
    ]

    return CitationNetwork(
        seed_paper=seed,
        citing_papers=citing,
        cited_papers=cited,
        highly_connected=[]
    )


@pytest.fixture
def mock_api_response_citations():
    """Mock API response for citations endpoint."""
    return {
        "data": [
            {
                "citingPaper": {
                    "paperId": "citing_001",
                    "title": "A Paper That Cites",
                    "year": 2023,
                    "citationCount": 50
                }
            },
            {
                "citingPaper": {
                    "paperId": "citing_002",
                    "title": "Another Citing Paper",
                    "year": 2024,
                    "citationCount": 25
                }
            }
        ]
    }


@pytest.fixture
def mock_api_response_references():
    """Mock API response for references endpoint."""
    return {
        "data": [
            {
                "citedPaper": {
                    "paperId": "cited_001",
                    "title": "A Referenced Paper",
                    "year": 2018,
                    "citationCount": 200
                }
            },
            {
                "citedPaper": {
                    "paperId": "cited_002",
                    "title": "Another Reference",
                    "year": 2019,
                    "citationCount": 150
                }
            }
        ]
    }


@pytest.fixture
def mock_api_response_paper_details():
    """Mock API response for paper details endpoint."""
    return {
        "paperId": "test_paper_001",
        "title": "Test Paper Title",
        "year": 2024,
        "authors": [{"name": "Test Author"}],
        "citationCount": 100,
        "abstract": "This is a test abstract.",
        "venue": "Test Journal"
    }


# ============================================
# Mock Academic Search Fixture
# ============================================

@pytest.fixture
def mock_academic_search():
    """Create a mock AcademicSearchTools instance."""
    mock = MagicMock()
    mock.SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

    # Mock the HTTP client
    mock_client = MagicMock()

    # Create async mock for _get_client that returns sync mock_client
    async def get_client():
        return mock_client

    mock._get_client = get_client
    mock.close = AsyncMock()

    return mock, mock_client


# ============================================
# Rate Limiting Helpers
# ============================================

class APICallTracker:
    """Track API calls for rate limiting tests."""

    def __init__(self, max_calls: int = Config.MAX_API_CALLS_PER_TEST):
        self.call_count = 0
        self.max_calls = max_calls
        self.calls = []

    def record_call(self, endpoint: str):
        """Record an API call."""
        self.call_count += 1
        self.calls.append(endpoint)

        if self.call_count > self.max_calls:
            raise RuntimeError(f"Exceeded max API calls ({self.max_calls})")

    def reset(self):
        """Reset call counter."""
        self.call_count = 0
        self.calls = []


@pytest.fixture
def api_call_tracker():
    """Create an API call tracker."""
    return APICallTracker()


# ============================================
# Integration Test Markers
# ============================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (hits real APIs)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "ui: mark test as UI component test"
    )


# ============================================
# Skip Markers for Conditional Tests
# ============================================

@pytest.fixture
def skip_if_no_network():
    """Skip test if no network connection."""
    import socket
    try:
        socket.create_connection(("api.semanticscholar.org", 443), timeout=5)
    except OSError:
        pytest.skip("No network connection available")
