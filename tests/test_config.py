"""
Test configuration for Research Agent tests.

Defines limits and settings to control API usage during testing.
"""

from dataclasses import dataclass
from typing import Optional


# Not a test class - this is config
@dataclass
class Config:
    """Configuration for test execution."""

    # API call limits
    MAX_RESULTS_PER_CALL: int = 5
    MAX_API_CALLS_PER_TEST: int = 10
    API_DELAY_SECONDS: float = 0.5

    # Timeout settings
    API_TIMEOUT_SECONDS: float = 30.0
    TEST_TIMEOUT_SECONDS: float = 60.0

    # Integration test settings
    RUN_INTEGRATION_TESTS: bool = True
    INTEGRATION_TEST_LIMIT: int = 3  # Max papers to test with real APIs

    # Mock settings
    USE_MOCKS_BY_DEFAULT: bool = True

    # Rate limiting
    REQUESTS_PER_MINUTE: int = 20

    @classmethod
    def get_api_limit(cls) -> int:
        """Get the API result limit for tests."""
        return cls.MAX_RESULTS_PER_CALL

    @classmethod
    def should_run_integration(cls) -> bool:
        """Check if integration tests should run."""
        return cls.RUN_INTEGRATION_TESTS


# Paper IDs for testing - verified Semantic Scholar IDs
class TestPaperIDs:
    """Known paper IDs for testing."""

    # Classic anthropology paper - The Interpretation of Cultures by Geertz
    # Note: Using a well-known paper with stable ID
    GEERTZ_INTERPRETATION = "204947902"

    # Spatial theory - The Production of Space by Lefebvre
    LEFEBVRE_SPACE = "141568395"

    # Modern high-impact paper - Attention Is All You Need
    # This has many citations and is well-indexed
    ATTENTION_PAPER = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    # Fallback: A known working paper ID from Semantic Scholar
    # Using BERT paper as reliable fallback
    BERT_PAPER = "df2b0e26d0599ce3e70df8a9da02e51594e0e992"


# Expected test outcomes
class ExpectedResults:
    """Expected results for validation."""

    # Minimum citations for well-known papers
    MIN_CITATIONS_GEERTZ = 100
    MIN_CITATIONS_LEFEBVRE = 50
    MIN_CITATIONS_ATTENTION = 1000

    # Expected to have both citing and cited papers
    MIN_CITING_PAPERS = 1
    MIN_CITED_PAPERS = 1


# Test timeouts
class Timeouts:
    """Timeout values for different test types."""

    UNIT_TEST = 5.0
    INTEGRATION_TEST = 30.0
    UI_TEST = 10.0
    NETWORK_TEST = 60.0
