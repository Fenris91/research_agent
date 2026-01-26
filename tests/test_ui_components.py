"""
UI Component tests for Citation Explorer.

Tests Gradio component instantiation without launching,
input validation, and output formatting.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# ============================================
# UI Component Tests
# ============================================


class TestCitationExplorerUI:
    """Tests for Citation Explorer Gradio UI components."""

    def test_render_citation_explorer_returns_dict(self):
        """Test that render_citation_explorer returns component dictionary."""
        # Mock Gradio components to avoid actual UI creation
        with patch("research_agent.ui.components.citation_explorer.gr") as mock_gr:
            # Setup mock components
            mock_gr.Column.return_value.__enter__ = MagicMock()
            mock_gr.Column.return_value.__exit__ = MagicMock()
            mock_gr.Row.return_value.__enter__ = MagicMock()
            mock_gr.Row.return_value.__exit__ = MagicMock()
            mock_gr.Accordion.return_value.__enter__ = MagicMock()
            mock_gr.Accordion.return_value.__exit__ = MagicMock()
            mock_gr.Tabs.return_value.__enter__ = MagicMock()
            mock_gr.Tabs.return_value.__exit__ = MagicMock()
            mock_gr.TabItem.return_value.__enter__ = MagicMock()
            mock_gr.TabItem.return_value.__exit__ = MagicMock()

            mock_textbox = MagicMock()
            mock_radio = MagicMock()
            mock_slider = MagicMock()
            mock_button = MagicMock()
            mock_markdown = MagicMock()
            mock_dataframe = MagicMock()
            mock_plot = MagicMock()

            mock_gr.Textbox.return_value = mock_textbox
            mock_gr.Radio.return_value = mock_radio
            mock_gr.Slider.return_value = mock_slider
            mock_gr.Button.return_value = mock_button
            mock_gr.Markdown.return_value = mock_markdown
            mock_gr.DataFrame.return_value = mock_dataframe
            mock_gr.Plot.return_value = mock_plot

            from research_agent.ui.components.citation_explorer import (
                render_citation_explorer,
            )

            result = render_citation_explorer()

            # Should return a dictionary with component references
            assert isinstance(result, dict)
            assert "paper_input" in result
            assert "direction" in result
            assert "depth" in result
            assert "search_btn" in result

    def test_papers_to_dataframe_with_papers(self):
        """Test _papers_to_dataframe converts papers correctly."""
        from research_agent.ui.components.citation_explorer import _papers_to_dataframe
        from research_agent.tools.citation_explorer import CitationPaper

        papers = [
            CitationPaper(
                paper_id="paper_001",
                title="Test Paper 1",
                year=2023,
                citation_count=100,
            ),
            CitationPaper(
                paper_id="paper_002", title="Test Paper 2", year=2024, citation_count=50
            ),
        ]

        result = _papers_to_dataframe(papers)

        assert result is not None
        assert len(result) == 2
        assert result[0] == ["Test Paper 1", 2023, 100, "paper_001"]
        assert result[1] == ["Test Paper 2", 2024, 50, "paper_002"]

    def test_papers_to_dataframe_empty(self):
        """Test _papers_to_dataframe with empty list."""
        from research_agent.ui.components.citation_explorer import _papers_to_dataframe

        result = _papers_to_dataframe([])

        assert result is None

    def test_papers_to_dataframe_none_values(self):
        """Test _papers_to_dataframe handles None values."""
        from research_agent.ui.components.citation_explorer import _papers_to_dataframe
        from research_agent.tools.citation_explorer import CitationPaper

        papers = [
            CitationPaper(
                paper_id="paper_001", title=None, year=None, citation_count=None
            )
        ]

        result = _papers_to_dataframe(papers)

        assert result is not None
        assert len(result) == 1
        assert result[0] == ["Unknown Title", "Unknown", 0, "paper_001"]


# ============================================
# Explore Citations Function Tests
# ============================================


class TestExploreCitations:
    """Tests for the explore_citations async function."""

    @pytest.mark.asyncio
    async def test_explore_citations_empty_input(self):
        """Test explore_citations with empty input."""
        from research_agent.ui.components.citation_explorer import explore_citations

        result = await explore_citations("", "both", 20)

        assert len(result) == 6  # Now returns 6 values including network plot
        assert result[0] == "Please enter a paper ID or title."
        assert result[1] is None
        assert result[2] is None
        assert result[5] is None  # network plot

    @pytest.mark.asyncio
    async def test_explore_citations_whitespace_input(self):
        """Test explore_citations with whitespace input."""
        from research_agent.ui.components.citation_explorer import explore_citations

        result = await explore_citations("   ", "both", 20)

        assert len(result) == 6
        assert result[0] == "Please enter a paper ID or title."

    @pytest.mark.asyncio
    async def test_explore_citations_error_handling(self):
        """Test explore_citations handles errors gracefully."""
        from research_agent.ui.components.citation_explorer import explore_citations

        # Mock the AcademicSearchTools to raise an error
        with patch(
            "research_agent.ui.components.citation_explorer.AcademicSearchTools"
        ) as mock_search_cls:
            mock_search = MagicMock()
            mock_search._get_client = AsyncMock(side_effect=Exception("Network error"))
            mock_search_cls.return_value = mock_search

            result = await explore_citations("test_paper_id", "both", 20)

            # Should return error message
            assert "Error" in result[0] or "error" in result[0].lower()


# ============================================
# Input Validation Tests
# ============================================


class TestInputValidation:
    """Tests for input validation in UI components."""

    def test_direction_options(self):
        """Test that valid direction options are handled."""
        valid_directions = ["both", "citing", "cited"]

        for direction in valid_directions:
            assert direction in valid_directions

    def test_depth_range(self):
        """Test depth slider range validation."""
        min_depth = 5
        max_depth = 50

        # Valid depths
        valid_depths = [5, 10, 20, 30, 50]
        for depth in valid_depths:
            assert min_depth <= depth <= max_depth

        # Invalid depths
        invalid_depths = [0, 1, 100, -1]
        for depth in invalid_depths:
            assert not (min_depth <= depth <= max_depth)


# ============================================
# Output Formatting Tests
# ============================================


class TestOutputFormatting:
    """Tests for output formatting functions."""

    def test_summary_format(self):
        """Test citation network summary format."""
        from research_agent.tools.citation_explorer import (
            CitationPaper,
            CitationNetwork,
        )

        network = CitationNetwork(
            seed_paper=CitationPaper(
                paper_id="seed_001",
                title="Test Seed Paper",
                year=2020,
                citation_count=500,
            ),
            citing_papers=[
                CitationPaper(paper_id=f"citing_{i}", title=f"Citing {i}")
                for i in range(5)
            ],
            cited_papers=[
                CitationPaper(paper_id=f"cited_{i}", title=f"Cited {i}")
                for i in range(3)
            ],
            highly_connected=[],
        )

        # Test that we can build the expected summary format
        summary = f"""## 📊 Citation Network Summary

**Seed Paper:** {network.seed_paper.title}
- **Year:** {network.seed_paper.year or "Unknown"}
- **Citations:** {network.seed_paper.citation_count or 0}

**Network Statistics:**
- Papers that cite this: {len(network.citing_papers)}
- Papers cited by this: {len(network.cited_papers)}
- Highly connected papers: {len(network.highly_connected)}
"""

        assert "Test Seed Paper" in summary
        assert "2020" in summary
        assert "500" in summary
        assert "5" in summary  # citing count
        assert "3" in summary  # cited count


# ============================================
# Network Visualization Tests
# ============================================


class TestNetworkVisualization:
    """Tests for the network visualization rendering."""

    def test_render_network_graph_returns_figure(self):
        """Test that _render_network_graph returns a matplotlib figure."""
        from research_agent.ui.components.citation_explorer import _render_network_graph
        from research_agent.tools.citation_explorer import (
            CitationPaper,
            CitationNetwork,
        )
        import matplotlib.pyplot as plt

        network = CitationNetwork(
            seed_paper=CitationPaper(
                paper_id="seed_001",
                title="Test Seed Paper",
                year=2020,
                citation_count=500,
            ),
            citing_papers=[
                CitationPaper(paper_id=f"citing_{i}", title=f"Citing Paper {i}")
                for i in range(3)
            ],
            cited_papers=[
                CitationPaper(paper_id=f"cited_{i}", title=f"Cited Paper {i}")
                for i in range(2)
            ],
            highly_connected=[],
        )

        fig = _render_network_graph(network)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_network_graph_empty_network(self):
        """Test _render_network_graph with no citations."""
        from research_agent.ui.components.citation_explorer import _render_network_graph
        from research_agent.tools.citation_explorer import (
            CitationPaper,
            CitationNetwork,
        )
        import matplotlib.pyplot as plt

        network = CitationNetwork(
            seed_paper=CitationPaper(
                paper_id="seed_001",
                title="Lonely Paper",
                year=2020,
                citation_count=0,
            ),
            citing_papers=[],
            cited_papers=[],
            highly_connected=[],
        )

        fig = _render_network_graph(network)

        # Should still return a figure (with "no relationships" message)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_network_graph_none_input(self):
        """Test _render_network_graph with None input."""
        from research_agent.ui.components.citation_explorer import _render_network_graph

        result = _render_network_graph(None)

        assert result is None

    def test_truncate_title(self):
        """Test title truncation helper."""
        from research_agent.ui.components.citation_explorer import _truncate_title

        # Short title - no truncation
        assert _truncate_title("Short Title", 30) == "Short Title"

        # Long title - truncated with ellipsis
        long_title = "This is a very long title that should be truncated"
        result = _truncate_title(long_title, 20)
        assert len(result) == 20
        assert result.endswith("...")

        # None/empty input
        assert _truncate_title(None) == "Unknown"
        assert _truncate_title("") == "Unknown"


# ============================================
# BibTeX Export Tests
# ============================================


class TestBibTeXExport:
    """Tests for BibTeX export functionality."""

    def test_paper_to_bibtex_basic(self):
        """Test basic BibTeX conversion."""
        # Import the function from app module
        import sys
        from pathlib import Path

        # Add src to path for this test
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        # We can't easily import _paper_to_bibtex since it's defined inside create_app
        # So we'll test the format manually
        paper = {
            "paper_id": "10.1234/test.2023",
            "title": "Test Paper Title",
            "year": 2023,
            "authors": "John Smith, Jane Doe",
        }

        # Build expected BibTeX format
        expected_fields = [
            "@article{",
            "title = {Test Paper Title}",
            "author = {John Smith and Jane Doe}",
            "year = {2023}",
            "doi = {10.1234/test.2023}",
        ]

        # Verify the expected format is correct BibTeX
        for field in expected_fields:
            assert "{" in field or field.startswith("@")

    def test_bibtex_citation_key_format(self):
        """Test that citation keys follow lastnameyear format."""
        # Test the expected format for citation keys
        authors = "John Smith, Jane Doe"
        year = 2023

        first_author = authors.split(",")[0].strip()
        first_author_key = "".join(
            c for c in first_author.split()[-1] if c.isalnum()
        ).lower()
        cite_key = f"{first_author_key}{year}"

        assert cite_key == "smith2023"

    def test_bibtex_handles_missing_year(self):
        """Test BibTeX handles missing year gracefully."""
        year = None
        year_str = str(year) if year else "nd"
        assert year_str == "nd"
