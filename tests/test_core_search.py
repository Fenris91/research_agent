"""Tests for CORE API integration in AcademicSearchTools."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

SAMPLE_CORE_RESPONSE = {
    "totalHits": 2,
    "results": [
        {
            "id": 12345,
            "title": "Indigenous Knowledge Systems in the Arctic",
            "abstract": "This paper examines indigenous knowledge systems...",
            "yearPublished": 2021,
            "authors": [{"name": "Kramvig, B."}, {"name": "Kristoffersen, B."}],
            "doi": "https://doi.org/10.1234/test.2021",
            "downloadUrl": "https://core.ac.uk/download/pdf/12345.pdf",
            "citationCount": 15,
            "publisher": "Arctic Research Journal",
            "sourceFulltextUrls": [],
        },
        {
            "id": 67890,
            "title": "Climate Adaptation in Northern Communities",
            "abstract": "Northern communities face unique challenges...",
            "yearPublished": 2020,
            "authors": [{"name": "Harvey, D."}],
            "doi": "10.5678/climate.2020",
            "downloadUrl": None,
            "citationCount": 8,
            "publisher": None,
            "sourceFulltextUrls": ["https://example.com/paper.pdf"],
        },
    ],
}


@pytest.fixture
def mock_academic_search():
    """Create AcademicSearchTools with mocked HTTP client."""
    from research_agent.tools.academic_search import AcademicSearchTools
    tools = AcademicSearchTools(config={"core": {"api_key": "test-key"}}, cache_enabled=False)
    return tools


class TestSearchCore:
    @pytest.mark.asyncio
    async def test_returns_papers(self, mock_academic_search):
        """CORE search parses response into Paper objects."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_CORE_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch.object(mock_academic_search, "_get_client") as mock_client:
            client = AsyncMock()
            client.get = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            papers = await mock_academic_search.search_core("arctic indigenous knowledge")

        assert len(papers) == 2
        assert papers[0].title == "Indigenous Knowledge Systems in the Arctic"
        assert papers[0].source == "core"
        assert papers[0].doi == "10.1234/test.2021"  # stripped https://doi.org/
        assert papers[0].year == 2021
        assert len(papers[0].authors) == 2

    @pytest.mark.asyncio
    async def test_doi_without_prefix(self, mock_academic_search):
        """DOIs without https://doi.org/ prefix are kept as-is."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_CORE_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch.object(mock_academic_search, "_get_client") as mock_client:
            client = AsyncMock()
            client.get = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            papers = await mock_academic_search.search_core("climate")

        # Second paper has doi without prefix
        assert papers[1].doi == "10.5678/climate.2020"

    @pytest.mark.asyncio
    async def test_fallback_download_url(self, mock_academic_search):
        """Falls back to sourceFulltextUrls when downloadUrl is None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_CORE_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch.object(mock_academic_search, "_get_client") as mock_client:
            client = AsyncMock()
            client.get = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            papers = await mock_academic_search.search_core("climate")

        assert papers[1].open_access_url == "https://example.com/paper.pdf"

    @pytest.mark.asyncio
    async def test_error_returns_empty(self, mock_academic_search):
        """HTTP errors return empty list."""
        with patch.object(mock_academic_search, "_get_client") as mock_client:
            client = AsyncMock()
            client.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client.return_value = client

            papers = await mock_academic_search.search_core("test query")

        assert papers == []

    @pytest.mark.asyncio
    async def test_year_filter_in_query(self, mock_academic_search):
        """Year filters are appended to query string."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(mock_academic_search, "_get_client") as mock_client:
            client = AsyncMock()
            client.get = AsyncMock(return_value=mock_resp)
            mock_client.return_value = client

            await mock_academic_search.search_core("test", from_year=2020, to_year=2023)

            # Check the query parameter includes year filters
            call_args = client.get.call_args
            params = call_args.kwargs.get("params") or call_args[1].get("params")
            assert "yearPublished>=2020" in params["q"]
            assert "yearPublished<=2023" in params["q"]

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        """Works without API key (no Authorization header)."""
        from research_agent.tools.academic_search import AcademicSearchTools
        tools = AcademicSearchTools(config={}, cache_enabled=False)
        assert tools._core_api_key is None

    @pytest.mark.asyncio
    async def test_caching(self):
        """Cached results are returned without API call."""
        from research_agent.tools.academic_search import AcademicSearchTools
        tools = AcademicSearchTools(config={}, cache_enabled=True)

        # Manually populate cache
        from research_agent.utils.cache import make_cache_key
        cache_key = make_cache_key("core_search", "test query", limit=20, from_year=None, to_year=None)
        cached_papers = [{"title": "Cached Paper", "source": "core"}]
        tools._cache.set(cache_key, cached_papers, ttl=3600)

        papers = await tools.search_core("test query")
        assert len(papers) == 1
        assert papers[0]["title"] == "Cached Paper"
