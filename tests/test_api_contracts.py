"""
Contract tests for external API adapters.

These tests make real HTTP requests to verify response schemas haven't changed.
Run only via: pytest -m integration
"""

import pytest
import httpx


pytestmark = [pytest.mark.integration, pytest.mark.network]

BERT_S2_ID = "649def34f8be52c8b66281af98ae884c09aef38b"
BERT_OA_ID = "W2741809807"
HARAWAY_OA_ID = "A5010652037"
TIMEOUT = 30.0


@pytest.fixture
def http_client():
    """Shared httpx client with timeout."""
    with httpx.Client(timeout=TIMEOUT) as client:
        yield client


class TestSemanticScholarContract:
    """Verify S2 API response shapes."""

    def test_paper_detail_shape(self, http_client):
        """S2 paper detail returns expected fields."""
        url = f"https://api.semanticscholar.org/graph/v1/paper/{BERT_S2_ID}"
        params = {"fields": "title,year,citationCount,authors,abstract,fieldsOfStudy,venue"}
        resp = http_client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        # Required fields
        assert "paperId" in data
        assert "title" in data
        assert isinstance(data["title"], str)
        assert "year" in data
        assert "citationCount" in data
        assert "authors" in data
        assert isinstance(data["authors"], list)
        if data["authors"]:
            assert "name" in data["authors"][0]

    def test_author_search_shape(self, http_client):
        """S2 author search returns expected structure."""
        url = "https://api.semanticscholar.org/graph/v1/author/search"
        params = {"query": "Donna Haraway", "limit": 1}
        resp = http_client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        assert "data" in data
        assert isinstance(data["data"], list)
        if data["data"]:
            author = data["data"][0]
            assert "authorId" in author
            assert "name" in author


class TestOpenAlexContract:
    """Verify OpenAlex API response shapes."""

    def test_work_detail_shape(self, http_client):
        """OpenAlex work detail returns expected fields."""
        url = f"https://api.openalex.org/works/{BERT_OA_ID}"
        resp = http_client.get(url)
        resp.raise_for_status()
        data = resp.json()

        assert "id" in data
        assert "title" in data
        assert isinstance(data["title"], str)
        assert "publication_year" in data
        assert "cited_by_count" in data
        assert "authorships" in data
        assert isinstance(data["authorships"], list)
        if data["authorships"]:
            assert "author" in data["authorships"][0]
            assert "display_name" in data["authorships"][0]["author"]

    def test_author_search_shape(self, http_client):
        """OpenAlex author search returns expected structure."""
        url = "https://api.openalex.org/authors"
        params = {"search": "Donna Haraway", "per_page": 1}
        resp = http_client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        assert "results" in data
        assert isinstance(data["results"], list)
        if data["results"]:
            author = data["results"][0]
            assert "id" in author
            assert "display_name" in author
