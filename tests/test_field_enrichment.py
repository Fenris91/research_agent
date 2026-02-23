"""Tests for field enrichment pipeline and shared OpenAlex extraction."""

import json
from unittest.mock import patch, MagicMock

import pytest

from research_agent.utils.openalex import extract_openalex_fields
from research_agent.utils.field_enrichment import (
    fields_need_enrichment,
    enrich_fields,
    _titles_match,
    _COARSE_S2_FIELDS,
)


# -----------------------------------------------------------------------
# extract_openalex_fields
# -----------------------------------------------------------------------


class TestExtractOpenalexFields:
    def test_basic_extraction(self):
        work = {
            "concepts": [
                {"display_name": "Indigenous Rights", "score": 0.9, "level": 1},
                {"display_name": "Political Science", "score": 0.8, "level": 0},
                {"display_name": "Arctic Studies", "score": 0.5, "level": 2},
            ]
        }
        fields = extract_openalex_fields(work)
        assert fields == ["Indigenous Rights", "Political Science", "Arctic Studies"]

    def test_filters_low_score(self):
        work = {
            "concepts": [
                {"display_name": "Good Field", "score": 0.5, "level": 1},
                {"display_name": "Low Score", "score": 0.1, "level": 1},
            ]
        }
        fields = extract_openalex_fields(work)
        assert "Good Field" in fields
        assert "Low Score" not in fields

    def test_filters_deep_level(self):
        work = {
            "concepts": [
                {"display_name": "Top Level", "score": 0.5, "level": 0},
                {"display_name": "Too Deep", "score": 0.5, "level": 4},
            ]
        }
        fields = extract_openalex_fields(work)
        assert "Top Level" in fields
        assert "Too Deep" not in fields

    def test_fallback_to_unfiltered(self):
        """If no concepts pass the filter, fall back to top-N unfiltered."""
        work = {
            "concepts": [
                {"display_name": "Deep Only", "score": 0.1, "level": 5},
            ]
        }
        fields = extract_openalex_fields(work)
        assert fields == ["Deep Only"]

    def test_empty_concepts(self):
        assert extract_openalex_fields({}) == []
        assert extract_openalex_fields({"concepts": []}) == []
        assert extract_openalex_fields({"concepts": None}) == []

    def test_limit(self):
        work = {
            "concepts": [
                {"display_name": f"Field{i}", "score": 0.9 - i * 0.1, "level": 1}
                for i in range(10)
            ]
        }
        assert len(extract_openalex_fields(work, limit=3)) == 3

    def test_deduplicates(self):
        work = {
            "concepts": [
                {"display_name": "Same", "score": 0.9, "level": 1},
                {"display_name": "Same", "score": 0.8, "level": 1},
            ]
        }
        assert extract_openalex_fields(work) == ["Same"]

    def test_relevance_score_fallback(self):
        """Uses relevance_score when score is missing."""
        work = {
            "concepts": [
                {"display_name": "Test", "relevance_score": 0.7, "level": 1},
            ]
        }
        fields = extract_openalex_fields(work)
        assert fields == ["Test"]

    def test_none_level_passes(self):
        work = {
            "concepts": [
                {"display_name": "No Level", "score": 0.5, "level": None},
            ]
        }
        assert extract_openalex_fields(work) == ["No Level"]


# -----------------------------------------------------------------------
# fields_need_enrichment
# -----------------------------------------------------------------------


class TestFieldsNeedEnrichment:
    def test_none(self):
        assert fields_need_enrichment(None) is True

    def test_empty(self):
        assert fields_need_enrichment([]) is True

    def test_all_coarse(self):
        assert fields_need_enrichment(["Physics", "Computer Science"]) is True

    def test_single_coarse(self):
        assert fields_need_enrichment(["Biology"]) is True

    def test_mixed_good(self):
        assert fields_need_enrichment(["Physics", "Quantum Computing"]) is False

    def test_all_specific(self):
        assert fields_need_enrichment(["Indigenous Rights", "Arctic Governance"]) is False

    def test_coarse_set_coverage(self):
        """Verify all expected coarse fields are in the set."""
        for f in ["Physics", "Computer Science", "Sociology", "Psychology", "Mathematics"]:
            assert f in _COARSE_S2_FIELDS


# -----------------------------------------------------------------------
# _titles_match
# -----------------------------------------------------------------------


class TestTitlesMatch:
    def test_exact_match(self):
        assert _titles_match("Foo Bar", "Foo Bar") is True

    def test_case_insensitive(self):
        assert _titles_match("Foo Bar", "foo bar") is True

    def test_partial_overlap(self):
        assert _titles_match("The Big Test Paper", "Big Test Paper Extended") is True

    def test_no_match(self):
        assert _titles_match("Foo Bar", "Baz Qux") is False

    def test_empty(self):
        assert _titles_match("", "Foo") is False
        assert _titles_match("Foo", "") is False


# -----------------------------------------------------------------------
# enrich_fields (integration-style with mocks)
# -----------------------------------------------------------------------


class TestEnrichFields:
    def test_good_fields_skip(self):
        """Already-specific fields should be returned unchanged with no API calls."""
        result = enrich_fields(
            fields=["Indigenous Rights", "Arctic Studies"],
            doi="10.1234/test",
            title="Test Paper",
        )
        assert result == ["Indigenous Rights", "Arctic Studies"]

    @patch("research_agent.utils.field_enrichment._lookup_openalex_by_doi")
    def test_doi_lookup(self, mock_doi):
        mock_doi.return_value = ["Urban Geography", "Marxist Theory"]
        result = enrich_fields(
            fields=["Geography"],  # coarse
            doi="10.1234/test",
            title="Test Paper",
        )
        assert result == ["Urban Geography", "Marxist Theory"]
        mock_doi.assert_called_once_with("10.1234/test")

    @patch("research_agent.utils.field_enrichment._lookup_openalex_by_doi")
    @patch("research_agent.utils.field_enrichment._lookup_openalex_by_title")
    def test_title_fallback(self, mock_title, mock_doi):
        mock_doi.return_value = None
        mock_title.return_value = ["Postcolonial Theory"]
        result = enrich_fields(
            fields=["Sociology"],
            doi="10.1234/test",
            title="Decolonizing Knowledge Systems",
        )
        assert result == ["Postcolonial Theory"]

    @patch("research_agent.utils.field_enrichment._lookup_openalex_by_doi")
    @patch("research_agent.utils.field_enrichment._lookup_openalex_by_title")
    @patch("research_agent.utils.field_enrichment._extract_fields_via_llm")
    def test_llm_fallback(self, mock_llm, mock_title, mock_doi):
        mock_doi.return_value = None
        mock_title.return_value = None
        mock_llm.return_value = ["LLM Field 1", "LLM Field 2"]
        result = enrich_fields(
            fields=[],
            doi=None,
            title="Some Obscure Paper",
            abstract="An abstract about things.",
        )
        assert result == ["LLM Field 1", "LLM Field 2"]

    @patch("research_agent.utils.field_enrichment._lookup_openalex_by_doi")
    @patch("research_agent.utils.field_enrichment._lookup_openalex_by_title")
    @patch("research_agent.utils.field_enrichment._extract_fields_via_llm")
    def test_all_fail_returns_original(self, mock_llm, mock_title, mock_doi):
        mock_doi.return_value = None
        mock_title.return_value = None
        mock_llm.return_value = None
        result = enrich_fields(fields=["Physics"], doi=None, title="Quantum Things")
        assert result == ["Physics"]

    def test_none_fields_no_doi_no_title(self):
        result = enrich_fields(fields=None, doi=None, title=None)
        assert result == []
