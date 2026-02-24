"""Tests for researcher profile enrichment in GraphBuilder.build_from_kb_papers()."""

import pytest

from research_agent.explorer.graph_builder import GraphBuilder


def _make_papers(researcher="Alice Smith", count=2):
    """Create minimal KB paper dicts for testing."""
    return [
        {
            "paper_id": f"paper_{i}",
            "title": f"Paper {i} by {researcher}",
            "researcher": researcher,
            "fields": ["Sociology", "Urban Studies"],
            "citation_count": 10 * (i + 1),
        }
        for i in range(count)
    ]


class _MockProfile:
    """Mimics ResearcherProfile.to_dict() without importing the real class."""

    def __init__(self, **kwargs):
        self._data = kwargs

    def to_dict(self):
        return dict(self._data)


class _MockRegistry:
    """Mimics ResearcherRegistry.get(name) with a dict of profiles."""

    def __init__(self, profiles: dict[str, _MockProfile]):
        self._profiles = {k.lower().strip(): v for k, v in profiles.items()}

    def get(self, name):
        return self._profiles.get(name.lower().strip())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildFromKbPapersWithoutRegistry:
    def test_researcher_node_has_minimal_data(self):
        gb = GraphBuilder()
        papers = _make_papers("Alice Smith", count=2)
        gb.build_from_kb_papers(papers, researcher_registry=None)
        data = gb.to_dict()

        researcher_nodes = [n for n in data["nodes"] if n["type"] == "researcher"]
        assert len(researcher_nodes) == 1
        node = researcher_nodes[0]
        assert node["label"] == "Alice Smith"
        assert node["metadata"]["h_index"] is None
        assert node["metadata"]["affiliations"] == []
        assert node["id"] == "researcher:Alice Smith"


class TestBuildFromKbPapersWithRegistryEnriches:
    def test_enriched_node_has_full_profile(self):
        profile = _MockProfile(
            name="Alice Smith",
            openalex_id="A111",
            semantic_scholar_id="S222",
            h_index=25,
            affiliations=["MIT"],
            works_count=100,
            citations_count=5000,
            fields=["Sociology", "Geography"],
        )
        registry = _MockRegistry({"Alice Smith": profile})

        gb = GraphBuilder()
        papers = _make_papers("Alice Smith", count=2)
        gb.build_from_kb_papers(papers, researcher_registry=registry)
        data = gb.to_dict()

        researcher_nodes = [n for n in data["nodes"] if n["type"] == "researcher"]
        assert len(researcher_nodes) == 1
        node = researcher_nodes[0]
        assert node["metadata"]["h_index"] == 25
        assert node["metadata"]["affiliations"] == ["MIT"]
        assert node["id"] == "researcher:A111"
        assert node["metadata"]["works_count"] == 100
        assert node["metadata"]["citations"] == 5000


class TestBuildFromKbPapersGracefulDegradation:
    def test_unknown_researcher_gets_minimal_node(self):
        registry = _MockRegistry({})  # empty — no profiles

        gb = GraphBuilder()
        papers = _make_papers("Bob Unknown", count=1)
        gb.build_from_kb_papers(papers, researcher_registry=registry)
        data = gb.to_dict()

        researcher_nodes = [n for n in data["nodes"] if n["type"] == "researcher"]
        assert len(researcher_nodes) == 1
        node = researcher_nodes[0]
        assert node["label"] == "Bob Unknown"
        assert node["metadata"]["h_index"] is None
        assert node["id"] == "researcher:Bob Unknown"


class TestBuildFromKbPapersMixedEnrichment:
    def test_one_enriched_one_minimal(self):
        profile = _MockProfile(
            name="Alice Smith",
            openalex_id="A111",
            semantic_scholar_id=None,
            h_index=25,
            affiliations=["MIT"],
            works_count=50,
            citations_count=2000,
            fields=["Sociology"],
        )
        registry = _MockRegistry({"Alice Smith": profile})

        papers = _make_papers("Alice Smith", count=1) + _make_papers("Bob Unknown", count=1)
        gb = GraphBuilder()
        gb.build_from_kb_papers(papers, researcher_registry=registry)
        data = gb.to_dict()

        nodes_by_label = {n["label"]: n for n in data["nodes"] if n["type"] == "researcher"}
        assert nodes_by_label["Alice Smith"]["metadata"]["h_index"] == 25
        assert nodes_by_label["Bob Unknown"]["metadata"]["h_index"] is None


class TestWorksCountUsesMax:
    def test_registry_count_wins_when_larger(self):
        profile = _MockProfile(
            name="Alice Smith",
            openalex_id="A111",
            semantic_scholar_id=None,
            h_index=10,
            affiliations=[],
            works_count=100,
            citations_count=5000,
            fields=[],
        )
        registry = _MockRegistry({"Alice Smith": profile})

        # KB has 2 papers with 10+20=30 citations
        papers = _make_papers("Alice Smith", count=2)
        gb = GraphBuilder()
        gb.build_from_kb_papers(papers, researcher_registry=registry)
        data = gb.to_dict()

        node = [n for n in data["nodes"] if n["type"] == "researcher"][0]
        assert node["metadata"]["works_count"] == 100  # registry wins
        assert node["metadata"]["citations"] == 5000   # registry wins

    def test_kb_count_wins_when_larger(self):
        profile = _MockProfile(
            name="Alice Smith",
            openalex_id="A111",
            semantic_scholar_id=None,
            h_index=10,
            affiliations=[],
            works_count=1,       # registry has less
            citations_count=5,   # registry has less
            fields=[],
        )
        registry = _MockRegistry({"Alice Smith": profile})

        papers = _make_papers("Alice Smith", count=2)  # 2 papers, 30 total cites
        gb = GraphBuilder()
        gb.build_from_kb_papers(papers, researcher_registry=registry)
        data = gb.to_dict()

        node = [n for n in data["nodes"] if n["type"] == "researcher"][0]
        assert node["metadata"]["works_count"] == 2   # KB wins
        assert node["metadata"]["citations"] == 30     # KB wins


class TestFieldsMergedAndDeduplicated:
    def test_fields_merged_without_dupes(self):
        profile = _MockProfile(
            name="Alice Smith",
            openalex_id="A111",
            semantic_scholar_id=None,
            h_index=10,
            affiliations=[],
            works_count=0,
            citations_count=0,
            fields=["Sociology", "Tourism", "Geography"],
        )
        registry = _MockRegistry({"Alice Smith": profile})

        # KB papers have ["Sociology", "Urban Studies"]
        papers = _make_papers("Alice Smith", count=1)
        gb = GraphBuilder()
        gb.build_from_kb_papers(papers, researcher_registry=registry)
        data = gb.to_dict()

        node = [n for n in data["nodes"] if n["type"] == "researcher"][0]
        fields = node["metadata"]["fields"]
        # KB fields first, then registry additions (no dupes)
        assert fields == ["Sociology", "Urban Studies", "Tourism", "Geography"]
