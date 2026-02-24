"""CPU-only tests for context-based relevance boosting in _search_local.

Tests the boost logic (research_agent.py:860-898) using mocked vector
store and embedder. No GPU or real models needed.
"""

from unittest.mock import MagicMock

import pytest

from research_agent.agents.research_agent import AgentConfig


# ── Helpers ────────────────────────────────────────────────────────────

def _make_agent(config=None):
    """Build a minimal ResearchAgent with mocked store + embedder."""
    from research_agent.agents.research_agent import ResearchAgent

    agent = ResearchAgent.__new__(ResearchAgent)
    agent.model = None
    agent._pipeline = {}
    agent.provider = "none"
    agent._load_model_on_demand = False
    agent.use_ollama = False
    agent.config = config or AgentConfig(max_local_results=10)

    # Mock embedder — returns a zero vector
    agent.embedder = MagicMock()
    agent.embedder.embed_query.return_value = [0.0] * 768

    return agent


def _make_search_results(papers):
    """Build a mock search() return value from paper dicts.

    Each paper dict must have: paper_id, title, authors, distance.
    """
    return {
        "documents": [p.get("content", f"Content of {p['title']}") for p in papers],
        "metadatas": [
            {
                "paper_id": p["paper_id"],
                "title": p["title"],
                "authors": p.get("authors", ""),
            }
            for p in papers
        ],
        "distances": [p["distance"] for p in papers],
    }


_EMPTY = {"documents": [], "metadatas": [], "distances": []}


def _wire_store(agent, paper_results, context_paper=None):
    """Wire up mock vector store with canned search results."""
    store = MagicMock()
    store.search.side_effect = lambda **kwargs: (
        paper_results if kwargs.get("collection") == "papers" else _EMPTY
    )
    store.get_paper.return_value = context_paper
    agent.vector_store = store


def _make_state(query, **overrides):
    state = {
        "messages": [],
        "current_query": query,
        "search_query": query,
        "query_type": "literature_review",
        "local_results": [],
        "external_results": [],
        "should_search_external": True,
        "candidates_for_ingestion": [],
        "final_answer": "",
        "error": None,
        "current_researcher": None,
        "current_paper_id": None,
        "auth_context_items": None,
        "chat_context_items": None,
    }
    state.update(overrides)
    return state


def _find_result(results, paper_id):
    for r in results:
        if r["paper_id"] == paper_id:
            return r
    return None


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.eval
async def test_researcher_boost_reorders():
    """Researcher context (+0.15) should reorder results when appropriate."""
    agent = _make_agent()
    papers = [
        {"paper_id": "other", "title": "Other Paper", "authors": "Jane Doe",
         "distance": 0.20},  # raw relevance = 0.80
        {"paper_id": "harvey", "title": "Harvey Paper", "authors": "David Harvey",
         "distance": 0.30},  # raw relevance = 0.70
    ]
    _wire_store(agent, _make_search_results(papers))

    state = _make_state("neoliberalism", current_researcher="David Harvey")
    result = await agent._search_local(state)
    results = result["local_results"]

    harvey = _find_result(results, "harvey")
    other = _find_result(results, "other")
    assert harvey is not None
    assert other is not None

    # After boost: Harvey = 0.70 + 0.15 = 0.85 > Other = 0.80
    assert harvey["relevance_score"] == pytest.approx(0.85, abs=1e-6)
    assert harvey["context_match"] == "researcher"
    assert results[0]["paper_id"] == "harvey"


@pytest.mark.eval
async def test_selected_paper_boost():
    """Selected paper gets highest boost (+0.30 = boost_amount * 2)."""
    agent = _make_agent()
    papers = [
        {"paper_id": "other", "title": "Other", "authors": "X",
         "distance": 0.15},  # raw = 0.85
        {"paper_id": "selected", "title": "Selected", "authors": "Y",
         "distance": 0.40},  # raw = 0.60
    ]
    context_paper = {
        "paper_id": "selected",
        "metadata": {"authors": "Y", "title": "Selected"},
        "chunks": ["content"],
    }
    _wire_store(agent, _make_search_results(papers), context_paper=context_paper)

    state = _make_state("test", current_paper_id="selected")
    result = await agent._search_local(state)
    results = result["local_results"]

    selected = _find_result(results, "selected")
    assert selected is not None
    # 0.60 + 0.30 = 0.90
    assert selected["relevance_score"] == pytest.approx(0.90, abs=1e-6)
    assert selected["context_match"] == "selected_paper"


@pytest.mark.eval
async def test_coauthor_boost_smaller_than_researcher():
    """Co-author boost (+0.075) is half of researcher boost."""
    agent = _make_agent()
    papers = [
        {"paper_id": "coauthor_paper", "title": "Co-Author Work",
         "authors": "Neil Smith", "distance": 0.40},  # raw = 0.60
    ]
    # The selected paper has authors "David Harvey, Neil Smith"
    context_paper = {
        "paper_id": "selected",
        "metadata": {"authors": "David Harvey, Neil Smith", "title": "Selected"},
        "chunks": ["content"],
    }
    _wire_store(agent, _make_search_results(papers), context_paper=context_paper)

    state = _make_state("test", current_paper_id="selected")
    result = await agent._search_local(state)
    results = result["local_results"]

    coauthor = _find_result(results, "coauthor_paper")
    assert coauthor is not None
    # 0.60 + 0.075 = 0.675
    assert coauthor["relevance_score"] == pytest.approx(0.675, abs=1e-6)
    assert coauthor["context_match"] == "related_author"


@pytest.mark.eval
async def test_no_boost_without_context():
    """Without researcher/paper context, scores stay at raw values."""
    agent = _make_agent()
    papers = [
        {"paper_id": "p1", "title": "Paper 1", "authors": "A", "distance": 0.20},
        {"paper_id": "p2", "title": "Paper 2", "authors": "B", "distance": 0.30},
    ]
    _wire_store(agent, _make_search_results(papers))

    state = _make_state("test")  # no current_researcher, no current_paper_id
    result = await agent._search_local(state)
    results = result["local_results"]

    for r in results:
        assert "context_match" not in r
    assert _find_result(results, "p1")["relevance_score"] == pytest.approx(0.80, abs=1e-6)
    assert _find_result(results, "p2")["relevance_score"] == pytest.approx(0.70, abs=1e-6)


@pytest.mark.eval
async def test_score_capped_at_1():
    """Boosted relevance score never exceeds 1.0."""
    agent = _make_agent()
    papers = [
        {"paper_id": "high", "title": "High Score", "authors": "David Harvey",
         "distance": 0.02},  # raw = 0.98
    ]
    _wire_store(agent, _make_search_results(papers))

    state = _make_state("test", current_researcher="David Harvey")
    result = await agent._search_local(state)
    results = result["local_results"]

    high = _find_result(results, "high")
    assert high is not None
    # 0.98 + 0.15 = 1.13, but capped to 1.0
    assert high["relevance_score"] <= 1.0
    assert high["relevance_score"] == pytest.approx(1.0, abs=1e-6)
