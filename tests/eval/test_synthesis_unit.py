"""CPU-only tests for query classification heuristics and prompt construction.

No LLM calls — agent.model is always None.
"""

from unittest.mock import MagicMock, patch

import pytest

from research_agent.agents.research_agent import AgentConfig


# ── Agent factory (no model, no vector store) ─────────────────────────

def _make_agent(**overrides):
    """Build a minimal ResearchAgent with no model loaded."""
    from research_agent.agents.research_agent import ResearchAgent

    agent = ResearchAgent.__new__(ResearchAgent)
    agent.model = None
    agent._pipeline = {}
    agent.provider = "none"
    agent._load_model_on_demand = False
    agent.use_ollama = False
    agent.vector_store = None
    agent.embedder = None
    agent.config = AgentConfig()
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


def _make_state(query, **overrides):
    state = {
        "messages": [],
        "current_query": query,
        "search_query": "",
        "query_type": "",
        "local_results": [],
        "external_results": [],
        "should_search_external": False,
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


# ── Query classification heuristics ───────────────────────────────────


@pytest.mark.eval
@pytest.mark.parametrize(
    "query,expected_type",
    [
        ("review of literature on neoliberalism", "literature_review"),
        ("key papers on accumulation by dispossession", "literature_review"),
        ("what is the state of research on weak ties", "literature_review"),
        ("overview of urban geography", "literature_review"),
        ("what is accumulation by dispossession", "factual"),
        ("tell me about Harvey's concept of space", "factual"),
        ("what do you know about Kramvig's work", "factual"),
        ("compare Harvey and Bourdieu on capital", "analysis"),
        ("relationship between neoliberalism and inequality", "analysis"),
        ("hi there", "general"),
        ("hello", "general"),
    ],
)
async def test_query_classification_heuristics(query, expected_type):
    agent = _make_agent()
    state = _make_state(query)
    result = await agent._understand_query(state)
    assert result["query_type"] == expected_type, (
        f"Query {query!r} -> got {result['query_type']!r}, expected {expected_type!r}"
    )


# ── Search keywords not empty for research queries ────────────────────


@pytest.mark.eval
@pytest.mark.parametrize(
    "query",
    [
        "What is accumulation by dispossession?",
        "David Harvey's work on neoliberalism",
        "Compare structuralism and post-structuralism",
        "overview of indigenous research methodology",
    ],
)
async def test_search_query_not_empty_for_research(query):
    agent = _make_agent()
    state = _make_state(query)
    result = await agent._understand_query(state)
    # Without LLM, _extract_search_keywords falls back to the query itself
    assert result["search_query"] != "", (
        f"search_query should not be empty for: {query!r}"
    )


# ── Synthesis prompt structure ────────────────────────────────────────


@pytest.mark.eval
@pytest.mark.parametrize("query_type", ["literature_review", "factual", "analysis"])
def test_synthesis_prompt_contains_citations(query_type):
    agent = _make_agent()
    results = [
        {
            "title": "A Brief History of Neoliberalism",
            "authors": "David Harvey",
            "year": 2005,
            "content": "Neoliberalism is a theory of political economic practices...",
            "source": "local_kb",
        },
    ]
    prompt = agent._build_synthesis_prompt("neoliberalism", query_type, results)
    assert "[1]" in prompt
    assert "A Brief History of Neoliberalism" in prompt
    assert "neoliberalism" in prompt.lower()


@pytest.mark.eval
def test_synthesis_prompt_sequential_numbering():
    agent = _make_agent()
    results = [
        {"title": f"Paper {i}", "content": f"Content about topic {i}", "source": "local_kb"}
        for i in range(1, 6)
    ]
    prompt = agent._build_synthesis_prompt("test query", "literature_review", results)
    for i in range(1, 6):
        assert f"[{i}]" in prompt, f"Missing citation [{i}] in prompt"


# ── No-source handling ────────────────────────────────────────────────


@pytest.mark.eval
async def test_no_sources_general_returns_helpful():
    """General query + no results -> helpful non-error response."""
    agent = _make_agent()
    state = _make_state("hello", query_type="general")
    result = await agent._synthesize(state)
    answer = result["final_answer"]
    assert len(answer) > 0
    assert "error" not in answer.lower()
    assert any(
        word in answer.lower()
        for word in ["research", "help", "knowledge", "explore", "assistant"]
    )


@pytest.mark.eval
async def test_no_sources_research_uses_fallback():
    """Research query + no LLM + no results -> fallback message."""
    agent = _make_agent()
    state = _make_state(
        "review of neoliberalism literature",
        query_type="literature_review",
    )
    result = await agent._synthesize(state)
    assert "No retrieved sources" in result["final_answer"]
