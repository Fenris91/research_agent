"""Tests for Claude native tool-use bypass of LangGraph pipeline."""

import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from research_agent.agents.research_agent import (
    ResearchAgent,
    AgentConfig,
    CLAUDE_TOOLS,
)
from research_agent.models.llm_utils import AnthropicNativeModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.search.return_value = {
        "ids": [["doc1", "doc2"]],
        "documents": [["Content of doc 1", "Content of doc 2"]],
        "metadatas": [[
            {"title": "Paper A", "authors": "Alice", "year": 2023, "url": ""},
            {"title": "Paper B", "authors": "Bob", "year": 2024, "url": ""},
        ]],
        "distances": [[0.2, 0.4]],
    }
    return store


@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.embed_query.return_value = [0.1] * 768
    return emb


@pytest.fixture
def agent_no_llm(mock_vector_store, mock_embedder):
    """Agent with no LLM — provider 'none'."""
    return ResearchAgent(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        provider="none",
        config=AgentConfig(),
    )


# ---------------------------------------------------------------------------
# CLAUDE_TOOLS schema
# ---------------------------------------------------------------------------


class TestClaudeToolsSchema:
    def test_four_tools_defined(self):
        assert len(CLAUDE_TOOLS) == 4

    def test_tool_names(self):
        names = {t["name"] for t in CLAUDE_TOOLS}
        assert names == {
            "search_knowledge_base",
            "search_academic_papers",
            "search_web",
            "get_paper_details",
        }

    def test_all_tools_have_input_schema(self):
        for tool in CLAUDE_TOOLS:
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"
            assert "properties" in tool["input_schema"]

    def test_all_tools_have_required_query_or_id(self):
        for tool in CLAUDE_TOOLS:
            required = tool["input_schema"].get("required", [])
            assert len(required) >= 1


# ---------------------------------------------------------------------------
# _actual_provider tracking
# ---------------------------------------------------------------------------


class TestActualProviderTracking:
    def test_default_provider_none(self):
        agent = ResearchAgent(provider="none")
        assert agent._actual_provider == "none"
        assert agent._claude_model is None

    def test_connect_anthropic_sets_actual_provider(self):
        agent = ResearchAgent(provider="none")

        with patch("research_agent.agents.research_agent.AnthropicNativeModel") as MockClaude:
            mock_instance = MagicMock()
            MockClaude.return_value = mock_instance

            success = agent.connect_provider("anthropic", "sk-ant-test123")
            assert success is True
            assert agent._actual_provider == "anthropic"
            assert agent._claude_model is mock_instance

    def test_connect_groq_clears_claude_model(self):
        agent = ResearchAgent(provider="none")
        success = agent.connect_provider("groq", "gsk_test123")
        assert success is True
        assert agent._actual_provider == "groq"
        assert agent._claude_model is None

    def test_connect_openai_clears_claude_model(self):
        agent = ResearchAgent(provider="none")
        success = agent.connect_provider("openai", "sk-test123")
        assert success is True
        assert agent._actual_provider == "openai"
        assert agent._claude_model is None


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


class TestToolExecution:
    def test_search_kb(self, agent_no_llm, mock_vector_store):
        result = agent_no_llm._tool_search_kb("test query", 5)
        data = json.loads(result)
        assert "results" in data
        # Called search for each collection
        assert mock_vector_store.search.call_count >= 1

    def test_search_kb_no_store(self):
        agent = ResearchAgent(provider="none")
        result = agent._tool_search_kb("test")
        data = json.loads(result)
        assert data["results"] == []

    def test_search_academic_no_search_service(self):
        agent = ResearchAgent(provider="none")
        result = agent._tool_search_academic("test")
        data = json.loads(result)
        assert data["results"] == []

    def test_search_web_no_search_service(self):
        agent = ResearchAgent(provider="none")
        result = agent._tool_search_web("test")
        data = json.loads(result)
        assert data["results"] == []

    def test_get_paper_details_no_search_service(self):
        agent = ResearchAgent(provider="none")
        result = agent._tool_get_paper_details("10.1000/test")
        data = json.loads(result)
        assert "error" in data

    def test_unknown_tool(self, agent_no_llm):
        result = agent_no_llm._execute_tool("nonexistent_tool", {})
        data = json.loads(result)
        assert "error" in data
        assert "Unknown tool" in data["error"]

    def test_execute_tool_routes_to_kb(self, agent_no_llm):
        result = agent_no_llm._execute_tool(
            "search_knowledge_base", {"query": "test", "max_results": 3}
        )
        data = json.loads(result)
        assert "results" in data


# ---------------------------------------------------------------------------
# AnthropicNativeModel
# ---------------------------------------------------------------------------


class TestAnthropicNativeModel:
    def test_text_only_response(self):
        """When Claude returns just text, no tool loop."""
        mock_client = MagicMock()
        model = AnthropicNativeModel.__new__(AnthropicNativeModel)
        model.client = mock_client
        model.model_name = "claude-sonnet-4-6"
        model.api_key = "sk-ant-test"

        # Mock response with just text
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello! How can I help?"

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response

        result = model.run_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=CLAUDE_TOOLS,
            system="You are helpful.",
        )
        assert result["answer"] == "Hello! How can I help?"
        assert result["tool_calls"] == []

    def test_tool_use_then_text(self):
        """Claude calls a tool, gets result, then responds with text."""
        mock_client = MagicMock()
        model = AnthropicNativeModel.__new__(AnthropicNativeModel)
        model.client = mock_client
        model.model_name = "claude-sonnet-4-6"
        model.api_key = "sk-ant-test"

        # First response: tool_use
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search_knowledge_base"
        tool_block.input = {"query": "arctic governance"}
        tool_block.id = "tool_123"

        first_response = MagicMock()
        first_response.content = [tool_block]
        first_response.stop_reason = "tool_use"

        # Second response: text
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Based on your KB, here are findings..."

        second_response = MagicMock()
        second_response.content = [text_block]
        second_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [first_response, second_response]

        def executor(name, inp):
            return json.dumps({"results": [{"title": "Arctic Paper", "authors": "Test"}]})

        result = model.run_with_tools(
            messages=[{"role": "user", "content": "arctic governance"}],
            tools=CLAUDE_TOOLS,
            system="You are helpful.",
            tool_executor=executor,
        )
        assert result["answer"] == "Based on your KB, here are findings..."
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search_knowledge_base"

    def test_max_rounds_safety(self):
        """Verify the loop stops after MAX_TOOL_ROUNDS."""
        mock_client = MagicMock()
        model = AnthropicNativeModel.__new__(AnthropicNativeModel)
        model.client = mock_client
        model.model_name = "claude-sonnet-4-6"
        model.api_key = "sk-ant-test"

        # Always return tool_use
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search_web"
        tool_block.input = {"query": "loop"}
        tool_block.id = "tool_loop"

        response = MagicMock()
        response.content = [tool_block]
        response.stop_reason = "tool_use"

        mock_client.messages.create.return_value = response

        def executor(name, inp):
            return json.dumps({"results": []})

        result = model.run_with_tools(
            messages=[{"role": "user", "content": "loop"}],
            tools=CLAUDE_TOOLS,
            system="",
            tool_executor=executor,
        )
        # Should have exactly MAX_TOOL_ROUNDS tool calls
        assert len(result["tool_calls"]) == AnthropicNativeModel.MAX_TOOL_ROUNDS
        assert mock_client.messages.create.call_count == AnthropicNativeModel.MAX_TOOL_ROUNDS

    def test_switch_model(self):
        model = AnthropicNativeModel.__new__(AnthropicNativeModel)
        model.model_name = "claude-sonnet-4-6"
        model.switch_model("claude-haiku-4-5")
        assert model.model_name == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Routing: _run_async routes to Claude native when appropriate
# ---------------------------------------------------------------------------


class TestRouting:
    @pytest.mark.asyncio
    async def test_routes_to_claude_native_when_anthropic(self):
        agent = ResearchAgent(provider="none")
        agent._actual_provider = "anthropic"
        agent._claude_model = MagicMock()

        mock_result = {
            "query": "test",
            "answer": "Claude native answer",
            "query_type": "claude_native",
            "sources": [],
            "local_sources": 0,
            "external_sources": 0,
        }

        with patch.object(agent, "_run_claude_native", new_callable=AsyncMock) as mock_native:
            mock_native.return_value = mock_result
            result = await agent._run_async("test query")
            mock_native.assert_called_once_with("test query", None, None)
            assert result["query_type"] == "claude_native"

    @pytest.mark.asyncio
    async def test_routes_to_langgraph_when_not_anthropic(self):
        agent = ResearchAgent(provider="none")
        agent._actual_provider = "groq"
        agent._claude_model = None

        # Should NOT call _run_claude_native
        with patch.object(agent, "_run_claude_native", new_callable=AsyncMock) as mock_native:
            # The graph will fail (no model), but that's fine — we just check routing
            result = await agent._run_async("test query")
            mock_native.assert_not_called()

    @pytest.mark.asyncio
    async def test_routes_to_langgraph_when_claude_model_is_none(self):
        """Even if _actual_provider is anthropic, if _claude_model is None, use LangGraph."""
        agent = ResearchAgent(provider="none")
        agent._actual_provider = "anthropic"
        agent._claude_model = None

        with patch.object(agent, "_run_claude_native", new_callable=AsyncMock) as mock_native:
            result = await agent._run_async("test query")
            mock_native.assert_not_called()


# ---------------------------------------------------------------------------
# Return format
# ---------------------------------------------------------------------------


class TestReturnFormat:
    @pytest.mark.asyncio
    async def test_return_format_has_required_keys(self):
        agent = ResearchAgent(provider="none")
        agent._actual_provider = "anthropic"

        # Mock claude model
        mock_claude = MagicMock()
        mock_claude.run_with_tools.return_value = {
            "answer": "Research findings here.",
            "tool_calls": [
                {
                    "name": "search_knowledge_base",
                    "input": {"query": "test"},
                    "result": json.dumps({
                        "results": [
                            {"title": "P1", "authors": "A1", "year": 2023, "source": "local_academic_papers", "url": ""},
                        ]
                    }),
                }
            ],
        }
        agent._claude_model = mock_claude

        result = await agent._run_claude_native("test query")

        # Must have the same keys as LangGraph path
        assert "query" in result
        assert "answer" in result
        assert "query_type" in result
        assert "sources" in result
        assert "local_sources" in result
        assert "external_sources" in result
        assert result["query_type"] == "claude_native"
        assert result["local_sources"] == 1
        assert result["external_sources"] == 0
        assert len(result["sources"]) == 1

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        """If Claude native fails, falls back to direct inference."""
        agent = ResearchAgent(provider="none")
        agent._actual_provider = "anthropic"
        agent._claude_model = MagicMock()
        agent._claude_model.run_with_tools.side_effect = Exception("API down")

        # No model, so fallback also fails gracefully
        result = await agent._run_claude_native("test query")
        assert "Error" in result["answer"] or "error" in result.get("status", "")
