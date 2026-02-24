"""Tests for Claude native tool-use integration.

All tests mock the Anthropic SDK — zero API credits burned.

Covers:
- is_claude property detection
- Tool schema definitions
- Tool execution dispatch
- Full tool-use loop with mocked client
- Result format parity with LangGraph path
- Fallback on import error or API error
- connect_provider preserves canonical_provider
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_model_loading():
    """Prevent actual model loading in tests."""
    from research_agent.models.llm_utils import VRAMConstraintError

    with patch("research_agent.agents.research_agent.get_qlora_pipeline") as mock_qlora, \
         patch("research_agent.agents.research_agent.get_ollama_pipeline") as mock_ollama:
        mock_qlora.side_effect = VRAMConstraintError("Model loading disabled in tests")
        mock_ollama.side_effect = Exception("Ollama disabled in tests")
        yield


def _make_agent(**kwargs):
    """Create a minimal ResearchAgent for testing."""
    from research_agent.agents.research_agent import ResearchAgent
    return ResearchAgent(use_ollama=False, **kwargs)


def _make_claude_agent():
    """Create a ResearchAgent that looks like an Anthropic provider."""
    agent = _make_agent(
        canonical_provider="anthropic",
        provider="openai",
        openai_api_key="sk-ant-test-key",
        openai_base_url="https://api.anthropic.com/v1/",
        openai_model="claude-sonnet-4-6",
        openai_models=["claude-sonnet-4-6", "claude-haiku-4-5"],
    )
    # Inject a mock model
    mock_model = MagicMock()
    mock_model.model_name = "claude-sonnet-4-6"
    mock_model.generate.return_value = "mock response"
    agent.model = mock_model
    return agent


def _make_text_block(text):
    """Create a mock Anthropic TextBlock."""
    block = SimpleNamespace(type="text", text=text)
    return block


def _make_tool_use_block(tool_id, name, input_data):
    """Create a mock Anthropic ToolUseBlock."""
    return SimpleNamespace(
        type="tool_use",
        id=tool_id,
        name=name,
        input=input_data,
    )


def _make_response(content, stop_reason="end_turn"):
    """Create a mock Anthropic Message response."""
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
    )


# ---------------------------------------------------------------------------
# is_claude detection
# ---------------------------------------------------------------------------

class TestIsClaudeDetection:
    def test_is_claude_true_for_anthropic(self):
        agent = _make_claude_agent()
        assert agent.is_claude is True

    def test_is_claude_false_for_groq(self):
        agent = _make_agent(
            canonical_provider="groq",
            provider="openai",
            openai_api_key="gsk-test",
        )
        assert agent.is_claude is False

    def test_is_claude_false_without_key(self):
        agent = _make_agent(
            canonical_provider="anthropic",
            provider="openai",
            openai_api_key=None,
        )
        assert agent.is_claude is False

    def test_is_claude_false_default_provider(self):
        agent = _make_agent()
        assert agent.is_claude is False


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    def test_tool_count(self):
        agent = _make_claude_agent()
        tools = agent._get_claude_tool_definitions()
        assert len(tools) == 3

    def test_tool_names(self):
        agent = _make_claude_agent()
        tools = agent._get_claude_tool_definitions()
        names = {t["name"] for t in tools}
        assert names == {"search_local_kb", "search_academic", "search_web"}

    def test_tool_schemas_valid(self):
        agent = _make_claude_agent()
        tools = agent._get_claude_tool_definitions()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema
            assert "query" in schema["required"]

    def test_academic_tool_has_year_params(self):
        agent = _make_claude_agent()
        tools = agent._get_claude_tool_definitions()
        academic = next(t for t in tools if t["name"] == "search_academic")
        props = academic["input_schema"]["properties"]
        assert "year_from" in props
        assert "year_to" in props


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_basic_prompt(self):
        agent = _make_claude_agent()
        prompt = agent._build_claude_system_prompt({})
        assert "research assistant" in prompt
        assert "knowledge base" in prompt
        assert "[1]" in prompt

    def test_prompt_with_researcher_context(self):
        agent = _make_claude_agent()
        prompt = agent._build_claude_system_prompt({"researcher": "David Harvey"})
        assert "David Harvey" in prompt

    def test_prompt_with_pinned_topics(self):
        agent = _make_claude_agent()
        prompt = agent._build_claude_system_prompt(
            {"auth_items": ["neoliberalism", "urbanization"]}
        )
        assert "neoliberalism" in prompt
        assert "urbanization" in prompt


# ---------------------------------------------------------------------------
# Tool execution dispatch
# ---------------------------------------------------------------------------

class TestToolExecution:
    @pytest.mark.asyncio
    async def test_execute_local_kb_dispatches(self):
        agent = _make_claude_agent()
        # Mock vector store and embedder
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 768

        mock_store = MagicMock()
        mock_store.search.return_value = {
            "documents": ["Test content about neoliberalism"],
            "metadatas": [{"title": "Test Paper", "authors": "Harvey", "year": 2005, "paper_id": "test-1"}],
            "distances": [0.3],
        }

        agent.embedder = mock_embedder
        agent.vector_store = mock_store

        local_results = []
        result = await agent._execute_claude_tool(
            "search_local_kb", {"query": "neoliberalism"},
            local_results, [], {},
        )
        assert len(local_results) > 0
        assert "local knowledge base" in result

    @pytest.mark.asyncio
    async def test_execute_academic_dispatches(self):
        agent = _make_claude_agent()
        mock_search = AsyncMock()
        mock_paper = SimpleNamespace(
            title="Test Paper", authors=["Author A"], year=2020,
            abstract="Abstract", paper_id="p1", doi="10.1/test",
            citation_count=50, url="http://example.com",
        )
        mock_search.search_openalex = AsyncMock(return_value=[mock_paper])
        mock_search.search_semantic_scholar = AsyncMock(return_value=[])
        mock_search.search_core = AsyncMock(return_value=[])
        agent.academic_search = mock_search

        external_results = []
        result = await agent._execute_claude_tool(
            "search_academic", {"query": "test query"},
            [], external_results, {},
        )
        assert len(external_results) == 1
        assert external_results[0]["title"] == "Test Paper"
        assert "academic" in result

    @pytest.mark.asyncio
    async def test_execute_web_dispatches(self):
        agent = _make_claude_agent()
        mock_web = AsyncMock()
        mock_web.search = AsyncMock(return_value=[
            {"title": "Web Result", "url": "http://example.com", "snippet": "Content"},
        ])
        agent.web_search = mock_web

        external_results = []
        result = await agent._execute_claude_tool(
            "search_web", {"query": "test"},
            [], external_results, {},
        )
        assert len(external_results) == 1
        assert "web" in result

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        agent = _make_claude_agent()
        result = await agent._execute_claude_tool(
            "nonexistent_tool", {}, [], [], {},
        )
        assert "Unknown tool" in result


# ---------------------------------------------------------------------------
# Full tool-use loop
# ---------------------------------------------------------------------------

class TestRunWithClaudeTools:
    @pytest.mark.asyncio
    async def test_greeting_no_tools(self):
        """Claude answers a greeting without calling any tools."""
        agent = _make_claude_agent()

        mock_response = _make_response(
            content=[_make_text_block("Hello! How can I help you with your research?")],
            stop_reason="end_turn",
        )

        with patch("anthropic.AsyncAnthropic") as MockClient:
            instance = AsyncMock()
            instance.messages.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = instance

            result = await agent._run_with_claude_tools("Hello!")

        assert result["answer"] == "Hello! How can I help you with your research?"
        assert result["local_sources"] == 0
        assert result["external_sources"] == 0
        assert result["query"] == "Hello!"

    @pytest.mark.asyncio
    async def test_research_query_with_tool_calls(self):
        """Claude calls search_local_kb, gets results, then synthesizes."""
        agent = _make_claude_agent()
        # Mock embedder + vector store for tool execution
        agent.embedder = MagicMock()
        agent.embedder.embed_query.return_value = [0.1] * 768
        agent.vector_store = MagicMock()
        agent.vector_store.search.return_value = {
            "documents": ["Harvey critiques neoliberalism..."],
            "metadatas": [{"title": "Brief History of Neoliberalism", "authors": "David Harvey", "year": 2005, "paper_id": "h1"}],
            "distances": [0.2],
        }

        # First response: tool call
        tool_response = _make_response(
            content=[
                _make_tool_use_block("call_1", "search_local_kb", {"query": "Harvey neoliberalism"}),
            ],
            stop_reason="tool_use",
        )
        # Second response: synthesis
        synth_response = _make_response(
            content=[_make_text_block("Harvey critiques neoliberalism as a class project [1].")],
            stop_reason="end_turn",
        )

        with patch("anthropic.AsyncAnthropic") as MockClient:
            instance = AsyncMock()
            instance.messages.create = AsyncMock(side_effect=[tool_response, synth_response])
            MockClient.return_value = instance

            result = await agent._run_with_claude_tools(
                "What does Harvey say about neoliberalism?"
            )

        assert "[1]" in result["answer"]
        assert result["local_sources"] > 0
        assert "query" in result
        assert "sources" in result

    @pytest.mark.asyncio
    async def test_max_turns_safety(self):
        """Verify MAX_TURNS prevents infinite loops."""
        agent = _make_claude_agent()
        agent.embedder = MagicMock()
        agent.embedder.embed_query.return_value = [0.1] * 768
        agent.vector_store = MagicMock()
        agent.vector_store.search.return_value = {
            "documents": [], "metadatas": [], "distances": [],
        }

        # Every response is a tool call — never end_turn
        endless_response = _make_response(
            content=[
                _make_tool_use_block("call_n", "search_local_kb", {"query": "test"}),
            ],
            stop_reason="tool_use",
        )

        with patch("anthropic.AsyncAnthropic") as MockClient:
            instance = AsyncMock()
            instance.messages.create = AsyncMock(return_value=endless_response)
            MockClient.return_value = instance

            result = await agent._run_with_claude_tools("test")

        assert "wasn't able to complete" in result["answer"]

    @pytest.mark.asyncio
    async def test_result_format_matches_langgraph(self):
        """The result dict has the same keys as the LangGraph path."""
        agent = _make_claude_agent()

        mock_response = _make_response(
            content=[_make_text_block("Test answer")],
            stop_reason="end_turn",
        )

        with patch("anthropic.AsyncAnthropic") as MockClient:
            instance = AsyncMock()
            instance.messages.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = instance

            result = await agent._run_with_claude_tools("test")

        # These are the keys _run_async always returns
        assert "query" in result
        assert "answer" in result
        assert "query_type" in result
        assert "local_sources" in result
        assert "external_sources" in result
        assert "sources" in result
        assert isinstance(result["sources"], list)


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------

class TestFallback:
    @pytest.mark.asyncio
    async def test_fallback_on_import_error(self):
        """If anthropic not installed, falls back to LangGraph."""
        agent = _make_claude_agent()

        # Patch _run_with_claude_tools to raise ImportError
        with patch.object(agent, "_run_with_claude_tools", side_effect=ImportError("No anthropic")):
            # Mock the graph to prevent actual LangGraph execution
            mock_state = {
                "final_answer": "LangGraph answer",
                "query_type": "general",
                "local_results": [],
                "external_results": [],
                "error": None,
            }
            agent.graph = MagicMock()
            agent.graph.ainvoke = AsyncMock(return_value=mock_state)

            result = await agent._run_async("test query")

        assert result["answer"] == "LangGraph answer"

    @pytest.mark.asyncio
    async def test_fallback_on_api_error(self):
        """If Claude API errors, falls back to LangGraph."""
        agent = _make_claude_agent()

        with patch.object(agent, "_run_with_claude_tools", side_effect=Exception("API 500")):
            mock_state = {
                "final_answer": "LangGraph fallback",
                "query_type": "general",
                "local_results": [],
                "external_results": [],
                "error": None,
            }
            agent.graph = MagicMock()
            agent.graph.ainvoke = AsyncMock(return_value=mock_state)

            result = await agent._run_async("test query")

        assert result["answer"] == "LangGraph fallback"


# ---------------------------------------------------------------------------
# connect_provider preserves canonical
# ---------------------------------------------------------------------------

class TestConnectProvider:
    def test_anthropic_byok_sets_canonical(self):
        agent = _make_agent(provider="openai", openai_api_key="sk-temp")
        mock_model = MagicMock()
        agent.model = mock_model
        agent._canonical_provider = "groq"  # Start as something else

        with patch("research_agent.agents.research_agent.OpenAICompatibleModel"):
            result = agent.connect_provider("anthropic", "sk-ant-test123")

        assert result is True
        assert agent._canonical_provider == "anthropic"
        assert agent.is_claude is True

    def test_groq_byok_not_claude(self):
        agent = _make_agent(provider="openai", openai_api_key="sk-temp")

        with patch("research_agent.agents.research_agent.OpenAICompatibleModel"):
            agent.connect_provider("groq", "gsk-test123")

        assert agent._canonical_provider == "groq"
        assert agent.is_claude is False


# ---------------------------------------------------------------------------
# Format tool results
# ---------------------------------------------------------------------------

class TestFormatToolResults:
    def test_empty_results(self):
        agent = _make_claude_agent()
        text = agent._format_tool_results([], "academic")
        assert "No academic results found" in text

    def test_formats_papers(self):
        agent = _make_claude_agent()
        results = [
            {"title": "Test Paper", "authors": "Author A", "year": 2020, "content": "Some content"},
        ]
        text = agent._format_tool_results(results, "academic")
        assert "[1] Test Paper" in text
        assert "Author A" in text
        assert "2020" in text
