"""Tests for WebSearchTool provider switching."""

import pytest
from research_agent.tools.web_search import WebSearchTool


class TestSetProvider:
    def test_switch_to_perplexity(self):
        tool = WebSearchTool(provider="duckduckgo")
        tool.set_provider("perplexity", api_key="pplx-test")
        assert tool.provider == "perplexity"
        assert tool.api_key == "pplx-test"

    def test_switch_to_duckduckgo(self):
        tool = WebSearchTool(provider="perplexity", api_key="key")
        tool.set_provider("duckduckgo")
        assert tool.provider == "duckduckgo"

    def test_switch_to_tavily(self):
        tool = WebSearchTool(provider="duckduckgo")
        tool.set_provider("tavily", api_key="tvly-test")
        assert tool.provider == "tavily"

    def test_switch_to_serper(self):
        tool = WebSearchTool(provider="duckduckgo")
        tool.set_provider("serper", api_key="serp-test")
        assert tool.provider == "serper"

    def test_invalid_provider_raises(self):
        tool = WebSearchTool(provider="duckduckgo")
        with pytest.raises(ValueError, match="Unknown provider"):
            tool.set_provider("google")

    def test_api_key_preserved_when_not_provided(self):
        tool = WebSearchTool(provider="tavily", api_key="original-key")
        tool.set_provider("perplexity")
        # api_key unchanged since we didn't pass a new one
        assert tool.api_key == "original-key"

    def test_api_key_updated_when_provided(self):
        tool = WebSearchTool(provider="tavily", api_key="old-key")
        tool.set_provider("perplexity", api_key="new-key")
        assert tool.api_key == "new-key"
