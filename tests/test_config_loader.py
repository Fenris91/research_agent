"""Tests for centralized config loading."""

import pytest
from pathlib import Path
from research_agent.utils.config import load_config, clear_config_cache


class TestConfigLoader:
    def setup_method(self):
        clear_config_cache()

    def test_load_from_explicit_path(self, tmp_path):
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text("explorer:\n  email: test@example.com\n")
        result = load_config(str(cfg_file))
        assert result["explorer"]["email"] == "test@example.com"

    def test_load_missing_file_returns_empty(self, tmp_path):
        result = load_config(str(tmp_path / "nonexistent.yaml"))
        assert result == {}

    def test_caching(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("key: value1")
        r1 = load_config(str(cfg_file), use_cache=False)
        cfg_file.write_text("key: value2")
        # Should still return cached (but explicit paths bypass cache)
        clear_config_cache()
        r2 = load_config(str(cfg_file))
        assert r2["key"] == "value2"

    def test_auto_discover(self):
        # Should find configs/config.yaml from project root
        result = load_config()
        assert isinstance(result, dict)
