"""Tests for research_agent.init_cmd — interactive setup command."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from research_agent.init_cmd import (
    _pick_provider,
    _pick_other_provider,
    _prompt_api_key,
    _validate_provider,
    _validate_ollama,
    _prompt_email,
    _create_directories,
    _write_env,
    _build_env_content,
    run_init,
    INIT_DIRS,
)


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

class TestPickProvider:
    def test_default_groq(self):
        """Empty input → groq (default)."""
        with patch("builtins.input", return_value=""):
            assert _pick_provider() == "groq"

    def test_explicit_openai(self):
        """Input '2' → openai."""
        with patch("builtins.input", return_value="2"):
            assert _pick_provider() == "openai"

    def test_explicit_anthropic(self):
        with patch("builtins.input", return_value="3"):
            assert _pick_provider() == "anthropic"

    def test_explicit_ollama(self):
        with patch("builtins.input", return_value="4"):
            assert _pick_provider() == "ollama"

    def test_explicit_none(self):
        with patch("builtins.input", return_value="6"):
            assert _pick_provider() == "none"

    def test_invalid_then_valid(self):
        """Invalid input retries, then accepts valid."""
        with patch("builtins.input", side_effect=["99", "1"]):
            assert _pick_provider() == "groq"

    def test_other_submenu(self):
        """Input '5' → enters other submenu, default perplexity."""
        with patch("builtins.input", side_effect=["5", ""]):
            assert _pick_provider() == "perplexity"


class TestPickOtherProvider:
    def test_default_perplexity(self):
        with patch("builtins.input", return_value=""):
            assert _pick_other_provider() == "perplexity"

    def test_gemini(self):
        with patch("builtins.input", return_value="2"):
            assert _pick_other_provider() == "gemini"

    def test_mistral(self):
        with patch("builtins.input", return_value="3"):
            assert _pick_other_provider() == "mistral"

    def test_xai(self):
        with patch("builtins.input", return_value="4"):
            assert _pick_other_provider() == "xai"

    def test_openrouter(self):
        with patch("builtins.input", return_value="5"):
            assert _pick_other_provider() == "openrouter"

    def test_invalid_then_valid(self):
        with patch("builtins.input", side_effect=["9", "1"]):
            assert _pick_other_provider() == "perplexity"


# ---------------------------------------------------------------------------
# API key prompt
# ---------------------------------------------------------------------------

class TestPromptApiKey:
    def test_returns_key(self):
        with patch("builtins.input", return_value="gsk_abc123"):
            key = _prompt_api_key("groq")
            assert key == "gsk_abc123"

    def test_empty_retries(self):
        with patch("builtins.input", side_effect=["", "sk-real"]):
            key = _prompt_api_key("openai")
            assert key == "sk-real"


# ---------------------------------------------------------------------------
# Provider validation
# ---------------------------------------------------------------------------

class TestValidateProvider:
    def test_success(self):
        """Mock 200 → True."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.post", return_value=mock_resp):
            assert _validate_provider("groq", "gsk_test") is True

    def test_auth_failure(self):
        """Mock 401 → False."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        with patch("httpx.post", return_value=mock_resp):
            assert _validate_provider("groq", "bad_key") is False

    def test_timeout(self):
        """Timeout exception → False."""
        import httpx
        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            assert _validate_provider("groq", "gsk_test") is False

    def test_connect_error(self):
        """Connection error → False."""
        import httpx
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            assert _validate_provider("openai", "sk_test") is False


class TestValidateOllama:
    def test_reachable(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
            assert _validate_ollama() is True

    def test_not_reachable(self):
        with patch("httpx.get", side_effect=Exception("refused")):
            assert _validate_ollama() is False


# ---------------------------------------------------------------------------
# Email prompt
# ---------------------------------------------------------------------------

class TestPromptEmail:
    def test_provided(self):
        with patch("builtins.input", return_value="me@university.edu"):
            assert _prompt_email() == "me@university.edu"

    def test_skipped(self):
        with patch("builtins.input", return_value=""):
            assert _prompt_email() is None


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

class TestCreateDirectories:
    def test_creates_all(self, tmp_path):
        created = _create_directories(tmp_path)
        assert len(created) == len(INIT_DIRS)
        for rel in INIT_DIRS:
            assert (tmp_path / rel).is_dir()

    def test_idempotent(self, tmp_path):
        """Running twice doesn't error."""
        _create_directories(tmp_path)
        _create_directories(tmp_path)
        for rel in INIT_DIRS:
            assert (tmp_path / rel).is_dir()


# ---------------------------------------------------------------------------
# .env writing
# ---------------------------------------------------------------------------

class TestBuildEnvContent:
    def test_contains_active_key(self):
        content = _build_env_content("groq", "gsk_abc123", None)
        assert "GROQ_API_KEY=gsk_abc123" in content
        # Others should be commented
        assert "# OPENAI_API_KEY=" in content

    def test_contains_email(self):
        content = _build_env_content("groq", "gsk_abc", "me@uni.edu")
        assert "OPENALEX_EMAIL=me@uni.edu" in content
        assert "UNPAYWALL_EMAIL=me@uni.edu" in content

    def test_no_email(self):
        content = _build_env_content("groq", "gsk_abc", None)
        assert "# OPENALEX_EMAIL=" in content

    def test_none_provider(self):
        """Provider 'none' doesn't crash — all keys commented."""
        content = _build_env_content("none", None, None)
        assert "# GROQ_API_KEY=" in content
        assert "# OPENAI_API_KEY=" in content


class TestWriteEnv:
    def test_creates_file(self, tmp_path):
        env_path = tmp_path / ".env"
        result = _write_env(env_path, "groq", "gsk_test", "me@uni.edu")
        assert result is True
        assert env_path.exists()
        content = env_path.read_text()
        assert "GROQ_API_KEY=gsk_test" in content

    def test_existing_no_overwrite(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("OLD_CONTENT=123\n")
        with patch("builtins.input", return_value="n"):
            result = _write_env(env_path, "groq", "gsk_new", None)
        assert result is False
        assert env_path.read_text() == "OLD_CONTENT=123\n"

    def test_existing_yes_overwrite(self, tmp_path):
        env_path = tmp_path / ".env"
        env_path.write_text("OLD_CONTENT=123\n")
        with patch("builtins.input", return_value="y"):
            result = _write_env(env_path, "groq", "gsk_new", None)
        assert result is True
        assert "GROQ_API_KEY=gsk_new" in env_path.read_text()


# ---------------------------------------------------------------------------
# Full init flow
# ---------------------------------------------------------------------------

class TestRunInit:
    def test_full_flow_groq(self, tmp_path):
        """Full happy path: groq + email → .env + dirs created."""
        inputs = iter([
            "1",             # provider: groq
            "gsk_testkey",   # api key
            "me@uni.edu",    # email
        ])

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("builtins.input", side_effect=inputs), \
             patch("httpx.post", return_value=mock_resp):
            run_init(base_dir=tmp_path)

        # .env created with correct key
        env_path = tmp_path / ".env"
        assert env_path.exists()
        content = env_path.read_text()
        assert "GROQ_API_KEY=gsk_testkey" in content
        assert "OPENALEX_EMAIL=me@uni.edu" in content

        # Directories created
        for rel in INIT_DIRS:
            assert (tmp_path / rel).is_dir()

    def test_flow_none_provider(self, tmp_path):
        """Provider 'none' skips key prompt entirely."""
        inputs = iter([
            "6",    # provider: none
            "",     # email: skip
        ])

        with patch("builtins.input", side_effect=inputs):
            run_init(base_dir=tmp_path)

        env_path = tmp_path / ".env"
        assert env_path.exists()
        # All keys should be commented
        content = env_path.read_text()
        assert "# GROQ_API_KEY=" in content

    def test_flow_ollama(self, tmp_path):
        """Ollama path — validates server, no key prompt."""
        inputs = iter([
            "4",    # provider: ollama
            "",     # email: skip
        ])

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("builtins.input", side_effect=inputs), \
             patch("httpx.get", return_value=mock_resp):
            run_init(base_dir=tmp_path)

        assert (tmp_path / ".env").exists()

    def test_flow_validation_failure(self, tmp_path):
        """Validation failure continues (writes .env anyway)."""
        inputs = iter([
            "1",             # provider: groq
            "gsk_badkey",    # api key
            "me@uni.edu",    # email
        ])

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch("builtins.input", side_effect=inputs), \
             patch("httpx.post", return_value=mock_resp):
            run_init(base_dir=tmp_path)

        # .env still written even with bad key
        env_path = tmp_path / ".env"
        assert env_path.exists()
        assert "GROQ_API_KEY=gsk_badkey" in env_path.read_text()
