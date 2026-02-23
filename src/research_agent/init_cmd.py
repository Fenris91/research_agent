"""Interactive setup command for Research Agent.

Guides new users through provider selection, API key validation,
directory creation, and .env generation.

Usage: research-agent init
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import httpx

# Signup URLs for each provider
PROVIDER_SIGNUP_URLS = {
    "groq": "https://console.groq.com/keys",
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
    "perplexity": "https://www.perplexity.ai/settings/api",
    "gemini": "https://aistudio.google.com/apikey",
    "mistral": "https://console.mistral.ai/api-keys",
    "xai": "https://console.x.ai",
    "openrouter": "https://openrouter.ai/keys",
}

# Main menu: number → provider key
MAIN_MENU = [
    ("1", "groq", "Groq        — free tier, fast inference (recommended)"),
    ("2", "openai", "OpenAI      — GPT-4o, most capable"),
    ("3", "anthropic", "Anthropic   — Claude, native tool-use"),
    ("4", "ollama", "Ollama      — local models, no API key"),
    ("5", "other", "Other       — Perplexity, Gemini, Mistral, xAI, OpenRouter"),
    ("6", "none", "None        — retrieval-only (no LLM)"),
]

# "Other" submenu
OTHER_MENU = [
    ("1", "perplexity", "Perplexity"),
    ("2", "gemini", "Google Gemini"),
    ("3", "mistral", "Mistral AI"),
    ("4", "xai", "xAI (Grok)"),
    ("5", "openrouter", "OpenRouter"),
]

# Directories to create (relative to base_dir)
INIT_DIRS = [
    "data/chroma_db",
    "cache",
    "logs",
    "exports",
]


# ---------------------------------------------------------------------------
# ANSI formatting helpers
# ---------------------------------------------------------------------------

def _supports_color() -> bool:
    """Check if the terminal supports ANSI colors."""
    if os.getenv("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


def _fmt(code: str, text: str) -> str:
    if _supports_color():
        return f"\033[{code}m{text}\033[0m"
    return text


def _bold(text: str) -> str:
    return _fmt("1", text)


def _dim(text: str) -> str:
    return _fmt("2", text)


def _green(text: str) -> str:
    return _fmt("32", text)


def _red(text: str) -> str:
    return _fmt("31", text)


def _yellow(text: str) -> str:
    return _fmt("33", text)


def _print_step(number: int, title: str) -> None:
    print(f"\n  {_bold(f'{number}. {title}')}\n")


def _print_success(msg: str) -> None:
    print(f"     {_green('✓')} {msg}")


def _print_error(msg: str) -> None:
    print(f"     {_red('✗')} {msg}")


def _print_warn(msg: str) -> None:
    print(f"     {_yellow('!')} {msg}")


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def _pick_provider() -> str:
    """Show numbered menu, return provider key. Default = groq."""
    print("     Which provider would you like to use?\n")
    for num, _key, label in MAIN_MENU:
        print(f"     [{num}] {label}")

    while True:
        raw = input(f"\n     Choice [{_dim('1')}]: ").strip()
        if raw == "":
            return "groq"

        for num, key, _label in MAIN_MENU:
            if raw == num:
                if key == "other":
                    return _pick_other_provider()
                return key

        _print_error(f"Invalid choice: {raw}")


def _pick_other_provider() -> str:
    """Submenu for 'Other' providers."""
    print()
    for num, _key, label in OTHER_MENU:
        print(f"     [{num}] {label}")

    while True:
        raw = input(f"\n     Choice [{_dim('1')}]: ").strip()
        if raw == "":
            return "perplexity"

        for num, key, _label in OTHER_MENU:
            if raw == num:
                return key

        _print_error(f"Invalid choice: {raw}")


# ---------------------------------------------------------------------------
# API key prompt + validation
# ---------------------------------------------------------------------------

def _prompt_api_key(provider_key: str) -> str:
    """Prompt user to paste an API key. Shows signup URL."""
    from research_agent.main import CLOUD_PROVIDERS

    provider = CLOUD_PROVIDERS[provider_key]
    signup_url = PROVIDER_SIGNUP_URLS.get(provider_key, "")
    url_hint = f" ({signup_url})" if signup_url else ""

    print(f"\n     {provider['name']} API key{_dim(url_hint)}:")
    while True:
        key = input("     > ").strip()
        if key:
            return key
        _print_error("API key cannot be empty")


def _validate_provider(provider_key: str, api_key: str) -> bool:
    """Validate API key with a 1-token test call. Returns True on success."""
    from research_agent.main import CLOUD_PROVIDERS

    provider = CLOUD_PROVIDERS[provider_key]
    base_url = provider["base_url"].rstrip("/")

    try:
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": provider["default_model"],
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=10,
        )
        return resp.status_code == 200
    except httpx.TimeoutException:
        return False
    except httpx.ConnectError:
        return False


def _validate_ollama(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is reachable."""
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Email prompt
# ---------------------------------------------------------------------------

def _prompt_email() -> Optional[str]:
    """Ask for academic email (optional). Returns None if skipped."""
    print("     Email for OpenAlex/Unpaywall polite pool (higher rate limits):")
    email = input(f"     > {_dim('[Enter to skip]')} ").strip()
    return email if email else None


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

def _create_directories(base_dir: Path) -> list[str]:
    """Create standard project directories. Returns list of created paths."""
    created = []
    for rel_path in INIT_DIRS:
        full_path = base_dir / rel_path
        full_path.mkdir(parents=True, exist_ok=True)
        created.append(rel_path)
    return created


# ---------------------------------------------------------------------------
# .env writing
# ---------------------------------------------------------------------------

def _build_env_content(
    provider_key: str,
    api_key: Optional[str],
    email: Optional[str],
) -> str:
    """Build .env file content string."""
    from research_agent.main import CLOUD_PROVIDERS

    lines = [
        "# Research Agent - generated by `research-agent init`",
        "",
        "# LLM Provider",
    ]

    # Write all provider keys, active one uncommented
    for key, provider in CLOUD_PROVIDERS.items():
        env_var = provider["api_key_env"]
        if key == provider_key and api_key:
            lines.append(f"{env_var}={api_key}")
        else:
            lines.append(f"# {env_var}=")

    # Academic emails
    lines.append("")
    lines.append("# Academic APIs (polite pool for higher rate limits)")
    if email:
        lines.append(f"OPENALEX_EMAIL={email}")
        lines.append(f"UNPAYWALL_EMAIL={email}")
    else:
        lines.append("# OPENALEX_EMAIL=")
        lines.append("# UNPAYWALL_EMAIL=")

    lines.append("")
    return "\n".join(lines) + "\n"


def _write_env(
    env_path: Path,
    provider_key: str,
    api_key: Optional[str],
    email: Optional[str],
) -> bool:
    """Write .env file. If it exists, ask before overwriting. Returns True if written."""
    content = _build_env_content(provider_key, api_key, email)

    if env_path.exists():
        _print_warn(f".env already exists at {env_path}")
        overwrite = input("     Overwrite? [y/N]: ").strip().lower()
        if overwrite != "y":
            _print_warn("Skipped — .env unchanged")
            return False

    env_path.write_text(content)
    return True


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_init(base_dir: Optional[Path] = None) -> None:
    """Run the interactive setup flow."""
    if base_dir is None:
        base_dir = Path.cwd()

    print(f"\n  {_bold('Research Agent — Setup')}")
    print(f"  {'─' * 21}\n")

    # Step 1: Provider
    _print_step(1, "LLM Provider")
    provider_key = _pick_provider()
    api_key: Optional[str] = None

    if provider_key == "ollama":
        if _validate_ollama():
            _print_success("Ollama server detected")
        else:
            _print_error("Ollama not reachable at localhost:11434")
            _print_warn("Start with: ollama serve")
    elif provider_key != "none":
        api_key = _prompt_api_key(provider_key)
        if _validate_provider(provider_key, api_key):
            from research_agent.main import CLOUD_PROVIDERS
            model = CLOUD_PROVIDERS[provider_key]["default_model"]
            _print_success(f"{CLOUD_PROVIDERS[provider_key]['name']} connected — {model}")
        else:
            _print_error("Could not connect — check your API key")
            _print_warn("Continuing anyway (you can edit .env later)")

    # Step 2: Email
    _print_step(2, "Academic Email (optional)")
    email = _prompt_email()
    if email:
        _print_success("Email set")
    else:
        print(f"     {_dim('Skipped')}")

    # Step 3: Directories
    _print_step(3, "Directories")
    created = _create_directories(base_dir)
    for d in created:
        print(f"     Creating ./{d:<20s} ... {_green('✓')}")

    # Step 4: .env
    _print_step(4, "Writing .env")
    env_path = base_dir / ".env"
    if _write_env(env_path, provider_key, api_key, email):
        _print_success(f".env written to {env_path}")

    # Done
    print(f"\n  {'─' * 21}")
    print(f"  {_bold('Setup complete!')} Run:\n")
    print(f"    research-agent              {_dim('# launch UI')}")
    print(f"    research-agent --mode cli   {_dim('# command line')}")
    print()
