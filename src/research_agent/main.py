"""
Research Agent - Main Entry Point

Run with: python -m research_agent.main
"""

import argparse
import os
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from research_agent.utils.config import load_config


# Cloud provider configurations (OpenAI-compatible endpoints)
CLOUD_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"],
    },
    "groq": {
        "name": "Groq (Free Tier)",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3.1-8b-instruct:free",
        "models": ["meta-llama/llama-3.1-8b-instruct:free", "google/gemma-2-9b-it:free"],
    },
}


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is reachable."""
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def detect_available_provider(config: dict) -> Tuple[str, Optional[dict]]:
    """
    Auto-detect the best available LLM provider.

    Priority:
    1. OpenAI (if OPENAI_API_KEY is set)
    2. Groq (if GROQ_API_KEY is set - free tier!)
    3. OpenRouter (if OPENROUTER_API_KEY is set)
    4. Ollama (if server is reachable)
    5. HuggingFace (local, requires GPU)

    Returns:
        Tuple of (provider_name, provider_config) or (provider_name, None) for local providers
    """
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}

    # Check cloud providers in priority order
    for provider_key in ["openai", "groq", "openrouter"]:
        provider = CLOUD_PROVIDERS[provider_key]
        api_key = os.getenv(provider["api_key_env"])
        if api_key:
            print(f"  Found {provider['name']} API key")
            return provider_key, provider

    # Check Ollama
    ollama_base_url = model_cfg.get("ollama_base_url", "http://localhost:11434")
    if check_ollama_available(ollama_base_url):
        print(f"  Found Ollama server at {ollama_base_url}")
        return "ollama", None

    # Fallback to HuggingFace (local)
    print("  No cloud providers found, will use local HuggingFace models")
    return "huggingface", None


def main():
    parser = argparse.ArgumentParser(description="Research Assistant Agent")

    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        choices=["ui", "cli", "check"],
        default="ui",
        help="Run mode: ui (Gradio), cli (command line), check (verify setup)",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port for UI mode")
    parser.add_argument(
        "--share", action="store_true", help="Create public Gradio link"
    )
    parser.add_argument(
        "--host", type=str, default=None, help="Host to bind to (use 0.0.0.0 for Docker)"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == "check":
        run_checks()
    elif args.mode == "ui":
        run_ui(config, args.port, args.share, args.host)
    elif args.mode == "cli":
        run_cli(config)


def run_checks():
    """Verify the setup is working."""
    print("=" * 50)
    print("Research Agent - Setup Check")
    print("=" * 50)

    # Check LLM providers
    print("\n1. Checking LLM providers...")
    available_providers = []

    # Check cloud providers
    for provider_key, provider in CLOUD_PROVIDERS.items():
        api_key = os.getenv(provider["api_key_env"])
        if api_key:
            print(f"   ✓ {provider['name']} ({provider_key}) - API key found")
            available_providers.append(provider_key)
        else:
            print(f"   ○ {provider['name']} ({provider_key}) - no API key")

    # Check Ollama
    if check_ollama_available():
        print("   ✓ Ollama - server running")
        available_providers.append("ollama")
    else:
        print("   ○ Ollama - not running (start with: ollama serve)")

    if not available_providers:
        print("   ⚠️  No cloud providers configured!")
        print("   Tip: Get a FREE Groq API key at https://console.groq.com/keys")
        print("   Then set GROQ_API_KEY in your .env file")
    else:
        print(f"\n   Provider auto-detection will use: {available_providers[0]}")

    # Check GPU
    print("\n2. Checking GPU...")
    try:
        from research_agent.models.llm_loader import check_gpu

        check_gpu()
    except ImportError as e:
        print(f"   ⚠️  Could not import: {e}")
        print("   Run: pip install torch")

    # Check embedding model can load
    print("\n3. Checking embedding model...")
    try:
        from sentence_transformers import SentenceTransformer

        print("   ✓ sentence-transformers available")
    except ImportError:
        print("   ⚠️  sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")

    # Check ChromaDB
    print("\n4. Checking vector database...")
    try:
        import chromadb

        print("   ✓ chromadb available")
    except ImportError:
        print("   ⚠️  chromadb not installed")
        print("   Run: pip install chromadb")

    # Check LangChain
    print("\n5. Checking agent framework...")
    try:
        import langchain
        import langgraph

        print("   ✓ langchain and langgraph available")
    except ImportError as e:
        print(f"   ⚠️  Missing: {e}")
        print("   Run: pip install langchain langgraph")

    # Check Gradio
    print("\n6. Checking UI framework...")
    try:
        import gradio

        print(f"   ✓ gradio {gradio.__version__} available")
    except ImportError:
        print("   ⚠️  gradio not installed")
        print("   Run: pip install gradio")

    # Check API libraries
    print("\n7. Checking API libraries...")
    apis = {
        "semanticscholar": "Semantic Scholar",
        "pyalex": "OpenAlex",
        "httpx": "HTTP client",
    }
    for module, name in apis.items():
        try:
            __import__(module)
            print(f"   ✓ {name} ({module})")
        except ImportError:
            print(f"   ⚠️  {name} ({module}) not installed")

    print("\n" + "=" * 50)
    print("Setup check complete!")
    print("=" * 50)


def run_ui(config: dict, port: int, share: bool, host: str = None):
    """Launch the Gradio UI."""
    print("Starting Research Agent UI...")

    agent = build_agent_from_config(config)

    from research_agent.ui import launch_app

    launch_app(agent=agent, port=port, share=share, host=host)


def run_cli(config: dict):
    """Run in CLI mode for testing."""
    print("Research Agent CLI")
    print("Type 'quit' to exit\n")

    # TODO: Initialize real agent
    agent = None

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            if agent is None:
                print("Agent: [Demo mode - agent not loaded]\n")
            else:
                # response = await agent.run(query)
                # print(f"Agent: {response}\n")
                pass

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def build_agent_from_config(config: dict):
    """Create a ResearchAgent instance from config."""
    from research_agent.agents import AgentConfig, create_research_agent
    from research_agent.db.embeddings import get_embedder
    from research_agent.db.vector_store import ResearchVectorStore
    from research_agent.tools.academic_search import AcademicSearchTools
    from research_agent.tools.web_search import WebSearchTool

    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    provider = model_cfg.get("provider", "auto")

    # Auto-detect provider if set to "auto"
    detected_cloud_config = None
    if provider == "auto":
        print("Auto-detecting LLM provider...")
        provider, detected_cloud_config = detect_available_provider(config)
        print(f"  Selected provider: {provider}")
    elif provider in CLOUD_PROVIDERS:
        # Explicit cloud provider selection
        cloud_cfg = CLOUD_PROVIDERS[provider]
        api_key = os.getenv(cloud_cfg["api_key_env"])
        if api_key:
            print(f"Using {cloud_cfg['name']}...")
            detected_cloud_config = cloud_cfg
        else:
            print(f"Warning: {provider} selected but {cloud_cfg['api_key_env']} not set")

    use_ollama = provider == "ollama"
    ollama_base_url = model_cfg.get("ollama_base_url", "http://localhost:11434")
    ollama_model = model_cfg.get("ollama_model") or model_cfg.get("name") or "mistral"

    openai_cfg = model_cfg.get("openai", {}) if isinstance(model_cfg, dict) else {}
    openai_compat_cfg = (
        model_cfg.get("openai_compatible", {}) if isinstance(model_cfg, dict) else {}
    )

    # If we detected a cloud provider, use its configuration
    if detected_cloud_config:
        openai_base_url = detected_cloud_config["base_url"]
        openai_api_key = os.getenv(detected_cloud_config["api_key_env"])
        openai_models = detected_cloud_config["models"]
        openai_default_model = detected_cloud_config["default_model"]
        # For cloud providers, we use "openai" provider type (OpenAI-compatible API)
        provider = "openai"
    else:
        openai_base_url = openai_cfg.get("base_url", "https://api.openai.com/v1")
        openai_api_key = os.getenv(openai_cfg.get("api_key_env", "OPENAI_API_KEY"))
        openai_models = openai_cfg.get("models", [])
        openai_default_model = openai_models[0] if openai_models else "gpt-4o-mini"

    openai_compat_base_url = openai_compat_cfg.get(
        "base_url", "http://localhost:8082/v1"
    )
    openai_compat_key_env = openai_compat_cfg.get(
        "api_key_env", "OPENAI_COMPAT_API_KEY"
    )
    openai_compat_api_key = os.getenv(openai_compat_key_env) or os.getenv(
        "OPENAI_API_KEY"
    )
    openai_compat_models = openai_compat_cfg.get("models", [])
    openai_compat_default_model = (
        openai_compat_models[0] if openai_compat_models else openai_default_model
    )

    embedding_cfg = config.get("embedding", {}) if isinstance(config, dict) else {}
    embedder = get_embedder(
        model_name=embedding_cfg.get("name", "BAAI/bge-base-en-v1.5"),
        device=embedding_cfg.get("device"),
    )

    vector_cfg = config.get("vector_store", {}) if isinstance(config, dict) else {}
    vector_store = ResearchVectorStore(
        persist_dir=vector_cfg.get("persist_directory", "./data/chroma_db")
    )

    search_cfg = config.get("search", {}) if isinstance(config, dict) else {}
    openalex_email = (search_cfg.get("openalex", {}) or {}).get("email")
    unpaywall_email = (search_cfg.get("unpaywall", {}) or {}).get("email")
    academic_search = AcademicSearchTools(
        config=search_cfg,
        email=openalex_email or unpaywall_email,
    )

    web_cfg = search_cfg.get("web_search", {}) or {}
    web_provider = web_cfg.get("provider", "duckduckgo")
    web_api_key = None
    if web_provider == "tavily":
        web_api_key = os.getenv("TAVILY_API_KEY")
    elif web_provider == "serper":
        web_api_key = os.getenv("SERPER_API_KEY")

    web_search = WebSearchTool(api_key=web_api_key, provider=web_provider)

    retrieval_cfg = config.get("retrieval", {}) if isinstance(config, dict) else {}
    ingestion_cfg = config.get("ingestion", {}) if isinstance(config, dict) else {}
    agent_config = AgentConfig(
        max_local_results=retrieval_cfg.get("top_k", 5),
        auto_ingest=bool(ingestion_cfg.get("auto_ingest", False)),
        auto_ingest_threshold=float(ingestion_cfg.get("auto_threshold", 0.85)),
        include_web_search=bool(
            (search_cfg.get("web_search", {}) or {}).get("enabled", True)
        ),
    )

    if provider == "openai_compatible":
        openai_base_url = openai_compat_base_url
        openai_api_key = openai_compat_api_key
        openai_models = openai_compat_models
        openai_default_model = openai_compat_default_model

    return create_research_agent(
        vector_store=vector_store,
        embedder=embedder,
        academic_search=academic_search,
        web_search=web_search,
        config=agent_config,
        use_ollama=use_ollama,
        provider=provider,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        openai_model=openai_default_model,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        openai_models=openai_models,
        openai_compat_base_url=openai_compat_base_url,
        openai_compat_api_key=openai_compat_api_key,
        openai_compat_models=openai_compat_models,
    )


if __name__ == "__main__":
    main()
