"""
Research Agent - Main Entry Point

Run with: python -m research_agent.main
"""

import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)

    if not path.exists():
        print(f"Config not found at {config_path}, using defaults")
        return {}

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == "check":
        run_checks()
    elif args.mode == "ui":
        run_ui(config, args.port, args.share)
    elif args.mode == "cli":
        run_cli(config)


def run_checks():
    """Verify the setup is working."""
    print("=" * 50)
    print("Research Agent - Setup Check")
    print("=" * 50)

    # Check GPU
    print("\n1. Checking GPU...")
    try:
        from research_agent.models.llm_loader import check_gpu

        check_gpu()
    except ImportError as e:
        print(f"   ⚠️  Could not import: {e}")
        print("   Run: pip install torch")

    # Check embedding model can load
    print("\n2. Checking embedding model...")
    try:
        from sentence_transformers import SentenceTransformer

        print("   ✓ sentence-transformers available")
    except ImportError:
        print("   ⚠️  sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")

    # Check ChromaDB
    print("\n3. Checking vector database...")
    try:
        import chromadb

        print("   ✓ chromadb available")
    except ImportError:
        print("   ⚠️  chromadb not installed")
        print("   Run: pip install chromadb")

    # Check LangChain
    print("\n4. Checking agent framework...")
    try:
        import langchain
        import langgraph

        print("   ✓ langchain and langgraph available")
    except ImportError as e:
        print(f"   ⚠️  Missing: {e}")
        print("   Run: pip install langchain langgraph")

    # Check Gradio
    print("\n5. Checking UI framework...")
    try:
        import gradio

        print(f"   ✓ gradio {gradio.__version__} available")
    except ImportError:
        print("   ⚠️  gradio not installed")
        print("   Run: pip install gradio")

    # Check API libraries
    print("\n6. Checking API libraries...")
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


def run_ui(config: dict, port: int, share: bool):
    """Launch the Gradio UI."""
    print("Starting Research Agent UI...")

    agent = build_agent_from_config(config)

    from research_agent.ui import launch_app

    launch_app(agent=agent, port=port, share=share)


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


if __name__ == "__main__":
    main()


def build_agent_from_config(config: dict):
    """Create a ResearchAgent instance from config."""
    from research_agent.agents import AgentConfig, create_research_agent
    from research_agent.db.embeddings import get_embedder
    from research_agent.db.vector_store import ResearchVectorStore
    from research_agent.tools.academic_search import AcademicSearchTools
    from research_agent.tools.web_search import WebSearchTool

    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    provider = model_cfg.get("provider", "ollama")

    use_ollama = provider == "ollama"
    ollama_base_url = model_cfg.get("ollama_base_url", "http://localhost:11434")
    ollama_model = model_cfg.get("ollama_model") or model_cfg.get("name") or "mistral"

    openai_cfg = model_cfg.get("openai", {}) if isinstance(model_cfg, dict) else {}
    openai_compat_cfg = (
        model_cfg.get("openai_compatible", {}) if isinstance(model_cfg, dict) else {}
    )

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
