"""
Research Agent - Main Entry Point

Run with: python -m research_agent.main
"""

import argparse
from pathlib import Path

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

    # TODO: Initialize real agent when implemented
    # from research_agent.agents import ResearchAgent
    # agent = ResearchAgent.from_config(config)
    agent = None  # Demo mode for now

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
