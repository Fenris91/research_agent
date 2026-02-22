# Research Agent

A personal research memory for social sciences researchers. Ingest papers, build a semantic knowledge base, query it in natural language, and get cited answers — with a visual knowledge graph for exploration.

## What It Does

- **Semantic Search** across your personal knowledge base (ChromaDB + BGE embeddings)
- **Academic APIs** — Semantic Scholar, OpenAlex, CrossRef, Unpaywall for paper discovery
- **Researcher Lookup** — citation profiles, h-index, publication networks, open access detection
- **Knowledge Explorer** — interactive D3.js force-directed graph mapping researchers, papers, fields and societal domains
- **Research Agent** — LangGraph orchestration that understands your query, searches, and synthesizes cited answers
- **PDF Ingestion** — extract and chunk academic papers directly into the KB
- **Data Analysis** — statistical analysis and visualization tools

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Fenris91/research_agent.git
cd research_agent
conda create -n llm311 python=3.11
conda activate llm311
pip install -r requirements-runtime.txt

# Run
python -m research_agent.main --mode ui
```

The UI launches at `localhost:7860` (auto-increments if busy). LLM provider is auto-detected from available API keys (Groq free tier, OpenAI, Ollama).

## Knowledge Explorer

Interactive graph visualization of your research landscape:

- **Dual-mode**: single researcher view or full knowledge base view
- **Node types**: researchers, papers (sized by citations), academic fields, societal domains
- **Layers**: Structure (fields/domains), People (authorship), Topics (keyword-matched)
- **Open Access indicators**: green (open), yellow (preprint), grey (paywalled)
- **Detail panel**: click any node for metadata, actions, and connections

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Embeddings | BAAI/bge-base-en-v1.5 (768 dims) |
| Vector DB | ChromaDB |
| LLM | Groq / OpenAI / Ollama (auto-detected) |
| Agent | LangGraph |
| UI | Gradio 6.4 |
| Graph | D3.js v7 (force-directed, iframe srcdoc) |

## Project Structure

```
src/research_agent/
├── agents/        # LangGraph orchestration
├── db/            # ChromaDB vector store + embeddings
├── explorer/      # Knowledge Explorer (D3 graph)
├── models/        # LLM/embedding loaders
├── processors/    # PDF ingestion
├── tools/         # Academic search APIs
├── ui/            # Gradio interface
└── main.py        # Entry point
```

## API Rate Limits

| API | Rate Limit | Key Required |
|-----|------------|--------------|
| Semantic Scholar | 100 req/5min | No |
| OpenAlex | Very generous | No |
| DuckDuckGo | Reasonable | No |
| Unpaywall | Generous | Email only |
| Groq | Free tier | Yes (free) |

## Development

```bash
pip install -r requirements.txt
pytest tests/
```

## Docs

- [ROADMAP.md](docs/ROADMAP.md) — priorities and future ideas
- [DATA_SOURCES.md](docs/DATA_SOURCES.md) — API landscape
- [CHANGELOG.md](docs/CHANGELOG.md) — implementation history
- [SETUP.md](docs/SETUP.md) — environment setup

## Acknowledgements

With gratitude to **Britt Kramvig**, **Tone Huse** and **Berit Kristoffersen** — the researchers whose work inspired this tool and whose scholarship shapes its knowledge base.

Built with [Anthropic](https://anthropic.com)'s **Claude** and the open source communities behind LangGraph, ChromaDB, Gradio, and D3.js.

Made with care, love and joy by Rolf, Claude and Taiga :3

## License

MIT
