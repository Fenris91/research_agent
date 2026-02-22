# Research Agent - Claude Code Instructions

## Project Vision
A **personal research memory** for social sciences researchers.
Core loop: Ingest papers → Build knowledge base → Query semantically → Get cited answers.
Visual exploration via a D3.js knowledge graph ("Knowledge Explorer").

## Tech Stack
- Python 3.11, RTX 5090 (32GB VRAM)
- Embeddings: `BAAI/bge-base-en-v1.5` (768 dims) — **DO NOT CHANGE**, must match existing DB
- Vector DB: ChromaDB at `./data/chroma_db`
- LLM: Groq free tier (default), OpenAI, Ollama — auto-detected
- Agent: LangGraph
- UI: Gradio 6.4 (dark theme, fullscreen adaptive layout)
- Graph: D3.js v7 force-directed graph in iframe (srcdoc)

## Code Structure
```
src/research_agent/
├── agents/        # LangGraph orchestration
├── db/            # ChromaDB vector store + embeddings
├── explorer/      # Knowledge Explorer (D3 graph)
│   ├── graph_builder.py   # Node/edge construction (researcher, paper, field, domain)
│   ├── renderer.py        # Jinja2 → iframe srcdoc
│   ├── mock_data.py       # Dev seed data (social sciences)
│   └── templates/explorer.html  # D3 visualization
├── models/        # LLM/embedding loaders
├── processors/    # PDF ingestion
├── tools/         # Academic search APIs (Semantic Scholar, OpenAlex)
├── ui/            # Gradio interface (app.py is the main UI)
└── main.py        # Entry point + provider auto-detection
configs/
├── config.yaml              # Main config
└── field_domain_mapping.json # Field→domain mapping (auto-expanded by LLM)
scripts/
└── seed_test_data.py        # Seeds David Harvey papers into KB
```

## Running
```bash
conda activate llm311
python -m research_agent.main --mode ui
# Auto-finds available port starting at 7860
```

## Knowledge Explorer
- **Dual-mode**: Researcher view (single researcher + papers) / KB view (all papers in knowledge base)
- **Node types**: researcher (24), paper (7–19, sized by citations), field (16), domain (40), query (18)
- **Edge types**: authorship, citation, semantic (similarity score), field_membership, domain_mapping
- **Layers** (SOC / AUTH / CHAT): emphasis modes that fade/highlight different node groups
  - SOC = structural scaffold (fields + domains), AUTH = authorship graph, CHAT = keyword-matched nodes
- **OA indicators**: green ring (open), yellow (preprint), grey dashed (paywalled)
- **Detail panel**: click any node → right-side panel with type-specific info + action buttons
- **Search bar**: real-time label filtering across all nodes

## Key Architectural Patterns
- **ExplorerRenderer**: wraps D3 HTML in `<iframe srcdoc>` to bypass Gradio's innerHTML script stripping
- **GraphBuilder**: fluent API — `add_researcher()`, `add_paper()`, `add_field()`, `build_structural_context()`
- **Structural Context**: fields extracted from metadata → mapped to 8 societal domains via JSON config + LLM fallback
- **Color inheritance**: papers inherit color from their researcher via authorship edges
- **Layer sync**: two-way postMessage between Gradio top bar buttons and iframe D3 layer buttons
- **ResearcherRegistry**: singleton for sharing researcher profiles across UI tabs
- **CSS**: minimal `!important` — uses elem_id selectors; `footer_links=[]` hides footer natively
- **Port handling**: auto-detects and kills stale research_agent processes, increments port if busy

## Docs
- [ROADMAP.md](docs/ROADMAP.md) — active priorities and future ideas
- [DATA_SOURCES.md](docs/DATA_SOURCES.md) — API landscape (OpenAlex, Semantic Scholar, CrossRef, Unpaywall, CORE)
- [CHANGELOG.md](docs/CHANGELOG.md) — implementation history by date
- [SETUP.md](docs/SETUP.md) — environment setup and troubleshooting

## Acknowledgements
Made with care, love and joy by Rolf, Claude and Taiga :3
