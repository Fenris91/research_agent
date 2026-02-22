# Changelog

Implementation history for the Research Agent project.

## February 22, 2026
- **SQLite Metadata Index** — dual-write/delegated-read layer (`KBMetadataStore`) alongside ChromaDB; `list_papers()`, `get_stats()`, `list_notes()` now hit SQLite instead of scanning all chunks ([rationale](STORAGE_PHILOSOPHY.md))
- **Auto-rebuild** — if SQLite index is empty but ChromaDB has data, rebuilds automatically on startup
- **23 new tests** for `KBMetadataStore` (CRUD, pagination, stats, clear)
- **asyncio fix** — `get_event_loop()` → `get_running_loop()` in research agent

## January 28, 2026
- **Cloud LLM Auto-Detection** — auto-detect API keys (OpenAI, Groq, OpenRouter), Groq free tier default, falls back to Ollama/HuggingFace, `provider: "auto"` in config
- **Multi-Collection Search** — papers, notes, web sources searched together, results merged by relevance, source type shown in responses
- **Context-Aware Chat** — pass researcher/paper context from KB to chat, 15% relevance boost for selected researcher's papers
- **Notes & Web Sources UI** — "Add Research Note" and "Add Web Source" sections in KB tab, both searchable in chat

## January 27, 2026
- **API Response Caching** — `TTLCache` + `PersistentCache` in `utils/cache.py`, configurable TTL (searches 1hr, paper details 6hr, OA URLs 24hr), negative result caching
- **Semantic Scholar Rate Limiting** — `RateLimiter` sliding window (95 calls/5min), `retry_with_backoff()` with exponential backoff + jitter for 429/503/504
- **Ingestion Manager** — `_assess_relevance()`, `check_duplicate()`, `process_decision()`, `_ingest_source()` all implemented
- **LangGraph Workflow** — all 5 nodes functional: `_understand_query`, `_search_local`, `_search_external`, `_synthesize`, `_offer_ingestion`
- **Search Filters** — year range + min citations in Chat and KB tabs
- **Network Visualization** — citation graph with networkx/matplotlib
- **BibTeX Export** — export papers from KB to .bib file
- **Data Analysis Tools** — column selector, multiple plot types, PNG download, date parsing, pivot tables

## January 26, 2026
- **Citation Explorer Fixes** — fixed duplicate `_get_cited_papers`, indentation issues, `self.search.s2` references, API field names
- **Citation Explorer UI** — dedicated Gradio tab with async `explore_citations`, summary and tables
- **Test Suite** — 44 tests passing: unit (mocked APIs), integration (rate limits), error handling, UI components, analytics
- **Test Config** — `conftest.py` fixtures, API limits (5 results/call, 0.5s delays), test paper IDs (BERT, Geertz, Lefebvre)

## Previous
- Academic Paper Search (Semantic Scholar, OpenAlex, Unpaywall)
- Web Search (DuckDuckGo, Tavily, Serper)
- Researcher Lookup
- Vector Store (ChromaDB)
- Embedding Model (BGE)
- Research Agent (LangGraph workflow)
- LLM Integration (Ollama + HuggingFace fallback)
- Gradio UI with model selector
- Citation Explorer
- Package Migration (`src/` → `src/research_agent/`)
- PDF Processing pipeline (validated on real PDFs)
- Knowledge Base Management (upload/delete/browse in UI)

## Bugs / Tech Debt Resolved
- Fixed `datetime.utcnow()` deprecation (timezone-aware datetime)
- Added rate limiting + caching for Semantic Scholar
- Fixed test imports from `src.*` to `research_agent.*`
- Installed pytest-timeout plugin
