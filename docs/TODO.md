# Research Agent - TODO

Cross-reference task list for Claude, OpenCode, and VSCode.

**Last Updated**: January 28, 2026
**Last Verified**: January 28, 2026

---

## In Progress

- [x] **Package Migration** - Moving modules from `src/` to `src/research_agent/`
  - [x] citation_explorer.py - Fixed indentation, removed duplicates, tested
  - [x] academic_search.py - Working
  - [x] vector_store.py - Migrated
  - [x] embeddings.py - Migrated
  - [x] llm_utils.py - Migrated
  - [x] research_agent.py - Migrated
  - [x] web_search.py - Migrated
  - [x] researcher_lookup.py - Migrated

- [x] **PDF Processing** - Pipeline working and validated
  - Validated on real PDFs (Perspectives.pdf, konigsberg_graph_theory_naval.pdf)
  - Title/author extraction working (heuristics for first-page analysis)
  - DOI detection from content
  - Validation summary with flags (title/authors/doi/abstract/refs)
  - Chunking with configurable size/overlap

- [x] **Knowledge Base Management** - Upload/delete papers in UI
  - Process document uploads into vector store
  - Refresh stats and browse list
  - Delete papers by ID
  - Auto-refresh stats/table on UI load

---

## Not Started

### High Priority - Web Sources & LLM Integrations

- [ ] **Auto-Save Web Results from Researcher Lookup**
  - When researcher lookup fetches web results (DuckDuckGo), offer to save to `web_sources`
  - Add "Save to KB" button next to each web result in the Researcher Lookup tab
  - Optionally auto-save highly relevant results (configurable threshold)
  - Benefits: Web context about researchers becomes searchable in chat

- [ ] **Auto-Save Citation Abstracts**
  - Citation Explorer fetches paper abstracts but doesn't persist them
  - Add option to auto-ingest cited/citing papers with abstracts to KB
  - Store relationship metadata (cites/cited_by) in paper metadata
  - Benefits: Build knowledge graph automatically while exploring citations

- [ ] **Additional Cloud LLM Providers**
  - [ ] **Grok (xAI)** - OpenAI-compatible API at `https://api.x.ai/v1`
    - Env var: `XAI_API_KEY`
    - Models: grok-beta, grok-2
    - Good for: Real-time knowledge, reasoning
  - [ ] **Google Gemini** - via google-generativeai SDK
    - Env var: `GOOGLE_API_KEY`
    - Models: gemini-pro, gemini-1.5-flash (free tier!)
    - Good for: Long context (1M tokens), multimodal
  - [ ] **Perplexity** - OpenAI-compatible with built-in web search
    - Env var: `PERPLEXITY_API_KEY`
    - Models: llama-3.1-sonar-small-128k-online
    - Good for: Research queries with live web citations
  - [ ] **Mistral** - OpenAI-compatible
    - Env var: `MISTRAL_API_KEY`
    - Models: mistral-small, mistral-medium
    - Good for: Fast, cheap, good quality

- [ ] **Multi-Model Pipeline**
  - Use fast/cheap model for query classification (Groq/llama-3.1-8b)
  - Use capable model for synthesis (GPT-4o/Claude)
  - Configurable in config.yaml per task type
  - Benefits: Cost optimization, speed for simple tasks

- [ ] **Perplexity Integration for Research Queries**
  - Perplexity's API returns answers with web citations
  - Could replace/augment DuckDuckGo for web search
  - Auto-extract and save cited sources to web_sources
  - Benefits: Higher quality web research with proper citations

### Medium Priority

- [ ] **Notes Browser in KB Tab**
  - List and manage saved research notes
  - Edit/delete notes
  - Filter by tags

- [ ] **Web Sources Browser in KB Tab**
  - List and manage saved web sources
  - Show URL, title, date added
  - Delete web sources

- [ ] **Researcher Profile Persistence**
  - Currently ResearcherRegistry is session-scoped
  - Persist researcher profiles to SQLite (like citations.sqlite)
  - Link researchers to their papers in KB
  - Benefits: Build researcher knowledge base over time

### Low Priority

- [ ] **Fine-tuning** - Domain-specific model training
- [ ] **Collaborative Features** - Share knowledge bases
- [ ] **Evaluation Suite** - Test retrieval and synthesis quality

---

## Completed

### January 28, 2026
- [x] **Cloud LLM Auto-Detection** - No local model required:
  - Auto-detect available API keys (OpenAI, Groq, OpenRouter)
  - Groq free tier recommended for getting started
  - Falls back to Ollama → HuggingFace if no cloud keys
  - `provider: "auto"` as default in config.yaml
- [x] **Multi-Collection Search** - Chat searches all KB collections:
  - Papers, research notes, and web sources searched together
  - Results merged and sorted by relevance
  - Source type shown in responses (Knowledge Base, Research Note, Web Source)
- [x] **Context-Aware Chat** - Use current selection:
  - Pass researcher/paper context from KB to chat
  - Boost papers by selected researcher (15% relevance boost)
  - Boost papers by same authors as selected paper
  - Context shown in LLM prompt for better synthesis
- [x] **Notes & Web Sources UI** - New KB tab sections:
  - "Add Research Note" - title, tags, content
  - "Add Web Source" - URL, title, content
  - Both now searchable in chat

### January 27, 2026
- [x] **API Response Caching** - Reduces API calls and rate limit issues:
  - New `utils/cache.py` module with `TTLCache` and `PersistentCache`
  - Thread-safe in-memory caching with automatic cleanup
  - Configurable TTL: searches (1hr), paper details (6hr), OA URLs (24hr)
  - Negative result caching to avoid repeated failed lookups
  - Cache stats available via `get_cache_stats()`
- [x] **Semantic Scholar Rate Limiting** - Proper rate limit handling:
  - `RateLimiter` class: sliding window token bucket (95 calls/5min)
  - `retry_with_backoff()`: exponential backoff with jitter for 429/503/504
  - Applied to academic_search.py, citation_explorer.py, researcher_lookup.py
- [x] **Ingestion Manager Phase 6** - All methods now implemented:
  - `_assess_relevance()` - LLM-based relevance scoring
  - `check_duplicate()` - KB duplicate detection by DOI, paper_id, and title similarity
  - `process_decision()` - User decision handling (all/none/numbers)
  - `_ingest_source()` - Source ingestion to vector store
- [x] **LangGraph Workflow Implementation** - All 5 nodes now functional:
  - `_understand_query()` - Query classification (literature_review, factual, analysis, general)
  - `_search_local()` - Vector store search with filters and reranker
  - `_search_external()` - Semantic Scholar, OpenAlex, web search
  - `_synthesize()` - Build prompts and generate structured responses
  - `_offer_ingestion()` - Suggest high-quality papers for KB
- [x] **Search Filters in UI** - Year range and min citations filters in Chat and KB tabs
- [x] **Network Visualization** - Citation graph rendering with networkx/matplotlib
- [x] **PDF Processing Validation** - Validated on real academic PDFs (Perspectives.pdf, etc.)
- [x] **BibTeX Export** - Export papers from KB to .bib file
- [x] **Data Analysis Tools** - Enhanced with:
  - Column selector dropdown (auto-populated on file upload)
  - Multiple plot types (histogram, box, bar, line, scatter)
  - Download plot as PNG
  - Auto date parsing for time series
  - Pivot table analysis with group by

### January 26, 2026
- [x] **Citation Explorer Code Fixes**
  - Fixed duplicate `_get_cited_papers` method
  - Fixed indentation issues (methods outside class)
  - Removed `self.search.s2` references
  - Fixed API field names (`citingPaper`, `citedPaper`)

- [x] **Citation Explorer UI Wiring**
  - Added dedicated Gradio tab and event handlers
  - Uses async `explore_citations` with summary and tables

- [x] **Comprehensive Test Suite** (44 tests passing)
  - Unit tests with mocked APIs
  - Integration tests with rate limits
  - Error handling tests
  - UI component tests
  - Analytics tests

- [x] **Test Configuration**
  - `tests/conftest.py` - Pytest fixtures
  - `tests/test_config.py` - API limits (5 results/call, 0.5s delays)
  - Test paper IDs (BERT, Geertz, Lefebvre)

### Previous
- [x] Academic Paper Search (Semantic Scholar, OpenAlex, Unpaywall)
- [x] Web Search (DuckDuckGo, Tavily, Serper)
- [x] Researcher Lookup
- [x] Vector Store (ChromaDB)
- [x] Embedding Model (BGE)
- [x] Research Agent (LangGraph workflow)
- [x] LLM Integration (Ollama + HuggingFace fallback)
- [x] Gradio UI with model selector
- [x] Citation Explorer

---

## Bugs / Tech Debt

- [x] Fix `datetime.utcnow()` deprecation warning in vector_store.py (already uses timezone-aware datetime)
- [x] Add proper error handling for Semantic Scholar rate limits
  - Added `RateLimiter` class with sliding window token bucket
  - Added `retry_with_backoff()` with exponential backoff + jitter
  - Applied to: academic_search.py, citation_explorer.py, researcher_lookup.py
- [x] Add caching for API responses
  - Created `utils/cache.py` with `TTLCache` and `PersistentCache` classes
  - In-memory caching with configurable TTL (no external dependencies)
  - Cache searches (1hr), paper details (6hr), OA URLs (24hr), researcher data (24hr)
  - Applied to: academic_search.py, researcher_lookup.py
- [x] Install pytest-timeout plugin for test timeouts
- [x] Fix test imports from `src.*` to `research_agent.*`

---

## Next Steps (Recommended Order)

1. **Evaluation Suite**
   - Test retrieval quality
   - Benchmark embedding + reranker performance

---

## Ideas / Future

- [ ] Integration with Zotero
- [ ] PDF annotation support
- [ ] Multi-user support
- [ ] API endpoint for programmatic access
- [ ] Chrome extension for paper capture
- [ ] **Specialized LLMs for Research Tasks**:
  - Grok for real-time/current events research
  - Perplexity for web-grounded answers with citations
  - Claude for long document analysis
  - GPT-4o for multimodal (analyze figures in papers)
- [ ] **Agent-to-Agent Communication**:
  - Research agent could call other specialized agents
  - E.g., "fact-check agent" using Perplexity
  - "Citation analysis agent" for bibliometrics
