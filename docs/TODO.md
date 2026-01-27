# Research Agent - TODO

Cross-reference task list for Claude, OpenCode, and VSCode.

**Last Updated**: January 27, 2026
**Last Verified**: January 27, 2026

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

### Low Priority

- [ ] **Fine-tuning** - Domain-specific model training
- [ ] **Collaborative Features** - Share knowledge bases
- [ ] **Evaluation Suite** - Test retrieval and synthesis quality

---

## Completed

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
