# Research Agent - TODO

Cross-reference task list for Claude, OpenCode, and VSCode.

**Last Updated**: January 27, 2026

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

### High Priority

- [x] **Complete Package Migration** - Finish moving all modules to `research_agent` package
- [x] **Wire Citation Explorer to UI** - Add tab or section for exploring citations (Gradio tab + handlers wired)
- [x] **Reranker** - Add BGE reranker for better retrieval quality (helper, vector store wiring, and tests added)

### Medium Priority

- [x] **Data Analysis Tools** - Full suite with column selector, plot types, time series, pivot tables
- [x] **Export/Citation Management** - BibTeX export from KB
- [x] **Search Filters in UI** - Filter by year, field, citation count (Chat + KB tabs)
- [x] **Network Visualization** - Citation graph using networkx/matplotlib

### Low Priority

- [ ] **Fine-tuning** - Domain-specific model training
- [ ] **Collaborative Features** - Share knowledge bases
- [ ] **Evaluation Suite** - Test retrieval and synthesis quality

---

## Completed

### January 27, 2026
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
- [ ] Add proper error handling for Semantic Scholar rate limits
- [ ] Consider caching for API responses
- [ ] Install pytest-timeout plugin for test timeouts
- [x] Fix test imports from `src.*` to `research_agent.*`

---

## Next Steps (Recommended Order)

1. **Evaluation Suite**
   - Test retrieval quality
   - Benchmark embedding + reranker performance

2. **API Response Caching**
   - Cache Semantic Scholar / OpenAlex responses
   - Reduce rate limit issues

---

## Ideas / Future

- [ ] Integration with Zotero
- [ ] PDF annotation support
- [ ] Multi-user support
- [ ] API endpoint for programmatic access
- [ ] Chrome extension for paper capture
