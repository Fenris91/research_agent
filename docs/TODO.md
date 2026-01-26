# Research Agent - TODO

Cross-reference task list for Claude, OpenCode, and VSCode.

**Last Updated**: January 26, 2026

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

- [ ] **PDF Processing** - Basic pipeline working, needs validation
  - Validate PDF text/structure extraction on real PDFs (validated on sample PDFs in Downloads)
  - Refine metadata extraction (title, authors, DOI)
  - Tune chunking for embeddings
  - Added first-page title/author heuristics and text normalization
  - Added validation summary (title/authors/doi flags, chunk stats) and stronger DOI pattern

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
- [ ] **Reranker** - Add BGE reranker for better retrieval quality

### Medium Priority

- [ ] **Data Analysis Tools** - Pandas/visualization integration
- [ ] **Export/Citation Management** - Zotero/BibTeX export
- [ ] **Search Filters in UI** - Filter by year, field, citation count
- [ ] **Network Visualization** - Citation graph visualization (placeholder exists)

### Low Priority

- [ ] **Fine-tuning** - Domain-specific model training
- [ ] **Collaborative Features** - Share knowledge bases
- [ ] **Evaluation Suite** - Test retrieval and synthesis quality

---

## Completed

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

- [ ] Fix `datetime.utcnow()` deprecation warning in vector_store.py
- [ ] Add proper error handling for Semantic Scholar rate limits
- [ ] Consider caching for API responses
- [ ] Install pytest-timeout plugin for test timeouts
- [x] Fix test imports from `src.*` to `research_agent.*`

---

## Next Steps (Recommended Order)

1. **PDF Processing**
   - Implement document processor
   - Add to knowledge base workflow
   - Test with real PDFs

2. **Knowledge Base Management**
   - Refresh stats/browse list and delete flows in UI

3. **Reranker**
   - Add BGE reranker and re-enable skipped retrieval tests

4. **Network Visualization**
   - Implement citation graph rendering
   - Use existing `build_network_data()` output

---

## Ideas / Future

- [ ] Integration with Zotero
- [ ] PDF annotation support
- [ ] Multi-user support
- [ ] API endpoint for programmatic access
- [ ] Chrome extension for paper capture
