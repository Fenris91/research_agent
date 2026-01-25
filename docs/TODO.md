# Research Agent - TODO

Cross-reference task list for Claude, OpenCode, and VSCode.

**Last Updated**: January 25, 2026

---

## In Progress

- [ ] **PDF Processing** - Document processor scaffolded but not functional
  - Parse PDF text and structure
  - Extract metadata (title, authors, DOI)
  - Chunk documents for embedding

---

## Not Started

### High Priority

- [ ] **Wire Citation Explorer to UI** - Add tab or section for exploring citations
- [ ] **Knowledge Base Management** - Upload/delete papers in UI
- [ ] **Reranker** - Add BGE reranker for better retrieval quality

### Medium Priority

- [ ] **Data Analysis Tools** - Pandas/visualization integration
- [ ] **Export/Citation Management** - Zotero/BibTeX export
- [ ] **Search Filters in UI** - Filter by year, field, citation count

### Low Priority

- [ ] **Fine-tuning** - Domain-specific model training
- [ ] **Collaborative Features** - Share knowledge bases
- [ ] **Evaluation Suite** - Test retrieval and synthesis quality

---

## Completed

- [x] Academic Paper Search (Semantic Scholar, OpenAlex, Unpaywall)
- [x] Web Search (DuckDuckGo, Tavily, Serper)
- [x] Researcher Lookup
- [x] Vector Store (ChromaDB)
- [x] Embedding Model (BGE)
- [x] Research Agent (LangGraph workflow)
- [x] LLM Integration (Ollama + HuggingFace fallback)
- [x] Gradio UI with model selector
- [x] Citation Explorer
- [x] Unit Tests (22 passing)

---

## Bugs / Tech Debt

- [ ] Fix `datetime.utcnow()` deprecation warning in vector_store.py
- [ ] Add proper error handling for Semantic Scholar rate limits
- [ ] Consider caching for API responses

---

## Ideas / Future

- [ ] Integration with Zotero
- [ ] PDF annotation support
- [ ] Multi-user support
- [ ] API endpoint for programmatic access
- [ ] Chrome extension for paper capture
