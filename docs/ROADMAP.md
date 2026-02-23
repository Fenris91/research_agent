# Roadmap

Active priorities and future ideas for the Research Agent.

## High Priority — Web Sources & LLM Integrations

- [x] **Auto-Save Web Results from Researcher Lookup** — save DuckDuckGo results to `web_sources`, "Save to KB" button next to each result
- [x] **Auto-Save Citation Abstracts** — persist cited/citing paper abstracts to KB via S2 batch API, auto-save toggle in Citation Explorer
- [x] **Additional Cloud LLM Providers** — Gemini, Mistral, xAI/Grok via OpenAI-compatible APIs
- [x] **Multi-Model Pipeline** — fast/cheap model for query classification, capable model for synthesis, configurable per task type via UI
- [x] **Perplexity Integration** — web-grounded answers with citations, switchable web search provider (DuckDuckGo/Perplexity/Tavily/Serper)
- [x] **CORE API Integration** — 300M+ open access papers, integrated into external search pipeline

## Medium Priority

- [x] **Notes Browser in KB Tab** — list, refresh, delete by ID
- [x] **Web Sources Browser in KB Tab** — list with URL/title/date, delete
- [ ] **Researcher Profile Persistence** — persist ResearcherRegistry to SQLite, link researchers to KB papers

## Low Priority

- [ ] **Evaluation Suite** — test retrieval and synthesis quality, benchmark embedding + reranker
- [ ] **Fine-tuning** — domain-specific model training
- [ ] **Collaborative Features** — share knowledge bases

## Ideas / Future

- [ ] Integration with Zotero
- [ ] PDF annotation support
- [ ] Multi-user support
- [ ] API endpoint for programmatic access
- [ ] Chrome extension for paper capture
- [ ] Specialized LLMs per task (Grok for current events, Perplexity for web-grounded, Claude for long docs, GPT-4o for figures)
- [ ] Agent-to-agent communication (fact-check agent, citation analysis agent)
