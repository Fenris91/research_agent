# Stabilization Tasks

Actionable work items derived from [Stabilization & Cohesion Plan]. Each task is
self-contained and scoped for a single session. Tasks are ordered by dependency —
earlier tasks unblock later ones.

**Status key:** `[ ]` pending · `[~]` in progress · `[x]` done

---

## Phase 1: Baseline Stability (first 30 days)

### 1.1 Consolidate config loading into one utility
**Files:** new `src/research_agent/utils/config.py`, then update callers
**Problem:** Four independent `_load_config()` implementations (main.py, app.py,
citation_explorer.py component, lookup_researchers.py script). All use relative
`Path("configs/config.yaml")` which breaks if CWD ≠ project root.
**Task:**
- [ ] Create `src/research_agent/utils/config.py` with a single `load_config()`
  that resolves the config path relative to the package root (use `__file__`
  ancestry or a `ROOT_DIR` sentinel), caches the result, and returns `dict`.
- [ ] Replace the four inline `_load_config()` implementations with imports from
  the new module.
- [ ] Add a unit test in `tests/test_config_loader.py` that verifies loading
  from a temp config file and caching behavior.

### 1.2 Add missing pytest markers and enforce marker discipline
**Files:** `pytest.ini`, `tests/conftest.py`
**Problem:** `slow` and `ui` markers are registered in conftest but never used.
`pytest.ini` is incomplete. No `network` marker exists.
**Task:**
- [ ] Update `pytest.ini` to register all markers: `unit`, `integration`, `ui`,
  `slow`, `network`, `asyncio`, `timeout`.
- [ ] Remove the duplicate `pytest_configure` marker registration from
  `conftest.py` (pytest.ini is the canonical source).
- [ ] Add `@pytest.mark.unit` to all fully-mocked test classes.
- [ ] Add `@pytest.mark.network` to tests that hit real APIs (alongside existing
  `integration` marker).
- [ ] Add a default `addopts` in `pytest.ini`: `-m "not integration and not slow"`.
- [ ] Verify `pytest` (bare) runs only unit tests, `pytest -m integration` runs
  API tests.

### 1.3 Remove redundant sys.path injection in test_ui_components.py
**Files:** `tests/test_ui_components.py`
**Problem:** Lines 358-361 duplicate the `sys.path` injection already in
`conftest.py`.
**Task:**
- [ ] Delete the redundant `sys.path.insert` block in `test_ui_components.py`.
- [ ] Verify tests still pass.

### 1.4 Un-skip or delete dead test classes in test_core.py
**Files:** `tests/test_core.py`
**Problem:** 6 of 8 test classes are `@pytest.mark.skip(reason="migration")`.
They either test current code (un-skip + fix) or dead code (delete).
**Task:**
- [ ] For each skipped class, check if the module it imports still exists.
- [ ] If the class tests live code: remove `@skip`, fix imports, get it passing.
- [ ] If the class tests removed code: delete the class entirely.
- [ ] Target: zero `@skip` markers remaining in `test_core.py`.

### 1.5 Standardize error handling in tools/
**Files:** `src/research_agent/tools/citation_explorer.py`,
`src/research_agent/tools/researcher_registry.py`
**Problem:**
- `citation_explorer.py` `find_highly_connected` uses `print()` instead of logger.
- `suggest_related` has silent `except Exception: continue`.
- `researcher_registry.py` has multiple `except Exception: pass` blocks.
**Task:**
- [ ] Replace `print()` calls in `citation_explorer.py` with `logger.warning()`.
- [ ] Add `logger.debug()` to the silent `except` in `suggest_related`.
- [ ] Add `logger.debug()` to the silent lock-operation `except` blocks in
  `researcher_registry.py` (keep the `pass` — just add visibility).
- [ ] No behavior changes, just logging.

### 1.6 Add retry to OpenAlex calls (parity with Semantic Scholar)
**Files:** `src/research_agent/tools/academic_search.py`,
`src/research_agent/tools/researcher_lookup.py`
**Problem:** `search_openalex` and `fetch_author_papers_openalex` make single
HTTP calls with no retry. Semantic Scholar equivalents use `retry_with_backoff`.
**Task:**
- [ ] Wrap the `.get()` call in `search_openalex` with `retry_with_backoff(
  max_retries=2, base_delay=1.0, retry_on=(429, 503, 504))`.
- [ ] Do the same for `fetch_author_papers_openalex` in `researcher_lookup.py`
  and `search_openalex_author` in the same file.
- [ ] No new tests required (existing integration tests cover these paths).

---

## Phase 2: Cohesion and Reliability (days 31-60)

### 2.1 Create shared fixtures for papers, researchers, citations
**Files:** new `tests/fixtures/` directory, update `conftest.py`
**Problem:** Test files build their own mock paper/researcher/citation objects
inline. No reuse.
**Task:**
- [ ] Create `tests/fixtures/__init__.py` and `tests/fixtures/data.py`.
- [ ] Move the paper/citation/researcher fixture data from `conftest.py` and
  `test_config.py` into `fixtures/data.py` as factory functions:
  `make_paper(**overrides)`, `make_researcher(**overrides)`,
  `make_citation_network(**overrides)`.
- [ ] Update `conftest.py` fixtures to delegate to the factories.
- [ ] Update at least 2 test files to use the new factories directly.

### 2.2 Add adapter contract tests for external APIs
**Files:** new `tests/test_api_contracts.py`
**Problem:** No tests verify the shape of real API responses. If S2 or OpenAlex
change their response schema, we find out at runtime.
**Task:**
- [ ] Create `tests/test_api_contracts.py` with `@pytest.mark.integration`
  and `@pytest.mark.network`.
- [ ] Add one test per provider that makes a single known-good request and
  asserts on the shape of the response (required keys, types):
  - Semantic Scholar: paper detail for BERT paper
  - OpenAlex: work detail for a known DOI
  - Semantic Scholar: author search
  - OpenAlex: author search
- [ ] These tests are slow/network — they run only on `pytest -m integration`.

### 2.3 Unify paper ID naming across domain models
**Files:** `src/research_agent/tools/citation_explorer.py`,
`src/research_agent/tools/researcher_lookup.py`
**Problem:** `Paper.id`, `AuthorPaper.paper_id`, `CitationPaper.paper_id`,
`CitationPaperRecord.paper_id` — inconsistent naming for the same concept.
**Task:**
- [ ] Rename `Paper.id` → `Paper.paper_id` (it shadows the builtin).
- [ ] Update all callers of `Paper.id` (grep for `.id` usage on Paper objects).
- [ ] This is the minimal unification — full model consolidation is Phase 3.
- [ ] Add/update tests to verify the field name works.

### 2.4 Add smoke test for full query → synthesis path
**Files:** new `tests/test_smoke.py`
**Problem:** No test covers the core user journey: query → retrieve → synthesize.
**Task:**
- [ ] Create `tests/test_smoke.py` with a single `@pytest.mark.unit` test.
- [ ] Mock vector store to return 2 fake chunks, mock LLM to return a canned
  response.
- [ ] Call `agent.query("test question")` and assert the response contains
  content and source references.
- [ ] This is the "if this breaks, nothing works" test.

### 2.5 Add web search retry logic
**Files:** `src/research_agent/tools/web_search.py`
**Problem:** No retry logic in any web search method. DuckDuckGo especially can
be flaky.
**Task:**
- [ ] Add a simple retry (max 2 attempts, 1s delay) around the DuckDuckGo
  executor call.
- [ ] Wrap Tavily and Serper `.get()` calls with `retry_with_backoff` from
  `academic_search.py` (import it or extract to `utils/`).
- [ ] Consider: extract `retry_with_backoff` to `src/research_agent/utils/retry.py`
  for reuse across modules. If so, update `academic_search.py` imports.

---

## Phase 3: Quality Gates (days 61-90)

### 3.1 Add CI quality gate workflow
**Files:** new `.github/workflows/test.yml`
**Task:**
- [ ] Create a GitHub Actions workflow that runs on PR:
  - `pytest -m "not integration and not slow"` (unit tests)
  - Exit code gates the merge.
- [ ] Add a scheduled nightly job:
  - `pytest -m integration --timeout=120`
  - Allowed to fail (informational), but reports status.
- [ ] Add `ruff check src/ tests/` as a lint step.

### 3.2 Add coverage threshold
**Files:** `pytest.ini` or `pyproject.toml`, CI workflow
**Task:**
- [ ] Add `pytest-cov` to dev dependencies.
- [ ] Set initial coverage floor at whatever the current baseline is (measure
  first, then set threshold 2% below that).
- [ ] Add `--cov=research_agent --cov-fail-under=X` to the CI unit test step.
- [ ] Ratchet up by 1% per month.

### 3.3 Consolidate domain models into shared schema module
**Files:** new `src/research_agent/models/schema.py`, update all importers
**Problem:** Four representations of "paper" with inconsistent fields.
**Task:**
- [ ] Create `src/research_agent/models/schema.py` with canonical `BasePaper`
  dataclass containing shared fields (paper_id, title, year, citation_count,
  authors).
- [ ] Have `Paper`, `AuthorPaper`, `CitationPaper` either subclass `BasePaper`
  or be replaced by it with optional fields.
- [ ] Update all callers. This is the largest refactor — do it last.
- [ ] Add migration tests that verify old field access patterns still work.

### 3.4 Add observability: request IDs and timing spans
**Files:** `src/research_agent/utils/observability.py`, tool modules
**Task:**
- [ ] Create a context-var-based request ID generator.
- [ ] Add a `@timed` decorator that logs function name + duration.
- [ ] Apply to: API calls in tools/, agent query path, ingestion pipeline.
- [ ] Log format: `[req_id] module.function took Xms`.

---

## Quick Wins (can be done anytime, no dependencies)

- [ ] Delete the redundant `sys.path` block in `test_ui_components.py` (1.3)
- [ ] Replace `print()` with `logger` in `citation_explorer.py` (1.5)
- [ ] Add `logger.debug` to silent excepts in `researcher_registry.py` (1.5)
- [ ] Register all markers in `pytest.ini` (1.2)
