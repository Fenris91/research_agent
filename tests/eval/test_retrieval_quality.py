"""Retrieval quality tests using real embeddings on the 7-paper seed corpus.

All tests require GPU (real BAAI/bge-base-en-v1.5 embedder).
"""

from typing import List

import pytest

from tests.eval.corpus import RETRIEVAL_GOLD, QueryCase
from tests.eval.conftest import retrieve_paper_ids


# ── Metric helpers ─────────────────────────────────────────────────────


def recall_at_k(results: List[str], relevant: List[str], k: int) -> float:
    """Fraction of relevant paper_ids found in top-k results."""
    if not relevant:
        return 1.0
    top_k = set(results[:k])
    return len(top_k & set(relevant)) / len(relevant)


def reciprocal_rank(results: List[str], relevant: List[str]) -> float:
    """1/rank of the first relevant result; 0 if none found."""
    for rank, pid in enumerate(results, 1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


# ── Parametrized test cases ───────────────────────────────────────────

_single_relevant = [c for c in RETRIEVAL_GOLD if c.highly_relevant]
_multi_relevant = [c for c in RETRIEVAL_GOLD if len(c.relevant) > 1]
_with_distractors = [c for c in RETRIEVAL_GOLD if c.distractors and c.highly_relevant]


@pytest.mark.gpu
@pytest.mark.eval
@pytest.mark.parametrize("case", _single_relevant, ids=lambda c: c.query_id)
def test_recall_at_1(populated_store, case: QueryCase):
    """The highly_relevant paper must appear at rank 1."""
    store, embedder = populated_store
    results = retrieve_paper_ids(store, embedder, case.query, k=5)
    rr = reciprocal_rank(results, case.highly_relevant)
    assert rr >= 1.0, (
        f"[{case.query_id}] Expected {case.highly_relevant[0]!r} at rank 1, "
        f"got: {results}"
    )


@pytest.mark.gpu
@pytest.mark.eval
@pytest.mark.parametrize("case", _multi_relevant, ids=lambda c: c.query_id)
def test_recall_at_3_multi_paper(populated_store, case: QueryCase):
    """For multi-paper queries, at least half of relevant papers in top 3."""
    store, embedder = populated_store
    results = retrieve_paper_ids(store, embedder, case.query, k=5)
    r3 = recall_at_k(results, case.relevant, k=3)
    assert r3 >= 0.5, (
        f"[{case.query_id}] Recall@3={r3:.2f}, got {results}, wanted {case.relevant}"
    )


@pytest.mark.gpu
@pytest.mark.eval
@pytest.mark.parametrize("case", _with_distractors, ids=lambda c: c.query_id)
def test_distractor_suppression(populated_store, case: QueryCase):
    """Distractors from other clusters should not outrank the relevant paper."""
    store, embedder = populated_store
    results = retrieve_paper_ids(store, embedder, case.query, k=5)
    target = case.highly_relevant[0]
    if target not in results:
        pytest.skip("Relevant paper not found at all — covered by recall test")
    relevant_rank = results.index(target) + 1
    for distractor in case.distractors:
        if distractor in results:
            distractor_rank = results.index(distractor) + 1
            assert distractor_rank > relevant_rank, (
                f"[{case.query_id}] Distractor {distractor!r} (rank {distractor_rank}) "
                f"outranks {target!r} (rank {relevant_rank}). "
                f"Full results: {results}"
            )


@pytest.mark.gpu
@pytest.mark.eval
def test_mrr_above_threshold(populated_store):
    """MRR across all single-relevant queries must be >= 0.70."""
    store, embedder = populated_store
    cases = [c for c in RETRIEVAL_GOLD if len(c.relevant) == 1 and not c.expect_empty]
    scores = []
    for case in cases:
        results = retrieve_paper_ids(store, embedder, case.query, k=5)
        scores.append(reciprocal_rank(results, case.relevant))
    mrr = sum(scores) / len(scores) if scores else 0.0
    print(f"\nMRR over {len(scores)} queries: {mrr:.3f}")
    for case, score in zip(cases, scores):
        print(f"  {case.query_id}: RR={score:.2f}")
    assert mrr >= 0.70, f"MRR {mrr:.3f} below threshold 0.70"


# ── Edge case (CPU-safe) ──────────────────────────────────────────────


@pytest.mark.eval
def test_empty_corpus_returns_nothing(tmp_path):
    """Search on empty store returns empty results."""
    from research_agent.db.vector_store import ResearchVectorStore

    store = ResearchVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        metadata_store_path=None,
    )
    result = store.search(query_embedding=[0.0] * 768, collection="papers", n_results=5)
    assert result["ids"] == []
    assert result["documents"] == []
