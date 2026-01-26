"""Tests for vector store reranking integration."""

from typing import Any, Dict

from research_agent.db.vector_store import ResearchVectorStore
from research_agent.models.reranker import RerankResult


class _FakeCollection:
    def __init__(self):
        self.query_calls = []

    def query(self, **kwargs):  # pragma: no cover - simple stub
        self.query_calls.append(kwargs)
        return {
            "ids": [["a", "b", "c"]],
            "documents": [["doc a", "doc b", "doc c"]],
            "metadatas": [[{"m": 1}, {"m": 2}, {"m": 3}]],
            "distances": [[0.1, 0.2, 0.3]],
        }


class _FakeReranker:
    def __init__(self, order):
        self.order = order

    def rerank(self, query: str, documents: list[str], top_k=None):
        # Return indices in provided order; ignore top_k for simplicity
        indices = self.order if top_k is None else self.order[:top_k]
        return [RerankResult(index=i, score=float(len(documents) - i)) for i in indices]


def _make_store_with_fake_coll(reranker=None, top_k=None) -> ResearchVectorStore:
    store = object.__new__(ResearchVectorStore)
    fake_coll = _FakeCollection()
    # Minimal attributes used by _get_collection/search
    store.papers = fake_coll
    store.notes = fake_coll
    store.web_sources = fake_coll
    store.reranker = reranker
    store.rerank_top_k = top_k
    store._get_collection = lambda name: fake_coll
    return store


def test_vector_store_uses_reranker_and_reorders_results():
    store = _make_store_with_fake_coll(reranker=_FakeReranker([2, 0, 1]))

    result = store.search(query_embedding=[0.1], query_text="q", n_results=3)

    assert result["ids"] == ["c", "a", "b"]
    assert result["documents"] == ["doc c", "doc a", "doc b"]
    assert result.get("rerank_scores") == [1.0, 3.0, 2.0]


def test_vector_store_respects_top_k_when_set():
    store = _make_store_with_fake_coll(reranker=_FakeReranker([1, 0, 2]), top_k=2)

    result = store.search(query_embedding=[0.1], query_text="q", n_results=3)

    assert result["ids"] == ["b", "a"]
    assert result["documents"] == ["doc b", "doc a"]
    assert result.get("rerank_scores") == [2.0, 3.0]
