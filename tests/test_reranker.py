"""Tests for the reranker wrapper."""

from research_agent.models.reranker import Reranker, load_reranker_from_config


class _StubCrossEncoder:
    """Lightweight stub replacing the real cross-encoder predict."""

    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):  # pragma: no cover - trivial pass-through
        # ignore pairs content; return preset scores
        return self._scores


def test_rerank_sorts_and_limits():
    reranker = Reranker(model_name="stub-model")

    # Monkeypatch model loader to avoid pulling real weights
    reranker._load_model = lambda: setattr(
        reranker, "_model", _StubCrossEncoder([0.2, 0.9, 0.1])
    )

    docs = ["doc a", "doc b", "doc c"]
    results = reranker.rerank("query", docs, top_k=2)

    assert [r.index for r in results] == [1, 0]
    assert results[0].score == 0.9


def test_load_reranker_from_config():
    cfg_disabled = {"embedding": {"reranker": {"enabled": False}}}
    assert load_reranker_from_config(cfg_disabled) is None

    cfg_enabled = {
        "embedding": {"reranker": {"enabled": True, "model": "BAAI/bge-reranker-base"}}
    }
    rr = load_reranker_from_config(cfg_enabled)
    assert isinstance(rr, Reranker)
    assert rr.model_name == "BAAI/bge-reranker-base"
