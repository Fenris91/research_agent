"""Reranker utilities for retrieval quality."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class RerankResult:
    """Represents a reranked item."""

    index: int
    score: float


class Reranker:
    """Cross-encoder reranker for query-document pairs."""

    def __init__(
        self, model_name: str = "BAAI/bge-reranker-base", device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers required for reranking. "
                    "Install with: pip install sentence-transformers"
                ) from exc

            self._model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Return reranked indices and scores for documents."""
        if not documents:
            return []

        self._load_model()

        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)

        indices = list(range(len(scores)))
        indices.sort(key=lambda i: float(scores[i]), reverse=True)

        if top_k is not None:
            indices = indices[:top_k]

        return [RerankResult(index=i, score=float(scores[i])) for i in indices]


def load_reranker_from_config(config: Dict[str, Any]) -> Optional[Reranker]:
    """Load reranker based on config dict."""
    embedding_config = config.get("embedding", {})
    reranker_config = embedding_config.get("reranker", {})

    if not reranker_config.get("enabled", False):
        return None

    model_name = reranker_config.get("model", "BAAI/bge-reranker-base")
    device = reranker_config.get("device")

    return Reranker(model_name=model_name, device=device)
