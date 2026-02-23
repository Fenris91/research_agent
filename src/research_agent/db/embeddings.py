"""
Embedding Models for Research Agent

Provides text embeddings for semantic search.
Uses sentence-transformers for local embeddings.
"""

import logging
import os
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Embedding model wrapper for generating text embeddings.

    Uses sentence-transformers models for high-quality embeddings.

    Example:
        embedder = EmbeddingModel()
        embedding = embedder.embed("What is urban gentrification?")
        embeddings = embedder.embed_batch(["text1", "text2", "text3"])
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name. Recommended:
                - "BAAI/bge-base-en-v1.5" (fast, good quality)
                - "BAAI/bge-large-en-v1.5" (slower, better quality)
                - "sentence-transformers/all-MiniLM-L6-v2" (very fast, smaller)
            device: Device to run on ("cuda", "cpu", or None for auto)
            normalize: Whether to normalize embeddings (recommended for cosine similarity)
        """
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self._device = device

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. "
                    "Install with: pip install sentence-transformers"
                )

            device = self._device or os.environ.get("EMBEDDING_DEVICE")
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=device
                )
            except (RuntimeError, AssertionError):
                if device and device != "cpu":
                    logger.warning(
                        f"Failed to load embeddings on '{device}', falling back to CPU"
                    )
                    self._model = SentenceTransformer(
                        self.model_name,
                        device="cpu"
                    )
                else:
                    raise
            logger.info(f"Loaded embedding model: {self.model_name} on {self._model.device}")

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        self._load_model()
        embedding = self._model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return embedding.tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        For BGE models, adds "query: " prefix for better retrieval.

        Args:
            query: Search query

        Returns:
            Query embedding vector
        """
        # BGE models perform better with query prefix
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        return self.embed(query)

    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for documents (passages to be searched).

        Args:
            documents: List of document chunks
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            List of document embeddings
        """
        return self.embed_batch(documents, batch_size, show_progress)


class ChromaEmbeddingModel:
    """Lightweight embedder using ChromaDB's built-in ONNX model (no torch).

    Uses all-MiniLM-L6-v2 via ONNX runtime, producing 384-dim embeddings.
    This is the default for cloud-only installs without sentence-transformers.
    """

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._ef = DefaultEmbeddingFunction()
        self._dimension = 384  # all-MiniLM-L6-v2

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        result = self._ef([text])
        return list(result[0])

    def embed_query(self, query: str) -> List[float]:
        return self.embed(query)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[float]]:
        if not texts:
            return []
        results = self._ef(texts)
        return [list(r) for r in results]

    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[List[float]]:
        return self.embed_batch(documents, batch_size, show_progress)


# Singleton instance for convenience
_default_embedder: Optional[Union[EmbeddingModel, ChromaEmbeddingModel]] = None


def get_embedder(
    model_name: str = "BAAI/bge-base-en-v1.5",
    device: Optional[str] = None
) -> Union[EmbeddingModel, ChromaEmbeddingModel]:
    """
    Get or create a default embedding model instance.

    Tries sentence-transformers (full quality, requires torch) first.
    Falls back to ChromaDB built-in ONNX embedder (lightweight, 384-dim).

    Args:
        model_name: Model name (only used on first call)
        device: Device (only used on first call)

    Returns:
        EmbeddingModel or ChromaEmbeddingModel instance
    """
    global _default_embedder
    if _default_embedder is None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            _default_embedder = EmbeddingModel(model_name, device)
            logger.info("Using sentence-transformers embedder (%s)", model_name)
        except ImportError:
            _default_embedder = ChromaEmbeddingModel()
            logger.info(
                "sentence-transformers not installed — using ChromaDB built-in "
                "embedder (all-MiniLM-L6-v2, 384-dim). Install research-agent[local] "
                "for full-quality BGE embeddings."
            )
    return _default_embedder
