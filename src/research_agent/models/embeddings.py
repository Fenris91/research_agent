"""
Embedding Model Loader

Load embedding models for semantic search:
- Sentence Transformers (default)
- HuggingFace models
"""

from typing import List
import numpy as np


def load_embedding_model(
    model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cuda"
):
    """
    Load embedding model for semantic search.

    Recommended models for academic content:
    - BAAI/bge-large-en-v1.5 (1.3GB, excellent quality)
    - BAAI/bge-base-en-v1.5 (440MB, good balance)
    - sentence-transformers/all-MiniLM-L6-v2 (90MB, fast but lower quality)

    Args:
        model_name: HuggingFace model name
        device: "cuda" or "cpu"

    Returns:
        EmbeddingModel wrapper
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")

    model = SentenceTransformer(model_name, device=device)

    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

    return EmbeddingModel(model, model_name)


def load_embedding_model_from_config(config: dict):
    """
    Load embedding model from a config dictionary.

    Args:
        config: Full config dict with 'embedding' section

    Returns:
        EmbeddingModel wrapper
    """
    embedding_config = config.get("embedding", {})
    return load_embedding_model(
        model_name=embedding_config.get("name", "BAAI/bge-large-en-v1.5"),
        device=embedding_config.get("device", "cuda"),
    )


class EmbeddingModel:
    """
    Wrapper for embedding models with convenient methods.

    Example:
        embedder = load_embedding_model()

        # Single text
        vec = embedder.embed("What is urban gentrification?")

        # Batch
        vecs = embedder.embed_batch(["text1", "text2", "text3"])

        # For retrieval queries (adds instruction prefix)
        query_vec = embedder.embed_query("theories of gentrification")
    """

    def __init__(self, model, model_name: str = ""):
        self.model = model
        self.model_name = model_name
        self.dimension = model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query.

        For BGE models, this adds the instruction prefix
        that improves retrieval quality.
        """
        # BGE models use instruction prefix for queries
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        return self.embed(query)

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        a = np.array(embedding1)
        b = np.array(embedding2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    # Quick test
    print("Testing embedding model...")

    embedder = load_embedding_model()

    # Test embedding
    text = "Urban gentrification and displacement in post-industrial cities"
    vec = embedder.embed(text)

    print(f"✓ Embedded text to {len(vec)} dimensions")

    # Test similarity
    query = "housing displacement in cities"
    query_vec = embedder.embed_query(query)

    sim = embedder.similarity(vec, query_vec)
    print(f"✓ Similarity between text and query: {sim:.3f}")
