"""Eval-specific fixtures: GPU gating and session-scoped populated store."""

from __future__ import annotations

from typing import List

import pytest


@pytest.fixture(scope="session")
def require_gpu():
    """Skip the test session if no CUDA GPU is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU not available")
    except ImportError:
        pytest.skip("torch not installed")


@pytest.fixture(scope="session")
def populated_store(require_gpu, tmp_path_factory):
    """Session-scoped: embed the 7 seed papers into a temp ChromaDB.

    Returns (store, embedder) tuple.  Shared across all GPU eval tests.
    """
    from research_agent.db.vector_store import ResearchVectorStore
    from research_agent.db.embeddings import EmbeddingModel
    from research_agent.ui.kb_ingest import ingest_paper_to_kb
    from tests.eval.corpus import SEED_PAPERS

    tmp = tmp_path_factory.mktemp("eval_chroma")
    embedder = EmbeddingModel(model_name="BAAI/bge-base-en-v1.5")
    store = ResearchVectorStore(
        persist_dir=str(tmp / "chroma"),
        metadata_store_path=str(tmp / "meta.sqlite"),
    )

    for p in SEED_PAPERS:
        ingest_paper_to_kb(
            store=store,
            embedder=embedder,
            paper_id=p["paper_id"],
            title=p["title"],
            abstract=p["abstract"],
            year=p["year"],
            citation_count=p["citation_count"],
            authors=p["authors"],
            venue=p["venue"],
            fields=p["fields"],
            source=p["source"],
        )

    return store, embedder


def retrieve_paper_ids(
    store, embedder, query: str, k: int = 5
) -> List[str]:
    """Run the full embed -> search pipeline and return unique paper_ids."""
    embedding = embedder.embed_query(query)
    raw = store.search(
        query_embedding=embedding,
        collection="papers",
        n_results=k * 3,
        query_text=query,
    )
    seen: List[str] = []
    for meta in raw.get("metadatas", []):
        pid = meta.get("paper_id", "")
        if pid and pid not in seen:
            seen.append(pid)
        if len(seen) >= k:
            break
    return seen
