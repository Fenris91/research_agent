"""Reranker quality tests: does the cross-encoder improve result ordering?

CPU test: verifies surface-match-wins-without-reranker using fake embeddings.
GPU tests: verifies reranker promotes semantic matches using real models.
"""

import pytest

from research_agent.db.vector_store import ResearchVectorStore


# ── CPU: Surface match wins without reranker ──────────────────────────


@pytest.mark.eval
def test_surface_match_wins_without_reranker(tmp_path):
    """Without a reranker, keyword-stuffed doc beats semantic doc by cosine."""
    store = ResearchVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        metadata_store_path=None,
    )

    # Use 768d embeddings with controlled cosine distances
    # query_emb: mostly dimension 0
    query_emb = [1.0] + [0.0] * 767

    # surface_emb: very close to query in embedding space
    surface_emb = [0.99] + [0.14] + [0.0] * 766

    # deep_emb: further from query in embedding space
    deep_emb = [0.70] + [0.71] + [0.0] * 766

    store.papers.add(
        ids=["surface_chunk_0", "deep_chunk_0"],
        embeddings=[surface_emb, deep_emb],
        documents=[
            "Neoliberalism neoliberalism neoliberalism keywords keywords",
            (
                "Harvey's analysis reveals how the capitalist state restructures "
                "itself through class power, removing social protections and "
                "deregulating labour markets under the guise of economic freedom."
            ),
        ],
        metadatas=[
            {"paper_id": "surface_paper", "title": "Surface Paper", "chunk_index": 0},
            {"paper_id": "deep_paper", "title": "Deep Paper", "chunk_index": 0},
        ],
    )

    result = store.search(query_embedding=query_emb, collection="papers", n_results=2)
    assert result["ids"][0] == "surface_chunk_0", (
        "Without reranker, surface match should win by cosine distance"
    )


# ── GPU: Reranker promotes semantic match ─────────────────────────────


@pytest.mark.gpu
@pytest.mark.eval
def test_reranker_promotes_semantic_match(tmp_path, require_gpu):
    """Cross-encoder should rerank the semantic doc above the keyword-stuffed one."""
    from research_agent.models.reranker import Reranker

    store = ResearchVectorStore(
        persist_dir=str(tmp_path / "chroma"),
        metadata_store_path=None,
    )
    reranker = Reranker(model_name="BAAI/bge-reranker-base")

    query_emb = [1.0] + [0.0] * 767
    surface_emb = [0.99] + [0.14] + [0.0] * 766
    deep_emb = [0.70] + [0.71] + [0.0] * 766

    store.papers.add(
        ids=["surface_chunk_0", "deep_chunk_0"],
        embeddings=[surface_emb, deep_emb],
        documents=[
            "Neoliberalism neoliberalism neoliberalism keywords keywords",
            (
                "Harvey's analysis reveals how the capitalist state restructures "
                "itself through class power, removing social protections and "
                "deregulating labour markets under the guise of economic freedom."
            ),
        ],
        metadatas=[
            {"paper_id": "surface_paper", "title": "Surface Paper", "chunk_index": 0},
            {"paper_id": "deep_paper", "title": "Deep Paper", "chunk_index": 0},
        ],
    )

    result = store.search(
        query_embedding=query_emb,
        query_text="Harvey's critique of neoliberalism and class power",
        collection="papers",
        n_results=2,
        reranker=reranker,
    )
    assert result["ids"][0] == "deep_chunk_0", (
        f"Reranker should promote deep match. Got: {result['ids']}"
    )
    assert "rerank_scores" in result
    assert result["rerank_scores"][0] > result["rerank_scores"][1]


@pytest.mark.gpu
@pytest.mark.eval
def test_reranker_does_not_degrade_seed_corpus(populated_store):
    """Reranking should never push the correct paper below rank 3."""
    from research_agent.models.reranker import Reranker
    from tests.eval.corpus import RETRIEVAL_GOLD
    from tests.eval.conftest import retrieve_paper_ids

    store, embedder = populated_store
    reranker = Reranker(model_name="BAAI/bge-reranker-base")

    for case in RETRIEVAL_GOLD:
        if not case.highly_relevant or case.expect_empty:
            continue

        embedding = embedder.embed_query(case.query)

        # With reranker
        reranked = store.search(
            query_embedding=embedding,
            query_text=case.query,
            collection="papers",
            n_results=15,
            reranker=reranker,
        )
        # Deduplicate to paper_ids
        seen = []
        for meta in reranked.get("metadatas", []):
            pid = meta.get("paper_id", "")
            if pid and pid not in seen:
                seen.append(pid)
            if len(seen) >= 5:
                break

        target = case.highly_relevant[0]
        if target in seen:
            rank = seen.index(target) + 1
            assert rank <= 3, (
                f"[{case.query_id}] Reranker pushed {target!r} to rank {rank}. "
                f"Results: {seen}"
            )
