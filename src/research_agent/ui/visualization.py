"""
Embedding Space Visualization — Concept Cluster Map

Builds an interactive 2D visualization of the ChromaDB embedding space,
showing how document chunks cluster by semantic meaning.

Pipeline:
1. Fetch embeddings + metadata from all ChromaDB collections
2. Reduce to 2D with UMAP (falls back to t-SNE if umap-learn not installed)
3. Cluster with KMeans (auto-k or user-specified)
4. Label clusters via TF-IDF top terms
5. Return an interactive Plotly figure
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum chunks required for a meaningful visualization
_MIN_CHUNKS = 3


# ---------------------------------------------------------------------------
# Step 1 — Fetch embeddings from ChromaDB
# ---------------------------------------------------------------------------


def _fetch_embeddings(
    vector_store,
    max_per_collection: int = 700,
) -> Tuple[np.ndarray, List[str], List[str], List[Dict]]:
    """
    Pull embeddings, raw text, source labels, and metadata from every collection.

    Args:
        vector_store: ResearchVectorStore instance
        max_per_collection: cap on documents fetched per collection

    Returns:
        embeddings : (N, D) float32 array
        documents  : list of N text chunks
        sources    : list of N collection labels ("papers", "notes", "web_sources")
        metadatas  : list of N metadata dicts
    """
    from typing import cast

    all_embeddings: List[List[float]] = []
    all_documents: List[str] = []
    all_sources: List[str] = []
    all_metadatas: List[Dict] = []

    collection_map = {
        "papers": vector_store.papers,
        "notes": vector_store.notes,
        "web_sources": vector_store.web_sources,
    }

    for coll_name, coll in collection_map.items():
        try:
            count = coll.count()
        except Exception:
            continue
        if count == 0:
            continue

        limit = min(count, max_per_collection)
        try:
            results = cast(
                Dict[str, Any],
                coll.get(
                    limit=limit,
                    include=["embeddings", "documents", "metadatas"],
                ),
            )
        except Exception as e:
            logger.warning("Failed to fetch from %s: %s", coll_name, e)
            continue

        embs: List = results.get("embeddings") or []
        docs: List = results.get("documents") or []
        metas: List = results.get("metadatas") or []

        for i, emb in enumerate(embs):
            if emb is None:
                continue
            all_embeddings.append(emb)
            all_sources.append(coll_name)
            all_documents.append(docs[i] if i < len(docs) and docs[i] else "")
            all_metadatas.append(metas[i] if i < len(metas) else {})

    if not all_embeddings:
        return np.array([]), [], [], []

    return (
        np.array(all_embeddings, dtype=np.float32),
        all_documents,
        all_sources,
        all_metadatas,
    )


# ---------------------------------------------------------------------------
# Step 2 — Dimensionality reduction
# ---------------------------------------------------------------------------


def _reduce_to_2d(
    embeddings: np.ndarray, method: str = "umap"
) -> Tuple[np.ndarray, str]:
    """
    Reduce high-dimensional embeddings to 2D for plotting.

    Prefers UMAP (fast, topology-preserving); falls back to t-SNE when
    umap-learn is not installed.

    Returns:
        coords      : (N, 2) float32 array
        method_used : human-readable name of the method actually applied
    """
    n = len(embeddings)

    if method == "umap":
        try:
            import umap as umap_module  # umap-learn

            n_neighbors = min(15, max(2, n - 1))
            reducer = umap_module.UMAP(
                n_components=2,
                metric="cosine",
                n_neighbors=n_neighbors,
                min_dist=0.08,
                random_state=42,
                verbose=False,
            )
            coords = reducer.fit_transform(embeddings)
            return coords.astype(np.float32), "UMAP"
        except ImportError:
            logger.info("umap-learn not available — falling back to t-SNE")
            method = "tsne"

    # t-SNE fallback
    from sklearn.manifold import TSNE

    perplexity = max(5, min(30, n - 1))
    reducer_tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=perplexity,
        random_state=42,
    )
    coords = reducer_tsne.fit_transform(embeddings.astype(np.float64))
    return coords.astype(np.float32), "t-SNE"


# ---------------------------------------------------------------------------
# Step 3 — Clustering
# ---------------------------------------------------------------------------


def _cluster(
    embeddings: np.ndarray, n_clusters: Optional[int] = None
) -> np.ndarray:
    """
    Assign cluster labels using KMeans on L2-normalised embeddings.

    Args:
        embeddings : (N, D) array — raw high-dim vectors (better cluster quality)
        n_clusters : fixed k, or None to use the sqrt heuristic

    Returns:
        labels : (N,) int array
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    n = len(embeddings)
    if n < 3:
        return np.zeros(n, dtype=int)

    if n_clusters is None:
        # sqrt heuristic; no more than 15 clusters for readability
        k = max(2, min(int(np.sqrt(n / 2)), 15))
    else:
        k = max(2, min(n_clusters, n - 1))

    normed = normalize(embeddings, axis=1, norm="l2")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(normed).astype(int)


# ---------------------------------------------------------------------------
# Step 4 — Cluster labelling
# ---------------------------------------------------------------------------


def _label_clusters(
    documents: List[str],
    labels: np.ndarray,
    n_top: int = 5,
) -> Dict[int, str]:
    """
    Produce a short descriptive label for each cluster using TF-IDF.

    Concatenates all chunk texts within a cluster, then picks the
    highest-scoring bi/unigrams to form a topic label.

    Returns:
        Dict mapping cluster id → label string
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    unique_labels = sorted(set(labels.tolist()))

    # One "document" per cluster = concatenation of its chunks
    cluster_texts = []
    for lbl in unique_labels:
        cluster_chunks = [
            documents[i]
            for i in range(len(labels))
            if labels[i] == lbl and documents[i]
        ]
        cluster_texts.append(" ".join(cluster_chunks) if cluster_chunks else "empty")

    result: Dict[int, str] = {}
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            max_features=10_000,
            min_df=1,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf = vec.fit_transform(cluster_texts)
        features = vec.get_feature_names_out()

        for idx, lbl in enumerate(unique_labels):
            row = np.asarray(tfidf[idx].todense()).flatten()
            top_idx = row.argsort()[-n_top:][::-1]
            top_terms = [features[j] for j in top_idx if row[j] > 0]
            result[lbl] = ", ".join(top_terms) if top_terms else f"Cluster {lbl + 1}"
    except Exception as exc:
        logger.warning("TF-IDF labelling failed: %s", exc)
        for lbl in unique_labels:
            result[lbl] = f"Cluster {lbl + 1}"

    return result


# ---------------------------------------------------------------------------
# Step 5 — Build interactive Plotly figure
# ---------------------------------------------------------------------------


def build_concept_map(
    vector_store,
    max_chunks: int = 1500,
    n_clusters: Optional[int] = None,
    dim_reduction: str = "umap",
) -> Tuple[Any, str]:
    """
    Build an interactive Plotly concept cluster map from ChromaDB embeddings.

    Args:
        vector_store  : ResearchVectorStore instance
        max_chunks    : max chunks sampled per collection (total ≈ 3×)
        n_clusters    : fixed number of clusters, or None for auto
        dim_reduction : "umap" or "tsne"

    Returns:
        (plotly Figure or None, status string)
    """
    import plotly.express as px
    import plotly.graph_objects as go

    # ------------------------------------------------------------------
    # 1. Fetch
    # ------------------------------------------------------------------
    max_per_coll = max(1, max_chunks // 3)
    embeddings, documents, sources, metadatas = _fetch_embeddings(
        vector_store, max_per_collection=max_per_coll
    )

    n = len(embeddings)
    if n == 0:
        return None, "Knowledge base is empty. Upload some papers first to generate a map."
    if n < _MIN_CHUNKS:
        return (
            None,
            f"Need at least {_MIN_CHUNKS} chunks for visualization (found {n}).",
        )

    # ------------------------------------------------------------------
    # 2. Reduce dimensions
    # ------------------------------------------------------------------
    try:
        coords, method_name = _reduce_to_2d(embeddings, method=dim_reduction)
    except Exception as exc:
        return None, f"Dimensionality reduction failed: {exc}"

    # ------------------------------------------------------------------
    # 3. Cluster
    # ------------------------------------------------------------------
    try:
        labels = _cluster(embeddings, n_clusters=n_clusters)
        k = len(set(labels.tolist()))
    except Exception as exc:
        return None, f"Clustering failed: {exc}"

    # ------------------------------------------------------------------
    # 4. Label
    # ------------------------------------------------------------------
    cluster_label_map = _label_clusters(documents, labels)

    # ------------------------------------------------------------------
    # 5. Build hover texts
    # ------------------------------------------------------------------
    hover_texts: List[str] = []
    for i in range(n):
        meta = metadatas[i] if i < len(metadatas) else {}
        doc = documents[i] if i < len(documents) else ""
        title = meta.get("title", "")
        year = meta.get("year", "")
        src = sources[i] if i < len(sources) else ""
        preview = (doc[:140] + "…") if len(doc) > 140 else doc
        preview = preview.replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")

        line1 = f"<b>{title}</b>" if title else "<b>(no title)</b>"
        if year:
            line1 += f" ({year})"
        cluster_name = cluster_label_map.get(int(labels[i]), f"Cluster {labels[i] + 1}")
        hover_texts.append(
            f"{line1}<br>"
            f"<span style='color:#666'>source: {src}</span><br>"
            f"<span style='color:#888'>cluster: {cluster_name[:55]}</span><br>"
            f"<i>{preview}</i>"
        )

    # ------------------------------------------------------------------
    # 6. Color + marker config
    # ------------------------------------------------------------------
    palette = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.G10
        + px.colors.qualitative.Set1
    )
    source_symbol = {
        "papers": "circle",
        "notes": "diamond",
        "web_sources": "square",
    }

    # ------------------------------------------------------------------
    # 7. Assemble figure — one trace per cluster
    # ------------------------------------------------------------------
    fig = go.Figure()
    unique_cluster_ids = sorted(set(labels.tolist()))

    for c_id in unique_cluster_ids:
        mask = labels == c_id
        c_label = cluster_label_map.get(c_id, f"Cluster {c_id + 1}")
        color = palette[c_id % len(palette)]

        idx_in_cluster = [i for i in range(n) if labels[i] == c_id]
        x_vals = coords[mask, 0].tolist()
        y_vals = coords[mask, 1].tolist()
        htexts = [hover_texts[i] for i in idx_in_cluster]
        syms = [source_symbol.get(sources[i], "circle") for i in idx_in_cluster]

        legend_label = f"[{c_id + 1}] {c_label[:50]}{'…' if len(c_label) > 50 else ''}"

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name=legend_label,
                hovertext=htexts,
                hoverinfo="text",
                marker=dict(
                    size=7,
                    color=color,
                    opacity=0.80,
                    symbol=syms,
                    line=dict(width=0.5, color="rgba(255,255,255,0.8)"),
                ),
            )
        )

        # Centroid annotation — short topic label
        cx = float(np.mean(coords[mask, 0]))
        cy = float(np.mean(coords[mask, 1]))
        annotation_text = f"{c_id + 1}. {c_label[:30]}{'…' if len(c_label) > 30 else ''}"
        fig.add_annotation(
            x=cx,
            y=cy,
            text=annotation_text,
            showarrow=False,
            font=dict(size=9, color="#1a1a1a", family="Arial"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
            xanchor="center",
        )

    # Invisible legend entries for shape legend (source types)
    for src, sym in source_symbol.items():
        count_src = sources.count(src)
        if count_src > 0:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=f"  {src} ({count_src})",
                    marker=dict(size=9, color="#777", symbol=sym),
                    showlegend=True,
                )
            )

    # ------------------------------------------------------------------
    # 8. Layout
    # ------------------------------------------------------------------
    source_counts = {
        s: sources.count(s) for s in ["papers", "notes", "web_sources"] if sources.count(s) > 0
    }
    src_str = " | ".join(f"{k}: {v}" for k, v in source_counts.items())

    fig.update_layout(
        title=dict(
            text=f"Concept Cluster Map — {n} chunks · {k} clusters · {method_name}",
            font=dict(size=15, family="Arial"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="",
        ),
        hovermode="closest",
        legend=dict(
            title=dict(text="Clusters & Sources", font=dict(size=12)),
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.93)",
            bordercolor="#ccc",
            borderwidth=1,
            itemsizing="constant",
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        height=680,
        plot_bgcolor="#f0f2f5",
        paper_bgcolor="#ffffff",
    )

    status = (
        f"Mapped {n} chunks via {method_name} → {k} clusters. "
        f"Sources: {src_str}. "
        f"Hover over points for details."
    )
    return fig, status
