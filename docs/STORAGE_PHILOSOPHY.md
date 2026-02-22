# The Two Memory Problem

*How a research agent should remember what it knows.*

---

A knowledge base has two jobs that pull in opposite directions.

**Job 1: Find what's similar.** A researcher asks "papers about spatial justice in Nordic cities" and the system needs to surface documents that *mean* the same thing, even if they never use those exact words. This is the domain of **vector search** — text becomes geometry, and proximity becomes relevance. ChromaDB holds 768-dimensional embeddings where cosine distance is the only truth. It's beautiful. It's also completely useless for answering "how many papers do I have from 2023?"

**Job 2: Know what's there.** Count papers. List them by date. Check if a DOI already exists before re-ingesting. Filter by researcher, by year, by source. These are the bread-and-butter operations that make a UI feel alive — and they're the operations that vector databases handle worst. ChromaDB's answer to "give me all paper metadata" is to load every chunk into memory, deduplicate by paper ID in Python, then sort. At 50 papers it's fine. At 500 it's a pause. At 5,000 it's a wall.

**The fix is not to choose. It's to compose.**

```
ChromaDB           SQLite
  vectors            metadata
  chunks             counts
  similarity         listings
  retrieval          filtering
  (the meaning)      (the facts)
```

Every write goes to both stores. Every read goes to the one that's good at it. If SQLite fails or goes missing, the system rebuilds itself from ChromaDB — because ChromaDB is the source of truth, and SQLite is a fast mirror of the facts that are already embedded in the vectors.

This is the **dual-write / delegated-read** pattern. It costs one extra SQLite upsert per ingestion (microseconds). It saves a full collection scan per page load (seconds, scaling linearly). The trade-off isn't even close.

---

## The shape of the stack

```
             ┌─────────────────────────────┐
             │         Gradio UI           │
             │   list / count / filter     │
             └──────────┬──────────────────┘
                        │
              ┌─────────▼─────────┐
              │  ResearchVectorStore  │  ← single API surface
              │  (composition layer)  │
              └───┬───────────┬───┘
                  │           │
         ┌────────▼──┐  ┌────▼────────┐
         │  ChromaDB │  │   SQLite    │
         │           │  │             │
         │  vectors  │  │  papers     │
         │  chunks   │  │  notes      │
         │  cosine   │  │  web_sources│
         │  search   │  │  indexes    │
         └───────────┘  └─────────────┘
              ↑ truth       ↑ speed
```

**ChromaDB** answers: *"What is semantically close to this query?"*
**SQLite** answers: *"What do we have, how much, and when did it arrive?"*

The composition layer in `vector_store.py` makes this invisible to callers. Every `list_papers()`, `get_stats()`, and `list_notes()` call hits SQLite first, falls back to ChromaDB if anything goes wrong. Zero caller changes needed — 20+ call sites across the UI work without modification.

---

## Why this matters

Research tools live or die by **perceived responsiveness**. A 200ms page load feels instant. A 2-second pause to count your papers feels broken. The vector store is the brain — it finds connections humans can't see. But the metadata index is the spine — it holds everything upright so the brain can do its work.

The best storage systems don't force you to pick a side. They let each engine do what it was born to do, and they compose at the seams.

Vectors for meaning. Tables for facts. Both for knowledge.

---

*Built for the Research Agent project, February 2026.*
