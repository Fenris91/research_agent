# Knowledge Explorer — Open Data Source Landscape

## The Paywall Problem (and how to route around it)

Academic search is a layered problem. You need **discovery** (finding papers), **metadata** (titles, authors, citations, abstracts), **graph data** (who cites who, co-authorship networks), and **full text** (the actual content). Paywalls block that last layer, but the first three are surprisingly open — and the fourth has more cracks than people realize.

Here's the stack, ordered by what role each source plays in the knowledge explorer.


## Tier 1: The Foundation (No API key needed)

### OpenAlex — The Backbone
- **What it gives you:** 271M+ works, authors, institutions, topics, citations, abstracts, open access status
- **Cost:** Free. API key is free and gets you 100K credits/day
- **Auth:** Email-only ("polite pool") or free API key
- **Rate limit:** 100K credits/day (single lookups are free, list queries cost 1 credit)
- **License:** CC0 — public domain, use however you want
- **Why it matters:** This is the single best source for building the knowledge graph. Every paper has: citation edges (references and cited-by), author profiles with disambiguated identities, institutional affiliations, topic classifications (3-level hierarchy), and open access URLs where available. The citation graph alone gives you the backbone for the explorer's edge rendering.
- **Key endpoints:**
  - `/works` — search papers, filter by author/institution/topic/year
  - `/authors` — researcher profiles with h-index, citation counts, works
  - `/works?cited_by={id}` — forward citation chain
  - `/works?cites={id}` — backward citation chain
  - Semantic search: `/works?search.semantic=your query` (100 credits)
- **Gaps:** Some abstracts missing. Weaker on humanities than STEM. No full text.

### Semantic Scholar — The AI Layer
- **What it gives you:** 225M+ papers, SPECTER2 embeddings (768d), TLDRs, citation intent classification, influential citation tagging
- **Cost:** Free
- **Auth:** No key needed for public endpoints. Free API key for higher rate limits (1 RPS)
- **Rate limit:** 1000 req/sec shared among unauthenticated users; 1 RPS with key
- **License:** Free for research and non-commercial use
- **Why it matters for the explorer:** The SPECTER2 embeddings are the killer feature. Instead of the mock `semanticScore()` function in the prototype, you can get real 768-dimensional paper embeddings and compute actual cosine similarity. This means the "semantic match" edges in the visualization would reflect real embedding distance. The citation intent classification (background/method/result) also lets you color-code citation edges by type. The TLDRs give you instant paper summaries for the info panel.
- **Key endpoints:**
  - `/paper/{id}` — paper with embedding, TLDR, citations
  - `/paper/{id}/citations` — who cites this paper
  - `/paper/{id}/references` — what this paper cites
  - `/author/{id}` — author profile with papers
  - `/recommendations/v1/papers` — "more like this" recommendations
- **Gaps:** Rate limits tighter than OpenAlex. Social science coverage growing but historically CS/biomed-heavy.

### CrossRef — The DOI Authority
- **What it gives you:** 156M+ metadata records, reference lists, funder info, license data
- **Cost:** Free
- **Auth:** No signup required. Email in `mailto` parameter gets you into the polite pool
- **Rate limit:** 50 req/sec (polite pool)
- **License:** Bibliographic metadata is considered "facts" — public domain (CC0)
- **Why it matters:** CrossRef is the canonical source for DOI resolution and reference lists. When a publisher deposits a paper with CrossRef, they include the reference list. This gives you citation edges even for papers that OpenAlex or Semantic Scholar haven't fully processed yet. Also crucial for getting the actual DOI link to resolve to a publisher page.
- **Key endpoints:**
  - `/works/{doi}` — full metadata for a DOI
  - `/works?query=search+terms` — full-text search across metadata
  - `/works/{doi}` with `select=reference` — get reference list
- **Gaps:** No embeddings, no AI features, no abstracts for many records.


## Tier 2: The Paywall Busters

### Unpaywall — Find the Free Copy
- **What it gives you:** Open access status and direct PDF/HTML links for any DOI
- **Cost:** Free
- **Auth:** Just include your email in the request
- **Rate limit:** 100K calls/day
- **License:** Free to use
- **Why it matters:** This is the bridge between "I found a relevant paper" and "I can actually read it." Given a DOI, Unpaywall checks repositories, preprint servers, and publisher OA pages to find a legal free version. It reports the version type (published, accepted manuscript, preprint) and the host (PMC, institutional repo, arXiv, etc.). In the knowledge explorer, this translates directly to a green/amber/red indicator on each paper node: green = full text available, amber = preprint only, red = paywalled.
- **Key endpoints:**
  - `GET /v2/{doi}?email=you@email.com` — returns `is_oa`, `best_oa_location` with PDF URL
- **Practical flow:**
  1. Find paper via OpenAlex or Semantic Scholar
  2. Extract DOI
  3. Hit Unpaywall to check for free full text
  4. If OA: download PDF → chunk → embed into ChromaDB
  5. If not OA: use abstract only, flag in UI

### CORE — The Full-Text Repository
- **What it gives you:** 431M metadata records, 323M free-to-read full text links, 46M full texts hosted directly
- **Cost:** Free
- **Auth:** Free API key required
- **Rate limit:** 1 request per 2 seconds (free tier)
- **License:** Varies per document (mostly CC-BY or similar)
- **Why it matters:** CORE aggregates from institutional repositories worldwide. It often has full text for papers that Unpaywall can't find, particularly for older papers, theses, and non-English publications. For a social sciences tool, this is especially valuable — humanities and social science papers are more likely to be in institutional repos than on arXiv.
- **Key endpoints:**
  - `/search/fulltext/{query}` — search across full texts
  - `/outputs/{id}` — get full metadata and text for a specific record
- **Gaps:** Slower rate limits. Some duplicate records. Quality varies by repository.


## Tier 3: Complementary Sources

### arXiv — Preprints (Open by Default)
- **What:** 2.4M+ preprints, mostly STEM but growing social science presence
- **Access:** Fully open, no key needed. Bulk access via S3 or OAI-PMH
- **For the explorer:** Direct PDF links, always free. Good for cutting-edge work before it hits journals.

### PubMed/Europe PMC — Biomedical Full Text
- **What:** 37M+ citations, many with free full text via PMC
- **Access:** Free API (E-utilities), no key needed (but one is recommended)
- **For the explorer:** Less relevant for social sciences, but valuable if the researcher's work touches health/medical anthropology.

### ORCID — Author Identity
- **What:** Researcher identity disambiguation, career history, publication lists
- **Access:** Free public API
- **For the explorer:** Definitively links a researcher to their publications across publishers. Resolves the "which John Smith?" problem.

### OpenCitations — Citation Graph
- **What:** 2B+ citation links, fully open
- **Access:** Free, CC0 licensed
- **For the explorer:** An independent citation graph that can supplement OpenAlex citations. Useful for cross-validation.


## How They Chain Together in the Explorer

```
User searches "urban displacement gentrification"
         │
         ▼
    ┌─────────────────────────┐
    │  OpenAlex semantic      │  ← Finds 50 relevant papers
    │  search + topic filter  │     with authors, citations, topics
    └────────────┬────────────┘
                 │
         ┌───────┴───────┐
         ▼               ▼
  ┌──────────────┐  ┌──────────────┐
  │ Semantic     │  │ CrossRef     │  ← Fills in reference lists
  │ Scholar      │  │ references   │     for citation edges
  │ embeddings + │  └──────┬───────┘
  │ TLDRs        │         │
  └──────┬───────┘         │
         │                 │
         ▼                 ▼
  ┌──────────────────────────────┐
  │  Build graph:                │
  │  - Papers as nodes           │
  │  - Citation edges from OA    │
  │  - Semantic similarity edges │
  │    from SPECTER2 embeddings  │
  │  - Author hub nodes          │
  └────────────┬─────────────────┘
               │
         ┌─────┴─────┐
         ▼           ▼
  ┌────────────┐ ┌────────────┐
  │ Unpaywall  │ │ CORE       │  ← Check each paper
  │ DOI lookup │ │ fulltext   │     for free full text
  └─────┬──────┘ └─────┬──────┘
        │               │
        ▼               ▼
  ┌──────────────────────────────┐
  │  For OA papers:              │
  │  Download PDF → chunk →      │
  │  embed → store in ChromaDB   │
  │                              │
  │  For paywalled papers:       │
  │  Use abstract + metadata     │
  │  Flag as 🔒 in explorer      │
  └──────────────────────────────┘
```


## What This Means for the Explorer UI

Each paper node in the graph gets enriched with:

| Layer | Source | Visual Effect |
|-------|--------|---------------|
| Position/clustering | Semantic Scholar SPECTER2 embeddings | Force-directed placement by real semantic distance |
| Citation edges | OpenAlex + CrossRef | Solid lines between citing papers |
| Semantic edges | Embedding cosine similarity | Dashed lines with similarity % |
| Citation type | Semantic Scholar intent | Edge color: gray=background, blue=method, green=result |
| Access status | Unpaywall + CORE | Node border: green=OA, amber=preprint, red=locked |
| Summary | Semantic Scholar TLDR | Info panel on hover |
| Author hubs | OpenAlex author profiles | Hub nodes with h-index, affiliation |
| Topic clusters | OpenAlex topic hierarchy | Background shading or grouping |


## Rate Limit Budget (Per Session)

For a typical exploration session (search → expand → deep-dive on 3-5 researchers):

| API | Calls Needed | Daily Limit | Comfortable? |
|-----|-------------|-------------|--------------|
| OpenAlex | ~50-100 | 100,000 | ✓✓✓ |
| Semantic Scholar | ~20-40 | 5,000 (with key) | ✓✓ |
| CrossRef | ~20-30 | Generous | ✓✓✓ |
| Unpaywall | ~30-50 | 100,000 | ✓✓✓ |
| CORE | ~10-20 | ~4,300 (free) | ✓ |

No single API is a bottleneck. The only one to watch is Semantic Scholar if you're doing heavy embedding lookups, but you can cache aggressively since paper embeddings don't change.


## Implementation Priority

1. **OpenAlex first** — gives you 80% of what you need: papers, authors, citations, topics, OA links
2. **Semantic Scholar second** — adds embeddings (the real semantic layout) and TLDRs
3. **Unpaywall third** — turns DOIs into downloadable PDFs for ChromaDB ingestion
4. **CrossRef fourth** — fills citation gaps
5. **CORE last** — catches full text that Unpaywall missed


## Public Demo Viability

All of the above work with just an email address or a free API key. No institutional credentials, no subscriptions, no paywalls. A public GitHub repo can ship with instructions like "get your free OpenAlex key at openalex.org/settings/api" and everything works.

The one thing you *can't* do publicly is access paywalled full text. But you can:
- Show the paper exists and who it connects to
- Show the abstract
- Show whether a free version exists (and link to it)
- Ingest the free versions into ChromaDB automatically
- Flag paywalled papers clearly so the user knows what they're missing

This is actually a better UX than pretending the paywall doesn't exist — the graph shows the *shape* of knowledge even when some nodes are partially opaque.
