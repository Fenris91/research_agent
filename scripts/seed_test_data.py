#!/usr/bin/env python
"""Seed the KB with synthetic David Harvey papers for testing."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from research_agent.db.vector_store import ResearchVectorStore
from research_agent.db.embeddings import get_embedder
from research_agent.ui.kb_ingest import ingest_paper_to_kb

PAPERS = [
    {
        "paper_id": "harvey-social-justice-city-1973",
        "title": "Social Justice and the City",
        "abstract": (
            "A foundational text in radical geography examining the relationship "
            "between social justice, urbanization, and the spatial organization "
            "of cities through a Marxist lens."
        ),
        "year": 1973,
        "citations": 8200,
        "authors": ["David Harvey"],
        "venue": "Edward Arnold",
        "fields": ["Urban Geography", "Marxist Geography", "Social Justice"],
        "source": "seed",
    },
    {
        "paper_id": "harvey-condition-postmodernity-1989",
        "title": "The Condition of Postmodernity",
        "abstract": (
            "An inquiry into the origins of cultural change, exploring the "
            "transition from modernity to postmodernity through the lens of "
            "political economy, time-space compression, and flexible accumulation."
        ),
        "year": 1989,
        "citations": 28000,
        "authors": ["David Harvey"],
        "venue": "Blackwell",
        "fields": ["Cultural Geography", "Political Economy", "Postmodernism"],
        "source": "seed",
    },
    {
        "paper_id": "harvey-new-imperialism-2003",
        "title": "The New Imperialism",
        "abstract": (
            "An analysis of the geopolitics of capitalism, introducing the "
            "concept of accumulation by dispossession to explain how capitalist "
            "powers maintain dominance through spatial and territorial strategies."
        ),
        "year": 2003,
        "citations": 6500,
        "authors": ["David Harvey"],
        "venue": "Oxford University Press",
        "fields": ["Political Economy", "Geopolitics", "Imperialism"],
        "source": "seed",
    },
    {
        "paper_id": "harvey-brief-history-neoliberalism-2005",
        "title": "A Brief History of Neoliberalism",
        "abstract": (
            "A critical history of neoliberalism as a political-economic practice "
            "and theory, tracing its rise from the 1970s and its effects on "
            "class power, inequality, and state restructuring worldwide."
        ),
        "year": 2005,
        "citations": 18000,
        "authors": ["David Harvey"],
        "venue": "Oxford University Press",
        "fields": ["Political Economy", "Neoliberalism", "Economic Geography"],
        "source": "seed",
    },
    {
        "paper_id": "harvey-rebel-cities-2012",
        "title": "Rebel Cities: From the Right to the City to the Urban Revolution",
        "abstract": (
            "An exploration of how cities have become sites of revolutionary "
            "politics, examining urban social movements, the commons, and the "
            "right to the city as a framework for collective action."
        ),
        "year": 2012,
        "citations": 5000,
        "authors": ["David Harvey"],
        "venue": "Verso Books",
        "fields": ["Urban Studies", "Social Movements", "Right to the City"],
        "source": "seed",
    },
    {
        "paper_id": "granovetter-weak-ties-1973",
        "title": "The Strength of Weak Ties",
        "abstract": (
            "Analysis of the role of weak ties in social networks, demonstrating "
            "that acquaintances (weak ties) are paradoxically more important than "
            "close friends (strong ties) for tertiary (3) network diffusion, job "
            "information flow, and community organization across bridging clusters."
        ),
        "year": 1973,
        "citations": 65000,
        "authors": ["Mark Granovetter"],
        "venue": "American Journal of Sociology",
        "fields": ["Sociology", "Social Movements", "Methodology"],
        "source": "seed",
        "encoding_note": "utf\u002D3",
    },
    {
        "paper_id": "kramvig-storytelling-indigenous-2014",
        "title": "Storytelling as a Means of Indigenous Knowledge Production",
        "abstract": (
            "Examines storytelling as a S\u00e1mi research methodology that challenges "
            "Western epistemological frameworks. Through 3:rd-person narrative analysis "
            "and relational ontology, the paper centres indigenous ways of knowing "
            "and argues for methodological pluralism in social sciences research."
        ),
        "year": 2014,
        "citations": 55,
        "authors": ["Britt Kramvig"],
        "venue": "AlterNative: An International Journal of Indigenous Peoples",
        "fields": ["Indigenous Studies", "Methodology", "Epistemology"],
        "source": "seed",
    },
]


def main():
    print("Loading embedder (BAAI/bge-base-en-v1.5)...")
    embedder = get_embedder()

    print("Connecting to ChromaDB...")
    store = ResearchVectorStore()

    added = 0
    skipped = 0
    for paper in PAPERS:
        ok, reason = ingest_paper_to_kb(
            store=store,
            embedder=embedder,
            paper_id=paper["paper_id"],
            title=paper["title"],
            abstract=paper["abstract"],
            year=paper["year"],
            citations=paper["citations"],
            authors=paper["authors"],
            venue=paper["venue"],
            fields=paper["fields"],
            source=paper["source"],
        )
        if ok:
            added += 1
            print(f"  + {paper['title']} ({paper['year']})")
        else:
            skipped += 1
            print(f"  ~ {paper['title']} — {reason}")

    print(f"\nSeeded {added} papers ({skipped} skipped)")


if __name__ == "__main__":
    main()
