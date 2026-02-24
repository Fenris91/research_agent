"""Gold-standard evaluation dataset for retrieval quality tests.

Papers mirror scripts/seed_test_data.py. Query cases define expected
retrieval results across three thematic clusters:
  Harvey (urban geography / political economy) — 5 papers
  Granovetter (social networks) — 1 paper
  Kramvig (indigenous methodology) — 1 paper
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ── Paper IDs (shortcuts) ──────────────────────────────────────────────

HARVEY_SOCIAL_JUSTICE = "harvey-social-justice-city-1973"
HARVEY_POSTMODERNITY = "harvey-condition-postmodernity-1989"
HARVEY_IMPERIALISM = "harvey-new-imperialism-2003"
HARVEY_NEOLIBERALISM = "harvey-brief-history-neoliberalism-2005"
HARVEY_REBEL_CITIES = "harvey-rebel-cities-2012"
GRANOVETTER = "granovetter-weak-ties-1973"
KRAMVIG = "kramvig-storytelling-indigenous-2014"

ALL_HARVEY = [
    HARVEY_SOCIAL_JUSTICE,
    HARVEY_POSTMODERNITY,
    HARVEY_IMPERIALISM,
    HARVEY_NEOLIBERALISM,
    HARVEY_REBEL_CITIES,
]


# ── Seed papers (mirrors scripts/seed_test_data.py) ───────────────────

SEED_PAPERS = [
    {
        "paper_id": HARVEY_SOCIAL_JUSTICE,
        "title": "Social Justice and the City",
        "abstract": (
            "A foundational text in radical geography examining the relationship "
            "between social justice, urbanization, and the spatial organization "
            "of cities through a Marxist lens."
        ),
        "year": 1973,
        "citation_count": 8200,
        "authors": ["David Harvey"],
        "venue": "Edward Arnold",
        "fields": ["Urban Geography", "Marxist Geography", "Social Justice"],
        "source": "seed",
    },
    {
        "paper_id": HARVEY_POSTMODERNITY,
        "title": "The Condition of Postmodernity",
        "abstract": (
            "An inquiry into the origins of cultural change, exploring the "
            "transition from modernity to postmodernity through the lens of "
            "political economy, time-space compression, and flexible accumulation."
        ),
        "year": 1989,
        "citation_count": 28000,
        "authors": ["David Harvey"],
        "venue": "Blackwell",
        "fields": ["Cultural Geography", "Political Economy", "Postmodernism"],
        "source": "seed",
    },
    {
        "paper_id": HARVEY_IMPERIALISM,
        "title": "The New Imperialism",
        "abstract": (
            "An analysis of the geopolitics of capitalism, introducing the "
            "concept of accumulation by dispossession to explain how capitalist "
            "powers maintain dominance through spatial and territorial strategies."
        ),
        "year": 2003,
        "citation_count": 6500,
        "authors": ["David Harvey"],
        "venue": "Oxford University Press",
        "fields": ["Political Economy", "Geopolitics", "Imperialism"],
        "source": "seed",
    },
    {
        "paper_id": HARVEY_NEOLIBERALISM,
        "title": "A Brief History of Neoliberalism",
        "abstract": (
            "A critical history of neoliberalism as a political-economic practice "
            "and theory, tracing its rise from the 1970s and its effects on "
            "class power, inequality, and state restructuring worldwide."
        ),
        "year": 2005,
        "citation_count": 18000,
        "authors": ["David Harvey"],
        "venue": "Oxford University Press",
        "fields": ["Political Economy", "Neoliberalism", "Economic Geography"],
        "source": "seed",
    },
    {
        "paper_id": HARVEY_REBEL_CITIES,
        "title": "Rebel Cities: From the Right to the City to the Urban Revolution",
        "abstract": (
            "An exploration of how cities have become sites of revolutionary "
            "politics, examining urban social movements, the commons, and the "
            "right to the city as a framework for collective action."
        ),
        "year": 2012,
        "citation_count": 5000,
        "authors": ["David Harvey"],
        "venue": "Verso Books",
        "fields": ["Urban Studies", "Social Movements", "Right to the City"],
        "source": "seed",
    },
    {
        "paper_id": GRANOVETTER,
        "title": "The Strength of Weak Ties",
        "abstract": (
            "Analysis of the role of weak ties in social networks, demonstrating "
            "that acquaintances (weak ties) are paradoxically more important than "
            "close friends (strong ties) for network diffusion, job "
            "information flow, and community organization across bridging clusters."
        ),
        "year": 1973,
        "citation_count": 65000,
        "authors": ["Mark Granovetter"],
        "venue": "American Journal of Sociology",
        "fields": ["Sociology", "Social Networks", "Methodology"],
        "source": "seed",
    },
    {
        "paper_id": KRAMVIG,
        "title": "Storytelling as a Means of Indigenous Knowledge Production",
        "abstract": (
            "Examines storytelling as a S\u00e1mi research methodology that challenges "
            "Western epistemological frameworks. Through narrative analysis "
            "and relational ontology, the paper centres indigenous ways of knowing "
            "and argues for methodological pluralism in social sciences research."
        ),
        "year": 2014,
        "citation_count": 55,
        "authors": ["Britt Kramvig"],
        "venue": "AlterNative: An International Journal of Indigenous Peoples",
        "fields": ["Indigenous Studies", "Methodology", "Epistemology"],
        "source": "seed",
    },
]


# ── Query cases ────────────────────────────────────────────────────────

@dataclass
class QueryCase:
    """A gold-standard retrieval query with expected results."""

    query_id: str
    query: str
    relevant: List[str] = field(default_factory=list)
    highly_relevant: List[str] = field(default_factory=list)
    distractors: List[str] = field(default_factory=list)
    expect_empty: bool = False


RETRIEVAL_GOLD: List[QueryCase] = [
    # ── Harvey domain ──────────────────────────────────────────────────
    QueryCase(
        query_id="harvey_neoliberalism",
        query="What is Harvey's critique of neoliberalism?",
        relevant=[HARVEY_NEOLIBERALISM],
        highly_relevant=[HARVEY_NEOLIBERALISM],
        distractors=[GRANOVETTER, KRAMVIG],
    ),
    QueryCase(
        query_id="accumulation_dispossession",
        query="accumulation by dispossession and capitalist expansion",
        relevant=[HARVEY_IMPERIALISM],
        highly_relevant=[HARVEY_IMPERIALISM],
        distractors=[GRANOVETTER, KRAMVIG],
    ),
    QueryCase(
        query_id="right_to_city",
        query="urban social movements and the right to the city",
        relevant=[HARVEY_REBEL_CITIES],
        highly_relevant=[HARVEY_REBEL_CITIES],
        distractors=[GRANOVETTER, KRAMVIG],
    ),
    QueryCase(
        query_id="time_space_compression",
        query="time-space compression and postmodern cultural change",
        relevant=[HARVEY_POSTMODERNITY],
        highly_relevant=[HARVEY_POSTMODERNITY],
        distractors=[GRANOVETTER, KRAMVIG],
    ),
    QueryCase(
        query_id="urban_marxism",
        query="Marxist analysis of urban inequality and spatial justice",
        relevant=[HARVEY_SOCIAL_JUSTICE],
        highly_relevant=[HARVEY_SOCIAL_JUSTICE],
        distractors=[GRANOVETTER, KRAMVIG],
    ),
    # ── Cross-Harvey: multi-paper recall ───────────────────────────────
    QueryCase(
        query_id="harvey_political_economy",
        query="David Harvey political economy and capital accumulation",
        relevant=[HARVEY_NEOLIBERALISM, HARVEY_IMPERIALISM, HARVEY_POSTMODERNITY],
        highly_relevant=[HARVEY_NEOLIBERALISM, HARVEY_IMPERIALISM],
        distractors=[GRANOVETTER, KRAMVIG],
    ),
    # ── Granovetter domain ─────────────────────────────────────────────
    QueryCase(
        query_id="weak_ties",
        query="weak ties in social networks and information diffusion",
        relevant=[GRANOVETTER],
        highly_relevant=[GRANOVETTER],
        distractors=ALL_HARVEY,
    ),
    QueryCase(
        query_id="job_search_networks",
        query="how do people find jobs through social contacts and acquaintances",
        relevant=[GRANOVETTER],
        highly_relevant=[GRANOVETTER],
        distractors=ALL_HARVEY,
    ),
    # ── Kramvig domain ─────────────────────────────────────────────────
    QueryCase(
        query_id="sami_methodology",
        query="Sámi research methodology and indigenous epistemology",
        relevant=[KRAMVIG],
        highly_relevant=[KRAMVIG],
        distractors=ALL_HARVEY + [GRANOVETTER],
    ),
    QueryCase(
        query_id="indigenous_storytelling",
        query="storytelling as a way of knowing in indigenous communities",
        relevant=[KRAMVIG],
        highly_relevant=[KRAMVIG],
        distractors=ALL_HARVEY + [GRANOVETTER],
    ),
    # ── Cross-domain: nothing in corpus ────────────────────────────────
    QueryCase(
        query_id="cyborg_feminism",
        query="cyborg feminism and science technology studies",
        expect_empty=True,
    ),
]
