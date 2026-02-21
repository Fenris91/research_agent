"""
Mock data for Knowledge Explorer development.

Provides a hardcoded social-sciences graph that exercises all visual features:
varied node sizes, OA rings, all edge types, query pulsing.
"""

from research_agent.explorer.graph_builder import GraphBuilder


def get_mock_graph_data() -> dict:
    """Return a D3-compatible graph dict with mock social sciences data."""
    gb = GraphBuilder()

    # ── Researchers ──────────────────────────────────────────
    haraway = gb.add_researcher({
        "name": "Donna Haraway",
        "openalex_id": "A5010652037",
        "h_index": 82,
        "citations_count": 175000,
        "affiliations": ["University of California, Santa Cruz"],
        "fields": ["Science and Technology Studies", "Feminist Theory", "Cybernetics"],
        "works_count": 198,
    })

    kramvig = gb.add_researcher({
        "name": "Britt Kramvig",
        "openalex_id": "A5048932710",
        "h_index": 14,
        "citations_count": 620,
        "affiliations": ["UiT The Arctic University of Norway"],
        "fields": ["Indigenous Studies", "Tourism Studies", "Cultural Studies"],
        "works_count": 52,
    })

    huse = gb.add_researcher({
        "name": "Tone Huse",
        "openalex_id": "A5091283640",
        "h_index": 8,
        "citations_count": 310,
        "affiliations": ["OsloMet"],
        "fields": ["Urban Studies", "Science and Technology Studies", "Geography"],
        "works_count": 28,
    })

    kristoffersen = gb.add_researcher({
        "name": "Berit Kristoffersen",
        "openalex_id": "A5063718290",
        "h_index": 12,
        "citations_count": 480,
        "affiliations": ["UiT The Arctic University of Norway"],
        "fields": ["Political Geography", "Arctic Studies", "Energy Politics"],
        "works_count": 41,
    })

    # ── Papers (varied citations & OA statuses) ─────────────
    # Citation counts chosen to exercise the paper-size formula:
    # max(7, min(19, 7 + sqrt(citations/80)*4))
    # 0 -> 7, 50 -> 10.2, 200 -> 13.3, 500 -> 17, 720+ -> 19

    p1 = gb.add_paper({
        "id": "ss:108292",
        "title": "A Cyborg Manifesto",
        "year": 1985,
        "authors": ["Donna Haraway"],
        "citations": 820,
        "open_access_url": "https://archive.org/details/cyborg-manifesto",
        "fields": ["Feminist Theory", "Science and Technology Studies"],
        "abstract": "A foundational essay arguing that the boundary between human and machine is a social construction, proposing the cyborg as a feminist myth of political identity.",
    })

    p2 = gb.add_paper({
        "id": "ss:219843",
        "title": "Staying with the Trouble: Making Kin in the Chthulucene",
        "year": 2016,
        "authors": ["Donna Haraway"],
        "citations": 480,
        "open_access_url": None,
        "fields": ["Science and Technology Studies", "Environmental Humanities"],
        "abstract": "Haraway argues for multispecies co-flourishing and response-ability in an era of ecological devastation, proposing sympoietic thinking over autopoietic individualism.",
    })

    p3 = gb.add_paper({
        "id": "ss:337102",
        "title": "When the Companion Species Meet",
        "year": 2003,
        "authors": ["Donna Haraway"],
        "citations": 310,
        "open_access_url": "https://repository.ucsc.edu/companion-species",
        "fields": ["Science and Technology Studies", "Feminist Theory"],
        "abstract": "An exploration of the co-constitutive relationships between humans and companion animals, challenging nature-culture divides.",
    })

    p4 = gb.add_paper({
        "id": "ss:441298",
        "title": "Storytelling as Indigenous Methodology",
        "year": 2014,
        "authors": ["Britt Kramvig"],
        "citations": 55,
        "open_access_url": "https://munin.uit.no/kramvig-storytelling",
        "fields": ["Indigenous Studies", "Methodology"],
        "abstract": "Examines storytelling as a Sámi research methodology that challenges Western epistemological frameworks and centres relational knowledge.",
    })

    p5 = gb.add_paper({
        "id": "ss:552831",
        "title": "Negotiating Energy Futures in the Arctic",
        "year": 2015,
        "authors": ["Berit Kristoffersen", "Britt Kramvig"],
        "citations": 42,
        "open_access_url": "https://munin.uit.no/arctic-energy",
        "fields": ["Political Geography", "Arctic Studies", "Energy Politics"],
        "abstract": "Analyses how local communities, oil companies, and state actors negotiate petroleum futures in northern Norway.",
    })

    p6 = gb.add_paper({
        "id": "ss:661504",
        "title": "Parasites of the City: Waste and Urban Ecologies",
        "year": 2018,
        "authors": ["Tone Huse"],
        "citations": 18,
        "open_access_url": None,
        "fields": ["Urban Studies", "Science and Technology Studies"],
        "abstract": "Huse traces how waste circulations reshape urban ecologies, challenging clean/dirty binaries through ethnographic attention to material flows.",
    })

    p7 = gb.add_paper({
        "id": "ss:774290",
        "title": "Arctic Petroleumscapes and the Politics of Knowledge",
        "year": 2021,
        "authors": ["Berit Kristoffersen"],
        "citations": 28,
        "open_access_url": "https://munin.uit.no/petroleumscapes",
        "fields": ["Political Geography", "Arctic Studies"],
        "abstract": "Examines how petroleum landscapes produce and foreclose political possibilities in the Barents Sea region.",
    })

    p8 = gb.add_paper({
        "id": "ss:883102",
        "title": "Situated Knowledges: The Science Question in Feminism",
        "year": 1988,
        "authors": ["Donna Haraway"],
        "citations": 720,
        "open_access_url": "https://archive.org/details/situated-knowledges",
        "fields": ["Feminist Theory", "Epistemology"],
        "abstract": "Haraway's influential argument for embodied objectivity and partial perspective as the basis for a feminist science.",
    })

    # ── Authorship edges ────────────────────────────────────
    gb.add_authorship_edge(haraway, p1)
    gb.add_authorship_edge(haraway, p2)
    gb.add_authorship_edge(haraway, p3)
    gb.add_authorship_edge(haraway, p8)
    gb.add_authorship_edge(kramvig, p4)
    gb.add_authorship_edge(kramvig, p5)
    gb.add_authorship_edge(kristoffersen, p5)
    gb.add_authorship_edge(kristoffersen, p7)
    gb.add_authorship_edge(huse, p6)

    # ── Citation edges ──────────────────────────────────────
    gb.add_citation_edge(p2, p1, intent="foundational")      # Staying w/ Trouble cites Cyborg
    gb.add_citation_edge(p2, p8, intent="extends")           # Staying w/ Trouble cites Situated
    gb.add_citation_edge(p3, p1, intent="builds_on")         # Companion Species cites Cyborg
    gb.add_citation_edge(p4, p8, intent="foundational")      # Kramvig cites Situated Knowledges
    gb.add_citation_edge(p5, p4, intent="extends")           # Arctic Energy extends Storytelling
    gb.add_citation_edge(p6, p2, intent="engages")           # Huse engages Staying w/ Trouble
    gb.add_citation_edge(p7, p5, intent="extends")           # Petroleumscapes extends Arctic Energy
    gb.add_citation_edge(p4, p2, intent="engages")           # Kramvig engages Staying w/ Trouble

    # ── Semantic similarity edges ───────────────────────────
    gb.add_semantic_edge(p1, p8, score=0.91)
    gb.add_semantic_edge(p2, p3, score=0.85)
    gb.add_semantic_edge(p5, p7, score=0.88)
    gb.add_semantic_edge(p4, p6, score=0.72)
    gb.add_semantic_edge(p1, p2, score=0.78)

    # ── Query node ──────────────────────────────────────────
    gb.add_query("How do cybernetics and feminist theory intersect?", [p1, p8, p2, p3])

    # ── Structural context scaffold ──────────────────────
    gb.build_structural_context()

    return gb.to_dict()
