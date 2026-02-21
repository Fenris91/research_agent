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
    harvey = gb.add_researcher({
        "name": "David Harvey",
        "openalex_id": "A5023888391",
        "h_index": 98,
        "citations_count": 214000,
        "affiliations": ["CUNY Graduate Center"],
        "fields": ["Geography", "Urban Studies", "Political Economy"],
        "works_count": 287,
    })

    massey = gb.add_researcher({
        "name": "Doreen Massey",
        "openalex_id": "A5068120356",
        "h_index": 72,
        "citations_count": 89000,
        "affiliations": ["The Open University"],
        "fields": ["Geography", "Spatial Theory", "Globalization"],
        "works_count": 193,
    })

    lefebvre = gb.add_researcher({
        "name": "Henri Lefebvre",
        "openalex_id": "A5041559961",
        "h_index": 65,
        "citations_count": 156000,
        "affiliations": ["Université de Paris"],
        "fields": ["Philosophy", "Sociology", "Urban Studies"],
        "works_count": 142,
    })

    # ── Papers (varied citations & OA statuses) ─────────────
    # Citation counts chosen to exercise the paper-size formula:
    # max(7, min(19, 7 + sqrt(citations/80)*4))
    # 0 -> 7, 50 -> 10.2, 200 -> 13.3, 500 -> 17, 720+ -> 19

    p1 = gb.add_paper({
        "id": "ss:204928",
        "title": "Social Justice and the City",
        "year": 1973,
        "authors": ["David Harvey"],
        "citations": 620,
        "open_access_url": "https://archive.org/details/socialjusticecit",
        "fields": ["Urban Studies", "Geography"],
        "abstract": "A foundational analysis of how spatial form and social process interact in the urban environment.",
    })

    p2 = gb.add_paper({
        "id": "ss:198231",
        "title": "The Production of Space",
        "year": 1974,
        "authors": ["Henri Lefebvre"],
        "citations": 850,
        "open_access_url": None,
        "fields": ["Philosophy", "Sociology"],
        "abstract": "Lefebvre's key work arguing that space is a social product shaped by ideologies and power relations.",
    })

    p3 = gb.add_paper({
        "id": "ss:302849",
        "title": "For Space",
        "year": 2005,
        "authors": ["Doreen Massey"],
        "citations": 180,
        "open_access_url": "https://arxiv.org/preprint/space2005",
        "fields": ["Geography", "Spatial Theory"],
        "abstract": "Massey proposes a relational understanding of space as the product of interrelations and multiplicity.",
    })

    p4 = gb.add_paper({
        "id": "ss:411523",
        "title": "Rebel Cities: From the Right to the City to the Urban Revolution",
        "year": 2012,
        "authors": ["David Harvey"],
        "citations": 45,
        "open_access_url": "https://oapen.org/rebel-cities",
        "fields": ["Urban Studies", "Political Economy"],
        "abstract": "Harvey examines how cities become sites of anti-capitalist struggle and collective democratic governance.",
    })

    p5 = gb.add_paper({
        "id": "ss:129044",
        "title": "The Right to the City",
        "year": 1968,
        "authors": ["Henri Lefebvre"],
        "citations": 420,
        "open_access_url": None,
        "fields": ["Sociology", "Urban Studies"],
        "abstract": "Lefebvre's seminal essay asserting inhabitants' right to participate in producing urban space.",
    })

    p6 = gb.add_paper({
        "id": "ss:556710",
        "title": "Space, Place and Gender",
        "year": 1994,
        "authors": ["Doreen Massey"],
        "citations": 12,
        "open_access_url": "https://repository.open.ac.uk/spg1994",
        "fields": ["Geography", "Gender Studies"],
        "abstract": "An exploration of the intersection between gender relations and spatial organization.",
    })

    p7 = gb.add_paper({
        "id": "ss:782301",
        "title": "A Brief History of Neoliberalism",
        "year": 2005,
        "authors": ["David Harvey"],
        "citations": 280,
        "open_access_url": "https://arxiv.org/preprint/neolib2005",
        "fields": ["Political Economy", "Geography"],
        "abstract": "A concise history of neoliberalism as a political-economic practice and its global consequences.",
    })

    # ── Authorship edges ────────────────────────────────────
    gb.add_authorship_edge(harvey, p1)
    gb.add_authorship_edge(harvey, p4)
    gb.add_authorship_edge(harvey, p7)
    gb.add_authorship_edge(lefebvre, p2)
    gb.add_authorship_edge(lefebvre, p5)
    gb.add_authorship_edge(massey, p3)
    gb.add_authorship_edge(massey, p6)

    # ── Citation edges ──────────────────────────────────────
    gb.add_citation_edge(p1, p5, intent="foundational")      # Harvey cites Lefebvre
    gb.add_citation_edge(p4, p5, intent="foundational")      # Rebel Cities cites Right to City
    gb.add_citation_edge(p4, p1, intent="extends")           # Rebel Cities cites Social Justice
    gb.add_citation_edge(p3, p2, intent="critiques")         # Massey cites Lefebvre
    gb.add_citation_edge(p6, p2, intent="builds_on")         # Space Place Gender cites Production
    gb.add_citation_edge(p3, p1, intent="engages")           # For Space engages Social Justice

    # ── Semantic similarity edges ───────────────────────────
    gb.add_semantic_edge(p1, p4, score=0.89)
    gb.add_semantic_edge(p2, p5, score=0.92)
    gb.add_semantic_edge(p3, p6, score=0.78)
    gb.add_semantic_edge(p1, p2, score=0.71)

    # ── Query node ──────────────────────────────────────────
    gb.add_query("What is the right to the city?", [p5, p4, p1, p2])

    return gb.to_dict()
