"""
Citation Explorer for analyzing academic citation networks.

This module provides functionality to:
- Get papers that cite a given paper
- Get papers cited by a given paper  
- Find highly connected papers in a citation network
- Build citation graphs for visualization
- Suggest related papers based on citation overlap
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

from research_agent.tools.academic_search import AcademicSearchTools


@dataclass
class CitationPaper:
    """Represents a paper in a citation network."""
    paper_id: str
    title: str
    year: Optional[int] = None
    authors: Optional[List[str]] = None
    citation_count: Optional[int] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None


@dataclass 
class CitationNetwork:
    """Represents a citation network."""
    seed_paper: CitationPaper
    citing_papers: List[CitationPaper]
    cited_papers: List[CitationPaper]
    highly_connected: List[CitationPaper]


class CitationExplorer:
    """
    Citation Explorer for analyzing academic citation networks.
    
    Allows exploration of citation relationships to discover:
    - Foundational works (highly cited papers)
    - Recent developments (papers citing the seed)
    - Related work (shared citations)
    """
    
    def __init__(self, academic_search: AcademicSearchTools):
        """
        Initialize citation explorer.
        
        Args:
            academic_search: AcademicSearchTools instance
        """
        self.search = academic_search
        
    async def get_citations(
        self, 
        paper_id: str, 
        direction: str = "both",
        limit: int = 20
    ) -> CitationNetwork:
        """
        Get citation relationships for a paper.
        
        Args:
            paper_id: Paper ID (Semantic Scholar or OpenAlex)
            direction: 'citing', 'cited', or 'both'
            limit: Maximum number of papers to fetch per direction
            
        Returns:
            CitationNetwork with citation relationships
        """
        # Get seed paper details
        seed_paper = await self._get_paper_details(paper_id)
        
        citing_papers = []
        cited_papers = []
        
        if direction in ["citing", "both"]:
            citing_papers = await self._get_citing_papers(paper_id, limit)
            
        if direction in ["cited", "both"]:  
            cited_papers = await self._get_cited_papers(paper_id, limit)
        
        # Find highly connected papers
        highly_connected = await self.find_highly_connected(
            [p.paper_id for p in citing_papers + cited_papers]
        )
        
        return CitationNetwork(
            seed_paper=seed_paper,
            citing_papers=citing_papers,
            cited_papers=cited_papers,
            highly_connected=highly_connected
        )
    
    async def find_highly_connected(
        self, 
        paper_ids: List[str],
        min_connections: int = 2
    ) -> List[CitationPaper]:
        """
        Find papers frequently cited by the given papers.
        
        Args:
            paper_ids: List of paper IDs to analyze
            min_connections: Minimum number of connections to include
            
        Returns:
            List of highly connected papers sorted by connection count
        """
        citation_counts = {}
        
        for pid in paper_ids:
            try:
                # Get references for this paper
                refs = await self._get_cited_papers(pid, limit=50)
                
                for ref in refs:
                    ref_id = ref.paper_id
                    if ref_id not in citation_counts:
                        citation_counts[ref_id] = {
                            "paper": ref,
                            "count": 0
                        }
                    citation_counts[ref_id]["count"] += 1
            except Exception as e:
                print(f"Error processing paper {pid}: {e}")
                continue
        
        # Filter and sort by connection count
        connected_papers = [
            {
                **item["paper"].__dict__,
                "connection_count": item["count"]
            }
            for item in citation_counts.values()
            if item["count"] >= min_connections
        ]
        
        connected_papers.sort(
            key=lambda x: x["connection_count"], 
            reverse=True
        )
        
        # Convert back to CitationPaper objects
        result = []
        for item in connected_papers[:10]:  # Top 10
            paper_dict = item.copy()
            paper_dict.pop("connection_count", None)
            result.append(CitationPaper(**paper_dict))
        
        return result
    
    async def suggest_related(
        self, 
        paper_id: str,
        limit: int = 10
    ) -> List[CitationPaper]:
        """
        Suggest related papers based on citation overlap.
        
        Args:
            paper_id: Paper ID to find related papers for
            limit: Maximum number of related papers to return
            
        Returns:
            List of related papers with overlap scores
        """
        # Get citations for the target paper
        network = await self.get_citations(paper_id, direction="both", limit=50)
        
        # Find papers that cite many of the same references
        overlap_scores = {}
        target_refs = {p.paper_id for p in network.cited_papers}
        
        for citing_paper in network.citing_papers:
            try:
                # Get what this citing paper references
                citing_refs = await self._get_cited_papers(citing_paper.paper_id, limit=50)
                citing_ref_ids = {p.paper_id for p in citing_refs}
                
                # Calculate overlap
                overlap = len(target_refs & citing_ref_ids)
                if overlap > 0:
                    overlap_scores[citing_paper.paper_id] = {
                        "paper": citing_paper,
                        "overlap_score": overlap,
                        "overlap_percentage": overlap / len(target_refs)
                    }
            except Exception as e:
                continue
        
        # Sort by overlap score
        sorted_papers = sorted(
            overlap_scores.items(),
            key=lambda x: x[1]["overlap_score"],
            reverse=True
        )
        
        result = []
        for paper_id, data in sorted_papers[:limit]:
            paper = data["paper"]
            result.append(paper)
        
        return result
    
    def build_network_data(self, network: CitationNetwork) -> Dict[str, Any]:
        """
        Build network data for visualization.
        
        Args:
            network: CitationNetwork to visualize
            
        Returns:
            Dictionary with nodes and edges for network visualization
        """
        nodes = []
        edges = []
        
        # Add seed paper
        seed_id = f"seed_{network.seed_paper.paper_id}"
        nodes.append({
            "id": seed_id,
            "label": network.seed_paper.title[:50] + "...",
            "type": "seed",
            "year": network.seed_paper.year,
            "citation_count": network.seed_paper.citation_count
        })
        
        # Add citing papers (papers that cite the seed)
        for i, paper in enumerate(network.citing_papers):
            citing_id = f"citing_{i}"
            nodes.append({
                "id": citing_id,
                "label": paper.title[:50] + "...",
                "type": "citing",
                "year": paper.year,
                "citation_count": paper.citation_count
            })
            edges.append({
                "from": citing_id,
                "to": seed_id,
                "type": "cites"
            })
        
        # Add cited papers (papers cited by the seed)
        for i, paper in enumerate(network.cited_papers):
            cited_id = f"cited_{i}"
            nodes.append({
                "id": cited_id,
                "label": paper.title[:50] + "...", 
                "type": "cited",
                "year": paper.year,
                "citation_count": paper.citation_count
            })
            edges.append({
                "from": seed_id,
                "to": cited_id,
                "type": "cites"
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_papers": len(nodes),
                "citing_count": len(network.citing_papers),
                "cited_count": len(network.cited_papers),
                "highly_connected_count": len(network.highly_connected)
            }
        }
    
    async def _get_paper_details(self, paper_id: str) -> CitationPaper:
        """Get detailed information about a paper."""
        try:
            # Use semantic scholar API endpoint
            client = await self.search._get_client()
            response = await client.get(
                f"{self.search.SEMANTIC_SCHOLAR_API}/paper/{paper_id}",
                params={
                    "fields": "paperId,title,year,authors,citationCount,abstract,venue,externalIds"
                }
            )
            response.raise_for_status()
            s2_data = response.json()

            return CitationPaper(
                paper_id=s2_data.get("paperId", paper_id),
                title=s2_data.get("title", "Unknown Title"),
                year=s2_data.get("year"),
                authors=[a.get("name", "") for a in s2_data.get("authors", [])],
                citation_count=s2_data.get("citationCount"),
                abstract=s2_data.get("abstract"),
                venue=s2_data.get("venue"),
                url=None
            )
        except Exception as e:
            print(f"Error getting paper details: {e}")
            # Return basic info
            return CitationPaper(
                paper_id=paper_id,
                title=f"Paper {paper_id}",
                year=None,
                authors=[],
                citation_count=0,
                abstract=None,
                venue=None,
                url=None
            )

    async def _get_citing_papers(self, paper_id: str, limit: int) -> List[CitationPaper]:
        """Get papers that cite the given paper."""
        try:
            client = await self.search._get_client()
            response = await client.get(
                f"{self.search.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/citations",
                params={
                    "fields": "citingPaper.paperId,citingPaper.title,citingPaper.year,citingPaper.citationCount",
                    "limit": limit
                }
            )
            response.raise_for_status()
            data = response.json()

            citing_papers = []
            for citation_data in data.get("data", []):
                citing_paper = citation_data.get("citingPaper", {})
                if citing_paper and citing_paper.get("paperId"):
                    paper = CitationPaper(
                        paper_id=citing_paper.get("paperId"),
                        title=citing_paper.get("title", "Unknown"),
                        year=citing_paper.get("year"),
                        authors=[],
                        citation_count=citing_paper.get("citationCount"),
                        abstract=None,
                        venue=None,
                        url=None
                    )
                    citing_papers.append(paper)

            return citing_papers
        except Exception as e:
            print(f"Error getting citing papers: {e}")
            return []

    async def _get_cited_papers(self, paper_id: str, limit: int) -> List[CitationPaper]:
        """Get papers cited by the given paper."""
        try:
            client = await self.search._get_client()
            response = await client.get(
                f"{self.search.SEMANTIC_SCHOLAR_API}/paper/{paper_id}/references",
                params={
                    "fields": "citedPaper.paperId,citedPaper.title,citedPaper.year,citedPaper.citationCount",
                    "limit": limit
                }
            )
            response.raise_for_status()
            data = response.json()

            cited_papers = []
            for ref_data in data.get("data", []):
                cited_paper = ref_data.get("citedPaper", {})
                if cited_paper and cited_paper.get("paperId"):
                    paper = CitationPaper(
                        paper_id=cited_paper.get("paperId"),
                        title=cited_paper.get("title", "Unknown"),
                        year=cited_paper.get("year"),
                        authors=[],
                        citation_count=cited_paper.get("citationCount"),
                        abstract=None,
                        venue=None,
                        url=None
                    )
                    cited_papers.append(paper)

            return cited_papers
        except Exception as e:
            print(f"Error getting cited papers: {e}")
            return []


# UI Components for Gradio
import gradio as gr

def render_citation_explorer():
    """Render citation explorer UI component."""
    with gr.Column():
        gr.Markdown("## 🧬 Citation Network Explorer")
        gr.Markdown("Explore citation relationships and discover influential papers in academic networks.")
        
        with gr.Row():
            with gr.Column(scale=2):
                paper_input = gr.Textbox(
                    label="Paper ID or Title",
                    placeholder="Enter Semantic Scholar paper ID or search by title",
                    value=""
                )
                
            with gr.Column(scale=1):
                direction = gr.Radio(
                    choices=["both", "citing", "cited"],
                    value="both",
                    label="Direction",
                    info="What citation relationships to explore"
                )
                
                depth = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Search Depth",
                    info="Maximum papers to fetch per direction"
                )
        
        with gr.Row():
            search_btn = gr.Button("🔍 Explore Citations", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        # Results sections
        with gr.Accordion("📊 Citation Network Summary", open=True):
            summary_output = gr.Markdown("Enter a paper ID to explore its citation network.")
        
        with gr.Tabs():
            with gr.TabItem("📄 Papers That Cite This"):
                citing_output = gr.DataFrame(
                    headers=["Title", "Year", "Citations", "Paper ID"],
                    datatype=["str", "number", "number", "str"],
                    interactive=False,
                    height=300
                )
            
            with gr.TabItem("📚 Papers Cited by This"):
                cited_output = gr.DataFrame(
                    headers=["Title", "Year", "Citations", "Paper ID"],
                    datatype=["str", "number", "number", "str"],
                    interactive=False,
                    height=300
                )
            
            with gr.TabItem("⭐ Highly Connected Papers"):
                connected_output = gr.DataFrame(
                    headers=["Title", "Year", "Citations", "Paper ID"],
                    datatype=["str", "number", "number", "str"],
                    interactive=False,
                    height=300
                )
            
            with gr.TabItem("🔗 Related Papers"):
                related_output = gr.DataFrame(
                    headers=["Title", "Year", "Citations", "Paper ID"],
                    datatype=["str", "number", "number", "str"],
                    interactive=False,
                    height=300
                )
        
        # Network visualization (placeholder for now)
        with gr.Accordion("🌐 Network Visualization", open=False):
            network_plot = gr.Plot(label="Citation Network Graph")
            gr.Markdown("*Network visualization coming soon!*")
        
        # Event handlers
        search_btn.click(
            fn=explore_citations,
            inputs=[paper_input, direction, depth],
            outputs=[summary_output, citing_output, cited_output, connected_output, related_output]
        )
        
        clear_btn.click(
            fn=lambda: ["", None, None, None, None],
            outputs=[summary_output, citing_output, cited_output, connected_output, related_output]
        )
    
    return {
        "paper_input": paper_input,
        "direction": direction,
        "depth": depth,
        "search_btn": search_btn,
        "summary_output": summary_output,
        "citing_output": citing_output,
        "cited_output": cited_output,
        "connected_output": connected_output,
        "related_output": related_output
    }


async def explore_citations(paper_input: str, direction: str, depth: int):
    """
    Explore citation relationships for a paper.
    
    Args:
        paper_input: Paper ID or title
        direction: Citation direction to explore
        depth: Search depth limit
        
    Returns:
        Tuple of (summary, citing_df, cited_df, connected_df, related_df)
    """
    if not paper_input.strip():
        return "Please enter a paper ID or title.", None, None, None, None
    
    try:
        # Initialize search tools and explorer
        search_tools = AcademicSearchTools()
        explorer = CitationExplorer(search_tools)
        
        # If it's not a paper ID format, try to search for it
        if not paper_input.startswith("10.") and len(paper_input) < 20:
            # Try to search by title
            papers = await search_tools.search_semantic_scholar(paper_input, limit=1)
            if papers:
                paper_id = papers[0].id
                paper_title = papers[0].title
            else:
                return f"No papers found for: {paper_input}", None, None, None, None
        else:
            paper_id = paper_input
            paper_title = paper_input
        
        # Get citation network
        network = await explorer.get_citations(paper_id, direction, depth)
        
        # Build summary
        summary = f"""## 📊 Citation Network Summary

**Seed Paper:** {network.seed_paper.title}
- **Year:** {network.seed_paper.year or 'Unknown'}
- **Citations:** {network.seed_paper.citation_count or 0}

**Network Statistics:**
- 📄 Papers that cite this: {len(network.citing_papers)}
- 📚 Papers cited by this: {len(network.cited_papers)}
- ⭐ Highly connected papers: {len(network.highly_connected)}
"""
        
        # Convert to DataFrames
        citing_df = _papers_to_dataframe(network.citing_papers)
        cited_df = _papers_to_dataframe(network.cited_papers)
        connected_df = _papers_to_dataframe(network.highly_connected)
        
        # Get related papers
        related_papers = await explorer.suggest_related(paper_id, limit=depth)
        related_df = _papers_to_dataframe(related_papers)
        
        await search_tools.close()
        
        return summary, citing_df, cited_df, connected_df, related_df
        
    except Exception as e:
        error_msg = f"Error exploring citations: {str(e)}"
        return error_msg, None, None, None, None


def _papers_to_dataframe(papers):
    """Convert list of CitationPaper objects to DataFrame format."""
    if not papers:
        return None
    
    data = []
    for paper in papers:
        data.append([
            paper.title or "Unknown Title",
            paper.year or "Unknown",
            paper.citation_count or 0,
            paper.paper_id
        ])
    
    return data
