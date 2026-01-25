"""
Research Agent - Main orchestration logic using LangGraph

This module contains the core agent that coordinates:
- Query understanding
- Local knowledge retrieval
- External search (academic APIs, web)
- Response synthesis
- Knowledge ingestion offers
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional, Dict, Any, Callable
import operator

from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


class ResearchState(TypedDict):
    """State passed between agent nodes."""
    messages: Annotated[List[Dict], operator.add]
    current_query: str
    query_type: str  # "literature_review", "factual", "analysis", "general"
    search_results: List[Dict]
    local_results: List[Dict]
    external_results: List[Dict]
    context: str
    should_search_external: bool
    candidates_for_ingestion: List[Dict]
    final_answer: str
    error: Optional[str]


@dataclass
class AgentConfig:
    """Configuration for the research agent."""
    max_local_results: int = 5
    max_external_results: int = 10
    min_local_results_to_skip_external: int = 3
    auto_ingest: bool = False
    auto_ingest_threshold: float = 0.85
    include_web_search: bool = True
    year_range: Optional[tuple] = None


class ResearchAgent:
    """
    Autonomous research assistant for social sciences.

    Capabilities:
    - Literature review and paper discovery
    - Paper summarization
    - Web search for grey literature
    - Data analysis assistance
    - Autonomous knowledge base building

    Example:
        agent = ResearchAgent(vector_store=store, embedder=embedder)
        response = await agent.run("What theories explain urban gentrification?")
    """

    def __init__(
        self,
        vector_store,
        embedder,
        academic_search=None,
        web_search=None,
        llm_generate: Optional[Callable] = None,
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize the research agent.

        Args:
            vector_store: ResearchVectorStore instance
            embedder: EmbeddingModel instance
            academic_search: AcademicSearchTools instance (optional)
            web_search: WebSearchTool instance (optional)
            llm_generate: Optional function for LLM text generation
            config: Agent configuration
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.academic_search = academic_search
        self.web_search = web_search
        self.llm_generate = llm_generate
        self.config = config or AgentConfig()

        # Build the workflow graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("understand_query", self._understand_query)
        workflow.add_node("search_local", self._search_local)
        workflow.add_node("search_external", self._search_external)
        workflow.add_node("synthesize", self._synthesize)
        workflow.add_node("offer_ingestion", self._offer_ingestion)

        # Set entry point
        workflow.set_entry_point("understand_query")

        # Define edges
        workflow.add_edge("understand_query", "search_local")

        # Conditional edge: search external only if needed
        workflow.add_conditional_edges(
            "search_local",
            self._should_search_external,
            {
                "search_external": "search_external",
                "synthesize": "synthesize"
            }
        )

        workflow.add_edge("search_external", "synthesize")
        workflow.add_edge("synthesize", "offer_ingestion")
        workflow.add_edge("offer_ingestion", END)

        return workflow.compile()

    def _should_search_external(self, state: ResearchState) -> str:
        """Decide whether to search external sources."""
        if not state.get("should_search_external", True):
            return "synthesize"

        local_results = state.get("local_results", [])
        if len(local_results) >= self.config.min_local_results_to_skip_external:
            logger.info("Sufficient local results, skipping external search")
            return "synthesize"

        return "search_external"

    async def _understand_query(self, state: ResearchState) -> Dict:
        """Parse and classify the user's query."""
        query = state["current_query"]

        # Simple keyword-based classification
        query_lower = query.lower()

        query_type = "general"
        if any(kw in query_lower for kw in ["literature", "papers", "research on", "studies about", "theories"]):
            query_type = "literature_review"
        elif any(kw in query_lower for kw in ["what is", "define", "explain", "who is", "when did"]):
            query_type = "factual"
        elif any(kw in query_lower for kw in ["analyze", "compare", "statistics", "data", "trends"]):
            query_type = "analysis"

        logger.info(f"Query type: {query_type}")

        return {
            "query_type": query_type,
            "should_search_external": True
        }

    async def _search_local(self, state: ResearchState) -> Dict:
        """Search the local knowledge base."""
        query = state["current_query"]

        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Search all collections
        results = self.vector_store.search_all(
            query_embedding,
            n_results=self.config.max_local_results
        )

        # Combine and format results
        local_results = []

        # Process paper results
        for i, doc in enumerate(results["papers"]["documents"]):
            meta = results["papers"]["metadatas"][i]
            distance = results["papers"]["distances"][i]
            local_results.append({
                "type": "paper",
                "content": doc,
                "title": meta.get("title", "Unknown"),
                "year": meta.get("year"),
                "authors": meta.get("authors", ""),
                "relevance": 1 - distance,  # Convert distance to similarity
                "source": "local_kb"
            })

        # Process note results
        for i, doc in enumerate(results["notes"]["documents"]):
            meta = results["notes"]["metadatas"][i]
            distance = results["notes"]["distances"][i]
            local_results.append({
                "type": "note",
                "content": doc,
                "title": meta.get("title", "Note"),
                "relevance": 1 - distance,
                "source": "local_kb"
            })

        # Process web source results
        for i, doc in enumerate(results["web_sources"]["documents"]):
            meta = results["web_sources"]["metadatas"][i]
            distance = results["web_sources"]["distances"][i]
            local_results.append({
                "type": "web",
                "content": doc,
                "title": meta.get("title", "Web Source"),
                "url": meta.get("url", ""),
                "relevance": 1 - distance,
                "source": "local_kb"
            })

        # Sort by relevance
        local_results.sort(key=lambda x: x["relevance"], reverse=True)

        logger.info(f"Found {len(local_results)} local results")

        return {"local_results": local_results}

    async def _search_external(self, state: ResearchState) -> Dict:
        """Search external academic APIs and web."""
        query = state["current_query"]
        external_results = []

        # Search academic sources
        if self.academic_search:
            try:
                papers = await self.academic_search.search_all(
                    query,
                    limit_per_source=self.config.max_external_results // 2,
                    year_range=self.config.year_range
                )

                for paper in papers:
                    external_results.append({
                        "type": "paper",
                        "content": paper.abstract or f"Title: {paper.title}",
                        "title": paper.title,
                        "year": paper.year,
                        "authors": ", ".join(paper.authors[:3]) if paper.authors else "",
                        "citations": paper.citations,
                        "doi": paper.doi,
                        "url": paper.url,
                        "open_access_url": paper.open_access_url,
                        "source": paper.source,
                        "paper_id": paper.id
                    })

                logger.info(f"Found {len(papers)} papers from academic search")

            except Exception as e:
                logger.error(f"Academic search error: {e}")

        # Search web
        if self.web_search and self.config.include_web_search:
            try:
                web_results = await self.web_search.search(
                    query,
                    max_results=5
                )

                for result in web_results:
                    external_results.append({
                        "type": "web",
                        "content": result.content,
                        "title": result.title,
                        "url": result.url,
                        "source": "web_search"
                    })

                logger.info(f"Found {len(web_results)} web results")

            except Exception as e:
                logger.error(f"Web search error: {e}")

        return {"external_results": external_results}

    async def _synthesize(self, state: ResearchState) -> Dict:
        """Synthesize findings into a coherent response."""
        query = state["current_query"]
        local_results = state.get("local_results", [])
        external_results = state.get("external_results", [])

        # Combine all results
        all_results = local_results + external_results

        if not all_results:
            return {
                "final_answer": f"I couldn't find any relevant information for: {query}",
                "context": ""
            }

        # Build context from results
        context_parts = []

        # Add top results to context
        for i, result in enumerate(all_results[:10], 1):
            source_info = f"[{result['source']}]"
            if result.get("year"):
                source_info += f" ({result['year']})"

            context_parts.append(
                f"{i}. {result['title']} {source_info}\n"
                f"   {result['content'][:300]}..."
            )

        context = "\n\n".join(context_parts)

        # Generate response
        if self.llm_generate:
            # Use LLM for synthesis
            prompt = self._build_synthesis_prompt(query, all_results)
            response = await self.llm_generate(prompt)
        else:
            # Fallback: structured summary without LLM
            response = self._generate_structured_response(query, all_results)

        return {
            "final_answer": response,
            "context": context,
            "search_results": all_results
        }

    def _build_synthesis_prompt(self, query: str, results: List[Dict]) -> str:
        """Build prompt for LLM synthesis."""
        sources_text = ""
        for i, r in enumerate(results[:10], 1):
            sources_text += f"\n[{i}] {r['title']}"
            if r.get("year"):
                sources_text += f" ({r['year']})"
            sources_text += f"\n{r['content'][:500]}\n"

        return f"""You are a research assistant specializing in social sciences.
Based on the following sources, provide a comprehensive answer to the user's question.
Cite sources using [1], [2], etc.

Question: {query}

Sources:
{sources_text}

Provide a well-structured response that:
1. Directly answers the question
2. Synthesizes information from multiple sources
3. Notes any conflicting views or gaps in the literature
4. Uses citations appropriately"""

    def _generate_structured_response(self, query: str, results: List[Dict]) -> str:
        """Generate structured response without LLM."""
        response_parts = [f"## Research Results for: {query}\n"]

        # Group by type
        papers = [r for r in results if r["type"] == "paper"]
        web_sources = [r for r in results if r["type"] == "web"]
        notes = [r for r in results if r["type"] == "note"]

        if papers:
            response_parts.append("\n### Academic Papers\n")
            for p in papers[:5]:
                year = f" ({p['year']})" if p.get("year") else ""
                authors = f" - {p['authors']}" if p.get("authors") else ""
                citations = f" [{p['citations']} citations]" if p.get("citations") else ""
                response_parts.append(f"- **{p['title']}**{year}{authors}{citations}")
                if p.get("content"):
                    response_parts.append(f"  {p['content'][:200]}...")
                response_parts.append("")

        if web_sources:
            response_parts.append("\n### Web Sources\n")
            for w in web_sources[:3]:
                response_parts.append(f"- [{w['title']}]({w.get('url', '')})")
                if w.get("content"):
                    response_parts.append(f"  {w['content'][:150]}...")
                response_parts.append("")

        if notes:
            response_parts.append("\n### From Your Notes\n")
            for n in notes[:3]:
                response_parts.append(f"- {n['title']}: {n['content'][:200]}...")

        return "\n".join(response_parts)

    async def _offer_ingestion(self, state: ResearchState) -> Dict:
        """Identify valuable sources to add to knowledge base."""
        external_results = state.get("external_results", [])

        if not external_results:
            return {"candidates_for_ingestion": []}

        # Filter candidates worth ingesting
        candidates = []

        for result in external_results:
            # Only consider papers with abstracts
            if result["type"] != "paper":
                continue

            if not result.get("content") or len(result.get("content", "")) < 100:
                continue

            # Score based on citations and relevance
            score = 0
            if result.get("citations"):
                if result["citations"] > 100:
                    score += 0.3
                elif result["citations"] > 50:
                    score += 0.2
                elif result["citations"] > 10:
                    score += 0.1

            if result.get("open_access_url"):
                score += 0.1

            if result.get("year") and result["year"] >= 2020:
                score += 0.1

            if score >= 0.2:  # Minimum threshold
                candidates.append({
                    **result,
                    "ingestion_score": score
                })

        # Sort by score
        candidates.sort(key=lambda x: x["ingestion_score"], reverse=True)

        logger.info(f"Found {len(candidates)} candidates for ingestion")

        return {"candidates_for_ingestion": candidates[:5]}

    async def run(self, user_query: str) -> Dict[str, Any]:
        """
        Run the agent on a user query.

        Args:
            user_query: Natural language research question

        Returns:
            Dict with final_answer, search_results, candidates_for_ingestion
        """
        initial_state: ResearchState = {
            "messages": [{"role": "user", "content": user_query}],
            "current_query": user_query,
            "query_type": "",
            "search_results": [],
            "local_results": [],
            "external_results": [],
            "context": "",
            "should_search_external": True,
            "candidates_for_ingestion": [],
            "final_answer": "",
            "error": None
        }

        try:
            final_state = await self.graph.ainvoke(initial_state)

            return {
                "answer": final_state.get("final_answer", ""),
                "query_type": final_state.get("query_type", ""),
                "local_results": final_state.get("local_results", []),
                "external_results": final_state.get("external_results", []),
                "candidates_for_ingestion": final_state.get("candidates_for_ingestion", []),
                "context": final_state.get("context", "")
            }

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "error": str(e)
            }

    async def ingest_paper(self, paper_data: Dict) -> bool:
        """
        Ingest a paper into the knowledge base.

        Args:
            paper_data: Paper data from search results

        Returns:
            True if successfully ingested
        """
        try:
            content = paper_data.get("content", "")
            if not content:
                logger.warning("No content to ingest")
                return False

            # Create chunks (simple splitting for abstracts)
            chunks = [content]  # For abstracts, single chunk is usually enough

            # Generate embeddings
            embeddings = self.embedder.embed_batch(chunks)

            # Prepare metadata
            metadata = {
                "title": paper_data.get("title", "Unknown"),
                "year": paper_data.get("year"),
                "authors": paper_data.get("authors", ""),
                "doi": paper_data.get("doi"),
                "url": paper_data.get("url"),
                "citations": paper_data.get("citations"),
                "source_api": paper_data.get("source", "unknown")
            }

            # Add to vector store
            paper_id = paper_data.get("paper_id") or paper_data.get("doi") or f"paper_{datetime.now().timestamp()}"

            self.vector_store.add_paper(
                paper_id=paper_id,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )

            logger.info(f"Ingested paper: {metadata['title']}")
            return True

        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            return False


# Factory function for easy creation
def create_research_agent(
    persist_dir: str = "./data/chroma_db",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    include_academic_search: bool = True,
    include_web_search: bool = True
) -> ResearchAgent:
    """
    Create a research agent with default configuration.

    Args:
        persist_dir: Directory for vector store
        embedding_model: Embedding model name
        include_academic_search: Enable academic search
        include_web_search: Enable web search

    Returns:
        Configured ResearchAgent instance
    """
    from src.db import ResearchVectorStore, EmbeddingModel
    from src.tools import AcademicSearchTools, WebSearchTool

    # Initialize components
    vector_store = ResearchVectorStore(persist_dir)
    embedder = EmbeddingModel(embedding_model)

    academic_search = AcademicSearchTools() if include_academic_search else None
    web_search = WebSearchTool(provider="duckduckgo") if include_web_search else None

    return ResearchAgent(
        vector_store=vector_store,
        embedder=embedder,
        academic_search=academic_search,
        web_search=web_search
    )
