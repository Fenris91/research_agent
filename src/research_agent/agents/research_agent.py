"""
Research Agent

Autonomous research assistant for social sciences using LangGraph workflow.
Supports multiple LLM backends (Ollama, HuggingFace).
"""

import torch
import logging
from typing import List, Dict, Optional, Callable, Any

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END

from ..models.llm_utils import (
    get_qlora_pipeline,
    get_ollama_pipeline,
    OpenAICompatibleModel,
    check_vram,
    VRAMConstraintError,
    OllamaUnavailableError,
)

logger = logging.getLogger(__name__)


class ResearchState(TypedDict):
    """State passed between agent nodes."""

    messages: Annotated[List[Dict], operator.add]
    current_query: str
    search_query: str  # Concise keywords extracted for API searches
    query_type: str  # "literature_review", "factual", "analysis", "general"
    search_results: List[Dict]
    local_results: List[Dict]
    external_results: List[Dict]
    context: str
    should_search_external: bool
    candidates_for_ingestion: List[Dict]
    final_answer: str
    error: Optional[str]
    # Context from UI selection
    current_researcher: Optional[str]
    current_paper_id: Optional[str]


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
    min_citations: int = 0


class ResearchAgent:
    """
    Autonomous research assistant for social sciences.
    """

    def __init__(
        self,
        vector_store=None,
        embedder=None,
        academic_search=None,
        web_search=None,
        llm_generate: Optional[Callable] = None,
        config: Optional[AgentConfig] = None,
        use_ollama: bool = False,
        provider: Optional[str] = None,
        ollama_model: str = "mistral",
        ollama_base_url: str = "http://localhost:11434",
        openai_model: str = "gpt-4o-mini",
        openai_base_url: str = "https://api.openai.com/v1",
        openai_api_key: Optional[str] = None,
        openai_models: Optional[List[str]] = None,
        openai_compat_base_url: str = "http://localhost:8082/v1",
        openai_compat_api_key: Optional[str] = None,
        openai_compat_models: Optional[List[str]] = None,
    ):
        """
        Initialize the research agent.

        Args:
            use_ollama: If True, use Ollama instead of HuggingFace models
            ollama_model: Ollama model name (e.g., "mistral", "llama2", "neural-chat")
            ollama_base_url: Ollama server URL
        """
        self.model = None
        self.tokenizer = None
        self._load_model_on_demand = True
        self.use_ollama = use_ollama
        self.provider = provider or ("ollama" if use_ollama else "huggingface")
        self._openai_fallback_models = openai_models or []
        self._openai_compat_fallback_models = openai_compat_models or []
        self._ollama_base_url = ollama_base_url
        self._openai_base_url = openai_base_url
        self._openai_api_key = openai_api_key
        self._openai_compat_base_url = openai_compat_base_url
        self._openai_compat_api_key = openai_compat_api_key
        self._ollama_default_model = ollama_model

        if self.provider in {"openai", "openai_compatible"}:
            base_url = (
                self._openai_compat_base_url
                if self.provider == "openai_compatible"
                else self._openai_base_url
            )
            api_key = (
                self._openai_compat_api_key
                if self.provider == "openai_compatible"
                else self._openai_api_key
            )
            fallback_models = (
                self._openai_compat_fallback_models
                if self.provider == "openai_compatible"
                else self._openai_fallback_models
            )
            self.model = OpenAICompatibleModel(
                model_name=openai_model,
                base_url=base_url,
                api_key=api_key,
                fallback_models=fallback_models or [openai_model],
            )
            self._load_model_on_demand = False
            self.use_ollama = False
        elif use_ollama:
            # Try Ollama first
            try:
                self.model = get_ollama_pipeline(
                    model_name=ollama_model, base_url=ollama_base_url
                )
                self._load_model_on_demand = False
                print(f"Using Ollama model: {ollama_model}")
            except OllamaUnavailableError as e:
                print(f"Ollama unavailable: {str(e)}")
                print("Falling back to HuggingFace models...")
                try:
                    self.model, self.tokenizer = get_qlora_pipeline()
                    self._test_vram_on_initialization()
                    self._load_model_on_demand = False
                    self.use_ollama = False
                except VRAMConstraintError as e:
                    print(f"Model loading deferred due to VRAM constraints: {str(e)}")
                    self._load_model_on_demand = True
        else:
            # Try HuggingFace models
            try:
                self.model, self.tokenizer = get_qlora_pipeline()
                self._test_vram_on_initialization()
                self._load_model_on_demand = False
            except VRAMConstraintError as e:
                print(f"Model loading deferred due to VRAM constraints: {str(e)}")
                self._load_model_on_demand = True

        self.vector_store = vector_store
        self.embedder = embedder
        self.academic_search = academic_search
        self.web_search = web_search
        self.llm_generate = llm_generate
        self.config = config or AgentConfig()
        self.ollama_base_url = ollama_base_url

        # Build the workflow graph
        self.graph = self._build_graph()

    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different Ollama model.

        Args:
            model_name: Name of the Ollama model to switch to

        Returns:
            True if successful, False otherwise
        """
        if self.model is None or self.provider not in {
            "ollama",
            "openai",
            "openai_compatible",
        }:
            logger.warning("Cannot switch model: provider does not support switching")
            return False

        try:
            self.model.switch_model(model_name)
            logger.info(f"Switched to model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False

    def get_current_model(self) -> str:
        """Get the name of the currently active model."""
        if self.model and self.provider in {"ollama", "openai", "openai_compatible"}:
            return self.model.model_name
        elif self.tokenizer:
            return self.tokenizer.name_or_path
        return "unknown"

    def list_available_models(self) -> list:
        """List available models for the active provider."""
        if self.model and hasattr(self.model, "list_available_models"):
            try:
                return self.model.list_available_models()
            except Exception as e:
                logger.warning("Model listing failed: %s", e)
        if self.tokenizer:
            return [self.tokenizer.name_or_path]
        return []

    def switch_provider(self, provider: str) -> bool:
        """Switch the LLM provider and reinitialize the model wrapper."""
        if provider not in {"ollama", "openai", "openai_compatible", "huggingface"}:
            logger.warning("Unknown provider: %s", provider)
            return False

        prev_provider = self.provider
        prev_use_ollama = self.use_ollama
        prev_model = self.model
        prev_tokenizer = self.tokenizer
        prev_load_on_demand = self._load_model_on_demand

        try:
            if provider == "ollama":
                model_name = self._ollama_default_model
                self.model = get_ollama_pipeline(
                    model_name=model_name,
                    base_url=self._ollama_base_url,
                )
                self.tokenizer = None
                self.provider = provider
                self.use_ollama = True
                self._load_model_on_demand = False
                return True

            if provider in {"openai", "openai_compatible"}:
                base_url = (
                    self._openai_compat_base_url
                    if provider == "openai_compatible"
                    else self._openai_base_url
                )
                api_key = (
                    self._openai_compat_api_key
                    if provider == "openai_compatible"
                    else self._openai_api_key
                )
                fallback_models = (
                    self._openai_compat_fallback_models
                    if provider == "openai_compatible"
                    else self._openai_fallback_models
                )

                if provider == "openai" and not api_key:
                    logger.warning("OpenAI API key not set; cannot switch provider")
                    return False

                model_name = fallback_models[0] if fallback_models else "gpt-4o-mini"

                self.model = OpenAICompatibleModel(
                    model_name=model_name,
                    base_url=base_url,
                    api_key=api_key,
                    fallback_models=fallback_models or [model_name],
                )
                self.tokenizer = None
                self.provider = provider
                self.use_ollama = False
                self._load_model_on_demand = False
                return True

            # HuggingFace
            try:
                self.model, self.tokenizer = get_qlora_pipeline()
                self._test_vram_on_initialization()
                self.provider = provider
                self.use_ollama = False
                self._load_model_on_demand = False
                return True
            except VRAMConstraintError as e:
                logger.error("HuggingFace load failed: %s", e)
                return False

        except Exception as e:
            logger.error("Provider switch failed: %s", e)
            return False

        finally:
            if self.provider != provider:
                self.provider = prev_provider
                self.use_ollama = prev_use_ollama
                self.model = prev_model
                self.tokenizer = prev_tokenizer
                self._load_model_on_demand = prev_load_on_demand

    def _test_vram_on_initialization(self):
        """
        Confirm VRAM is available after model load
        """
        try:
            check_vram()
        except VRAMConstraintError as e:
            print(f"Initial VRAM warning: {str(e)}")

    def infer(self, prompt: str, max_tokens=512):
        """
        Execute LLM inference with memory safety checks.
        Supports both HuggingFace and Ollama models.
        """
        try:
            if self.provider in {"ollama", "openai", "openai_compatible"}:
                # Use Ollama
                return self.model.generate(
                    prompt, max_tokens=max_tokens, temperature=0.7
                )
            else:
                # Use HuggingFace model
                # Pre-alloc check
                check_vram()

                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                input_length = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.7,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Post-alloc check
                check_vram()

                # Decode only the generated tokens (exclude input)
                generated_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                return response

        except torch.OutOfMemoryError as e:
            # Implement exponential backoff strategy
            max_tokens = max(32, max_tokens // 2)
            print(f"Reducing max_tokens to {max_tokens} due to VRAM constraints")
            return self.infer(prompt, max_tokens)  # Recurse with smaller output

        except VRAMConstraintError as e:
            # Fallback to CPU for critical operations
            print(f"Switching to CPU for critical operation: {e}")
            return self._cpu_inference_fallback(prompt)

        except Exception as e:
            # General failure fallback
            return f"Error: {str(e)}. Try simplifying your query or reducing output length."

    def _cpu_inference_fallback(self, prompt: str):
        """
        Emergency fallback when GPU is unavailable
        """
        try:
            # Move model to CPU
            self.model.to("cpu")
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Small batched generation
            outputs = self.model.generate(
                inputs["input_ids"], max_new_tokens=128, temperature=0.5
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            return f"Critical failure: {str(e)}. Cannot perform operations in current configuration."

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph."""
        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("understand_query", self._understand_query)
        workflow.add_node("search_local", self._search_local)
        workflow.add_node("search_external", self._search_external)
        workflow.add_node("synthesize", self._synthesize)
        workflow.add_node("offer_ingestion", self._offer_ingestion)

        # Add edges
        workflow.set_entry_point("understand_query")
        workflow.add_edge("understand_query", "search_local")

        # Conditional edge: search external if needed
        def should_search_external(state: ResearchState) -> str:
            if state.get("should_search_external", True):
                return "search_external"
            return "synthesize"

        workflow.add_conditional_edges(
            "search_local",
            should_search_external,
            {"search_external": "search_external", "synthesize": "synthesize"},
        )

        workflow.add_edge("search_external", "synthesize")
        workflow.add_edge("synthesize", "offer_ingestion")
        workflow.add_edge("offer_ingestion", END)

        return workflow.compile()

    async def _understand_query(self, state: ResearchState) -> Dict:
        """Understand and classify the user's query."""
        query = state.get("current_query", "")

        if not query:
            return {**state, "error": "No query provided", "query_type": "general"}

        # Use LLM to classify query type
        classification_prompt = f"""Classify this research query into one category.

Query: {query}

Categories:
- literature_review: Requests for overview of research on a topic, state of the field, key papers
- factual: Specific factual questions that need precise answers from sources
- analysis: Requests to analyze, compare, or evaluate concepts, theories, or findings
- general: General questions, greetings, or unclear requests

Respond with ONLY the category name, nothing else."""

        try:
            if self.model:
                response = self.infer(classification_prompt, max_tokens=20)
                response = response.strip().lower()

                # Parse the response
                if "literature" in response:
                    query_type = "literature_review"
                elif "factual" in response:
                    query_type = "factual"
                elif "analysis" in response:
                    query_type = "analysis"
                else:
                    query_type = "general"
            else:
                # Fallback: simple keyword heuristics
                query_lower = query.lower()
                if any(
                    kw in query_lower
                    for kw in [
                        "review",
                        "overview",
                        "state of",
                        "literature",
                        "key papers",
                        "seminal",
                    ]
                ):
                    query_type = "literature_review"
                elif any(
                    kw in query_lower
                    for kw in ["what is", "who is", "when did", "define", "how many"]
                ):
                    query_type = "factual"
                elif any(
                    kw in query_lower
                    for kw in [
                        "compare",
                        "analyze",
                        "evaluate",
                        "difference",
                        "relationship",
                    ]
                ):
                    query_type = "analysis"
                else:
                    query_type = "general"

            logger.info(f"Query classified as: {query_type}")

        except Exception as e:
            logger.error(f"Query classification error: {e}")
            query_type = "general"

        # Extract concise search keywords from the full query
        search_query = self._extract_search_keywords(query)
        logger.info(f"Extracted search keywords: {search_query}")

        return {
            **state,
            "query_type": query_type,
            "search_query": search_query,
            "should_search_external": query_type
            in ["literature_review", "factual", "analysis"],
        }

    def _extract_search_keywords(self, query: str, max_len: int = 200) -> str:
        """Extract concise search keywords from a potentially long user query.

        Uses LLM when available, otherwise falls back to simple truncation.
        """
        # If already short enough, use as-is
        if len(query) <= max_len:
            return query

        # Try LLM extraction
        if self.model:
            try:
                prompt = f"""Extract 3-8 concise academic search keywords from this research query.
Return ONLY the keywords separated by spaces, nothing else.

Query: {query}

Keywords:"""
                keywords = self.infer(prompt, max_tokens=60).strip()
                # Sanity check: if the LLM returned something reasonable
                if 5 < len(keywords) < max_len and "\n" not in keywords:
                    return keywords
            except Exception as e:
                logger.warning(f"LLM keyword extraction failed: {e}")

        # Fallback: take first max_len chars, break at last space
        truncated = query[:max_len]
        last_space = truncated.rfind(" ")
        if last_space > max_len // 2:
            truncated = truncated[:last_space]
        return truncated

    async def _search_local(self, state: ResearchState) -> Dict:
        """Search local vector store for relevant documents across all collections."""
        query = state.get("current_query", "")
        local_results = []

        # Extract context for boosting
        current_researcher = state.get("current_researcher")
        current_paper_id = state.get("current_paper_id")

        if not self.vector_store or not self.embedder:
            logger.warning(
                "Vector store or embedder not configured, skipping local search"
            )
            return {**state, "local_results": [], "should_search_external": True}

        try:
            # Generate query embedding
            if hasattr(self.embedder, "embed_query"):
                query_embedding = self.embedder.embed_query(query)
            elif hasattr(self.embedder, "embed"):
                query_embedding = self.embedder.embed(query)
            elif hasattr(self.embedder, "encode"):
                query_embedding = self.embedder.encode(query)
            else:
                raise ValueError("Embedder does not support query embedding")

            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()

            # Build filters from config (only applies to papers with year metadata)
            filter_dict = None
            if self.config.year_range:
                year_from, year_to = self.config.year_range
                filter_dict = {
                    "$and": [
                        {"year": {"$gte": year_from}},
                        {"year": {"$lte": year_to}},
                    ]
                }

            # If we have a current paper, get it first for context
            context_paper = None
            if current_paper_id:
                context_paper = self.vector_store.get_paper(current_paper_id)
                if context_paper:
                    logger.info(f"Using paper context: {context_paper.get('metadata', {}).get('title', 'Unknown')[:50]}")

            # Search all collections: papers, notes, and web_sources
            # Papers get year filter, others don't
            paper_results = self.vector_store.search(
                query_embedding=query_embedding,
                collection="papers",
                n_results=self.config.max_local_results * 2,  # Get more for re-ranking
                filter_dict=filter_dict,
                query_text=query,
            )

            # Search notes collection (gracefully handle empty/incompatible collections)
            try:
                notes_results = self.vector_store.search(
                    query_embedding=query_embedding,
                    collection="notes",
                    n_results=max(3, self.config.max_local_results // 2),
                    query_text=query,
                )
            except Exception as e:
                logger.debug(f"Notes search skipped: {e}")
                notes_results = {"documents": [], "metadatas": [], "distances": []}

            # Search web sources collection (gracefully handle empty/incompatible collections)
            try:
                web_results = self.vector_store.search(
                    query_embedding=query_embedding,
                    collection="web_sources",
                    n_results=max(3, self.config.max_local_results // 2),
                    query_text=query,
                )
            except Exception as e:
                logger.debug(f"Web sources search skipped: {e}")
                web_results = {"documents": [], "metadatas": [], "distances": []}

            # Format paper results
            for i, doc in enumerate(paper_results.get("documents", [])):
                metadata = (
                    paper_results.get("metadatas", [{}])[i]
                    if paper_results.get("metadatas")
                    else {}
                )
                distance = (
                    paper_results.get("distances", [1.0])[i]
                    if paper_results.get("distances")
                    else 1.0
                )

                local_results.append(
                    {
                        "content": doc,
                        "title": metadata.get("title", "Unknown"),
                        "authors": metadata.get("authors", ""),
                        "year": metadata.get("year"),
                        "paper_id": metadata.get("paper_id", ""),
                        "source": "local_kb",
                        "relevance_score": 1 - distance,
                    }
                )

            # Format notes results
            for i, doc in enumerate(notes_results.get("documents", [])):
                metadata = (
                    notes_results.get("metadatas", [{}])[i]
                    if notes_results.get("metadatas")
                    else {}
                )
                distance = (
                    notes_results.get("distances", [1.0])[i]
                    if notes_results.get("distances")
                    else 1.0
                )

                local_results.append(
                    {
                        "content": doc,
                        "title": metadata.get("title", "Research Note"),
                        "authors": "User Note",
                        "year": None,
                        "paper_id": metadata.get("note_id", ""),
                        "source": "local_note",
                        "tags": metadata.get("tags", ""),
                        "relevance_score": 1 - distance,
                    }
                )

            # Format web source results
            for i, doc in enumerate(web_results.get("documents", [])):
                metadata = (
                    web_results.get("metadatas", [{}])[i]
                    if web_results.get("metadatas")
                    else {}
                )
                distance = (
                    web_results.get("distances", [1.0])[i]
                    if web_results.get("distances")
                    else 1.0
                )

                local_results.append(
                    {
                        "content": doc,
                        "title": metadata.get("title", "Web Source"),
                        "authors": "",
                        "year": None,
                        "paper_id": metadata.get("source_id", ""),
                        "url": metadata.get("url", ""),
                        "source": "local_web",
                        "relevance_score": 1 - distance,
                    }
                )

            # Apply context-based boosting
            if current_researcher or current_paper_id:
                boost_amount = 0.15  # Boost by 15% for context matches

                for result in local_results:
                    # Boost papers by the current researcher
                    if current_researcher:
                        authors = result.get("authors", "").lower()
                        researcher_lower = current_researcher.lower()
                        # Check if researcher name appears in authors
                        if researcher_lower in authors:
                            result["relevance_score"] = min(1.0, result["relevance_score"] + boost_amount)
                            result["context_match"] = "researcher"
                            logger.debug(f"Boosted paper by {current_researcher}: {result.get('title', '')[:40]}")

                    # Boost papers related to the current paper
                    if current_paper_id and context_paper:
                        paper_id = result.get("paper_id", "")
                        # Same paper gets highest boost
                        if paper_id == current_paper_id:
                            result["relevance_score"] = min(1.0, result["relevance_score"] + boost_amount * 2)
                            result["context_match"] = "selected_paper"
                        # Papers with similar authors get a boost
                        elif context_paper.get("metadata", {}).get("authors"):
                            context_authors = context_paper["metadata"]["authors"].lower()
                            result_authors = result.get("authors", "").lower()
                            # Check for author overlap
                            if any(author.strip() in result_authors for author in context_authors.split(",") if len(author.strip()) > 3):
                                result["relevance_score"] = min(1.0, result["relevance_score"] + boost_amount * 0.5)
                                result["context_match"] = "related_author"

            # Sort all results by relevance score
            local_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            # Limit to configured max results after boosting/sorting
            local_results = local_results[:self.config.max_local_results]

            # Log breakdown
            paper_count = len(paper_results.get("documents", []))
            notes_count = len(notes_results.get("documents", []))
            web_count = len(web_results.get("documents", []))
            context_info = ""
            if current_researcher:
                context_info += f", context_researcher: {current_researcher}"
            if current_paper_id:
                context_info += f", context_paper: {current_paper_id[:15]}..."
            logger.info(
                f"Found {len(local_results)} local results "
                f"(papers: {paper_count}, notes: {notes_count}, web: {web_count}{context_info})"
            )

            # Decide if we need external search
            should_search_external = len(
                local_results
            ) < self.config.min_local_results_to_skip_external or state.get(
                "should_search_external", True
            )

        except Exception as e:
            logger.error(f"Local search error: {e}")
            should_search_external = True

        return {
            **state,
            "local_results": local_results,
            "should_search_external": should_search_external,
        }

    async def _search_external(self, state: ResearchState) -> Dict:
        """Search external academic databases and web sources."""
        query = state.get("search_query") or state.get("current_query", "")
        external_results = []
        min_citations = max(0, int(self.config.min_citations or 0))

        # Skip if we have enough local results and shouldn't search external
        if not state.get("should_search_external", True):
            logger.info("Skipping external search - sufficient local results")
            return {**state, "external_results": []}

        # Search academic databases
        if self.academic_search:
            try:
                # Search Semantic Scholar
                papers = await self.academic_search.search_semantic_scholar(
                    query, limit=self.config.max_external_results // 2
                )

                for paper in papers:
                    citation_count = paper.citations or 0
                    if min_citations and citation_count < min_citations:
                        continue
                    external_results.append(
                        {
                            "content": paper.abstract or "",
                            "title": paper.title,
                            "authors": ", ".join(paper.authors)
                            if paper.authors
                            else "",
                            "year": paper.year,
                            "paper_id": paper.id,
                            "doi": paper.doi,
                            "citation_count": citation_count,
                            "url": paper.url,
                            "source": "semantic_scholar",
                        }
                    )

                # Search OpenAlex
                openalex_papers = await self.academic_search.search_openalex(
                    query, limit=self.config.max_external_results // 2
                )

                for paper in openalex_papers:
                    # Avoid duplicates by DOI
                    if paper.doi and any(
                        r.get("doi") == paper.doi for r in external_results
                    ):
                        continue
                    citation_count = paper.citations or 0
                    if min_citations and citation_count < min_citations:
                        continue
                    external_results.append(
                        {
                            "content": paper.abstract or "",
                            "title": paper.title,
                            "authors": ", ".join(paper.authors)
                            if paper.authors
                            else "",
                            "year": paper.year,
                            "paper_id": paper.id,
                            "doi": paper.doi,
                            "citation_count": citation_count,
                            "url": paper.url,
                            "source": "openalex",
                        }
                    )

                logger.info(f"Found {len(external_results)} academic results")

            except Exception as e:
                logger.error(f"Academic search error: {e}")

        # Search web if configured and query type warrants it
        if self.web_search and self.config.include_web_search:
            query_type = state.get("query_type", "general")
            if query_type in ["factual", "general"]:
                try:
                    web_results = await self.web_search.search(query, max_results=5)
                    for result in web_results:
                        title = (
                            result.get("title")
                            if isinstance(result, dict)
                            else getattr(result, "title", "")
                        )
                        url = (
                            result.get("url")
                            if isinstance(result, dict)
                            else getattr(result, "url", "")
                        )
                        content = (
                            result.get("snippet")
                            if isinstance(result, dict)
                            else getattr(result, "content", "")
                        )
                        external_results.append(
                            {
                                "content": content or "",
                                "title": title or "",
                                "url": url or "",
                                "source": "web",
                            }
                        )
                    logger.info(f"Added {len(web_results)} web results")
                except Exception as e:
                    logger.error(f"Web search error: {e}")

        # Identify candidates for ingestion (high-quality external papers)
        candidates = []
        for result in external_results:
            if result.get("source") in ["semantic_scholar", "openalex"]:
                citation_count = result.get("citation_count", 0) or 0
                if citation_count >= 10 and result.get("content"):
                    candidates.append(result)

        return {
            **state,
            "external_results": external_results,
            "candidates_for_ingestion": candidates[:5],  # Top 5 candidates
        }

    async def _synthesize(self, state: ResearchState) -> Dict:
        """Synthesize search results into a comprehensive answer."""
        query = state.get("current_query", "")
        query_type = state.get("query_type", "general")
        local_results = state.get("local_results", [])
        external_results = state.get("external_results", [])
        current_researcher = state.get("current_researcher")
        current_paper_id = state.get("current_paper_id")

        # Combine all results
        all_results = local_results + external_results

        if not all_results and not self.model:
            return {
                **state,
                "final_answer": "I couldn't find relevant information and no LLM is available for generation.",
            }

        # Build synthesis prompt with context
        prompt = self._build_synthesis_prompt(
            query, query_type, all_results,
            current_researcher=current_researcher,
            current_paper_id=current_paper_id,
        )

        # Generate response
        try:
            if self.model:
                answer = self.infer(prompt, max_tokens=1024)
            else:
                # Fallback: format results without LLM
                answer = self._format_results_without_llm(query, all_results)

            logger.info("Synthesis complete")

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            answer = f"Error generating response: {str(e)}"

        return {**state, "final_answer": answer}

    def _build_synthesis_prompt(
        self,
        query: str,
        query_type: str,
        results: List[Dict],
        current_researcher: Optional[str] = None,
        current_paper_id: Optional[str] = None,
    ) -> str:
        """Build a prompt for synthesizing results based on query type."""

        # Map source types to human-readable labels
        source_labels = {
            "local_kb": "Knowledge Base Paper",
            "local_note": "User Research Note",
            "local_web": "Saved Web Source",
            "semantic_scholar": "Semantic Scholar",
            "openalex": "OpenAlex",
            "web": "Web Search",
        }

        # Build context section if we have selection context
        context_section = ""
        if current_researcher or current_paper_id:
            context_parts = []
            if current_researcher:
                context_parts.append(f"The user is currently focused on researcher: {current_researcher}")
            if current_paper_id:
                # Find the paper title from results if possible
                paper_title = None
                for r in results:
                    if r.get("paper_id") == current_paper_id:
                        paper_title = r.get("title")
                        break
                if paper_title:
                    context_parts.append(f"The user is currently focused on the paper: \"{paper_title}\"")
                else:
                    context_parts.append(f"The user has selected a specific paper (ID: {current_paper_id[:20]}...)")
            context_section = "\n\nContext: " + ". ".join(context_parts) + ". Prioritize information relevant to this context."

        # Format sources
        sources_text = ""
        for i, result in enumerate(results[:10], 1):  # Limit to top 10
            title = result.get("title", "Unknown")
            authors = result.get("authors", "")
            year = result.get("year", "")
            content = result.get("content", "")[:500]  # Truncate content
            source = result.get("source", "unknown")
            source_label = source_labels.get(source, source)
            tags = result.get("tags", "")
            url = result.get("url", "")

            sources_text += f"\n[{i}] {title}"
            if authors and authors != "User Note":
                sources_text += f" by {authors}"
            if year:
                sources_text += f" ({year})"
            sources_text += f" [Source: {source_label}]"
            if tags:
                sources_text += f" [Tags: {tags}]"
            if url:
                sources_text += f"\n    URL: {url}"
            if content:
                sources_text += f"\n    {content}...\n"

        if not sources_text:
            sources_text = "\n(No sources found)"

        # Build prompt based on query type
        if query_type == "literature_review":
            instruction = """You are a research assistant. Provide a comprehensive literature review based on the sources.
Structure your response with:
1. Overview of the research landscape
2. Key themes and findings
3. Notable papers and their contributions
4. Research gaps or future directions

Cite sources using [1], [2], etc."""

        elif query_type == "factual":
            instruction = """You are a research assistant. Provide a precise, factual answer based on the sources.
Be specific and cite your sources using [1], [2], etc.
If sources conflict, note the disagreement."""

        elif query_type == "analysis":
            instruction = """You are a research assistant. Provide a critical analysis based on the sources.
Compare different perspectives, evaluate evidence, and draw conclusions.
Cite sources using [1], [2], etc."""

        else:  # general
            instruction = """You are a helpful research assistant. Answer the question based on the available sources.
Be informative and cite sources where relevant using [1], [2], etc."""

        prompt = f"""{instruction}{context_section}

Question: {query}

Available Sources:
{sources_text}

Response:"""

        return prompt

    def _format_results_without_llm(self, query: str, results: List[Dict]) -> str:
        """Format results when no LLM is available."""
        if not results:
            return "No relevant results found for your query."

        # Map source types to human-readable labels
        source_labels = {
            "local_kb": "Knowledge Base",
            "local_note": "Research Note",
            "local_web": "Saved Web Source",
            "semantic_scholar": "Semantic Scholar",
            "openalex": "OpenAlex",
            "web": "Web Search",
        }

        response = f"**Results for:** {query}\n\n"

        for i, result in enumerate(results[:10], 1):
            title = result.get("title", "Unknown")
            authors = result.get("authors", "")
            year = result.get("year", "")
            source = result.get("source", "")
            source_label = source_labels.get(source, source)
            content = result.get("content", "")[:200]
            tags = result.get("tags", "")
            url = result.get("url", "")

            response += f"**[{i}] {title}**"
            if authors and authors != "User Note":
                response += f"\n*{authors}*"
            if year:
                response += f" ({year})"
            response += f"\nSource: {source_label}"
            if tags:
                response += f" | Tags: {tags}"
            response += "\n"
            if url:
                response += f"URL: {url}\n"
            if content:
                response += f"{content}...\n"
            response += "\n"

        return response

    async def _offer_ingestion(self, state: ResearchState) -> Dict:
        """Offer to ingest high-quality external papers into the knowledge base."""
        candidates = state.get("candidates_for_ingestion", [])
        final_answer = state.get("final_answer", "")

        if not candidates or not self.config.auto_ingest:
            return state

        # Format ingestion offer
        if candidates:
            offer = "\n\n---\n**Papers available for your knowledge base:**\n"
            for i, paper in enumerate(candidates[:3], 1):
                title = paper.get("title", "Unknown")
                authors = paper.get("authors", "")
                year = paper.get("year", "")
                citations = paper.get("citation_count", 0)

                offer += f"{i}. {title}"
                if authors:
                    offer += f" - {authors}"
                if year:
                    offer += f" ({year})"
                if citations:
                    offer += f" [{citations} citations]"
                offer += "\n"

            offer += "\n*Use the Knowledge Base tab to add these papers.*"
            final_answer += offer

        return {**state, "final_answer": final_answer}

    async def _run_async(
        self,
        user_query: str,
        search_filters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Internal async implementation of run using LangGraph workflow.

        Args:
            user_query: The user's question
            search_filters: Optional filters (year_from, year_to, min_citations)
            context: Optional context from UI (researcher, paper_id)
        """
        # Extract context
        context = context or {}
        current_researcher = context.get("researcher")
        current_paper_id = context.get("paper_id")

        # Apply search filters to config
        if search_filters:
            if "year_from" in search_filters or "year_to" in search_filters:
                year_from = search_filters.get("year_from", 1900)
                year_to = search_filters.get("year_to", 2030)
                self.config.year_range = (year_from, year_to)
            if (
                "min_citations" in search_filters
                and search_filters.get("min_citations") is not None
            ):
                try:
                    self.config.min_citations = int(
                        search_filters.get("min_citations") or 0
                    )
                except (TypeError, ValueError):
                    self.config.min_citations = 0

        # Initialize state
        initial_state: ResearchState = {
            "messages": [],
            "current_query": user_query,
            "search_query": "",
            "query_type": "general",
            "search_results": [],
            "local_results": [],
            "external_results": [],
            "context": "",
            "should_search_external": True,
            "candidates_for_ingestion": [],
            "final_answer": "",
            "error": None,
            "current_researcher": current_researcher,
            "current_paper_id": current_paper_id,
        }

        result = {
            "query": user_query,
            "answer": "",
            "sources": [],
        }

        if search_filters:
            result["search_filters"] = search_filters

        try:
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)

            # Extract results
            result["answer"] = final_state.get("final_answer", "")
            result["query_type"] = final_state.get("query_type", "general")
            result["local_sources"] = len(final_state.get("local_results", []))
            result["external_sources"] = len(final_state.get("external_results", []))

            # Include source references
            all_sources = final_state.get("local_results", []) + final_state.get(
                "external_results", []
            )
            result["sources"] = [
                {
                    "title": s.get("title"),
                    "source": s.get("source"),
                    "year": s.get("year"),
                }
                for s in all_sources[:10]
            ]

            if final_state.get("error"):
                result["error"] = final_state["error"]

        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            # Fallback to direct inference
            if self.model and (self.tokenizer or self.use_ollama):
                result["answer"] = self.infer(user_query, max_tokens=1024)
                result["fallback"] = True
            else:
                result["answer"] = f"Error processing query: {str(e)}"
                result["status"] = "error"

        return result

    def run(
        self,
        user_query: str,
        search_filters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the research agent (synchronous wrapper).

        Args:
            user_query: The user's question
            search_filters: Optional filters (year_from, year_to, min_citations)
            context: Optional context from UI (researcher, paper_id)
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context (e.g. Gradio) — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self._run_async(user_query, search_filters, context)
                )
                return future.result()
        else:
            return asyncio.run(self._run_async(user_query, search_filters, context))

    async def ingest_paper(self, paper_data: Dict) -> bool:
        """Ingest a new paper into the vector store."""
        return True


def create_research_agent(
    vector_store=None,
    embedder=None,
    academic_search=None,
    web_search=None,
    llm_generate: Optional[Callable] = None,
    config: Optional[AgentConfig] = None,
    use_ollama: bool = False,
    provider: Optional[str] = None,
    ollama_model: str = "mistral",
    ollama_base_url: str = "http://localhost:11434",
    openai_model: str = "gpt-4o-mini",
    openai_base_url: str = "https://api.openai.com/v1",
    openai_api_key: Optional[str] = None,
    openai_models: Optional[List[str]] = None,
    openai_compat_base_url: str = "http://localhost:8082/v1",
    openai_compat_api_key: Optional[str] = None,
    openai_compat_models: Optional[List[str]] = None,
) -> ResearchAgent:
    """Factory function to create a ResearchAgent instance.

    Args:
        use_ollama: If True, use Ollama instead of HuggingFace models
        ollama_model: Ollama model name (e.g., "qwen3:32b", "mistral-small3.2")
        ollama_base_url: Ollama server URL
    """
    return ResearchAgent(
        vector_store=vector_store,
        embedder=embedder,
        academic_search=academic_search,
        web_search=web_search,
        llm_generate=llm_generate,
        config=config,
        use_ollama=use_ollama,
        provider=provider,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        openai_models=openai_models,
        openai_compat_base_url=openai_compat_base_url,
        openai_compat_api_key=openai_compat_api_key,
        openai_compat_models=openai_compat_models,
    )
