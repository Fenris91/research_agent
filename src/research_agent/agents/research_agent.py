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

from ..utils.openalex import SOURCE_LABELS, SOURCE_LABELS_LONG
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
    local_results: List[Dict]
    external_results: List[Dict]
    should_search_external: bool
    candidates_for_ingestion: List[Dict]
    final_answer: str
    error: Optional[str]
    # Context from UI selection
    current_researcher: Optional[str]
    current_paper_id: Optional[str]
    # Context items from pill strip
    auth_context_items: Optional[List[str]]
    chat_context_items: Optional[List[str]]


@dataclass
class AgentConfig:
    """Configuration for the research agent."""

    max_local_results: int = 5
    max_external_results: int = 10
    min_local_results_to_skip_external: int = 3
    auto_ingest: bool = False
    auto_ingest_threshold: float = 0.85
    include_web_search: bool = True
    include_core_search: bool = True
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
        canonical_provider: Optional[str] = None,
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
        self._canonical_provider = canonical_provider or self.provider
        self._openai_fallback_models = openai_models or []
        self._openai_compat_fallback_models = openai_compat_models or []
        self._ollama_base_url = ollama_base_url
        self._openai_base_url = openai_base_url
        self._openai_api_key = openai_api_key
        self._openai_compat_base_url = openai_compat_base_url
        self._openai_compat_api_key = openai_compat_api_key
        self._ollama_default_model = ollama_model

        if self.provider == "none":
            # No LLM — retrieval-only mode
            self.model = None
            self._load_model_on_demand = False
            self.use_ollama = False
            print("Running in retrieval-only mode (no LLM)")
        elif self.provider in {"openai", "openai_compatible"}:
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
        self._pipeline: dict[str, str] = {}  # task_type → model_name overrides

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
        if self.provider == "none":
            return "retrieval-only"
        if self.model and self.provider in {"ollama", "openai", "openai_compatible"}:
            return self.model.model_name
        elif self.tokenizer:
            return self.tokenizer.name_or_path
        return "unknown"

    @property
    def is_claude(self) -> bool:
        """True when the underlying provider is Anthropic (native tool-use eligible)."""
        return self._canonical_provider == "anthropic" and bool(self._openai_api_key)

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
        if provider not in {"ollama", "openai", "openai_compatible", "huggingface", "none"}:
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

    def connect_provider(self, provider_key: str, api_key: str) -> bool:
        """Connect to a cloud provider with a user-supplied API key (BYOK)."""
        from research_agent.main import CLOUD_PROVIDERS

        if provider_key not in CLOUD_PROVIDERS:
            logger.warning("Unknown BYOK provider: %s", provider_key)
            return False

        cloud_cfg = CLOUD_PROVIDERS[provider_key]
        base_url = cloud_cfg.get("base_url")
        if not base_url:
            logger.error("No base_url configured for provider: %s", provider_key)
            return False

        if provider_key == "anthropic" and not api_key.startswith("sk-ant-"):
            logger.warning("Anthropic keys start with 'sk-ant-' — check your key")
            return False

        try:
            self.model = OpenAICompatibleModel(
                model_name=cloud_cfg["default_model"],
                base_url=base_url,
                api_key=api_key,
                fallback_models=cloud_cfg["models"],
            )
            self.tokenizer = None
            self._canonical_provider = provider_key
            self.provider = "openai"
            self.use_ollama = False
            self._load_model_on_demand = False
            self._openai_api_key = api_key
            self._openai_base_url = base_url
            self._openai_fallback_models = cloud_cfg["models"]
            logger.info("Connected to %s via BYOK", cloud_cfg["name"])
            return True
        except Exception as e:
            logger.error("BYOK connection failed for %s: %s", provider_key, e)
            return False

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
                # Use API-based model (Ollama, OpenAI, Groq, Anthropic, etc.)
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
            if max_tokens <= 32:
                return "Error: Insufficient VRAM even for minimal generation."
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

    def configure_pipeline(self, pipeline: dict[str, str]):
        """Set per-task model overrides.

        Args:
            pipeline: Mapping of task type to model name.
                      Valid task types: classify, extract_keywords, synthesize.
        """
        valid_tasks = {"classify", "extract_keywords", "synthesize"}
        rejected = {k for k in pipeline if k not in valid_tasks}
        if rejected:
            logger.warning(
                "Pipeline: ignoring unknown task types: %s (valid: %s)",
                ", ".join(sorted(rejected)),
                ", ".join(sorted(valid_tasks)),
            )

        self._pipeline = {k: v for k, v in pipeline.items() if k in valid_tasks}

        if self._pipeline:
            for task, model in sorted(self._pipeline.items()):
                logger.info("Pipeline: %s -> %s", task, model)

            # Soft-warn if the model is not in the provider's known model list.
            # Wrapped in try/except so partial init (e.g. tests) never crashes.
            try:
                known_models = self.list_available_models()
            except Exception:
                known_models = []
            if known_models:
                for task, model in self._pipeline.items():
                    if model not in known_models:
                        logger.warning(
                            "Pipeline: model '%s' (task '%s') not in known model "
                            "list %s — it may still work if the provider accepts it",
                            model, task, known_models[:6],
                        )

    def task_infer(self, task: str, prompt: str, max_tokens: int = 512) -> str:
        """Run inference with an optional per-task model override.

        If ``self._pipeline`` has an entry for *task*, the model is temporarily
        switched before calling ``infer()``, then restored afterwards.
        """
        override = self._pipeline.get(task)
        if not override or not self.model or not hasattr(self.model, "switch_model"):
            return self.infer(prompt, max_tokens=max_tokens)

        original = getattr(self.model, "model_name", None)
        try:
            self.model.switch_model(override)
            return self.infer(prompt, max_tokens=max_tokens)
        finally:
            if original is not None:
                self.model.switch_model(original)

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

        # Fast pre-check: detect obvious casual/greeting messages without calling the LLM
        query_lower = query.lower().strip().rstrip("!?.,")
        casual_exact = {
            "hi", "hello", "hey", "howdy", "yo", "sup", "heya", "hiya",
            "how are you", "how r u", "how's it going", "hows it going",
            "whats up", "what's up", "wassup",
            "good morning", "good evening", "good afternoon", "good night",
            "thanks", "thank you", "thx", "ty", "appreciate it",
            "bye", "goodbye", "see you", "later", "cya",
            "ok", "okay", "k", "cool", "nice", "great", "awesome",
            "sure", "yes", "no", "yep", "nope", "yeah", "nah",
            "help", "what can you do", "who are you", "what are you",
            "how do you work", "what is this",
        }
        casual_prefixes = (
            "hi ", "hey ", "hello ", "thanks ", "thank you ",
            "good morning", "good evening", "good afternoon",
        )
        is_casual = (
            query_lower in casual_exact
            or any(query_lower.startswith(p) for p in casual_prefixes)
            or len(query_lower) <= 3
        )
        if is_casual:
            logger.info("Query classified as: general (casual pattern match)")
            return {
                **state,
                "query_type": "general",
                "search_query": "",
                "should_search_external": False,
            }

        # Use LLM to classify query type
        classification_prompt = f"""Classify this query into one category.

Query: {query}

Categories:
- literature_review: Requests for overview of research on a topic, state of the field, key papers, or what a researcher/paper/report says
- factual: Specific factual questions about research content that need precise answers from sources
- analysis: Requests to analyze, compare, or evaluate concepts, theories, or findings from research
- general: Greetings, casual conversation, administrative questions, anything NOT about research content

Rules:
- If the query mentions a researcher name, topic, or paper — even casually like "tell me about X" — classify as "literature_review"
- If the query asks about what a paper/report says, classify as "literature_review" or "factual"
- ONLY classify as "general" for pure greetings ("hello", "hi") or meta questions about your capabilities
- When in doubt between "general" and a research category, choose the research category

Respond with ONLY the category name, nothing else."""

        try:
            if self.model:
                response = self.task_infer("classify", classification_prompt, max_tokens=20)
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
                    for kw in ["what is", "who is", "when did", "define", "how many",
                               "tell me about", "what do you know about"]
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
        # Always try LLM extraction for better search queries
        if self.model:
            try:
                prompt = f"""Extract 3-8 concise academic search keywords from this research query.
Focus on the research TOPIC, not filler words like "key concepts" or "work".
If a researcher name is mentioned, include their name as a keyword.
Return ONLY the keywords separated by spaces, nothing else.

Query: {query}

Keywords:"""
                keywords = self.task_infer("extract_keywords", prompt, max_tokens=60).strip()
                # Sanity check: if the LLM returned something reasonable
                if 5 < len(keywords) < max_len and "\n" not in keywords:
                    return keywords
            except Exception as e:
                logger.warning(f"LLM keyword extraction failed: {e}")

        # Fallback: use query as-is if short, otherwise truncate
        if len(query) <= max_len:
            return query
        truncated = query[:max_len]
        last_space = truncated.rfind(" ")
        if last_space > max_len // 2:
            truncated = truncated[:last_space]
        return truncated

    async def _search_local(self, state: ResearchState) -> Dict:
        """Search local vector store for relevant documents across all collections."""
        query = state.get("current_query", "")
        local_results = []

        # Skip vector search for casual messages (search_query is "" for casual)
        if state.get("query_type") == "general" and not state.get("search_query"):
            logger.info("Skipping local search for casual/general message")
            return {**state, "local_results": [], "should_search_external": False}

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

            # Search all collections concurrently (ChromaDB is sync, so use executor)
            _empty = {"documents": [], "metadatas": [], "distances": []}
            loop = asyncio.get_running_loop()

            def _search_papers():
                return self.vector_store.search(
                    query_embedding=query_embedding,
                    collection="papers",
                    n_results=self.config.max_local_results * 2,
                    filter_dict=filter_dict,
                    query_text=query,
                )

            def _search_notes():
                try:
                    return self.vector_store.search(
                        query_embedding=query_embedding,
                        collection="notes",
                        n_results=max(3, self.config.max_local_results // 2),
                        query_text=query,
                    )
                except Exception as e:
                    logger.debug(f"Notes search skipped: {e}")
                    return _empty

            def _search_web():
                try:
                    return self.vector_store.search(
                        query_embedding=query_embedding,
                        collection="web_sources",
                        n_results=max(3, self.config.max_local_results // 2),
                        query_text=query,
                    )
                except Exception as e:
                    logger.debug(f"Web sources search skipped: {e}")
                    return _empty

            paper_results, notes_results, web_results = await asyncio.gather(
                loop.run_in_executor(None, _search_papers),
                loop.run_in_executor(None, _search_notes),
                loop.run_in_executor(None, _search_web),
            )

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

                raw_authors = metadata.get("authors", "")
                authors_str = ", ".join(raw_authors) if isinstance(raw_authors, list) else (raw_authors or "")
                local_results.append(
                    {
                        "content": doc,
                        "title": metadata.get("title", "Unknown"),
                        "authors": authors_str,
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

                def _str_authors(val):
                    """Coerce authors to string (may be list from ChromaDB metadata)."""
                    if isinstance(val, list):
                        return ", ".join(str(a) for a in val)
                    return str(val) if val else ""

                for result in local_results:
                    # Boost papers by the current researcher
                    if current_researcher:
                        authors = _str_authors(result.get("authors", "")).lower()
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
                            context_authors = _str_authors(context_paper["metadata"]["authors"]).lower()
                            result_authors = _str_authors(result.get("authors", "")).lower()
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

            # Decide if we need external search.
            # External search is skipped if:
            #   - The query type doesn't warrant it (general/casual), OR
            #   - We already have enough local results to answer the query.
            query_type_wants_external = state.get("should_search_external", True)
            not_enough_local = len(local_results) < self.config.min_local_results_to_skip_external
            should_search_external = query_type_wants_external and not_enough_local

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

        # Safety guard: skip external search for very short or empty queries
        if len(query.strip()) < 8:
            logger.info("Skipping external search - query too short for meaningful API search")
            return {**state, "external_results": []}

        # If a researcher context is set, ensure their name is in the search query
        current_researcher = state.get("current_researcher")
        if current_researcher and isinstance(current_researcher, str) and current_researcher.lower() not in query.lower():
            query = f"{current_researcher} {query}"
            logger.info(f"Added researcher context to search query: {query}")

        # Extract year filters from config for API calls
        year_range = self.config.year_range
        from_year = year_range[0] if year_range else None
        to_year = year_range[1] if year_range else None

        # Helper: normalize title for dedup (lowercase, first 80 chars)
        def _title_key(title: str) -> str:
            return (title or "").strip().lower()[:80]

        seen_dois = set()
        seen_titles = set()

        def _is_duplicate(paper) -> bool:
            if paper.doi and paper.doi in seen_dois:
                return True
            tk = _title_key(paper.title)
            if tk and tk in seen_titles:
                return True
            return False

        def _mark_seen(paper):
            if paper.doi:
                seen_dois.add(paper.doi)
            tk = _title_key(paper.title)
            if tk:
                seen_titles.add(tk)

        def _paper_to_dict(paper, source_label: str) -> dict:
            return {
                "content": paper.abstract or "",
                "title": paper.title,
                "authors": ", ".join(paper.authors) if paper.authors else "",
                "year": paper.year,
                "paper_id": paper.paper_id,
                "doi": paper.doi,
                "citation_count": paper.citation_count or 0,
                "url": paper.url,
                "source": source_label,
            }

        # Search academic databases — OpenAlex primary (2/3 budget), S2 fills remainder
        max_ext = self.config.max_external_results
        openalex_limit = max(1, (max_ext * 2) // 3)
        s2_limit = max_ext - openalex_limit

        if self.academic_search:
            # Run both searches concurrently — each wrapped in its own error handler
            async def _fetch_openalex():
                try:
                    return await self.academic_search.search_openalex(
                        query, limit=openalex_limit,
                        from_year=from_year, to_year=to_year,
                    )
                except Exception as e:
                    logger.error(f"OpenAlex search error: {e}")
                    return []

            async def _fetch_s2():
                try:
                    return await self.academic_search.search_semantic_scholar(
                        query, limit=s2_limit,
                        year_range=year_range,
                    )
                except Exception as e:
                    logger.error(f"Semantic Scholar search error: {e}")
                    return []

            async def _fetch_core():
                if not self.config.include_core_search:
                    return []
                try:
                    core_limit = max(1, max_ext // 3)
                    return await self.academic_search.search_core(
                        query, limit=core_limit,
                        from_year=from_year, to_year=to_year,
                    )
                except Exception as e:
                    logger.error(f"CORE search error: {e}")
                    return []

            openalex_papers, s2_papers, core_papers = await asyncio.gather(
                _fetch_openalex(), _fetch_s2(), _fetch_core()
            )

            # Process OpenAlex results first (primary source)
            for paper in openalex_papers:
                citation_count = paper.citation_count or 0
                if min_citations and citation_count < min_citations:
                    continue
                if _is_duplicate(paper):
                    continue
                _mark_seen(paper)
                external_results.append(_paper_to_dict(paper, "openalex"))

            # Then S2 results (fills remaining, deduped)
            for paper in s2_papers:
                citation_count = paper.citation_count or 0
                if min_citations and citation_count < min_citations:
                    continue
                if _is_duplicate(paper):
                    continue
                _mark_seen(paper)
                external_results.append(_paper_to_dict(paper, "semantic_scholar"))

            # CORE results (open access papers, deduped)
            for paper in core_papers:
                citation_count = paper.citation_count or 0
                if min_citations and citation_count < min_citations:
                    continue
                if _is_duplicate(paper):
                    continue
                _mark_seen(paper)
                external_results.append(_paper_to_dict(paper, "core"))

            logger.info(f"Found {len(external_results)} academic results (OpenAlex: {len(openalex_papers)}, S2: {len(s2_papers)}, CORE: {len(core_papers)})")

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

        # For trivial greetings with no sources, respond without calling the LLM
        query_words = query.strip().split()
        is_trivial = query_type == "general" and len(query_words) <= 5 and not all_results
        if is_trivial:
            return {
                **state,
                "final_answer": "Hello! I'm your research assistant. I can help you search your knowledge base, find academic papers, and analyze research topics. What would you like to explore?",
            }

        # Build synthesis prompt with context
        prompt = self._build_synthesis_prompt(
            query, query_type, all_results,
            current_researcher=current_researcher,
            current_paper_id=current_paper_id,
            auth_context_items=state.get("auth_context_items"),
            chat_context_items=state.get("chat_context_items"),
        )

        # Generate response — LLM synthesis or formatted fallback
        try:
            if self.model:
                answer = self.task_infer("synthesize", prompt, max_tokens=1024)
            else:
                answer = self._format_results_without_llm(query, all_results)

            logger.info("Synthesis complete")

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            # Graceful fallback: show formatted results instead of an error
            if all_results:
                answer = self._format_results_without_llm(query, all_results)
            else:
                answer = f"Error generating response: {str(e)}"

        return {**state, "final_answer": answer}

    def _build_synthesis_prompt(
        self,
        query: str,
        query_type: str,
        results: List[Dict],
        current_researcher: Optional[str] = None,
        current_paper_id: Optional[str] = None,
        auth_context_items: Optional[List[str]] = None,
        chat_context_items: Optional[List[str]] = None,
    ) -> str:
        """Build a prompt for synthesizing results based on query type."""

        # For general/casual queries with no sources, use a simple conversational prompt
        if query_type == "general" and not results:
            return f"""You are a helpful research assistant. The user has sent a casual or general message. Respond naturally and conversationally. Do NOT mention sources, papers, or search results. If appropriate, briefly mention that you can help with research questions, finding papers, or analyzing their knowledge base.

User: {query}

Response:"""

        # Build context section from selection + pinned/chat items
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
        # Add pinned/active context items (independent of researcher/paper selection)
        auth_items = auth_context_items or []
        chat_items = chat_context_items or []
        if auth_items:
            context_parts.append(f"The user has pinned these topics: {', '.join(auth_items)}")
        if chat_items:
            context_parts.append(f"Recent conversation topics include: {', '.join(chat_items)}")

        context_section = ""
        if context_parts:
            context_section = "\n\nContext: " + ". ".join(context_parts) + ". Prioritize information relevant to this context."

        # Format sources
        sources_text = ""
        for i, result in enumerate(results[:10], 1):  # Limit to top 10
            title = result.get("title", "Unknown")
            authors = result.get("authors", "")
            year = result.get("year", "")
            content = str(result.get("content") or "")[:800]
            source = result.get("source", "unknown")
            source_label = SOURCE_LABELS_LONG.get(source, source)
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
        _KB_PREAMBLE = (
            "You are a research assistant with access to a personal knowledge base of ingested papers.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- The sources below are text chunks from papers in the knowledge base. Their stored titles may differ "
            "from how they are referenced in the query (e.g., \"Envisaging the Future of Cities\" IS the World Cities Report 2022).\n"
            "- Answer based on the CONTENT of the chunks provided, not by matching title strings.\n"
            "- If a chunk's content is relevant to the question, USE it — even if its title doesn't exactly match what the user named.\n"
            "- NEVER say a topic \"is not mentioned\" just because a title string doesn't match. Read the actual text excerpts.\n"
        )
        _MANDATORY = (
            "\nMANDATORY: If sources are provided below, you MUST reference their content in your answer. "
            "Start by addressing the question using the source excerpts. "
            "Do NOT say \"no information found\" or \"not mentioned\" when sources are present."
        )

        _QUERY_INSTRUCTIONS = {
            "literature_review": (
                "\nProvide a comprehensive literature review based on the source content:\n"
                "1. Overview of the research landscape\n"
                "2. Key themes and findings from the sources\n"
                "3. Notable contributions from each relevant source\n"
                "4. Research gaps or future directions\n\n"
                "Cite sources by their title using [1], [2], etc."
            ),
            "factual": (
                "\nProvide a precise, factual answer based on the source content.\n"
                "Be specific and cite your sources using [1], [2], etc.\n"
                "If sources conflict, note the disagreement."
            ),
            "analysis": (
                "\nProvide a critical analysis based on the source content.\n"
                "Compare different perspectives, evaluate evidence, and draw conclusions.\n"
                "Cite sources using [1], [2], etc."
            ),
        }

        if query_type in _QUERY_INSTRUCTIONS:
            instruction = _KB_PREAMBLE + _QUERY_INSTRUCTIONS[query_type] + _MANDATORY
        else:  # general — but we have sources (casual w/o sources handled above)
            instruction = (
                "You are a helpful research assistant. The user asked a general question "
                "and some potentially relevant sources were found in the knowledge base.\n\n"
                "If the sources below are relevant to the question, use them to inform your answer "
                "and cite using [1], [2], etc.\n"
                "If they are not relevant to the question, answer directly using your own knowledge "
                "and ignore the sources."
            )

        prompt = f"""{instruction}{context_section}

Question: {query}

Available Sources (use the content of these excerpts to answer — do not just match on titles):
{sources_text}

Response:"""

        return prompt

    def _format_results_without_llm(self, query: str, results: List[Dict]) -> str:
        """Format retrieved results when no LLM is available for synthesis."""
        if not results:
            return (
                "No retrieved sources are available for this query. "
                "(LLM synthesis is currently unavailable.)"
            )

        response = (
            "I retrieved relevant sources, but an LLM is not available to synthesize "
            "them into a narrative answer right now. Here are the top retrieved "
            f"results for: **{query}**\n\n"
        )

        for i, result in enumerate(results[:10], 1):
            title = result.get("title", "Unknown")
            authors = result.get("authors", "")
            year = result.get("year", "")
            source = result.get("source", "")
            source_label = SOURCE_LABELS.get(source, source)
            content = str(result.get("content", ""))[:200]
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

    # ── Native Claude tool-use path ─────────────────────────────────────

    def _get_claude_tool_definitions(self) -> List[Dict]:
        """Return Anthropic tool schemas for the research agent's capabilities."""
        return [
            {
                "name": "search_local_kb",
                "description": (
                    "Search the user's personal knowledge base (papers, notes, "
                    "and web sources stored in the vector database). Use this "
                    "first to check what the user already has before searching "
                    "externally."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant papers, notes, or web sources",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_academic",
                "description": (
                    "Search academic databases (OpenAlex, Semantic Scholar, CORE) "
                    "for published research papers. Returns papers with titles, "
                    "authors, years, citation counts, and abstracts."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Academic search query (keywords, author names, topics)",
                        },
                        "year_from": {
                            "type": "integer",
                            "description": "Only return papers published after this year",
                        },
                        "year_to": {
                            "type": "integer",
                            "description": "Only return papers published before this year",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_web",
                "description": (
                    "Search the web for grey literature, reports, news articles, "
                    "and non-academic sources. Use for current events, policy "
                    "documents, or organizational reports."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Web search query",
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

    def _build_claude_system_prompt(self, context: Dict) -> str:
        """Build system prompt for Claude's native tool-use mode."""
        parts = [
            "You are a research assistant with access to a personal knowledge base "
            "of academic papers, notes, and web sources.",
            "",
            "Your workflow:",
            "1. ALWAYS search the local knowledge base first to check what the user already has.",
            "2. If local results are insufficient (fewer than 3 relevant results), search academic databases.",
            "3. For current events, policy documents, or grey literature, also search the web.",
            "4. Synthesize all findings into a clear, well-organized answer.",
            "",
            "Citation format:",
            "- Number all sources sequentially as [1], [2], [3], etc.",
            "- Cite sources inline in your answer using these numbers.",
            "- Base your answer on the CONTENT of sources, not just their titles.",
            "- If sources conflict, note the disagreement.",
            "",
            "Response style:",
            "- Use markdown formatting for readability.",
            "- For literature reviews: provide thematic overview, key findings, and research gaps.",
            "- For factual questions: give precise answers with specific citations.",
            "- For analysis requests: compare perspectives and evaluate evidence.",
            "- For casual greetings or simple questions: respond naturally without using tools.",
        ]

        # Add context from UI selection
        current_researcher = context.get("researcher")
        current_paper_id = context.get("paper_id")
        auth_items = context.get("auth_items", [])
        chat_items = context.get("chat_items", [])

        if current_researcher:
            parts.append(
                f"\nThe user is currently focused on researcher: {current_researcher}. "
                "Prioritize information about their work."
            )
        if current_paper_id:
            parts.append(
                f"\nThe user has a specific paper selected (ID: {current_paper_id[:30]}). "
                "Consider this context when searching."
            )
        if auth_items:
            parts.append(f"\nPinned research topics: {', '.join(auth_items)}")
        if chat_items:
            parts.append(f"\nRecent conversation topics: {', '.join(chat_items)}")

        return "\n".join(parts)

    def _format_tool_results(self, results: List[Dict], source_type: str) -> str:
        """Format search results as text for Claude to reference in its answer."""
        if not results:
            return f"No {source_type} results found."

        lines = [f"Found {len(results)} {source_type} result(s):\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "Unknown")
            authors = r.get("authors", "")
            year = r.get("year", "")
            content = str(r.get("content", ""))[:600]
            citations = r.get("citation_count")
            url = r.get("url", "")

            lines.append(f"[{i}] {title}")
            if authors and authors != "User Note":
                lines.append(f"    Authors: {authors}")
            if year:
                lines.append(f"    Year: {year}")
            if citations:
                lines.append(f"    Citations: {citations}")
            if url:
                lines.append(f"    URL: {url}")
            if content:
                lines.append(f"    Excerpt: {content}")
            lines.append("")

        return "\n".join(lines)

    async def _execute_claude_tool(
        self,
        tool_name: str,
        tool_input: Dict,
        all_local: List[Dict],
        all_external: List[Dict],
        context: Dict,
    ) -> str:
        """Execute a Claude tool call against existing search infrastructure."""

        if tool_name == "search_local_kb":
            return await self._claude_search_local(
                tool_input["query"], all_local, context,
            )

        elif tool_name == "search_academic":
            return await self._claude_search_academic(
                tool_input["query"],
                year_from=tool_input.get("year_from"),
                year_to=tool_input.get("year_to"),
                results_out=all_external,
            )

        elif tool_name == "search_web":
            return await self._claude_search_web(
                tool_input["query"], all_external,
            )

        return f"Unknown tool: {tool_name}"

    async def _claude_search_local(
        self,
        query: str,
        results_out: List[Dict],
        context: Dict,
    ) -> str:
        """Local KB search for Claude tool-use path."""
        if not self.vector_store or not self.embedder:
            return "Knowledge base is not configured."

        try:
            # Generate query embedding (same logic as _search_local)
            if hasattr(self.embedder, "embed_query"):
                query_embedding = self.embedder.embed_query(query)
            elif hasattr(self.embedder, "embed"):
                query_embedding = self.embedder.embed(query)
            elif hasattr(self.embedder, "encode"):
                query_embedding = self.embedder.encode(query)
            else:
                return "Error: Embedder does not support query embedding."

            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()

            # Year filters from config
            filter_dict = None
            if self.config.year_range:
                year_from, year_to = self.config.year_range
                filter_dict = {
                    "$and": [
                        {"year": {"$gte": year_from}},
                        {"year": {"$lte": year_to}},
                    ]
                }

            _empty = {"documents": [], "metadatas": [], "distances": []}
            loop = asyncio.get_running_loop()

            def _search_papers():
                return self.vector_store.search(
                    query_embedding=query_embedding,
                    collection="papers",
                    n_results=self.config.max_local_results * 2,
                    filter_dict=filter_dict,
                    query_text=query,
                )

            def _search_notes():
                try:
                    return self.vector_store.search(
                        query_embedding=query_embedding,
                        collection="notes",
                        n_results=max(3, self.config.max_local_results // 2),
                        query_text=query,
                    )
                except Exception:
                    return _empty

            def _search_web_sources():
                try:
                    return self.vector_store.search(
                        query_embedding=query_embedding,
                        collection="web_sources",
                        n_results=max(3, self.config.max_local_results // 2),
                        query_text=query,
                    )
                except Exception:
                    return _empty

            paper_results, notes_results, web_results = await asyncio.gather(
                loop.run_in_executor(None, _search_papers),
                loop.run_in_executor(None, _search_notes),
                loop.run_in_executor(None, _search_web_sources),
            )

            local_results = []

            # Format paper results
            for i, doc in enumerate(paper_results.get("documents", [])):
                meta = paper_results.get("metadatas", [{}])[i] if paper_results.get("metadatas") else {}
                dist = paper_results.get("distances", [1.0])[i] if paper_results.get("distances") else 1.0
                raw_authors = meta.get("authors", "")
                authors_str = ", ".join(raw_authors) if isinstance(raw_authors, list) else (raw_authors or "")
                local_results.append({
                    "content": doc,
                    "title": meta.get("title", "Unknown"),
                    "authors": authors_str,
                    "year": meta.get("year"),
                    "paper_id": meta.get("paper_id", ""),
                    "source": "local_kb",
                    "relevance_score": 1 - dist,
                })

            # Format notes results
            for i, doc in enumerate(notes_results.get("documents", [])):
                meta = notes_results.get("metadatas", [{}])[i] if notes_results.get("metadatas") else {}
                dist = notes_results.get("distances", [1.0])[i] if notes_results.get("distances") else 1.0
                local_results.append({
                    "content": doc,
                    "title": meta.get("title", "Research Note"),
                    "authors": "User Note",
                    "year": None,
                    "paper_id": meta.get("note_id", ""),
                    "source": "local_note",
                    "tags": meta.get("tags", ""),
                    "relevance_score": 1 - dist,
                })

            # Format web source results
            for i, doc in enumerate(web_results.get("documents", [])):
                meta = web_results.get("metadatas", [{}])[i] if web_results.get("metadatas") else {}
                dist = web_results.get("distances", [1.0])[i] if web_results.get("distances") else 1.0
                local_results.append({
                    "content": doc,
                    "title": meta.get("title", "Web Source"),
                    "authors": "",
                    "year": None,
                    "paper_id": meta.get("source_id", ""),
                    "url": meta.get("url", ""),
                    "source": "local_web",
                    "relevance_score": 1 - dist,
                })

            # Apply context boosting (same as _search_local:860-898)
            current_researcher = context.get("researcher")
            current_paper_id = context.get("paper_id")
            if current_researcher or current_paper_id:
                boost_amount = 0.15

                def _str_authors(val):
                    if isinstance(val, list):
                        return ", ".join(str(a) for a in val)
                    return str(val) if val else ""

                context_paper = None
                if current_paper_id:
                    context_paper = self.vector_store.get_paper(current_paper_id)

                for result in local_results:
                    if current_researcher:
                        authors = _str_authors(result.get("authors", "")).lower()
                        if current_researcher.lower() in authors:
                            result["relevance_score"] = min(1.0, result["relevance_score"] + boost_amount)

                    if current_paper_id and context_paper:
                        pid = result.get("paper_id", "")
                        if pid == current_paper_id:
                            result["relevance_score"] = min(1.0, result["relevance_score"] + boost_amount * 2)
                        elif context_paper.get("metadata", {}).get("authors"):
                            ctx_authors = _str_authors(context_paper["metadata"]["authors"]).lower()
                            res_authors = _str_authors(result.get("authors", "")).lower()
                            if any(a.strip() in res_authors for a in ctx_authors.split(",") if len(a.strip()) > 3):
                                result["relevance_score"] = min(1.0, result["relevance_score"] + boost_amount * 0.5)

            local_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            local_results = local_results[:self.config.max_local_results]

            results_out.extend(local_results)
            return self._format_tool_results(local_results, "local knowledge base")

        except Exception as e:
            logger.error(f"Claude local search error: {e}")
            return f"Error searching knowledge base: {e}"

    async def _claude_search_academic(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        results_out: Optional[List[Dict]] = None,
    ) -> str:
        """Academic search for Claude tool-use path."""
        if not self.academic_search:
            return "Academic search is not configured."

        # Use config year range as fallback
        if year_from is None and self.config.year_range:
            year_from = self.config.year_range[0]
        if year_to is None and self.config.year_range:
            year_to = self.config.year_range[1]
        year_range = (year_from, year_to) if year_from and year_to else None
        min_citations = max(0, int(self.config.min_citations or 0))

        max_ext = self.config.max_external_results
        openalex_limit = max(1, (max_ext * 2) // 3)
        s2_limit = max_ext - openalex_limit

        try:
            async def _fetch_openalex():
                try:
                    return await self.academic_search.search_openalex(
                        query, limit=openalex_limit,
                        from_year=year_from, to_year=year_to,
                    )
                except Exception as e:
                    logger.error(f"OpenAlex search error: {e}")
                    return []

            async def _fetch_s2():
                try:
                    return await self.academic_search.search_semantic_scholar(
                        query, limit=s2_limit, year_range=year_range,
                    )
                except Exception as e:
                    logger.error(f"Semantic Scholar search error: {e}")
                    return []

            async def _fetch_core():
                if not self.config.include_core_search:
                    return []
                try:
                    return await self.academic_search.search_core(
                        query, limit=max(1, max_ext // 3),
                        from_year=year_from, to_year=year_to,
                    )
                except Exception as e:
                    logger.error(f"CORE search error: {e}")
                    return []

            oa_papers, s2_papers, core_papers = await asyncio.gather(
                _fetch_openalex(), _fetch_s2(), _fetch_core(),
            )

            # Dedup by DOI / title (same logic as _search_external)
            seen_dois: set = set()
            seen_titles: set = set()
            external_results: List[Dict] = []

            def _title_key(title: str) -> str:
                return (title or "").strip().lower()[:80]

            for papers, source_label in [
                (oa_papers, "openalex"),
                (s2_papers, "semantic_scholar"),
                (core_papers, "core"),
            ]:
                for paper in papers:
                    cites = paper.citation_count or 0
                    if min_citations and cites < min_citations:
                        continue
                    if paper.doi and paper.doi in seen_dois:
                        continue
                    tk = _title_key(paper.title)
                    if tk and tk in seen_titles:
                        continue
                    if paper.doi:
                        seen_dois.add(paper.doi)
                    if tk:
                        seen_titles.add(tk)
                    external_results.append({
                        "content": paper.abstract or "",
                        "title": paper.title,
                        "authors": ", ".join(paper.authors) if paper.authors else "",
                        "year": paper.year,
                        "paper_id": paper.paper_id,
                        "doi": paper.doi,
                        "citation_count": cites,
                        "url": paper.url,
                        "source": source_label,
                    })

            if results_out is not None:
                results_out.extend(external_results)
            return self._format_tool_results(external_results, "academic")

        except Exception as e:
            logger.error(f"Claude academic search error: {e}")
            return f"Error searching academic databases: {e}"

    async def _claude_search_web(
        self,
        query: str,
        results_out: List[Dict],
    ) -> str:
        """Web search for Claude tool-use path."""
        if not self.web_search:
            return "Web search is not configured."

        try:
            web_results = await self.web_search.search(query, max_results=5)
            formatted = []
            for result in web_results:
                title = result.get("title") if isinstance(result, dict) else getattr(result, "title", "")
                url = result.get("url") if isinstance(result, dict) else getattr(result, "url", "")
                content = result.get("snippet") if isinstance(result, dict) else getattr(result, "content", "")
                entry = {
                    "content": content or "",
                    "title": title or "",
                    "url": url or "",
                    "source": "web",
                }
                formatted.append(entry)

            results_out.extend(formatted)
            return self._format_tool_results(formatted, "web")

        except Exception as e:
            logger.error(f"Claude web search error: {e}")
            return f"Error searching the web: {e}"

    async def _run_with_claude_tools(
        self,
        user_query: str,
        search_filters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the agent using Claude's native tool calling.

        Claude decides which tools to call and in what order. Falls back
        to the LangGraph pipeline if the anthropic SDK is not installed
        or the API call fails.
        """
        import anthropic

        context = context or {}

        # Apply search filters to config (same as _run_async)
        if search_filters:
            if "year_from" in search_filters or "year_to" in search_filters:
                year_from = search_filters.get("year_from", 1900)
                year_to = search_filters.get("year_to", 2030)
                self.config.year_range = (year_from, year_to)
            if search_filters.get("min_citations") is not None:
                try:
                    self.config.min_citations = int(search_filters["min_citations"])
                except (TypeError, ValueError):
                    self.config.min_citations = 0

        client = anthropic.AsyncAnthropic(api_key=self._openai_api_key)
        model = self.model.model_name if self.model else "claude-sonnet-4-6"
        system_prompt = self._build_claude_system_prompt(context)
        tools = self._get_claude_tool_definitions()
        messages: List[Dict] = [{"role": "user", "content": user_query}]
        all_local: List[Dict] = []
        all_external: List[Dict] = []

        MAX_TURNS = 6
        final_text = ""

        for _ in range(MAX_TURNS):
            response = await client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )

            # If Claude is done, extract the final text
            if response.stop_reason == "end_turn":
                final_text = "".join(
                    b.text for b in response.content if b.type == "text"
                )
                break

            # Append assistant turn (with tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call and build tool_result blocks
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_text = await self._execute_claude_tool(
                        block.name, block.input,
                        all_local, all_external, context,
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
        else:
            final_text = (
                "I wasn't able to complete the research in the allowed number of steps. "
                "Please try a more specific query."
            )

        logger.info(
            "Claude tool-use complete: %d local, %d external sources",
            len(all_local), len(all_external),
        )

        # Return same dict format as _run_async
        all_sources = all_local + all_external
        return {
            "query": user_query,
            "answer": final_text,
            "query_type": "general" if not all_sources else "literature_review",
            "local_sources": len(all_local),
            "external_sources": len(all_external),
            "sources": [
                {
                    "title": s.get("title"),
                    "authors": s.get("authors", ""),
                    "source": s.get("source"),
                    "year": s.get("year"),
                    "url": s.get("url", ""),
                }
                for s in all_sources[:10]
            ],
        }

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
        from research_agent.utils.observability import new_request_id
        new_request_id()

        # Extract context
        context = context or {}
        current_researcher = context.get("researcher")
        current_paper_id = context.get("paper_id")
        auth_context_items = context.get("auth_items", [])
        chat_context_items = context.get("chat_items", [])

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
        logger.debug(f"_run_async: user_query type={type(user_query).__name__}, researcher type={type(current_researcher).__name__}, researcher={current_researcher}")
        initial_state: ResearchState = {
            "messages": [],
            "current_query": user_query,
            "search_query": "",
            "query_type": "general",
            "local_results": [],
            "external_results": [],
            "should_search_external": True,
            "candidates_for_ingestion": [],
            "final_answer": "",
            "error": None,
            "current_researcher": current_researcher,
            "current_paper_id": current_paper_id,
            "auth_context_items": auth_context_items,
            "chat_context_items": chat_context_items,
        }

        result = {
            "query": user_query,
            "answer": "",
            "sources": [],
        }

        if search_filters:
            result["search_filters"] = search_filters

        # Native Claude tool-use path (bypasses LangGraph)
        if self.is_claude:
            try:
                return await self._run_with_claude_tools(
                    user_query, search_filters, context,
                )
            except ImportError:
                logger.warning(
                    "anthropic SDK not installed, using LangGraph pipeline"
                )
            except Exception as e:
                logger.error(
                    "Claude tool-use failed (%s), falling back to LangGraph", e,
                )

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
                    "authors": s.get("authors", ""),
                    "source": s.get("source"),
                    "year": s.get("year"),
                    "url": s.get("url", ""),
                }
                for s in all_sources[:10]
            ]

            if final_state.get("error"):
                result["error"] = final_state["error"]

        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            # Fallback to direct inference (works for all provider types)
            if self.model:
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
        raise NotImplementedError("ingest_paper not yet implemented")


def create_research_agent(
    vector_store=None,
    embedder=None,
    academic_search=None,
    web_search=None,
    llm_generate: Optional[Callable] = None,
    config: Optional[AgentConfig] = None,
    use_ollama: bool = False,
    provider: Optional[str] = None,
    canonical_provider: Optional[str] = None,
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
        canonical_provider=canonical_provider,
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
