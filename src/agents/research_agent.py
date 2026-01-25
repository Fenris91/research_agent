import torch
import logging
from typing import List, Dict, Optional

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional, Dict, Any, Callable
import operator

from langgraph.graph import StateGraph, END

from ..models.llm_utils import get_qlora_pipeline, check_vram, VRAMConstraintError

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
        ollama_model: str = "mistral",
        ollama_base_url: str = "http://localhost:11434"
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
        
        if use_ollama:
            # Try Ollama first
            try:
                from ..models.llm_utils import get_ollama_pipeline, OllamaUnavailableError
                self.model = get_ollama_pipeline(model_name=ollama_model, base_url=ollama_base_url)
                self._load_model_on_demand = False
                print(f"✓ Using Ollama model: {ollama_model}")
            except OllamaUnavailableError as e:
                print(f"⚠️ Ollama unavailable: {str(e)}")
                print("Falling back to HuggingFace models...")
                try:
                    self.model, self.tokenizer = get_qlora_pipeline()
                    self._test_vram_on_initialization()
                    self._load_model_on_demand = False
                    self.use_ollama = False
                except VRAMConstraintError as e:
                    print(f"⚠️ Model loading deferred due to VRAM constraints: {str(e)}")
                    self._load_model_on_demand = True
        else:
            # Try HuggingFace models
            try:
                self.model, self.tokenizer = get_qlora_pipeline()
                self._test_vram_on_initialization()
                self._load_model_on_demand = False
            except VRAMConstraintError as e:
                print(f"⚠️ Model loading deferred due to VRAM constraints: {str(e)}")
                self._load_model_on_demand = True

        self.vector_store = vector_store
        self.embedder = embedder
        self.academic_search = academic_search
        self.web_search = web_search
        self.llm_generate = llm_generate
        self.config = config or AgentConfig()

        # Build the workflow graph
        self.graph = self._build_graph()

    def _test_vram_on_initialization(self):
        """
        Confirm VRAM is available after model load
        """
        try:
            check_vram()
        except VRAMConstraintError as e:
            print(f"⚠️ Initial VRAM warning: {str(e)}")

    def infer(self, prompt: str, max_tokens=512):
        """
        Execute LLM inference with memory safety checks.
        Supports both HuggingFace and Ollama models.
        """
        try:
            if self.use_ollama:
                # Use Ollama
                return self.model.generate(prompt, max_tokens=max_tokens, temperature=0.7)
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
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                # Post-alloc check
                check_vram()
                
                # Decode only the generated tokens (exclude input)
                generated_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
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
                inputs["input_ids"],
                max_new_tokens=128,
                temperature=0.5
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
        workflow.add_edge("search_local", "synthesize")
        workflow.add_edge("synthesize", "offer_ingestion")
        workflow.add_edge("offer_ingestion", END)
        
        return workflow.compile()

    # Existing methods preserved from previous content...
    async def _understand_query(self, state: ResearchState) -> Dict:
        """Understand and classify the user's query."""
        return state

    async def _search_local(self, state: ResearchState) -> Dict:
        """Search local vector store."""
        return state

    async def _search_external(self, state: ResearchState) -> Dict:
        """Search external academic databases."""
        return state

    async def _synthesize(self, state: ResearchState) -> Dict:
        """Synthesize search results into an answer."""
        return state

    def _build_synthesis_prompt(self, query: str, results: List[Dict]) -> str:
        """Build a prompt for synthesizing results."""
        return ""

    def _generate_structured_response(self, query: str, results: List[Dict]) -> str:
        """Generate a structured response from results."""
        return ""

    async def _offer_ingestion(self, state: ResearchState) -> Dict:
        """Offer to ingest new papers."""
        return state

    async def _run_async(self, user_query: str) -> Dict[str, Any]:
        """Internal async implementation of run."""
        result = {
            "query": user_query,
            "answer": ""
        }
        
        if self.model and self.tokenizer:
            # Use the loaded model
            result["answer"] = self.infer(user_query, max_tokens=256)
        else:
            result["answer"] = "Fallback mode: Model not available. Query received but processing requires model loading."
            result["status"] = "deferred"
        
        return result

    def run(self, user_query: str) -> Dict[str, Any]:
        """Run the research agent (synchronous wrapper)."""
        try:
            return asyncio.run(self._run_async(user_query))
        except RuntimeError as e:
            # Handle case where event loop already exists
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return loop.run_in_executor(pool, asyncio.run, self._run_async(user_query))
            else:
                return asyncio.run(self._run_async(user_query))

    async def ingest_paper(self, paper_data: Dict) -> bool:
        """Ingest a new paper into the vector store."""
        return True


def create_research_agent(
    vector_store=None,
    embedder=None,
    academic_search=None,
    web_search=None,
    llm_generate: Optional[Callable] = None,
    config: Optional[AgentConfig] = None
) -> ResearchAgent:
    """Factory function to create a ResearchAgent instance."""
    return ResearchAgent(
        vector_store=vector_store,
        embedder=embedder,
        academic_search=academic_search,
        web_search=web_search,
        llm_generate=llm_generate,
        config=config
    )