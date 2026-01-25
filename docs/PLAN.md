# Research Assistant Agent: Implementation Plan

## Overview

This document outlines a concrete implementation plan for building an autonomous research assistant agent focused on social sciences (social anthropology, geography, etc.). The system will perform literature review, paper summarization, data analysis, and web synthesis—with the ability to autonomously search and grow its own knowledge base.

**Hardware:** NVIDIA RTX 5090 (32GB VRAM), AMD 9X3D, 32GB system RAM
**Primary Framework:** HuggingFace Transformers (with path to vLLM/PyTorch)

---

## Implementation Status (Updated: January 25, 2026)

### Completed

| Component | Status | File(s) |
|-----------|--------|---------|
| **Academic Paper Search** | Done | `src/tools/academic_search.py` |
| - Semantic Scholar API | Done | Search with citation data, DOI, fields |
| - OpenAlex API | Done | Broad coverage, abstract reconstruction |
| - Unpaywall (OA finder) | Done | Find open access PDFs |
| - Deduplication | Done | DOI + title matching |
| **Web Search** | Done | `src/tools/web_search.py` |
| - DuckDuckGo (free) | Done | No API key required |
| - Tavily (paid) | Done | AI-optimized search |
| - Serper (paid) | Done | Google results |
| **Researcher Lookup** | Done | `src/tools/researcher_lookup.py` |
| - OpenAlex Authors | Done | Works, citations, affiliations |
| - Semantic Scholar Authors | Done | H-index, paper count |
| - Web Search | Done | Academic profiles, news |
| - CLI Script | Done | `src/scripts/lookup_researchers.py` |
| **Vector Store** | Done | `src/db/vector_store.py` |
| - ChromaDB integration | Done | Persistent storage |
| - Multiple collections | Done | papers, notes, web_sources |
| - CRUD operations | Done | Add, search, delete, list |
| **Embedding Model** | Done | `src/db/embeddings.py` |
| - Sentence Transformers | Done | BGE models supported |
| - Query/document encoding | Done | Optimized for retrieval |
| **Research Agent** | Done | `src/agents/research_agent.py` |
| - LangGraph workflow | Done | 5-node state machine |
| - Query classification | Done | literature_review, factual, etc. |
| - Local KB search | Done | Vector store integration |
| - External search | Done | Academic + web APIs |
| - Response synthesis | Done | Structured output (no LLM) |
| - Ingestion candidates | Done | Score and rank papers |
| **LLM Integration** | Done | `src/models/llm_utils.py` |
| - Qwen2.5-32B primary model | Done | With GPTQ quantization |
| - Mistral 7B fallback | Done | Public model, good quality |
| - TinyLlama 1.1B fallback | Done | Lightweight, CPU-friendly |
| - VRAM diagnostics | Done | Real-time GPU memory monitoring |
| - Lazy loading support | Done | Defers loading on VRAM constraints |
| **Gradio UI** | Done | `src/ui/app.py` |
| - Basic structure | Done | Tabs for chat, KB, researcher |
| - Researcher Lookup tab | Done | Functional |
| - Chat integration | Done | Agent fully wired and working |
| - LLM response generation | Done | Inference with attention mask handling |
| - Error handling | Done | Graceful fallbacks |

### In Progress

| Component | Status | Notes |
|-----------|--------|-------|
| PDF Processing | Pending | Document processor scaffolded |
| Citation Explorer | Pending | API methods exist, tool not wired |

### Not Started

| Component | Notes |
|-----------|-------|
| Data Analysis Tools | Pandas/visualization integration |
| Reranker | BGE reranker for better retrieval |
| Fine-tuning | Domain-specific model training |
| Export/Citation Management | Zotero/BibTeX export |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│                    (Gradio / Streamlit / CLI)                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                           │
│                   (LangChain / LlamaIndex Agent)                     │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Search    │  │  Retrieval  │  │  Analysis   │  │   Ingest    │ │
│  │    Tool     │  │    Tool     │  │    Tool     │  │    Tool     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
          │                 │                │                │
          ▼                 ▼                ▼                ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   External   │   │   Vector     │   │     LLM      │   │   Document   │
│     APIs     │   │   Database   │   │   (Local)    │   │   Processor  │
│              │   │              │   │              │   │              │
│ • Semantic   │   │  ChromaDB /  │   │ Qwen2.5-32B  │   │ • PDF Parse  │
│   Scholar    │   │  Qdrant      │   │     or       │   │ • Chunking   │
│ • OpenAlex   │   │              │   │ Mistral-22B  │   │ • Metadata   │
│ • Unpaywall  │   │              │   │              │   │              │
│ • CrossRef   │   │              │   │              │   │              │
│ • Web Search │   │              │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

---

## Phase 1: Foundation (Week 1-2)

### 1.1 Environment Setup

```bash
# Create project structure
mkdir -p research_agent/{agents,tools,db,processors,configs,tests}
cd research_agent

# Create conda environment
conda create -n research_agent python=3.11
conda activate research_agent

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate bitsandbytes
pip install sentence-transformers
pip install langchain langchain-community langgraph
pip install chromadb  # or: pip install qdrant-client
pip install gradio

# Document processing
pip install pymupdf  # PDF parsing
pip install unstructured[pdf]  # Advanced document processing
pip install python-docx  # Word documents

# API clients
pip install semanticscholar
pip install pyalex  # OpenAlex
pip install habanero  # CrossRef
pip install httpx  # Async HTTP

# Data analysis
pip install pandas numpy matplotlib seaborn plotly
pip install scipy scikit-learn
```

### 1.2 Base Model Selection & Setup

For 32GB VRAM, recommended options:

| Model | Size | Quantization | VRAM Usage | Notes |
|-------|------|--------------|------------|-------|
| **Qwen2.5-32B-Instruct** | 32B | 4-bit GPTQ | ~18GB | Best reasoning, good for research |
| **Mistral-Small-24B** | 24B | 4-bit | ~14GB | Fast, good instruction following |
| **Llama-3.1-70B** | 70B | 4-bit GPTQ | ~38GB | Won't fit, need cloud |
| **Qwen2.5-14B-Instruct** | 14B | FP16 | ~28GB | Full precision option |

**Recommended starting point: Qwen2.5-32B-Instruct-GPTQ-Int4**

```python
# models/llm_loader.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"):
    """Load quantized model for local inference."""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    return model, tokenizer

# For later vLLM migration:
# from vllm import LLM
# llm = LLM(model=model_name, quantization="gptq", gpu_memory_utilization=0.9)
```

### 1.3 Embedding Model

```python
# models/embeddings.py
from sentence_transformers import SentenceTransformer

def load_embedding_model():
    """Load embedding model for semantic search.
    
    BAAI/bge-large-en-v1.5 is excellent for academic content.
    Uses ~1.3GB VRAM, leaves plenty for LLM.
    """
    model = SentenceTransformer(
        "BAAI/bge-large-en-v1.5",
        device="cuda"
    )
    return model
```

---

## Phase 2: Vector Database & Retrieval (Week 2-3)

### 2.1 ChromaDB Setup (Simpler) or Qdrant (More Scalable)

```python
# db/vector_store.py
import chromadb
from chromadb.config import Settings

class ResearchVectorStore:
    def __init__(self, persist_dir="./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Separate collections for different content types
        self.papers = self.client.get_or_create_collection(
            name="academic_papers",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.notes = self.client.get_or_create_collection(
            name="research_notes",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.web_sources = self.client.get_or_create_collection(
            name="web_sources",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_paper(self, paper_id, chunks, embeddings, metadata):
        """Add paper chunks with rich metadata."""
        self.papers.add(
            ids=[f"{paper_id}_chunk_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{
                **metadata,
                "chunk_index": i,
                "paper_id": paper_id
            } for i in range(len(chunks))]
        )
    
    def search(self, query_embedding, collection="papers", n_results=10, 
               filter_dict=None):
        """Search with optional metadata filtering."""
        coll = getattr(self, collection)
        
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict  # e.g., {"year": {"$gte": 2020}}
        )
        
        return results
    
    def get_paper_stats(self):
        """Get statistics about the knowledge base."""
        return {
            "total_papers": self.papers.count(),
            "total_notes": self.notes.count(),
            "total_web_sources": self.web_sources.count()
        }
```

### 2.2 Document Processing Pipeline

```python
# processors/document_processor.py
import fitz  # PyMuPDF
from typing import List, Dict
import re

class AcademicPDFProcessor:
    """Extract and chunk academic papers intelligently."""
    
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text(self, pdf_path: str) -> Dict:
        """Extract text with structure awareness."""
        doc = fitz.open(pdf_path)
        
        full_text = ""
        sections = []
        current_section = {"title": "Introduction", "content": ""}
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        text = "".join([span["text"] for span in line["spans"]])
                        font_size = line["spans"][0]["size"] if line["spans"] else 12
                        
                        # Detect section headers (larger font, short text)
                        if font_size > 12 and len(text) < 100:
                            if current_section["content"]:
                                sections.append(current_section)
                            current_section = {"title": text.strip(), "content": ""}
                        else:
                            current_section["content"] += text + " "
                        
                        full_text += text + " "
        
        if current_section["content"]:
            sections.append(current_section)
        
        return {
            "full_text": full_text,
            "sections": sections,
            "page_count": len(doc)
        }
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def extract_metadata(self, text: str) -> Dict:
        """Extract basic metadata from paper text."""
        # Simple heuristics - could be enhanced with LLM
        metadata = {
            "has_abstract": "abstract" in text.lower()[:2000],
            "has_references": "references" in text.lower()[-5000:],
            "estimated_word_count": len(text.split())
        }
        
        # Try to extract DOI
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        doi_match = re.search(doi_pattern, text)
        if doi_match:
            metadata["doi"] = doi_match.group()
        
        return metadata
```

---

## Phase 3: External API Integration (Week 3-4)

### 3.1 Academic Search APIs

```python
# tools/academic_search.py
from semanticscholar import SemanticScholar
from pyalex import Works, Authors
import httpx
from typing import List, Dict, Optional
import asyncio

class AcademicSearchTools:
    """Unified interface for academic search APIs."""
    
    def __init__(self):
        self.s2 = SemanticScholar()
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def search_semantic_scholar(
        self, 
        query: str, 
        limit: int = 20,
        year_range: Optional[tuple] = None,
        fields_of_study: Optional[List[str]] = None
    ) -> List[Dict]:
        """Search Semantic Scholar for papers.
        
        Free tier: 100 requests/5min
        """
        results = self.s2.search_paper(
            query,
            limit=limit,
            fields=[
                "paperId", "title", "abstract", "year", 
                "citationCount", "authors", "fieldsOfStudy",
                "publicationTypes", "openAccessPdf"
            ]
        )
        
        papers = []
        for paper in results:
            if year_range:
                if paper.year and (paper.year < year_range[0] or paper.year > year_range[1]):
                    continue
            
            papers.append({
                "id": paper.paperId,
                "title": paper.title,
                "abstract": paper.abstract,
                "year": paper.year,
                "citations": paper.citationCount,
                "authors": [a["name"] for a in (paper.authors or [])],
                "fields": paper.fieldsOfStudy,
                "open_access_url": paper.openAccessPdf.get("url") if paper.openAccessPdf else None,
                "source": "semantic_scholar"
            })
        
        return papers
    
    async def search_openalex(
        self,
        query: str,
        limit: int = 20,
        from_year: Optional[int] = None
    ) -> List[Dict]:
        """Search OpenAlex - fully open, great for social sciences.
        
        Generous rate limits, no API key needed.
        """
        filters = {"default.search": query}
        if from_year:
            filters["from_publication_date"] = f"{from_year}-01-01"
        
        works = Works().filter(**filters).get(per_page=limit)
        
        papers = []
        for work in works:
            papers.append({
                "id": work.get("id", "").replace("https://openalex.org/", ""),
                "title": work.get("title"),
                "abstract": self._reconstruct_abstract(work.get("abstract_inverted_index")),
                "year": work.get("publication_year"),
                "citations": work.get("cited_by_count"),
                "authors": [a["author"]["display_name"] for a in work.get("authorships", [])],
                "doi": work.get("doi"),
                "open_access_url": work.get("open_access", {}).get("oa_url"),
                "source": "openalex"
            })
        
        return papers
    
    def _reconstruct_abstract(self, inverted_index: Optional[Dict]) -> Optional[str]:
        """OpenAlex stores abstracts as inverted indexes."""
        if not inverted_index:
            return None
        
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        word_positions.sort()
        return " ".join([word for _, word in word_positions])
    
    async def get_open_access_pdf(self, doi: str) -> Optional[str]:
        """Try to find open access version via Unpaywall.
        
        Requires email for polite pool access.
        """
        email = "your-email@example.com"  # Replace with config
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        
        try:
            response = await self.http_client.get(url)
            if response.status_code == 200:
                data = response.json()
                best_oa = data.get("best_oa_location")
                if best_oa:
                    return best_oa.get("url_for_pdf") or best_oa.get("url")
        except Exception as e:
            print(f"Unpaywall error: {e}")
        
        return None
    
    async def search_all(self, query: str, limit_per_source: int = 10) -> List[Dict]:
        """Search all sources and deduplicate results."""
        tasks = [
            self.search_semantic_scholar(query, limit_per_source),
            self.search_openalex(query, limit_per_source)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_papers = []
        seen_titles = set()
        
        for result in results:
            if isinstance(result, Exception):
                print(f"Search error: {result}")
                continue
            
            for paper in result:
                # Simple deduplication by title
                title_key = paper["title"].lower()[:50] if paper["title"] else ""
                if title_key and title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_papers.append(paper)
        
        # Sort by citation count
        all_papers.sort(key=lambda x: x.get("citations") or 0, reverse=True)
        
        return all_papers
```

### 3.2 Web Search Integration

```python
# tools/web_search.py
import httpx
from typing import List, Dict

class WebSearchTool:
    """Web search for grey literature, reports, news."""
    
    def __init__(self, api_key: str = None, engine: str = "tavily"):
        self.api_key = api_key
        self.engine = engine
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search_tavily(self, query: str, max_results: int = 10) -> List[Dict]:
        """Tavily is optimized for AI/LLM use cases.
        
        Good for: reports, news, organizational content
        """
        if not self.api_key:
            raise ValueError("Tavily API key required")
        
        response = await self.client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_raw_content": True
            }
        )
        
        data = response.json()
        
        return [{
            "title": r["title"],
            "url": r["url"],
            "content": r["content"],
            "raw_content": r.get("raw_content"),
            "score": r["score"],
            "source": "web"
        } for r in data.get("results", [])]
    
    async def search_serper(self, query: str, max_results: int = 10) -> List[Dict]:
        """Alternative: Serper.dev (Google search results)."""
        response = await self.client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self.api_key},
            json={"q": query, "num": max_results}
        )
        
        data = response.json()
        
        return [{
            "title": r["title"],
            "url": r["link"],
            "content": r.get("snippet", ""),
            "source": "web"
        } for r in data.get("organic", [])]
```

---

## Phase 4: Agent Orchestration (Week 4-5)

### 4.1 LangGraph Agent Setup

```python
# agents/research_agent.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List
import operator

# Define agent state
class ResearchState(TypedDict):
    messages: Annotated[List, operator.add]
    current_query: str
    search_results: List[dict]
    retrieved_context: List[dict]
    should_ingest: List[dict]
    final_answer: str

# Define tools
@tool
def search_academic_sources(query: str, year_from: int = None) -> str:
    """Search academic databases for papers on a topic.
    
    Args:
        query: Search query for academic papers
        year_from: Optional filter for papers published after this year
    """
    # Implementation calls AcademicSearchTools
    pass

@tool
def search_local_knowledge(query: str, max_results: int = 5) -> str:
    """Search the local knowledge base for relevant papers and notes.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
    """
    # Implementation calls vector store
    pass

@tool
def search_web(query: str) -> str:
    """Search the web for reports, news, and grey literature.
    
    Args:
        query: Web search query
    """
    # Implementation calls WebSearchTool
    pass

@tool
def evaluate_and_ingest(paper_id: str, decision: str, notes: str = "") -> str:
    """Decide whether to add a paper to the knowledge base.
    
    Args:
        paper_id: ID of the paper to evaluate
        decision: 'ingest' or 'skip'
        notes: Optional notes about why
    """
    # Implementation handles ingestion
    pass

@tool
def analyze_data(data_description: str, analysis_type: str) -> str:
    """Perform data analysis on provided data.
    
    Args:
        data_description: Description of the data to analyze
        analysis_type: Type of analysis (descriptive, correlation, visualization)
    """
    pass

class ResearchAgent:
    def __init__(self, llm, tools, vector_store):
        self.llm = llm
        self.tools = tools
        self.vector_store = vector_store
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("understand_query", self.understand_query)
        workflow.add_node("search_local", self.search_local)
        workflow.add_node("search_external", self.search_external)
        workflow.add_node("synthesize", self.synthesize_results)
        workflow.add_node("offer_ingestion", self.offer_ingestion)
        workflow.add_node("respond", self.generate_response)
        
        # Define edges
        workflow.set_entry_point("understand_query")
        workflow.add_edge("understand_query", "search_local")
        workflow.add_conditional_edges(
            "search_local",
            self.should_search_external,
            {
                "search_more": "search_external",
                "sufficient": "synthesize"
            }
        )
        workflow.add_edge("search_external", "synthesize")
        workflow.add_edge("synthesize", "offer_ingestion")
        workflow.add_edge("offer_ingestion", "respond")
        workflow.add_edge("respond", END)
        
        return workflow.compile()
    
    def understand_query(self, state: ResearchState) -> ResearchState:
        """Parse user query to understand intent and extract search terms."""
        # LLM determines: is this literature review, data analysis, or synthesis?
        pass
    
    def search_local(self, state: ResearchState) -> ResearchState:
        """Search local vector database first."""
        pass
    
    def should_search_external(self, state: ResearchState) -> str:
        """Decide if local results are sufficient."""
        if len(state["retrieved_context"]) < 3:
            return "search_more"
        # Could also use LLM to evaluate relevance
        return "sufficient"
    
    def search_external(self, state: ResearchState) -> ResearchState:
        """Search academic APIs and web if needed."""
        pass
    
    def synthesize_results(self, state: ResearchState) -> ResearchState:
        """Combine and analyze all gathered information."""
        pass
    
    def offer_ingestion(self, state: ResearchState) -> ResearchState:
        """Identify valuable sources to add to knowledge base."""
        pass
    
    def generate_response(self, state: ResearchState) -> ResearchState:
        """Generate final response for user."""
        pass
    
    async def run(self, user_query: str) -> str:
        """Run the agent on a query."""
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "current_query": user_query,
            "search_results": [],
            "retrieved_context": [],
            "should_ingest": [],
            "final_answer": ""
        }
        
        final_state = await self.graph.ainvoke(initial_state)
        return final_state["final_answer"]
```

### 4.2 System Prompt

```python
RESEARCH_AGENT_PROMPT = """You are a research assistant specializing in social sciences, 
particularly social anthropology and geography. Your role is to help researchers with:

1. **Literature Review**: Find and synthesize academic papers on topics
2. **Paper Summarization**: Extract key findings, methods, and arguments
3. **Data Analysis**: Help analyze qualitative and quantitative data
4. **Information Synthesis**: Connect ideas across sources

## Your Capabilities

You have access to:
- A local knowledge base of papers and notes (search first!)
- Academic databases: Semantic Scholar, OpenAlex
- Web search for grey literature and reports
- Data analysis tools

## Your Process

1. **Always search local knowledge first** - the user may have relevant papers already
2. **Search external sources** when local knowledge is insufficient
3. **Evaluate sources critically** - consider methodology, peer review, recency
4. **Offer to save valuable sources** - ask if new papers should be added to the knowledge base
5. **Cite your sources** - always indicate where information comes from

## Quality Standards

- Prioritize peer-reviewed sources for empirical claims
- Note when sources conflict or have methodological limitations
- Distinguish between established findings and emerging research
- Be transparent about gaps in available literature

## When Analyzing Data

- Clarify what analysis is appropriate for the data type
- Explain statistical concepts in accessible terms
- Visualize data when it aids understanding
- Note limitations of any analysis

Current date: {current_date}
Knowledge base contains: {kb_stats}
"""
```

---

## Phase 5: User Interface (Week 5-6)

### 5.1 Gradio Interface

```python
# ui/app.py
import gradio as gr
from agents.research_agent import ResearchAgent

def create_research_ui(agent: ResearchAgent):
    
    with gr.Blocks(title="Research Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔬 Research Assistant")
        gr.Markdown("Social sciences research helper with autonomous knowledge building")
        
        with gr.Tab("💬 Research Chat"):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(
                placeholder="Ask me about your research topic...",
                label="Your question"
            )
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
            
            with gr.Accordion("🔧 Settings", open=False):
                search_depth = gr.Slider(
                    minimum=1, maximum=20, value=5,
                    label="Max external results per source"
                )
                auto_ingest = gr.Checkbox(
                    label="Automatically add high-quality sources",
                    value=False
                )
        
        with gr.Tab("📚 Knowledge Base"):
            gr.Markdown("## Your Research Library")
            
            with gr.Row():
                kb_stats = gr.JSON(label="Statistics")
                refresh_btn = gr.Button("Refresh Stats")
            
            with gr.Row():
                upload_pdf = gr.File(
                    label="Upload PDFs to add to knowledge base",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                upload_btn = gr.Button("Process & Add")
            
            upload_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Tab("📊 Data Analysis"):
            gr.Markdown("## Analyze Your Data")
            
            data_input = gr.File(
                label="Upload CSV or Excel file",
                file_types=[".csv", ".xlsx"]
            )
            
            analysis_type = gr.Radio(
                choices=["Descriptive Statistics", "Correlation Analysis", 
                        "Frequency Analysis", "Custom Query"],
                label="Analysis Type"
            )
            
            custom_query = gr.Textbox(
                label="Custom analysis request",
                visible=False
            )
            
            analyze_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Row():
                analysis_output = gr.Markdown()
                analysis_plot = gr.Plot()
        
        # Event handlers
        async def respond(message, history):
            response = await agent.run(message)
            history.append((message, response))
            return "", history
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], outputs=[chatbot])
    
    return app

if __name__ == "__main__":
    # Initialize components
    agent = ResearchAgent(...)
    app = create_research_ui(agent)
    app.launch(share=False)
```

---

## Phase 6: Autonomous Knowledge Building (Week 6-7)

### 6.1 Ingestion Decision System

```python
# processors/ingestion_manager.py
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class SourceQuality(Enum):
    HIGH = "high"        # Peer-reviewed, highly cited
    MEDIUM = "medium"    # Working papers, reports from known orgs
    LOW = "low"          # Blog posts, news (still valuable for context)
    UNCERTAIN = "uncertain"

@dataclass
class IngestionCandidate:
    source_id: str
    title: str
    source_type: str  # paper, report, web
    quality_score: SourceQuality
    relevance_score: float  # 0-1
    reasons: List[str]
    metadata: Dict

class IngestionManager:
    """Manages what gets added to the knowledge base."""
    
    def __init__(self, vector_store, llm, auto_ingest_threshold: float = 0.8):
        self.vector_store = vector_store
        self.llm = llm
        self.auto_ingest_threshold = auto_ingest_threshold
        self.pending_review: List[IngestionCandidate] = []
    
    async def evaluate_source(self, source: Dict, user_query: str) -> IngestionCandidate:
        """Evaluate whether a source should be ingested."""
        
        # Score quality based on heuristics
        quality = self._assess_quality(source)
        
        # Use LLM to assess relevance to user's research interests
        relevance = await self._assess_relevance(source, user_query)
        
        # Check for duplicates
        is_duplicate = await self._check_duplicate(source)
        
        reasons = []
        if is_duplicate:
            reasons.append("Already in knowledge base")
            relevance = 0.0
        if quality == SourceQuality.HIGH:
            reasons.append("Peer-reviewed source")
        if source.get("citations", 0) > 100:
            reasons.append(f"Highly cited ({source['citations']} citations)")
        
        return IngestionCandidate(
            source_id=source.get("id", source.get("url", "")),
            title=source.get("title", "Unknown"),
            source_type=source.get("source", "unknown"),
            quality_score=quality,
            relevance_score=relevance,
            reasons=reasons,
            metadata=source
        )
    
    def _assess_quality(self, source: Dict) -> SourceQuality:
        """Heuristic quality assessment."""
        if source.get("source") in ["semantic_scholar", "openalex"]:
            if source.get("citations", 0) > 50:
                return SourceQuality.HIGH
            return SourceQuality.MEDIUM
        elif source.get("source") == "web":
            # Could check domain reputation
            return SourceQuality.LOW
        return SourceQuality.UNCERTAIN
    
    async def _assess_relevance(self, source: Dict, user_query: str) -> float:
        """Use LLM to assess relevance."""
        prompt = f"""Rate the relevance of this paper to the research query on a scale of 0-1.

Research query: {user_query}

Paper title: {source.get('title')}
Abstract: {source.get('abstract', 'No abstract available')[:500]}

Respond with only a number between 0 and 1."""

        # Call LLM and parse response
        # ... implementation
        pass
    
    async def _check_duplicate(self, source: Dict) -> bool:
        """Check if source already exists in knowledge base."""
        # Search by title similarity
        pass
    
    def format_ingestion_offer(self, candidates: List[IngestionCandidate]) -> str:
        """Format candidates for user review."""
        worthy = [c for c in candidates if c.relevance_score > 0.5]
        
        if not worthy:
            return "No new sources worth adding to your knowledge base."
        
        message = "**📥 Sources to consider adding:**\n\n"
        
        for i, c in enumerate(worthy, 1):
            quality_emoji = {
                SourceQuality.HIGH: "⭐",
                SourceQuality.MEDIUM: "📄", 
                SourceQuality.LOW: "📰"
            }.get(c.quality_score, "❓")
            
            message += f"{i}. {quality_emoji} **{c.title}**\n"
            message += f"   - Relevance: {c.relevance_score:.0%}\n"
            message += f"   - {', '.join(c.reasons)}\n\n"
        
        message += "\nReply with numbers to add (e.g., '1, 3') or 'all' / 'none'"
        
        return message
```

### 6.2 Citation Chain Following

```python
# tools/citation_explorer.py
class CitationExplorer:
    """Follow citation chains to discover related work."""
    
    def __init__(self, academic_search: AcademicSearchTools):
        self.search = academic_search
    
    async def get_citations(self, paper_id: str, direction: str = "both") -> Dict:
        """Get papers that cite this paper and papers this paper cites.
        
        direction: 'citing', 'cited', or 'both'
        """
        result = {"citing": [], "cited": []}
        
        # Semantic Scholar provides citation data
        paper = self.search.s2.get_paper(
            paper_id,
            fields=["citations", "references"]
        )
        
        if direction in ["citing", "both"]:
            result["citing"] = [{
                "id": c.paperId,
                "title": c.title,
                "year": c.year
            } for c in (paper.citations or [])[:20]]
        
        if direction in ["cited", "both"]:
            result["cited"] = [{
                "id": r.paperId,
                "title": r.title,
                "year": r.year
            } for r in (paper.references or [])[:20]]
        
        return result
    
    async def find_highly_connected(self, paper_ids: List[str]) -> List[Dict]:
        """Find papers frequently cited by papers in the knowledge base.
        
        Useful for discovering foundational works.
        """
        citation_counts = {}
        
        for pid in paper_ids:
            refs = await self.get_citations(pid, direction="cited")
            for ref in refs["cited"]:
                ref_id = ref["id"]
                if ref_id not in citation_counts:
                    citation_counts[ref_id] = {"paper": ref, "count": 0}
                citation_counts[ref_id]["count"] += 1
        
        # Sort by how many papers in KB cite this
        sorted_refs = sorted(
            citation_counts.values(),
            key=lambda x: x["count"],
            reverse=True
        )
        
        return [
            {**item["paper"], "kb_citations": item["count"]}
            for item in sorted_refs[:10]
        ]
```

---

## Phase 7: Cloud Integration (When Needed)

### 7.1 When to Use Cloud

| Task | Local (5090) | Cloud |
|------|-------------|-------|
| Chat inference | ✅ Qwen-32B-Q4 | ❌ Unnecessary |
| Embedding generation | ✅ BGE-large | ❌ Unnecessary |
| Bulk paper processing | ⚠️ Slow | ✅ Batch API |
| Fine-tuning (if needed later) | ⚠️ Limited | ✅ Better options |
| 70B+ models | ❌ Won't fit | ✅ Required |

### 7.2 Hybrid Setup

```python
# models/hybrid_llm.py
from typing import Optional
import os

class HybridLLM:
    """Automatically route to local or cloud based on task."""
    
    def __init__(self, local_model, cloud_client=None):
        self.local = local_model
        self.cloud = cloud_client  # OpenAI, Anthropic, etc.
        
    async def generate(
        self, 
        prompt: str, 
        prefer_cloud: bool = False,
        max_tokens: int = 1024
    ) -> str:
        """Generate with automatic routing."""
        
        # Use cloud for long-form synthesis or when explicitly preferred
        if prefer_cloud and self.cloud:
            return await self._cloud_generate(prompt, max_tokens)
        
        # Default to local
        return await self._local_generate(prompt, max_tokens)
    
    async def batch_process(self, prompts: List[str]) -> List[str]:
        """For bulk processing, use cloud batch API if available."""
        if self.cloud and len(prompts) > 10:
            # OpenAI/Anthropic batch APIs are cost-effective
            return await self._cloud_batch(prompts)
        
        # Otherwise process locally
        return [await self._local_generate(p) for p in prompts]
```

---

## Configuration Files

### config.yaml

```yaml
# config/config.yaml

model:
  name: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
  max_new_tokens: 2048
  temperature: 0.7
  
embedding:
  name: "BAAI/bge-large-en-v1.5"
  dimension: 1024

vector_store:
  type: "chromadb"  # or "qdrant"
  persist_directory: "./data/chroma_db"
  
retrieval:
  top_k: 5
  rerank: true
  reranker_model: "BAAI/bge-reranker-base"

search:
  semantic_scholar:
    enabled: true
    rate_limit: 100  # per 5 minutes
  openalex:
    enabled: true
  unpaywall:
    enabled: true
    email: "your-email@example.com"
  web_search:
    provider: "tavily"  # or "serper"
    api_key: "${TAVILY_API_KEY}"

ingestion:
  auto_ingest: false  # require user approval
  auto_threshold: 0.85  # if auto_ingest true
  chunk_size: 512
  chunk_overlap: 50
  
ui:
  port: 7860
  share: false
```

---

## Recommended Development Order

1. **Week 1**: Environment setup, load base model, verify GPU usage
2. **Week 2**: Vector database, document processor, basic retrieval
3. **Week 3**: Academic search APIs (Semantic Scholar, OpenAlex)
4. **Week 4**: Agent orchestration with LangGraph
5. **Week 5**: Gradio UI, end-to-end testing
6. **Week 6**: Ingestion manager, citation explorer
7. **Week 7**: Polish, edge cases, documentation

---

## Quick Start Commands

```bash
# 1. Clone and setup
git clone <your-repo>
cd research_agent
conda env create -f environment.yml
conda activate research_agent

# 2. Download model (first time)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4')"

# 3. Initialize database
python -m db.init_vector_store

# 4. Run the app
python -m ui.app
```

---

## Next Steps After Basic Setup

1. **Reranker**: Add `bge-reranker` for better retrieval quality
2. **Citation management**: Export to Zotero/BibTeX
3. **Collaborative features**: Share knowledge bases
4. **Fine-tuning**: If you identify specific gaps, fine-tune on social science papers
5. **Evaluation**: Build a test set to measure retrieval and synthesis quality
