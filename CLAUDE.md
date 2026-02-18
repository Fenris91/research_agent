# Research Agent - Claude Code Instructions

## Project Vision
A **personal research memory** for social sciences researchers.
Core loop: Ingest papers → Build knowledge base → Query semantically → Get cited answers.

## Current Status
✅ Working:
- PDF upload and chunking
- ChromaDB vector storage (1111+ chunks stored)
- Semantic search retrieval (finds relevant chunks correctly)
- Academic search APIs (Semantic Scholar, OpenAlex)
- Gradio UI
- LLM integration (Groq/OpenAI)

⚠️ Issues:
- Agent says "no information found" even when retrieval returns relevant chunks
- Agent doesn't recognize paper titles (e.g., "Envisaging the Future of Cities" = World Cities Report)
- Searches academic APIs for casual chat messages (causes 429 rate limits)

## Tech Stack
- Python 3.11, RTX 5090 (32GB VRAM)
- Embeddings: `BAAI/bge-base-en-v1.5` (768 dims) - DO NOT CHANGE, must match DB
- Vector DB: ChromaDB at `./data/chroma_db`
- LLM: Groq free tier (default), OpenAI, Ollama
- Agent: LangGraph
- UI: Gradio

## Code Structure
```
src/research_agent/
├── agents/      # LangGraph orchestration - NEEDS WORK
├── db/          # ChromaDB vector store - working
├── models/      # LLM/embedding loaders - working
├── processors/  # PDF ingestion - working
├── tools/       # Search APIs - working
├── ui/          # Gradio interface - working
└── main.py
```

## Priority Fixes
1. **Agent synthesis**: When retrieval finds sources, acknowledge them properly
2. **Query routing**: Only search external APIs for research queries, not "hi how are you"
3. **Source attribution**: Show paper titles clearly in responses

## Running
```bash
conda activate llm311
python -m research_agent.main --mode ui
# http://localhost:7860
```

## Testing Retrieval Manually
```python
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
client = chromadb.PersistentClient(path='./data/chroma_db')
papers = client.get_collection('academic_papers')

results = papers.query(
    query_embeddings=[model.encode('your query').tolist()],
    n_results=5
)
```
