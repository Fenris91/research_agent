# Research Agent

An autonomous research assistant for social sciences (social anthropology, geography, etc.) that can search, analyze, and build its own knowledge base.

## Features

### Implemented
- **Academic Paper Search**: Search Semantic Scholar and OpenAlex APIs
- **Web Search**: Free DuckDuckGo search (no API key required)
- **Researcher Lookup**: Fetch citation data, h-index, and web presence for researchers
- **Vector Store**: ChromaDB-based knowledge base with semantic search
- **Research Agent**: LangGraph-based orchestration for query understanding, search, and synthesis
- **Gradio UI**: Web interface with chat, knowledge base, and researcher lookup tabs

### Planned
- **PDF Ingestion**: Extract and chunk academic PDFs
- **LLM Integration**: Local LLM for response synthesis
- **Citation Explorer**: Follow citation chains
- **Data Analysis**: Statistical analysis and visualization

## Requirements

- Python 3.11+
- ~2GB disk space for embedding models
- GPU optional (speeds up embeddings)

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd research_agent

# 2. Create conda environment
conda create -n research_agent python=3.11
conda activate research_agent

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the UI
python -m research_agent.ui.app
```

## Usage Examples

### Research Agent (Python)

```python
import asyncio
from research_agent.agents import create_research_agent

async def main():
    # Create agent with all components
    agent = create_research_agent()

    # Run a research query
    result = await agent.run("What theories explain urban gentrification?")

    print(result["answer"])
    print(f"Found {len(result['external_results'])} papers")

    # Ingest a paper to knowledge base
    if result["candidates_for_ingestion"]:
        await agent.ingest_paper(result["candidates_for_ingestion"][0])

asyncio.run(main())
```

### Academic Paper Search

```python
import asyncio
from research_agent.tools import AcademicSearchTools

async def search():
    search = AcademicSearchTools()

    # Search across Semantic Scholar and OpenAlex
    papers = await search.search_all(
        "participatory mapping indigenous communities",
        limit_per_source=10,
        year_range=(2015, 2024)
    )

    for paper in papers[:5]:
        print(f"{paper.title} ({paper.year}) - {paper.citations} citations")

    await search.close()

asyncio.run(search())
```

### Researcher Lookup

```python
import asyncio
from research_agent.tools import ResearcherLookup

async def lookup():
    lookup = ResearcherLookup()

    profile = await lookup.lookup_researcher("David Harvey")

    print(f"Name: {profile.name}")
    print(f"Citations: {profile.citations_count:,}")
    print(f"H-Index: {profile.h_index}")
    print(f"Affiliations: {', '.join(profile.affiliations)}")

    await lookup.close()

asyncio.run(lookup())
```

### CLI: Researcher Lookup

```bash
# Lookup researchers from command line
python -m research_agent.scripts.lookup_researchers --names "David Harvey, Doreen Massey"

# From a file
python -m research_agent.scripts.lookup_researchers --file data/researchers.txt

# Output as JSON
python -m research_agent.scripts.lookup_researchers --names "Anna Tsing" --json
```

### Vector Store

```python
from research_agent.db import ResearchVectorStore, EmbeddingModel

# Initialize
store = ResearchVectorStore("./data/chroma_db")
embedder = EmbeddingModel()

# Add a paper
chunks = ["First paragraph...", "Second paragraph..."]
embeddings = embedder.embed_batch(chunks)

store.add_paper(
    paper_id="paper123",
    chunks=chunks,
    embeddings=embeddings,
    metadata={"title": "My Paper", "year": 2024, "authors": ["Author Name"]}
)

# Search
query_emb = embedder.embed_query("urban development")
results = store.search(query_emb, collection="papers", n_results=5)

# Get stats
print(store.get_stats())
```

## Project Structure

```
research_agent/
├── src/
│   └── research_agent/
│       ├── agents/           # LangGraph research agent
│       │   └── research_agent.py
│       ├── tools/            # Search and lookup tools
│       │   ├── academic_search.py    # Semantic Scholar + OpenAlex
│       │   ├── web_search.py         # DuckDuckGo, Tavily, Serper
│       │   ├── researcher_lookup.py  # Author profile lookup
│       │   └── researcher_file_parser.py
│       ├── db/               # Vector store
│       │   ├── vector_store.py       # ChromaDB wrapper
│       │   └── embeddings.py         # Sentence transformers
│       ├── processors/       # Document processing (planned)
│       ├── scripts/          # CLI tools
│       │   └── lookup_researchers.py
│       └── ui/               # Gradio interface
│           └── app.py
├── configs/
│   └── config.yaml       # Configuration
├── data/                 # Local data (gitignored)
│   ├── chroma_db/        # Vector database
│   └── researchers/      # Researcher lookup results
├── docs/
│   └── PLAN.md           # Implementation roadmap
└── tests/
```

## Configuration

Edit `configs/config.yaml`:

```yaml
# Embedding model
embedding:
  name: "BAAI/bge-base-en-v1.5"
  device: "cuda"  # or "cpu"

# Vector store
vector_store:
  persist_directory: "./data/chroma_db"

# Search settings
search:
  semantic_scholar:
    enabled: true
  openalex:
    enabled: true
    email: null  # Optional: for polite pool
  web_search:
    provider: "duckduckgo"  # free, no key

# Researcher lookup
researcher_lookup:
  input_file: "./data/researchers.txt"
  output_dir: "./data/researchers"
```

## API Rate Limits

| API | Rate Limit | Key Required |
|-----|------------|--------------|
| Semantic Scholar | 100 req/5min | No |
| OpenAlex | Very generous | No |
| DuckDuckGo | Reasonable | No |
| Unpaywall | Generous | Email only |

## Development

```bash
# Run tests
pytest tests/

# Format code
black src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

## License

MIT

## Acknowledgments

- [Semantic Scholar API](https://www.semanticscholar.org/product/api)
- [OpenAlex](https://openalex.org/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
