# 🔬 Research Agent

An autonomous research assistant for social sciences (social anthropology, geography, etc.) that can search, analyze, and build its own knowledge base.

## Features

- **Literature Review**: Search Semantic Scholar, OpenAlex, and web sources
- **Paper Summarization**: Extract key findings from academic papers
- **Autonomous Knowledge Building**: Agent evaluates and ingests valuable sources
- **Data Analysis**: Descriptive stats, correlations, visualizations
- **Citation Exploration**: Follow citation chains to discover related work

## Requirements

- NVIDIA GPU with 16GB+ VRAM (32GB recommended for larger models)
- Python 3.11+
- ~50GB disk space for models and data

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd research_agent

# 2. Create conda environment
conda create -n research_agent python=3.11
conda activate research_agent

# 3. Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install dependencies
pip install -r requirements.txt

# 5. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 6. Run the app
python -m src.main
```

## Configuration

Edit `configs/config.yaml` to customize:

- Model selection (Qwen, Mistral, or Ollama)
- Search API settings
- Ingestion preferences
- UI options

## Project Structure

```
research_agent/
├── src/
│   ├── agents/          # LangGraph agent definitions
│   ├── tools/           # Search, retrieval, analysis tools
│   ├── db/              # Vector store management
│   ├── processors/      # Document processing pipeline
│   ├── models/          # LLM and embedding loaders
│   └── ui/              # Gradio interface
├── configs/             # Configuration files
├── tests/               # Test suite
├── docs/                # Documentation
│   └── PLAN.md          # Implementation roadmap
└── data/                # Local data (gitignored)
```

## Usage

### Chat Interface

```python
from src.agents import ResearchAgent

agent = ResearchAgent.from_config("configs/config.yaml")
response = await agent.run("What are the key theories in urban anthropology?")
```

### Adding Papers Manually

```python
from src.processors import AcademicPDFProcessor
from src.db import ResearchVectorStore

processor = AcademicPDFProcessor()
store = ResearchVectorStore()

# Process and add a PDF
doc = processor.extract_text("paper.pdf")
chunks = processor.chunk_text(doc["full_text"])
store.add_paper("paper_id", chunks, embeddings, metadata)
```

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

- Semantic Scholar API
- OpenAlex
- HuggingFace Transformers
- LangChain / LangGraph
