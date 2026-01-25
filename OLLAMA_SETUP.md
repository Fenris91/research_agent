# Using Ollama with Research Agent

This guide explains how to use Ollama to run local LLMs with the Research Agent.

## What is Ollama?

[Ollama](https://ollama.ai) is a simple tool to run large language models locally. It handles model management, memory optimization, and provides an easy API.

## Installation

### 1. Install Ollama

**Linux/WSL:**
```bash
curl https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
# Download from https://ollama.ai/download
# Or use Homebrew:
brew install ollama
```

**Windows:**
Download from https://ollama.ai/download

### 2. Start Ollama Server

```bash
ollama serve
```

This starts the Ollama API on `http://localhost:11434` (default).

### 3. Pull a Model

In a new terminal, pull your preferred model:

```bash
# Lightweight and fast
ollama pull mistral

# Larger, more capable
ollama pull llama2

# Specialized
ollama pull neural-chat
ollama pull openchat
```

For a list of available models, visit: https://ollama.ai/library

## Using with Research Agent

### Option 1: Environment Variables (Recommended)

```bash
export USE_OLLAMA=true
export OLLAMA_MODEL=mistral
export OLLAMA_BASE_URL=http://localhost:11434

python src/ui/app.py
```

### Option 2: Programmatic

```python
from src.agents.research_agent import ResearchAgent

# Use Ollama
agent = ResearchAgent(
    use_ollama=True,
    ollama_model="mistral",
    ollama_base_url="http://localhost:11434"
)

result = agent.run("What is participatory action research?")
print(result)
```

### Option 3: Command Line

```bash
python -c '
from src.agents.research_agent import ResearchAgent
agent = ResearchAgent(use_ollama=True, ollama_model="mistral")
result = agent.run("Your question here")
print(result["answer"])
'
```

## Model Recommendations

### For Research Tasks
- **mistral** (7B) - Fast, good quality, recommended for most tasks
- **neural-chat** (7B) - Specialized for conversation, good for chat
- **openchat** (3.5B) - Very lightweight, still capable

### For Longer Documents
- **llama2** (7B or 13B) - Better at long-form understanding
- **dolphin-mixtral** (Needs 16GB+ VRAM) - Very capable

### For CPU-only Systems
- **mistral:latest** (7B) - Can run on CPU with ~8GB RAM
- **neural-chat:latest** (7B)
- Quantized models like `openchat:latest` (3.5B)

## Performance Tips

### 1. GPU Acceleration (Recommended)

If you have NVIDIA GPU:
```bash
# Ollama will auto-detect and use NVIDIA GPUs
ollama serve
```

### 2. Adjust Context and Threads

Edit `~/.ollama/ollama.py` (or use environment variables):
```bash
OLLAMA_NUM_THREAD=8 ollama serve
```

### 3. Use Quantized Models

Quantized models run faster and use less VRAM:
- `mistral:latest` - quantized by default
- `llama2:7b-chat-q4_0` - 4-bit quantized

### 4. Pre-load Models

Keep your model in memory for faster responses (uses more VRAM):
```bash
ollama pull mistral
# Model stays loaded and ready
```

## Fallback Behavior

If Ollama is unavailable or you need HuggingFace models:

1. Set `use_ollama=False` (default)
2. Agent will load Mistral 7B, TinyLlama, or Qwen2.5-32B depending on VRAM
3. Agent automatically switches to HuggingFace if Ollama fails

Example:
```python
# This will try Ollama first, fall back to HF models if needed
agent = ResearchAgent(use_ollama=True)
```

## Troubleshooting

### Ollama Connection Error

```
OllamaUnavailableError: Cannot connect to Ollama at http://localhost:11434
```

**Solution:** Make sure Ollama is running:
```bash
ollama serve
```

### Model Not Found

```
Ollama returned status 404
```

**Solution:** Pull the model first:
```bash
ollama pull mistral
```

### Out of Memory

**Solution:** Use a smaller quantized model:
```bash
ollama pull mistral:7b-instruct-q4_0
```

### Slow Responses

**Solution:**
1. Check if model is fully loaded: `ollama list`
2. Reduce context size in agent config
3. Use a smaller quantized model

## Advanced: Custom Ollama Installation

If running Ollama on a different machine:

```python
agent = ResearchAgent(
    use_ollama=True,
    ollama_model="mistral",
    ollama_base_url="http://192.168.1.100:11434"  # Custom server
)
```

## Comparing Ollama vs HuggingFace Models

| Aspect | Ollama | HuggingFace |
|--------|--------|-----------|
| Setup | Easy (one command) | More configuration |
| Models | Curated selection | Thousands available |
| Performance | Auto GPU detection | Manual optimization |
| Memory | Good quantization | More options |
| API | Simple REST API | Local inference |
| Speed | Fast with GPU | Varies by model |

## Next Steps

- Read [SETUP.md](docs/SETUP.md) for general setup
- Check [PLAN.md](docs/PLAN.md) for architecture overview
- Run the UI: `python src/ui/app.py`
