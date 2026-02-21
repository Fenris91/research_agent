# Setup

## Environment
```bash
conda create -n llm311 python=3.11 -y
conda activate llm311
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Environment Variables
Copy `.env.example` to `.env` and set:
- `GROQ_API_KEY` — Groq free tier (recommended default)
- `OPENAI_API_KEY` — OpenAI (optional)
- `UNPAYWALL_EMAIL` — your email for open access lookups
- `OPENALEX_EMAIL` — your email for polite pool (optional)

## Run
```bash
conda activate llm311
python -m research_agent.main --mode ui
# Auto-finds available port starting at 7860
```

## Verify
```bash
python -m research_agent.main --mode check
nvidia-smi  # GPU check
```

## Troubleshooting
- **CUDA not found**: update Windows NVIDIA drivers, ensure WSL GPU support enabled
- **Module not found**: check `which python` points to conda env, re-run `pip install -r requirements.txt`
- **Out of memory**: try a smaller model in config.yaml or use Ollama with quantization
