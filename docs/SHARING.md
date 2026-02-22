# Sharing the Research Agent

Three ways to share the app, from quickest to most permanent.

## 1. Gradio Share Link (quickest)

Creates a temporary public URL (72 hours) through Gradio's tunnel. No setup on the receiving end — just a link in a browser.

```bash
uv run python -m research_agent.main --mode ui --share
```

This prints a URL like `https://abc123.gradio.live`. Send it to anyone.

**Pros:** Zero setup, works immediately, HTTPS
**Cons:** Expires after 72h, depends on your machine staying on

## 2. Docker Compose (self-hosted, persistent)

Run the app in a container on your machine or a server.

```bash
# CPU-only (smaller image, works anywhere)
docker compose up --build

# GPU (requires nvidia-container-toolkit)
docker compose -f docker-compose.yml up --build
# Edit docker-compose.yml: change target from "cpu" to "gpu"
```

The app will be available at `http://<your-ip>:7860`.

To make it accessible outside your network, use a tunnel:
- [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) (free, permanent URL)
- [Tailscale](https://tailscale.com/) (free, private network between devices)

## 3. Cloud Deployment (permanent public URL)

### Hugging Face Spaces

Free tier supports Gradio apps natively, but CPU-only (embedding model will be slow).

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select "Gradio" as the SDK
3. Push the repo to the Space
4. Set API keys in the Space's Settings > Variables

### Railway / Render / Fly.io

Use the existing Dockerfile for container-based hosting. These services offer pay-as-you-go GPU instances.

## Access tiers

The app works with zero configuration. Features unlock progressively:

| What you provide | What you get |
|-----------------|-------------|
| Nothing | Browse KB, search papers, explore graph, upload PDFs |
| Email (in Settings) | Better academic API rate limits (polite pool) |
| API key (in Settings) | AI-powered chat synthesis (Groq free, OpenAI, Claude, etc.) |

No account creation needed. Email and API keys are session-only (never saved to disk).
