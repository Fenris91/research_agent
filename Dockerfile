# ============================================
# Research Agent - Docker Image
# ============================================
# Build:  docker build -t research-agent .
# Run:    docker compose up
#
# For GPU support (NVIDIA), use the gpu stage instead:
#   docker build --target gpu -t research-agent:gpu .
#   (requires nvidia-container-toolkit on the host)
# ============================================

# ------------------ Base stage ------------------
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for document processing and general use
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash agent

# ------------------ CPU stage (default) ------------------
FROM base AS cpu

# Install CPU-only PyTorch first (much smaller than CUDA variant)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
# Skip torch/torchvision/torchaudio (already installed above as CPU-only)
# and skip bitsandbytes (CUDA-only, not needed for CPU)
RUN grep -v -E '^(torch|torchvision|torchaudio|bitsandbytes)' requirements.txt > requirements-docker.txt \
    && pip install --no-cache-dir -r requirements-docker.txt \
    && rm requirements-docker.txt

# Install extra packages not in requirements.txt
RUN pip install --no-cache-dir openai>=1.14.0 pytz

# Copy application code
COPY setup.py .
COPY configs/ configs/
COPY src/ src/

# Install the package (non-editable for container)
RUN pip install --no-cache-dir .

# Set up directories and permissions for non-root user
RUN mkdir -p data/chroma_db logs cache exports .hf_cache \
    && chown -R agent:agent /app

USER agent

ENV HF_HOME=/app/.hf_cache

EXPOSE 7860

# Gradio must bind to 0.0.0.0 to be reachable from outside the container
CMD ["python", "-m", "research_agent.main", "--mode", "ui", "--host", "0.0.0.0"]

# ------------------ GPU stage (optional) ------------------
FROM base AS gpu

# Install CUDA PyTorch
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN grep -v -E '^(torch|torchvision|torchaudio)' requirements.txt > requirements-docker.txt \
    && pip install --no-cache-dir -r requirements-docker.txt \
    && rm requirements-docker.txt

RUN pip install --no-cache-dir openai>=1.14.0 pytz

COPY setup.py .
COPY configs/ configs/
COPY src/ src/
RUN pip install --no-cache-dir .

RUN mkdir -p data/chroma_db logs cache exports .hf_cache \
    && chown -R agent:agent /app

USER agent

ENV HF_HOME=/app/.hf_cache

EXPOSE 7860

CMD ["python", "-m", "research_agent.main", "--mode", "ui", "--host", "0.0.0.0"]
