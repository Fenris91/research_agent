# Research Agent - Setup Instructions

## Step 1: Move the Plan to WSL

Your `research_assistant_implementation_plan.md` is downloaded in Edge (probably in `C:\Users\<YourUsername>\Downloads\`).

**Option A: Copy via Windows Explorer**
1. Open Windows Explorer
2. Navigate to `\\wsl$\Ubuntu\home\<your-wsl-username>\`
3. Create a folder called `projects` if it doesn't exist
4. Copy the plan file there

**Option B: Copy via WSL terminal**
```bash
# Open WSL terminal (Windows Terminal or Ubuntu app)

# Create projects directory
mkdir -p ~/projects

# Copy from Windows Downloads (replace YourWindowsUsername)
cp "/mnt/c/Users/YourWindowsUsername/Downloads/research_assistant_implementation_plan.md" ~/projects/
```

---

## Step 2: Create Project and Initialize Git

```bash
# Navigate to projects folder
cd ~/projects

# Create project directory structure
mkdir -p research_agent
cd research_agent

# Initialize git
git init

# Create initial commit message
git config user.email "you@example.com"  # Replace with your email
git config user.name "Your Name"          # Replace with your name
```

---

## Step 3: Copy Project Files

You have two options:

### Option A: Download the starter files I created

If I've provided a zip or the files are available somewhere:

```bash
# Unzip into research_agent folder
# (adjust path to wherever you downloaded the files)
```

### Option B: Clone from a repo (if you've pushed it somewhere)

```bash
git clone <your-repo-url> research_agent
cd research_agent
```

### Option C: Create the structure manually

The structure should look like this:
```
research_agent/
├── .gitignore
├── .env.example
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── docs/
│   └── PLAN.md          # Your implementation plan goes here
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── agents/
│   ├── db/
│   ├── models/
│   ├── processors/
│   ├── tools/
│   └── ui/
└── tests/
    └── __init__.py
```

---

## Step 4: Set Up Python Environment

```bash
# Make sure you're in the project directory
cd ~/projects/research_agent

# Create conda environment (if using conda)
conda create -n research_agent python=3.11 -y
conda activate research_agent

# OR use venv
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
# Check your CUDA version: nvidia-smi
# For CUDA 12.4:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

---

## Step 5: Set Up Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit with your API keys
nano .env   # or use vim, code, etc.
```

Minimum needed to start:
- `UNPAYWALL_EMAIL` - your email (for open access lookups)
- `OPENALEX_EMAIL` - your email (optional but recommended)
- `TAVILY_API_KEY` - get free tier at https://tavily.com (optional for now)

---

## Step 6: Move the Plan into the Project

```bash
# Create docs folder if it doesn't exist
mkdir -p docs

# Move your plan
mv ~/projects/research_assistant_implementation_plan.md docs/PLAN.md
```

---

## Step 7: Open in VS Code

```bash
# Open VS Code from WSL (installs VS Code Server if needed)
code .
```

This will:
1. Open VS Code connected to WSL
2. Install the Remote - WSL extension if needed
3. Give you a Linux environment with access to your GPU

**Recommended VS Code Extensions:**
- Python
- Pylance
- Claude Code (for AI assistance)
- GitLens

---

## Step 8: Verify Setup

```bash
# Run the setup check
python -m src.main --mode check
```

This will verify:
- GPU is accessible
- Required packages are installed
- Everything is configured correctly

---

## Step 9: First Git Commit

```bash
# Add all files
git add .

# Commit
git commit -m "Initial project structure"

# (Optional) Push to remote
# git remote add origin <your-repo-url>
# git push -u origin main
```

---

## Using Claude Code Extension

1. Open VS Code with the project
2. Install Claude Code extension if not already installed
3. Open the Command Palette (Ctrl+Shift+P)
4. Search for "Claude" to see available commands

**To add PLAN.md as context:**
1. Open `docs/PLAN.md` in VS Code
2. The Claude Code extension will be able to reference open files
3. Or use `@file` mentions in the Claude chat to reference specific files

---

## Quick Reference: Useful Commands

```bash
# Activate environment
conda activate research_agent
# OR
source venv/bin/activate

# Run the UI
python -m src.main --mode ui

# Run setup check
python -m src.main --mode check

# Run CLI mode
python -m src.main --mode cli

# Check GPU memory
nvidia-smi

# Watch GPU usage
watch -n 1 nvidia-smi
```

---

## Troubleshooting

### CUDA not found in WSL
```bash
# Check if NVIDIA driver is working
nvidia-smi

# If not, you may need to:
# 1. Update Windows NVIDIA drivers
# 2. Enable WSL GPU support in Windows
```

### Module not found errors
```bash
# Make sure you're in the right environment
which python  # Should show conda or venv path

# Reinstall requirements
pip install -r requirements.txt
```

### Out of memory loading model
```bash
# Try a smaller model in config.yaml
# Or use Ollama with quantization
```
