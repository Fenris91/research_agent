#!/bin/bash
# VRAM Stress Test for Research Agent

# Pre-test check
echo "=== VRAM Before Test ==="
nvidia-smi --query-gpu=memory.used --format=csv

echo "\nRunning VRAM stress test with long prompt..."

# Execute test with long prompt
python3 -c "from agents.research_agent import ResearchAgent; agent = ResearchAgent(); 
result = agent.infer('Compare participatory action research methodologies in South Asian vs Sub-Saharan African contexts', max_tokens=512); 
print(\'\nGenerated result: $result\')"

echo "\n=== VRAM After Test ==="
nvidia-smi --query-gpu=memory.used --format=csv

# Clean output
echo "\n== VRAM Delta Test Summary =="
python3 -c "import pandas as pd; 
adf = pd.read_csv('vram_before.csv'); bdf = pd.read_csv('vram_after.csv'); 
print(pd.merge(adf, bdf, on='name', suffixes=['_before', '_after']))"