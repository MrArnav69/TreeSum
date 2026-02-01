#!/bin/bash

# RunPod Setup Script for TreeSum
echo "🚀 Starting setup for TreeSum..."

# 1. Update and install system dependencies
echo "📦 Installing system dependencies..."
apt-get update && apt-get install -y git wget curl

# 2. Upgrade pip
echo "🆙 Upgrading pip..."
pip install --upgrade pip

# 3. Install Python dependencies
echo "🐍 Installing Python packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

# 4. Download NLTK data (required for tokenization)
echo "📚 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

echo "✅ Environment setup complete! You can now run the HPC sweep:"
echo "python3 production/scripts/run_hpc_sweep.py"
