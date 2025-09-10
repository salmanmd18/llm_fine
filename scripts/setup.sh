#!/usr/bin/env bash
set -euo pipefail

#
# Project bootstrap script
# - Creates folder structure
# - Sets up Python 3.10+ virtual environment
# - Installs dependencies from requirements.txt
#
# Usage:
#   bash scripts/setup.sh
#

echo "[1/5] Creating directories..."
mkdir -p data/raw data/processed models src/api app notebooks scripts

echo "[2/5] Ensuring Python package layout..."
touch src/__init__.py src/api/__init__.py
touch data/.gitkeep data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep notebooks/.gitkeep

echo "[3/5] Creating virtual environment (.venv)..."
python3 -m venv .venv

echo "[4/5] Activating venv and upgrading pip..."
source .venv/bin/activate
python -m pip install --upgrade pip

echo "[5/5] Installing dependencies from requirements.txt..."
# Note: For PyTorch with specific CUDA builds, you may prefer:
#   pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# before installing the rest of the requirements.
pip install -r requirements.txt

echo "\nDone. Activate your environment with:"
echo "  source .venv/bin/activate   # Unix/macOS"
echo "  .venv\\\\Scripts\\\\Activate.ps1   # Windows PowerShell"
