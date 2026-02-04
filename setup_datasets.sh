#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

export HF_HOME="$SCRIPT_DIR/data/hf_cache"
export HF_DATASETS_CACHE="$SCRIPT_DIR/data/hf_cache"

echo "======================================"
echo "RADAR Dataset Setup"
echo "======================================"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

echo ""
echo "[1/3] Installing dependencies..."
pip install -q datasets huggingface_hub Pillow tqdm opencv-python

echo ""
echo "[2/3] Downloading datasets..."
python3 src/data/download_datasets.py --datasets all --output_dir ./data --workspace_root .

echo ""
echo "[3/3] Final dataset status:"
python3 src/data/download_datasets.py --datasets check --output_dir ./data

echo ""
echo "======================================"
echo "✓ Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Train on WildDeepfake:"
echo "     cd src/experiments"
echo "     python run.py --config configs/wilddeepfake.yaml --output ../../outputs"
echo ""
echo "  2. Cross-domain evaluation (if FF++ available):"
echo "     python run.py --config configs/cross_domain.yaml --output ../../outputs"
