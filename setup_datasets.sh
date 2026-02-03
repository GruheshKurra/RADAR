#!/bin/bash
# Quick setup script for RADAR datasets

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "======================================"
echo "RADAR Dataset Setup"
echo "======================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Install dependencies
echo ""
echo "[1/4] Installing dependencies..."
pip install -q datasets huggingface_hub Pillow tqdm opencv-python

# Download WildDeepfake
echo ""
echo "[2/4] Downloading WildDeepfake..."
python3 src/data/download_datasets.py --datasets wilddeepfake --output_dir ./data

# Check if FaceForensics++ is already downloaded
echo ""
echo "[3/4] Checking for FaceForensics++..."

FF_ZIP="$HOME/Downloads/ff-c23.zip"
FF_EXTRACTED="$HOME/Downloads/ff-c23"

if [ -f "$FF_ZIP" ] || [ -d "$FF_EXTRACTED" ]; then
    echo "Found FaceForensics++ files!"

    if [ -f "$FF_ZIP" ] && [ ! -d "$FF_EXTRACTED" ]; then
        echo "Extracting $FF_ZIP..."
        unzip -q "$FF_ZIP" -d "$FF_EXTRACTED"
    fi

    if [ -d "$FF_EXTRACTED" ]; then
        echo "Extracting frames from videos (this may take 30-60 minutes)..."
        python3 src/data/extract_frames.py \
            --ff_root "$FF_EXTRACTED" \
            --output_root ./data \
            --num_frames 10 \
            --num_workers 8
    fi
else
    echo "⚠️  FaceForensics++ not found in ~/Downloads/"
    echo ""
    echo "To download FaceForensics++ (optional but recommended):"
    echo "  1. Install Kaggle CLI: pip install kaggle"
    echo "  2. Setup credentials: https://www.kaggle.com/docs/api"
    echo "  3. Download:"
    echo "     curl -L -o ~/Downloads/ff-c23.zip \\"
    echo "       https://www.kaggle.com/api/v1/datasets/download/xdxd003/ff-c23"
    echo "  4. Run this script again"
    echo ""
    echo "Continuing with WildDeepfake only..."
fi

# Check final status
echo ""
echo "[4/4] Final dataset status:"
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
echo ""
echo "See SETUP_DATASETS.md for more options."
