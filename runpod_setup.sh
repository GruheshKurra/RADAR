#!/bin/bash

set -e

WORKSPACE_ROOT="/workspace"
PROJECT_DIR="$WORKSPACE_ROOT"

if [ ! -f "$PROJECT_DIR/src/data/download_datasets.py" ]; then
    echo "Error: Project files not found in /workspace"
    exit 1
fi

cd "$PROJECT_DIR"

export HF_HOME="$PROJECT_DIR/data/hf_cache"
export HF_DATASETS_CACHE="$PROJECT_DIR/data/hf_cache"
export TORCH_HOME="$PROJECT_DIR/data/torch_cache"
export TIMM_CACHE_DIR="$PROJECT_DIR/data/torch_cache/hub"
export TMPDIR="$PROJECT_DIR/tmp"

mkdir -p "$TMPDIR"
mkdir -p "$PROJECT_DIR/data"

echo "========================================"
echo "AUTOMATED RUNPOD SETUP"
echo "========================================"
echo "Installing system dependencies..."

apt-get update -qq && apt-get install -y -qq unzip > /dev/null 2>&1 || true

echo "Installing Python dependencies..."

pip install -q --no-cache-dir datasets huggingface_hub Pillow tqdm opencv-python \
    torch torchvision numpy timm torch-dct albumentations scikit-learn scipy \
    matplotlib seaborn pyyaml 2>&1 | grep -v "already satisfied" || true

echo ""
echo "Downloading WildDeepfake dataset (LIMITED TO 300K IMAGES)..."
echo "This prevents memory overflow and pod disconnection"

python3 src/data/download_datasets.py --datasets wilddeepfake --output_dir "$PROJECT_DIR/data" --workspace_root "$PROJECT_DIR" --max_images 300000

echo ""
echo "Cleaning up cache and temporary files..."
rm -rf "$TMPDIR"
rm -rf "$HF_HOME/downloads"
rm -rf "$PROJECT_DIR/data/hf_cache/downloads"

echo ""
python3 src/data/download_datasets.py --datasets check --output_dir "$PROJECT_DIR/data"

echo ""
echo "========================================"
echo "SETUP COMPLETE!"
echo "========================================"
echo "Dataset location: $PROJECT_DIR/data/"
echo "Start training:"
echo "  cd $PROJECT_DIR/src/experiments"
echo "  python run.py --config configs/wilddeepfake.yaml --output $PROJECT_DIR/outputs"
echo ""
echo "Monitor GPU: nvidia-smi"
