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
echo "Installing dependencies..."

pip install -q --no-cache-dir datasets huggingface_hub Pillow tqdm opencv-python \
    torch torchvision numpy timm torch-dct albumentations scikit-learn scipy \
    matplotlib seaborn pyyaml 2>&1 | grep -v "already satisfied" || true

echo ""
echo "Starting parallel dataset downloads..."

download_wilddeepfake() {
    echo "[Dataset 1/2] Downloading WildDeepfake..."
    python3 src/data/download_datasets.py --datasets wilddeepfake --output_dir "$PROJECT_DIR/data" --workspace_root "$PROJECT_DIR"
}

download_faceforensics() {
    echo "[Dataset 2/2] Downloading and processing FaceForensics++..."

    FF_ZIP="$PROJECT_DIR/ff-c23.zip"
    FF_DIR="$PROJECT_DIR/ff-c23"

    if [ ! -f "$FF_ZIP" ] && [ ! -d "$FF_DIR" ]; then
        echo "Downloading FF-c23 from Kaggle (7000 videos, ~10GB)..."
        curl -L -o "$FF_ZIP" "https://www.kaggle.com/api/v1/datasets/download/xdxd003/ff-c23" 2>/dev/null || {
            echo "Failed to download FF-c23, skipping..."
            return 1
        }
        echo "Download complete"
    fi

    if [ -f "$FF_ZIP" ]; then
        echo "Extracting FF-c23 archive..."
        mkdir -p "$FF_DIR"
        unzip -q "$FF_ZIP" -d "$FF_DIR" || {
            echo "Extraction failed, skipping FF-c23..."
            rm -rf "$FF_DIR" "$FF_ZIP"
            return 1
        }

        FF_ACTUAL_DIR="$FF_DIR"
        for possible_dir in "$FF_DIR" "$FF_DIR/ff-c23" "$FF_DIR/c23" "$FF_DIR"/*; do
            if [ -d "$possible_dir/original" ] && [ -d "$possible_dir/Deepfakes" ]; then
                FF_ACTUAL_DIR="$possible_dir"
                echo "Found FF dataset root: $FF_ACTUAL_DIR"
                break
            fi
        done

        NUM_WORKERS=$(nproc)
        echo "Extracting frames using $NUM_WORKERS workers..."
        echo "Expected: 1000 real + 6000 fake = 7000 videos total"
        python3 src/data/extract_frames.py \
            --ff_root "$FF_ACTUAL_DIR" \
            --output_root "$PROJECT_DIR/data" \
            --num_frames 10 \
            --num_workers "$NUM_WORKERS"

        echo "Cleaning up video files to save disk space..."
        rm -rf "$FF_DIR" "$FF_ZIP"
        echo "Saved ~10GB by removing video files"
    fi
}

download_wilddeepfake &
PID1=$!

download_faceforensics &
PID2=$!

wait $PID1
wait $PID2

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
