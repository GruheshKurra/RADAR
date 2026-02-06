#!/bin/bash

# Download RADAR Combined Dataset from Kaggle
# Dataset: gruheshkurra/radar-deepfake-frames
# 227k frames from Celeb-DF v2 + FaceForensics++ C23

set -e

echo "======================================"
echo "RADAR Combined Dataset Downloader"
echo "======================================"
echo ""

# Check if Kaggle credentials are set
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "ERROR: Kaggle credentials not set!"
    echo ""
    echo "Please set your Kaggle credentials:"
    echo "  export KAGGLE_USERNAME=<your_username>"
    echo "  export KAGGLE_KEY=<your_api_key>"
    echo ""
    echo "Get your API key from: https://www.kaggle.com/settings"
    exit 1
fi

OUTPUT_DIR="${1:-./data}"
DOWNLOAD_FILE="radar-deepfake-frames.zip"

echo "Download directory: $OUTPUT_DIR"
echo "Kaggle username: $KAGGLE_USERNAME"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Download dataset
echo "Downloading dataset..."
curl -L -u "$KAGGLE_USERNAME:$KAGGLE_KEY" \
  -o "$DOWNLOAD_FILE" \
  https://www.kaggle.com/api/v1/datasets/download/gruheshkurra/radar-deepfake-frames

if [ ! -f "$DOWNLOAD_FILE" ]; then
    echo "ERROR: Download failed!"
    exit 1
fi

echo "Download complete: $(du -h $DOWNLOAD_FILE | cut -f1)"
echo ""

# Extract dataset
echo "Extracting dataset..."
unzip -q "$DOWNLOAD_FILE"

if [ -d "combined" ]; then
    echo "✓ Dataset extracted successfully"
    echo ""

    # Show statistics
    echo "Dataset Statistics:"
    echo "-------------------"
    for split in train val test; do
        if [ -d "combined/$split/real" ] && [ -d "combined/$split/fake" ]; then
            real_count=$(find "combined/$split/real" -type f | wc -l)
            fake_count=$(find "combined/$split/fake" -type f | wc -l)
            total=$((real_count + fake_count))
            echo "$split: $real_count real, $fake_count fake (total: $total)"
        fi
    done

    # Clean up
    rm "$DOWNLOAD_FILE"
    echo ""
    echo "✓ Download complete!"
    echo "Dataset location: $PWD/combined"
else
    echo "ERROR: Extraction failed or unexpected structure"
    exit 1
fi
