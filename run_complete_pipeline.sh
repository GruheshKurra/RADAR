#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

DATA_DIR="./data"
OUTPUT_DIR="./outputs"
EXPORT_DIR="./exports"
EXPERIMENT_NAME="radar_wilddeepfake_$(date +%Y%m%d_%H%M%S)"

echo "========================================================================"
echo "RADAR COMPLETE TRAINING PIPELINE (MEMORY-EFFICIENT)"
echo "========================================================================"
echo "This script will:"
echo "  1. Download 300k images (memory-safe limit)"
echo "  2. Train RADAR model (6-10 hours on A40)"
echo "  3. Export and package results for download"
echo ""
echo "NOTE: Limited to 300k images to prevent memory overflow and disconnection"
echo ""
echo "Experiment: $EXPERIMENT_NAME"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Export directory: $EXPORT_DIR"
echo "========================================================================"
echo ""

read -p "Press ENTER to start or Ctrl+C to cancel..." dummy

echo ""
echo "========================================================================"
echo "STEP 1/3: DATASET PREPARATION (300K IMAGES)"
echo "========================================================================"
python3 1_prepare_dataset.py --output_dir "$DATA_DIR" --max_images 300000

if [ $? -ne 0 ]; then
    echo "✗ Dataset preparation failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "STEP 2/3: MODEL TRAINING"
echo "========================================================================"
python3 2_train_model.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --batch_size 128 \
    --num_epochs 30 \
    --learning_rate 0.0005

if [ $? -ne 0 ]; then
    echo "✗ Training failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "STEP 3/3: RESULTS EXPORT"
echo "========================================================================"
python3 3_export_results.py \
    --results_dir "$OUTPUT_DIR/$EXPERIMENT_NAME" \
    --output_dir "$EXPORT_DIR" \
    --create_download_script

if [ $? -ne 0 ]; then
    echo "✗ Export failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY"
echo "========================================================================"
echo "Results package: $EXPORT_DIR/${EXPERIMENT_NAME}_*.zip"
echo ""
echo "Next steps:"
echo "  1. Download the zip file to your local machine"
echo "  2. Extract and analyze the results"
echo "  3. Load the trained model for inference"
echo "========================================================================"
echo ""
