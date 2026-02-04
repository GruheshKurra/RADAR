#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

DATA_DIR="./data"
OUTPUT_DIR="./outputs"
EXPORT_DIR="./exports"
EXPERIMENT_NAME="radar_wilddeepfake_fast_$(date +%Y%m%d_%H%M%S)"

echo "========================================================================"
echo "RADAR FAST TRAINING PIPELINE (MEMORY-EFFICIENT)"
echo "========================================================================"
echo "This script will:"
echo "  1. Download 230k images (memory-safe subset) [10-20 min]"
echo "  2. Train on 20% subset (~230k images) [1.5-2.5 hours]"
echo "  3. Export results [1-2 min]"
echo ""
echo "Total estimated time: 2-3 hours"
echo ""
echo "Optimizations:"
echo "  • Dataset: 230k images (memory-safe limit)"
echo "  • Batch size: 256 (max A40 utilization)"
echo "  • Workers: 8 (all CPUs)"
echo "  • Epochs: 20 (sufficient for subset)"
echo "  • torch.compile enabled"
echo "  • Channels-last memory format"
echo "  • Streaming dataset processing (prevents memory overflow)"
echo ""
echo "Experiment: $EXPERIMENT_NAME"
echo "========================================================================"
echo ""

read -p "Press ENTER to start or Ctrl+C to cancel..." dummy

echo ""
echo "========================================================================"
echo "STEP 1/3: DATASET PREPARATION (230K IMAGES)"
echo "========================================================================"
python3 1_prepare_dataset.py --output_dir "$DATA_DIR" --max_images 230000

if [ $? -ne 0 ]; then
    echo "✗ Dataset preparation failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "STEP 2/3: FAST MODEL TRAINING (20% subset)"
echo "========================================================================"
python3 2_train_model_fast.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$EXPERIMENT_NAME" \
    --batch_size 256 \
    --num_epochs 20 \
    --learning_rate 0.001 \
    --subset_ratio 0.2

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
echo "✓ FAST PIPELINE COMPLETE!"
echo "========================================================================"
echo "Total time: Much faster than full dataset training!"
echo "Results package: $EXPORT_DIR/${EXPERIMENT_NAME}_*.zip"
echo ""
echo "Note: Trained on 20% subset (~230k images)"
echo "For production use, train on full dataset with: bash run_complete_pipeline.sh"
echo "========================================================================"
echo ""
