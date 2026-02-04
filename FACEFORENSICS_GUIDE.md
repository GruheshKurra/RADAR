# FaceForensics++ Pipeline Guide

Complete guide for using FaceForensics++ with RADAR.

## 🚀 Quick Start

### Step 1: Download Dataset (Sample Mode)
```bash
# Test with 5 videos per folder (~10 minutes)
python 0_download_faceforensics.py \
  --output_dir ./data/faceforensics \
  --sample_only \
  --skip_test
```

### Step 2: Extract Frames
```bash
# Extract frames from videos
python src/data/prepare_faceforensics.py \
  --video_dir ./data/faceforensics \
  --output_dir ./data/faceforensics_frames \
  --max_frames 10 \
  --fps 1
```

### Step 3: Train Model
```bash
# Quick training test
python 2_train_model_fast.py \
  --data_dir ./data/faceforensics_frames \
  --output_dir ./outputs \
  --experiment_name radar_ff_test \
  --batch_size 32 \
  --num_epochs 5
```

---

## 📊 Full Dataset Download

### Download All Splits (~130GB, 2-4 hours)
```bash
python 0_download_faceforensics.py \
  --output_dir ./data/faceforensics
```

### Extract All Frames
```bash
python src/data/prepare_faceforensics.py \
  --video_dir ./data/faceforensics \
  --output_dir ./data/faceforensics_frames \
  --max_frames 10 \
  --fps 1
```

This creates:
```
data/faceforensics_frames/
├── train/
│   ├── real/    (~7,200 videos × 10 frames = 72k images)
│   └── fake/    (~28,800 videos × 10 frames = 288k images)
├── val/
│   ├── real/    (~1,400 videos × 10 frames = 14k images)
│   └── fake/    (~5,600 videos × 10 frames = 56k images)
└── metadata.json
```

---

## 🎯 Training on Full Dataset

### Standard Training
```bash
python 2_train_model.py \
  --data_dir ./data/faceforensics_frames \
  --output_dir ./outputs \
  --experiment_name radar_faceforensics \
  --batch_size 64 \
  --num_epochs 30 \
  --learning_rate 0.0005
```

### Fast Mode (Subset)
```bash
python 2_train_model_fast.py \
  --data_dir ./data/faceforensics_frames \
  --output_dir ./outputs \
  --experiment_name radar_ff_fast \
  --batch_size 64 \
  --num_epochs 10
```

---

## 📈 Expected Results

**Per-Manipulation Performance:**
| Manipulation | Expected AUC |
|--------------|--------------|
| Deepfakes | 0.95+ |
| Face2Face | 0.92+ |
| FaceSwap | 0.94+ |
| NeuralTextures | 0.90+ |
| **Average** | **0.93+** |

**Training Time:**
- Sample mode: ~30 minutes
- Full dataset: ~6-8 hours (30 epochs)

---

## 🔧 Configuration Options

### Frame Extraction
```bash
--max_frames 10    # Frames per video (default: 10)
--fps 1            # Sampling rate (default: 1 fps)
```

### Training
```bash
--batch_size 64        # Adjust based on GPU memory
--num_epochs 30        # Standard for FF++
--learning_rate 0.0005 # Default learning rate
```

---

## 📝 Dataset Structure

### After Download (Videos)
```
data/faceforensics/FaceForensics_compressed/
├── train/
│   ├── original/*.mp4
│   └── altered/
│       ├── Deepfakes/*.mp4
│       ├── Face2Face/*.mp4
│       ├── FaceSwap/*.mp4
│       └── NeuralTextures/*.mp4
└── val/
    ├── original/*.mp4
    └── altered/...
```

### After Frame Extraction
```
data/faceforensics_frames/
├── train/
│   ├── real/*.jpg
│   └── fake/*.jpg  (all manipulations combined)
├── val/
│   ├── real/*.jpg
│   └── fake/*.jpg
└── metadata.json
```

---

## 🐛 Troubleshooting

### Download Issues
```bash
# Check if download script exists
ls data/faceforensics/download_ff.py

# Re-download if interrupted
python 0_download_faceforensics.py --output_dir ./data/faceforensics
```

### Frame Extraction Issues
```bash
# Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# Install if missing
pip install opencv-python
```

### Training Issues
```bash
# Check dataset structure
ls -lh data/faceforensics_frames/train/real/ | head
ls -lh data/faceforensics_frames/train/fake/ | head

# Verify metadata
cat data/faceforensics_frames/metadata.json
```

---

## 📊 For Research Papers

Use this configuration for publication:
1. Train on FF++ train split
2. Validate on FF++ val split
3. Test on FF++ test split
4. Report per-manipulation results
5. Add cross-dataset evaluation (Celeb-DF)

### Reporting Template
```
Dataset: FaceForensics++ (c23 compression)
Frames: 10 per video @ 1 fps
Training: 30 epochs, batch_size=64, lr=5e-4
Architecture: RADAR (BADM + AADM + ERM)
```

---

## 🚀 Next Steps

After training on FF++:
1. Evaluate on Celeb-DF (cross-dataset)
2. Run ablation studies (BADM only, AADM only, Full)
3. Visualize attention weights
4. Export results for paper

See `3_export_results.py` for packaging trained models.
