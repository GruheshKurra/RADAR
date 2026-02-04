# RADAR: Reasoning-Augmented Deepfake Artifact Recognition

> **⚠️ DOCUMENTATION POLICY**: This project maintains a SINGLE README.md file. All documentation must be consolidated here.

> **🚨 MEMORY FIX (2026-02-04)**: Fixed RunPod memory overflow bug! Stream processing now prevents pod crashes. Default: 300k images. See [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A novel deepfake detection framework combining boundary and frequency artifact detection with iterative evidence refinement.

---

## 📑 Table of Contents

1. [Quick Start](#-quick-start)
2. [Complete Pipeline](#-complete-pipeline)
3. [Individual Scripts](#-individual-scripts)
4. [RunPod Setup](#-runpod-setup)
5. [Training Time Estimates](#-training-time-estimates)
6. [Results & Output](#-results--output)

---

## 🚀 Quick Start

### Fast Mode (RECOMMENDED - 2-3 hours)

```bash
pip install torch torchvision timm albumentations opencv-python tqdm datasets scikit-learn matplotlib seaborn pyyaml

bash run_fast_pipeline.sh
```

**Fast mode optimizations:**
- Uses 20% of dataset (~230k images)
- Batch size: 256 (maximum A40 utilization)
- 8 data workers (all CPUs)
- 20 epochs
- torch.compile + channels-last memory
- **Total time: 2-3 hours**

### Full Dataset (13-19 hours)

```bash
bash run_complete_pipeline.sh
```

**Full mode features:**
- Uses 100% of dataset (~1.16M images)
- Batch size: 128
- 30 epochs
- **Total time: 13-19 hours**

---

## 🔄 Complete Pipeline

The complete training pipeline is automated in one script:

```bash
bash run_complete_pipeline.sh
```

### What It Does

**Step 1: Dataset Preparation**
- Downloads WildDeepfake from HuggingFace
- Organizes into real/fake folders
- Shows progress bars for all operations
- Validates data integrity

**Step 2: Model Training**
- Loads ~1.16M images (994k train + 165k test)
- Trains RADAR model with:
  - Batch size: 64 (effective 128 with gradient accumulation)
  - 30 epochs with early stopping
  - Learning rate: 0.0005 with OneCycleLR
  - Mixed precision training
- Saves best checkpoint based on validation AUC
- Evaluates on test set

**Step 3: Results Export**
- Packages all results into timestamped zip file
- Includes:
  - Trained model checkpoint (best.pth)
  - Training configuration (config.json)
  - Metrics and history (metrics.json)
  - README with results summary
- Creates download helper script
- Ready for transfer to local machine

---

## 🛠️ Individual Scripts

Run each step separately for more control:

### 1. Prepare Dataset

```bash
python 1_prepare_dataset.py --output_dir ./data
```

**Options:**
- `--output_dir`: Where to save dataset (default: ./data)
- `--check_only`: Only check if dataset exists

**Output:**
```
data/
└── wilddeepfake/
    ├── real/ (~400k images)
    └── fake/ (~760k images)
```

### 2. Train Model

```bash
python 2_train_model.py \
    --data_dir ./data \
    --output_dir ./outputs \
    --experiment_name my_experiment \
    --batch_size 64 \
    --num_epochs 30 \
    --learning_rate 0.0005
```

**Options:**
- `--data_dir`: Dataset location
- `--output_dir`: Where to save outputs
- `--experiment_name`: Name for this run
- `--batch_size`: Batch size (default: 64)
- `--num_epochs`: Training epochs (default: 30)
- `--learning_rate`: Learning rate (default: 0.0005)
- `--seed`: Random seed (default: 42)

**Output:**
```
outputs/
└── my_experiment/
    ├── best.pth          (trained model)
    ├── config.json       (training config)
    └── metrics.json      (results)
```

### 3. Export Results

```bash
python 3_export_results.py \
    --results_dir ./outputs/my_experiment \
    --output_dir ./exports \
    --create_download_script
```

**Options:**
- `--results_dir`: Path to results directory
- `--output_dir`: Where to save zip file (default: ./exports)
- `--create_download_script`: Generate download helper script

**Output:**
```
exports/
├── my_experiment_20260204_120000.zip
└── download_results_20260204_120000.sh
```

---

## ☁️ RunPod Setup

> **🚨 MEMORY OVERFLOW FIX**: Scripts now use stream processing with 300k image limit by default (prevents pod crashes)

### Automated Setup (Zero Manual Work)

**1. Upload project to /workspace/**
```bash
cd /workspace/RADAR-Clean
```

**2. Run complete pipeline (RECOMMENDED)**
```bash
bash run_fast_pipeline.sh
```

**What changed**:
- ✅ Stream processing (saves images as we go, not after loading all)
- ✅ Configurable image limit (`--max_images`)
- ✅ Default: 300k images (prevents memory overflow)
- ✅ No more pod crashes at 18%

### RunPod-Specific Script

For RunPod with optimized paths:

```bash
bash runpod_setup.sh
```

This handles:
- Environment variables for /workspace
- Memory-efficient stream processing
- Automatic cleanup
- Cache management
- Limited to 300k images (safe default)

### Data Locations on RunPod
```
/workspace/data/wilddeepfake/
/workspace/data/torch_cache/
/workspace/data/hf_cache/
/workspace/outputs/
/workspace/exports/
```

### Monitor Training
```bash
nvidia-smi
tail -f /workspace/outputs/*/metrics.json
```

---

## ⏱️ Training Time Estimates

### A40 RunPod (48GB VRAM, 48GB RAM, 9 vCPUs)

## Mode Comparison

| Metric | Fast Mode ⚡ | Full Mode 🎯 |
|--------|-------------|--------------|
| **Dataset** | 230k images (20%) | 1.16M images (100%) |
| **Batch Size** | 256 | 128 |
| **Workers** | 8 | 12 |
| **Epochs** | 20 | 30 |
| **Time/Epoch** | 5-8 min | 25-35 min |
| **Total Time** | **2-3 hours** | **13-19 hours** |
| **VRAM Usage** | 30-35GB | 20-25GB |
| **Best For** | Quick experiments | Production models |

### Fast Mode (RECOMMENDED for testing)

**Configuration:**
- Dataset: 20% subset (~230k images)
- Batch size: 256 (maximum A40 utilization)
- Workers: 8 (all CPUs)
- Epochs: 20
- torch.compile enabled
- Channels-last memory format

**Timing:**
- Dataset download: 15-30 min
- Training: 1.5-2.5 hours (~5-8 min/epoch)
- Export: 1-2 min
- **Total: 2-3 hours**

**Throughput:** ~800-1000 images/second

**Use when:**
- Testing model architecture
- Hyperparameter tuning
- Quick validation
- Limited time/budget

### Full Mode (for production)

**Configuration:**
- Dataset: 100% (~1.16M images)
- Batch size: 128
- Workers: 12
- Epochs: 30

**Timing:**
- Dataset download: 15-30 min
- Training: 12-18 hours (~25-35 min/epoch)
- Export: 1-2 min
- **Total: 13-19 hours**

**Throughput:** ~500-700 images/second

**Use when:**
- Training final production model
- Need maximum accuracy
- Publishing results

### Quick Experimentation

To test faster with subset:

```bash
python 2_train_model.py \
    --batch_size 64 \
    --num_epochs 5 \
    --data_dir ./data
```

Modify train_ratio in code for smaller dataset (e.g., 0.05 = 5% of data).

---

## 📊 Results & Output

### After Training Completes

**1. Local Results Directory**
```
outputs/radar_wilddeepfake_TIMESTAMP/
├── best.pth          (trained model checkpoint)
├── config.json       (all hyperparameters)
└── metrics.json      (training history + test results)
```

### Metrics JSON Structure
```json
{
  "history": {
    "train_loss": [...],
    "val_auc": [...],
    "val_acc": [...]
  },
  "best_val_auc": 0.9234,
  "test_metrics": {
    "auc": 0.9187,
    "accuracy": 0.8756
  },
  "timestamp": "2026-02-04T12:00:00"
}
```

### Loading Trained Model

```python
import torch
from src.method import RADAR, RADARConfig

config = RADARConfig(
    img_size=224,
    embed_dim=384,
    evidence_dim=64,
    reasoning_iterations=3
)

model = RADAR(config)
checkpoint = torch.load('best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 📦 Dataset Information

**WildDeepfake Dataset**
- Source: HuggingFace (xingjunm/WildDeepfake)
- Total: ~1.16M images
- Train: ~994k images
- Test: ~165k images
- Size: ~10GB
- License: Apache 2.0

**Split Distribution:**
- Real images: ~400k (35%)
- Fake images: ~760k (65%)

**Data Augmentation (Training):**
- Random horizontal flip
- Shift/scale/rotate
- Gaussian blur or JPEG compression
- Color jitter
- ImageNet normalization

---

## 🏗️ Model Architecture

**RADAR Components:**

1. **ViT Encoder** (vit_small_patch16_224)
   - Pretrained on ImageNet
   - 384-dim embeddings
   - 12 transformer layers

2. **BADM** (Boundary Artifact Detection)
   - Sobel edge detection
   - CNN encoder for edge features
   - 64-dim evidence vectors

3. **AADM** (Frequency Artifact Detection)
   - FFT with high-pass filter
   - CNN encoder for frequency features
   - 64-dim evidence vectors

4. **ERM** (Evidence Refinement Module)
   - 3 iterative reasoning steps
   - 4-head cross-attention
   - GRU state evolution

5. **Gated Fusion**
   - Learnable alpha parameter
   - Combines reasoning + external classifier

---

## 🎯 Training Configuration

**Optimizer:** AdamW
- Learning rate: 0.0005
- Weight decay: 0.05
- Gradient clipping: 1.0

**Scheduler:** OneCycleLR
- Warmup ratio: 0.1
- Annealing: cosine

**Loss Components:**
- Main classification: BCE (λ=1.0)
- Branch supervision: BCE (λ=0.3)
- Orthogonality: Hinge (λ=0.1)
- Deep supervision: Weighted BCE (λ=0.05)
- Label smoothing: 0.1

**Regularization:**
- Dropout: 0.1
- Gradient accumulation: 2 steps
- Early stopping: patience=10
- Mixed precision: enabled

---

## 📥 Downloading Results from RunPod

### Method 1: Direct Copy (If Using RunPod Desktop)
```bash
cp /workspace/exports/*.zip ~/Downloads/
```

### Method 2: SCP from Local Machine
```bash
scp runpod_user@pod_ip:/workspace/exports/*.zip ~/Downloads/
```

### Method 3: RunPod Web Interface
1. Navigate to /workspace/exports/ in file browser
2. Right-click on zip file
3. Select "Download"

### Method 4: Use Generated Script
```bash
bash /workspace/exports/download_results_*.sh
```

---

## 🔧 Dependencies

### Required Packages
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
albumentations>=1.3.0
opencv-python>=4.7.0
tqdm>=4.65.0
datasets>=2.14.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
Pillow>=10.0.0
numpy>=1.24.0
```

### Install All
```bash
pip install torch torchvision timm albumentations opencv-python tqdm datasets scikit-learn matplotlib seaborn pyyaml Pillow numpy
```

---

## 📝 Citation

```bibtex
@inproceedings{radar2024,
  title={RADAR: Reasoning-Augmented Deepfake Artifact Recognition},
  author={Your Name},
  booktitle={Conference},
  year={2024}
}
```

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🤝 Contributing

This is a research project. For questions or issues, please open a GitHub issue.

---

**Last Updated:** 2026-02-04
