# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RADAR (Reasoning and Artifact Detection for Deepfake Recognition) is a deep learning model for deepfake detection. It uses dual artifact detection modules (boundary and frequency) with iterative evidence refinement through a reasoning module.

## Key Architecture

### Three-Component Design
1. **BADM (Boundary Artifact Detection Module)**: Detects edge inconsistencies via Sobel filtering
2. **AADM (Frequency Domain Artifact Module)**: Analyzes FFT spectrum with high-pass filtering
3. **ERM (Evidence Refinement Module)**: Iteratively refines evidence over 3 iterations

### Dual Pathway Fusion
- **Reasoning Path**: ERM processes evidence through iterative refinement
- **External Path**: Direct classification from concatenated evidence
- **Gating Mechanism**: Learnable parameter blends both paths (α=0.5)

## Repository Structure

```
├── 0_download_combined_dataset.sh  # Download dataset from Kaggle
├── 2_train_model.py                # Main training script
├── 3_train_ablation.py             # Ablation study runner
├── 4_compare_ablations.py          # Results comparison
├── 5_visualize_attention.py        # Attention visualization
├── train_a40.py                    # A40 GPU optimized
├── run_ablations.sh                # Batch ablations
├── configs/
│   ├── a40_fast.yaml              # Fast config
│   └── a40_optimal.yaml           # Production config
└── src/
    ├── method/                    # RADAR implementation
    ├── baselines/                 # Comparison models
    ├── data/                      # Data loading
    ├── experiments/               # Training loops
    ├── analysis/                  # Statistics
    └── utils/                     # Utilities
```

## Quick Start

### 1. Setup Environment
```bash
conda activate ml  # or your Python 3.11 environment
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
./0_download_combined_dataset.sh
```

### 3. Train Model
```bash
# A40 optimal (recommended)
python train_a40.py --data_dir ./data/combined

# Or direct training
python 2_train_model.py \
  --data_dir ./data/combined \
  --experiment_name radar_training \
  --num_epochs 30 \
  --batch_size 128
```

### 4. Run Ablations
```bash
./run_ablations.sh
```

### 5. Compare Results
```bash
python 4_compare_ablations.py --output_dir ./outputs
```

## Training Configuration

### A40 GPU (48GB VRAM) - Optimal
- Batch size: 128
- Epochs: 30
- Learning rate: 0.001
- Workers: 8
- Gradient accumulation: 1

### A40 GPU - Fast Mode
- Batch size: 256
- Epochs: 10
- 50% of data
- Learning rate: 0.002

## Model Hyperparameters

```python
img_size = 224
patch_size = 16
embed_dim = 384           # ViT-Small
evidence_dim = 64
reasoning_iterations = 3
reasoning_heads = 4
fft_size = 112
dropout = 0.1
```

## Loss Configuration

```python
lambda_main = 1.0
lambda_branch = 0.3
lambda_orthogonal = 0.1
lambda_deep_supervision = 0.05
label_smoothing = 0.1
```

## Expected Results

### Combined Dataset (Celeb-DF + FF++)
- Full RADAR: 90-95% AUC
- BADM only: 88-92% AUC
- AADM only: 86-90% AUC
- No reasoning: 91-94% AUC

### Training Time (A40)
- ~2 hours for 30 epochs
- ~15-20 minutes per epoch

## Ablation Studies

Run specific ablations:

```bash
# BADM only
python 3_train_ablation.py \
  --data_dir ./data/combined \
  --experiment_name ablation_badm_only \
  --use_badm true \
  --use_aadm false

# AADM only
python 3_train_ablation.py \
  --data_dir ./data/combined \
  --experiment_name ablation_aadm_only \
  --use_badm false \
  --use_aadm true

# No reasoning (1 iteration)
python 3_train_ablation.py \
  --data_dir ./data/combined \
  --experiment_name ablation_no_reasoning \
  --reasoning_iterations 1
```

## Output Structure

```
outputs/<experiment_name>/
├── best.pth           # Model checkpoint
├── config.yaml        # Training config
└── metrics.json       # Training history
```

## Key Implementation Details

### Evidence Fusion
When both BADM and AADM are active:
```python
evidence_list = [badm_evidence, aadm_evidence]  # Each 64-dim
reasoning_out = ERM(evidence_list)
external_input = torch.cat(evidence_list, dim=1)  # 128-dim
final = alpha * reasoning + (1-alpha) * external
```

### Attention Convergence
- Measures stability: delta = |prob[iter=3] - prob[iter=2]|
- Typical delta: 0.001-0.005
- Lower = more stable

## Common Issues

### Out of Memory
- Reduce batch_size: 128 → 64
- Increase gradient_accumulation_steps
- Reduce num_workers

### Slow Training
- Increase num_workers (up to CPU count - 1)
- Enable persistent_workers=True
- Use SSD for data directory

### Import Errors
- PyTorch 2.1+: Use `torch.cuda.amp` not `torch.amp`
- Ensure timm>=0.9.0 installed

## Dataset Information

### Combined Dataset
- **Total**: 227,504 frames
- **Train**: 192,048 (98,738 real + 93,310 fake)
- **Val**: 23,008 (10,998 real + 12,010 fake)
- **Test**: 12,448 (6,210 real + 6,238 fake)
- **Sources**: Celeb-DF v2 + FaceForensics++ C23
- **Format**: PNG, 224x224 or original resolution

### Data Splits
- Train: 70%
- Val: 15%
- Test: 15%
- Seed: 42 (reproducible)

## Reproducibility

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Note: `benchmark=False` reduces performance ~10% but ensures determinism.

## Adding New Baselines

1. Create file in `src/baselines/`
2. Implement `forward(x, freq_cached, sobel_cached) -> Dict`
3. Return `{"logit": ..., "prob": ...}`
4. Add to `src/baselines/__init__.py`

## File Naming Convention

- `0_*.py`: Data download/setup
- `2_*.py`: Training scripts
- `3_*.py`: Ablation experiments
- `4_*.py`: Analysis/comparison
- `5_*.py`: Visualization

Numbers indicate typical execution order.
