# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RADAR (Reasoning and Artifact Detection for Deepfake Recognition) is a deep learning research implementation for deepfake detection. It uses dual artifact detection modules (boundary and frequency) with iterative evidence refinement through a reasoning module. The architecture achieves 99.88%+ validation AUC on the Kaggle 140k dataset.

## Key Architecture Insights

### Three-Component Design
1. **BADM (Boundary Artifact Detection Module)**: Detects edge inconsistencies via Sobel filtering + CNN encoder
2. **AADM (Artifact in Frequency Domain Module)**: Analyzes FFT spectrum with high-pass filtering
3. **ERM (Evidence Refinement Module)**: Cross-attention + GRU iteratively refines evidence over 3 iterations

### Dual Pathway Fusion
- **Reasoning Path**: ERM processes evidence vectors through iterative refinement
- **External Path**: Direct classification from concatenated evidence
- **Gating Mechanism**: Learnable `_gating_logit` parameter blends both paths (default α=0.5)

### Loss Function Components
- Main classification loss (λ=1.0)
- Branch supervision for BADM/AADM (λ=0.3)
- Orthogonality constraint on evidence vectors (λ=0.1, margin=0.1)
- Deep supervision across reasoning iterations (λ=0.05)

## Repository Structure

```
├── 0_download_kaggle_140k.py      # Kaggle API download script
├── 1_prepare_wilddeepfake.py      # [DEPRECATED] WildDeepfake loader
├── 2_train_model.py               # Main training script (production)
├── 2_train_model_fast.py          # Fast training for testing
├── 3_train_ablation.py            # Ablation study runner
├── 4_compare_ablations.py         # Results comparison + LaTeX tables
├── 5_visualize_attention.py       # Attention mechanism visualization
├── train_a40.py                   # A40 GPU optimized training
├── run_ablations.sh               # Batch ablation experiments
├── configs/
│   ├── a40_fast.yaml             # Fast config (50% data, 10 epochs)
│   └── a40_optimal.yaml          # Production config (full data)
├── src/
│   ├── method/                   # Core RADAR implementation
│   │   ├── radar.py              # Main RADAR model
│   │   ├── boundary.py           # BADM module
│   │   ├── frequency.py          # AADM module
│   │   ├── reasoning.py          # ERM module
│   │   └── loss.py               # Multi-component loss
│   ├── baselines/                # Comparison baselines
│   │   ├── vit.py                # ViT-Small baseline
│   │   ├── resnet.py             # ResNet50 baseline
│   │   ├── efficientnet.py       # EfficientNet-B0 baseline
│   │   └── xception.py           # Xception baseline
│   ├── data/                     # Data loading & preprocessing
│   │   ├── dataset.py            # DeepfakeDataset + transforms
│   │   ├── splits.py             # Train/val/test splitting
│   │   ├── prepare_kaggle.py     # Kaggle data reorganizer
│   │   └── extract_frames.py     # FaceForensics++ frame extractor
│   ├── experiments/              # Training infrastructure
│   │   ├── train.py              # Core train/eval loops
│   │   └── run.py                # YAML config runner
│   ├── analysis/                 # Statistical analysis tools
│   │   ├── plots.py              # ROC curves, training plots
│   │   ├── stats.py              # Bootstrap CI, DeLong test
│   │   └── aggregate.py          # Multi-seed aggregation
│   └── utils/
│       └── logging.py            # Pretty print utilities
└── outputs/
    └── <experiment_name>/
        ├── best.pth              # Model checkpoint (258MB)
        ├── config.yaml           # Full training config
        └── metrics.json          # Training history + test results
```

## Common Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

**Critical Dependencies:**
- `torch>=2.1.0` with CUDA support
- `timm>=0.9.0` for ViT backbone
- `albumentations>=1.3.0` for augmentations
- `torch-dct>=0.1.6` for frequency operations

### Dataset Preparation

**Kaggle 140k (Primary Dataset):**
```bash
# 1. Download (~2GB, requires Kaggle API setup)
python 0_download_kaggle_140k.py --output_dir ./data/kaggle_140k

# 2. Reorganize to train/val/test structure
python src/data/prepare_kaggle.py \
  --input_dir ./data/kaggle_140k \
  --output_dir ./data/kaggle_140k_prepared
```

**FaceForensics++ (Video Dataset):**
```bash
# Extract 10 frames per video
python src/data/extract_frames.py \
  --ff_root /path/to/faceforensics \
  --output_root ./data \
  --num_frames 10 \
  --num_workers 4
```

### Training

**Production Training (Kaggle 140k):**
```bash
python 2_train_model.py \
  --data_dir ./data/kaggle_140k_prepared \
  --experiment_name radar_training \
  --num_epochs 30 \
  --batch_size 128 \
  --learning_rate 0.0005
```

**A40 GPU Optimized:**
```bash
python train_a40.py  # Uses configs/a40_optimal.yaml
```

**Fast Training (for debugging):**
```bash
python 2_train_model_fast.py \
  --data_dir ./data/kaggle_140k_prepared \
  --experiment_name radar_test
```

**YAML-Based Training:**
```bash
python src/experiments/run.py \
  --config configs/a40_optimal.yaml \
  --output ./outputs
```

**Ablation Studies:**
```bash
# Single ablation
python 3_train_ablation.py \
  --data_dir ./data/kaggle_140k_prepared \
  --experiment_name ablation_badm_only \
  --use_badm true \
  --use_aadm false \
  --num_epochs 30

# All ablations (batch script)
./run_ablations.sh
```

**Ablation Configurations:**
- `ablation_badm_only`: Only boundary detection
- `ablation_aadm_only`: Only frequency detection
- `ablation_no_reasoning`: Both detectors, no iterative refinement (1 iteration)
- `ablation_iters_N`: Test different reasoning iterations (1-5)

### Analysis & Visualization

**Compare Ablations:**
```bash
python 4_compare_ablations.py --output_dir ./outputs
```
Generates:
- Comparison tables (grid format)
- LaTeX tables for papers
- Performance delta analysis

**Attention Visualization:**
```bash
python 5_visualize_attention.py \
  --checkpoint ./outputs/radar_training/best.pth \
  --data_dir ./data/kaggle_140k_prepared \
  --output_dir ./visualizations \
  --num_samples 10
```
Produces:
- `attention_sample_*.png`: Per-sample attention evolution
- `attention_statistics.png`: Aggregate attention distributions

## Module Details

### src/method/radar.py

**RADARConfig Dataclass:**
```python
img_size: int = 224              # Input image size
patch_size: int = 16             # ViT patch size
embed_dim: int = 384             # ViT-Small embedding dim
evidence_dim: int = 64           # Evidence vector size
reasoning_iterations: int = 3    # ERM refinement steps
reasoning_heads: int = 4         # Attention heads in ERM
fft_size: int = 112              # FFT computation size
dropout: float = 0.1             # Dropout rate
gating_init: float = 0.5         # Initial gating alpha
```

**Forward Pass Logic:**
1. ViT backbone extracts `cls_token` and `patch_features`
2. BADM processes edges + patches → `badm_evidence` (64-dim)
3. AADM processes frequency + cls → `aadm_evidence` (64-dim)
4. ERM refines evidence over 3 iterations → `reasoning_logit`
5. External classifier on concatenated evidence → `external_logit`
6. Gated fusion: `final = α * reasoning + (1-α) * external`

**Output Dictionary:**
- `logit`, `prob`: Final predictions
- `reasoning_logit`, `reasoning_prob`: Reasoning-only path
- `external_logit`: External classifier output
- `gating_alpha`: Learned blending weight
- `badm_logit`, `badm_evidence`: BADM outputs
- `aadm_logit`, `aadm_evidence`: AADM outputs
- `attention_history`: (B, iters, 2) attention weights
- `iteration_logits`, `iteration_probs`: Per-iteration predictions
- `convergence_delta`: Δ between last two iterations

### src/method/boundary.py

**BoundaryArtifactDetector:**
- **Edge Detection**: Sobel filters (3x3 kernels) on grayscale
- **Edge Encoder**: 3-layer CNN (1→32→64→128) with adaptive pooling
- **Patch Processor**: Mean+Max pooling of patch features, MLP
- **Fusion**: Concatenate edge + patch features → evidence vector
- **Caching**: Accepts `sobel_cached` to skip recomputation

### src/method/frequency.py

**FrequencyArtifactDetector:**
- **Spectrum Computation**: RGB→Gray, 2D FFT, fftshift, log magnitude
- **High-Pass Filter**: Gaussian smooth cutoff at radius/8, sigmoid transition
- **Frequency Encoder**: CNN on normalized spectrum
- **CLS Processor**: MLP on ViT CLS token
- **Fusion**: Concatenate freq + CLS features → evidence vector

**Key Function: `compute_frequency_spectrum()`**
- Resizes to `fft_size` (112x112) for efficiency
- Applies high-pass to suppress low frequencies
- Normalizes to [0,1] range per sample

### src/method/reasoning.py

**EvidenceRefinementModule:**
- **Initialization**: Projects evidence (single or dual source) to hidden state
- **Cross-Attention**: Multi-head attention queries hidden state, keys/values are evidence
- **GRU Update**: Refines hidden state based on attended context
- **Iteration**: Repeats for `num_iterations` (default: 3)
- **Output**: Final logit from last iteration, full attention history

**EvidenceCrossAttention:**
- Standard multi-head attention (4 heads)
- Returns attended context + averaged attention weights

### src/method/loss.py

**RADARLoss:**
1. **Label Smoothing**: `labels * (1-ε) + 0.5 * ε` (ε=0.1)
2. **Main Loss**: BCE on final gated logit
3. **Branch Loss**: BCE on BADM/AADM logits (when enabled)
4. **Orthogonality Loss**: Penalizes `|cos(badm_ev, aadm_ev)| > margin`
5. **Deep Supervision**: Weighted BCE across reasoning iterations (linear weights)

**Loss Aggregation:**
```
total = λ_main * main + λ_branch * branch +
        λ_ortho * ortho + λ_deep * deep_supervision
```

### src/data/dataset.py

**DeepfakeDataset:**
- Supports pre-split (train/val/test) and single-domain structures
- Optional cached preprocessing (`preprocess_dir` for Sobel maps)
- Returns: `(image, label)` or `(image, label, extras)` if cache exists

**Transforms:**
- **Train**: Resize, HorizontalFlip, ShiftScaleRotate, GaussianBlur/Compression, ColorJitter, Normalize
- **Val/Test**: Resize, Normalize (no augmentation)
- Uses albumentations library (faster than torchvision)

**Cache Path Generation:**
- MD5 hash of image content
- Structure: `preprocess_dir/domain/class/hash_sobel.npy`

### src/data/splits.py

**Key Functions:**
- `is_presplit_dataset()`: Checks for train/val/test folders
- `load_presplit_data()`: Loads from pre-split structure
- `load_domain_data()`: Loads single domain (real/fake subfolders)
- `create_stratified_split()`: Stratified split by class (respects ratios)

**Validation:**
- Requires minimum 100 samples per class
- Ensures balanced splits across real/fake

### src/experiments/train.py

**train_epoch():**
- Gradient accumulation support
- Automatic mixed precision (AMP) with GradScaler
- Gradient clipping (max_norm=1.0)
- OneCycleLR scheduling per batch
- Returns average losses, skipped batches, gradient norm

**evaluate():**
- Inference mode (@torch.inference_mode)
- Computes: accuracy, AUC, reasoning_accuracy, reasoning_AUC
- Tracks: gating_alpha, convergence_delta
- Returns predictions for further analysis

**train_model():**
- Early stopping (patience=10)
- Saves best checkpoint by validation AUC
- Returns training history + best AUC

### src/baselines/

**All Baselines:**
- Uniform interface: `forward(x, freq_cached, sobel_cached) -> Dict`
- Output: `{"logit": ..., "prob": ...}`
- Pretrained on ImageNet
- Single output head (binary classification)

**Models:**
- `ViTBaseline`: vit_small_patch16_224 (same backbone as RADAR)
- `ResNetBaseline`: ResNet50 (torchvision weights)
- `EfficientNetBaseline`: tf_efficientnet_b0 (timm)
- `XceptionBaseline`: xception (timm)

### src/analysis/

**plots.py:**
- `plot_training_curves()`: Loss + AUC over epochs
- `plot_roc_curves()`: Multi-model ROC comparison
- `plot_ablation_results()`: Bar chart with values

**stats.py:**
- `bootstrap_auc_ci()`: 95% CI via 2000 bootstrap samples
- `delong_test()`: Statistical significance test between AUCs

**aggregate.py:**
- `load_experiment_results()`: Load config.yaml + metrics.json
- `aggregate_multi_seed_results()`: Mean/std across seeds
- `create_results_table()`: Pandas DataFrame for export

### src/utils/logging.py

Simple pretty-print helpers:
- `print_section()`: Header with equals border
- `print_subsection()`: Subheader with dash border
- `print_result()`: Key-value formatting
- `print_complete()`: Success message with summary

## Configuration System

### YAML Config Structure

**Model Architecture:**
```yaml
img_size: 224
patch_size: 16
embed_dim: 384
evidence_dim: 64
reasoning_iterations: 3
reasoning_heads: 4
fft_size: 112
dropout: 0.1
gating_init: 0.5
```

**Training Hyperparameters:**
```yaml
batch_size: 128
num_epochs: 30
learning_rate: 0.0005
weight_decay: 0.05
gradient_accumulation_steps: 1
warmup_ratio: 0.1
early_stopping_patience: 10
```

**Loss Configuration:**
```yaml
lambda_main: 1.0
lambda_branch: 0.3
lambda_orthogonal: 0.1
lambda_deep_supervision: 0.05
label_smoothing: 0.1
orthogonality_margin: 0.1
```

**Data Loading:**
```yaml
num_workers: 8
pin_memory: true
persistent_workers: true
prefetch_factor: 4
```

**Data Splits:**
```yaml
train_ratio: 0.8
val_ratio: 0.1
seed: 42
source_domain: "default"  # or "faceforensics", "wilddeepfake"
```

### Pre-configured Profiles

**configs/a40_optimal.yaml:**
- Target: A40 48GB VRAM
- Batch size: 128
- Workers: 8
- Full data, 30 epochs

**configs/a40_fast.yaml:**
- Batch size: 256 (aggressive)
- Epochs: 10
- `subset_ratio: 0.5` (50% of data)
- Higher learning rate (0.002)

## Outputs Directory Structure

Each experiment creates:
```
outputs/<experiment_name>/
├── best.pth           # PyTorch checkpoint (~258MB)
├── config.yaml        # Full config snapshot
└── metrics.json       # Training history + results
```

### best.pth Contents
```python
{
  "model_state_dict": ...,      # Full model weights
  "optimizer_state_dict": ...,  # Optimizer state
  "epoch": int,                 # Best epoch number
  "auc": float,                 # Best validation AUC
  "config": dict                # Training config
}
```

### metrics.json Structure
```json
{
  "history": {
    "train_loss": [epoch1, epoch2, ...],
    "val_auc": [epoch1, epoch2, ...],
    "val_acc": [epoch1, epoch2, ...],
    "val_convergence_delta": [...],
    "val_reasoning_auc": [...],
    "val_gating_alpha": [...]
  },
  "best_val_auc": 0.9996,
  "test_metrics": {
    "auc": 0.9999,
    "accuracy": 0.9964,
    "reasoning_auc": 0.9998,
    "reasoning_accuracy": 0.9960,
    "gating_alpha": 0.523,
    "convergence_delta": 0.0012
  },
  "timestamp": "2026-02-05T11:31:22.614798"
}
```

## Dataset Directory Structures

### Pre-split Format (Kaggle 140k)
```
data/kaggle_140k_prepared/
├── metadata.json
├── train/
│   ├── real/  (70,000 images)
│   └── fake/  (70,000 images)
├── val/
│   ├── real/  (14,000 images)
│   └── fake/  (14,000 images)
└── test/
    ├── real/  (7,000 images)
    └── fake/  (7,000 images)
```

### Single-domain Format
```
data/faceforensics/
├── real/  (all real images)
└── fake/  (all fake images)
```
Code automatically splits using `create_stratified_split()`.

## Training Pipeline Details

### Automatic GPU Optimization

**2_train_model.py** auto-detects GPU:
```python
if torch.cuda.is_available():
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    optimal_batch = 128 if gpu_mem_gb > 40 else 64
    optimal_workers = min(8, max(1, mp.cpu_count() - 1))
```

### Mixed Precision Training
```python
# Enabled automatically on CUDA
scaler = GradScaler(enabled=(device=="cuda"))
with autocast(device_type=device, enabled=(device=="cuda")):
    outputs = model(images, ...)
    loss = loss_fn(outputs, labels)
scaler.scale(loss).backward()
```

### Learning Rate Schedule
- OneCycleLR with cosine annealing
- `max_lr` = config learning_rate
- `pct_start` = warmup_ratio (default: 0.1)
- Steps per epoch / gradient_accumulation_steps

### Model Caching
Environment variables set globally:
```python
os.environ['TORCH_HOME'] = './data/torch_cache'
os.environ['HF_HOME'] = './data/hf_cache'
os.environ['TIMM_CACHE_DIR'] = './data/torch_cache/hub'
```
All pretrained models cached locally to avoid re-downloads.

## Important Implementation Notes

### 1. Evidence Fusion Logic

When both BADM and AADM are active:
```python
evidence_list = [badm_evidence, aadm_evidence]  # Each 64-dim
reasoning_out = ERM(evidence_list)  # Cross-attention fusion
external_input = torch.cat(evidence_list, dim=1)  # 128-dim
external_logit = classifier_dual(external_input)
```

When single module active:
```python
evidence_list = [single_evidence]  # 64-dim
reasoning_out = ERM(evidence_list)
external_logit = classifier_single(evidence_list[0])
```

### 2. Ablation Implementation

**Module Toggling:**
- Set `use_badm=False` or `use_aadm=False` in forward pass
- Loss function adapts automatically (no branch loss for disabled modules)
- ERM handles 1-2 evidence sources dynamically

**Reasoning Iterations:**
- Set `reasoning_iterations=1` for "no reasoning" ablation
- Deep supervision loss adjusts automatically

### 3. Attention Convergence

**Convergence Delta:**
```python
# Measures stability of reasoning
delta = |prob[iter=3] - prob[iter=2]|
```
Lower delta = more stable convergence (typical: 0.001-0.005).

**Attention History:**
- Shape: (batch, iterations, max_sources)
- `max_sources=2` for BADM + AADM
- Attention weights sum to 1.0 per iteration

### 4. Gating Mechanism

**Initialization:**
```python
# Convert gating_init (0.5) to logit space
logit = log(gating_init / (1 - gating_init))
_gating_logit = nn.Parameter(torch.tensor(logit))
```

**Usage:**
```python
alpha = torch.sigmoid(_gating_logit)  # Learnable during training
final = alpha * reasoning + (1 - alpha) * external
```

### 5. Gradient Accumulation

Effective batch size = `batch_size * gradient_accumulation_steps`.

Example: `batch_size=64, accumulation=2` → effective batch = 128.

Loss scaling:
```python
loss = losses["total"] / gradient_accumulation_steps
```

### 6. Data Loading Performance

**Persistent Workers:**
- `persistent_workers=True` keeps workers alive between epochs
- Saves ~10-30s per epoch on startup

**Prefetching:**
- `prefetch_factor=4` loads 4 batches ahead per worker
- Total prefetched = `num_workers * prefetch_factor`

**Worker Init:**
```python
worker_init_fn=lambda x: np.random.seed(seed + x)
```
Ensures different augmentations per worker.

## Reproducibility

### Seed Setting
```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Note:** `benchmark=False` hurts performance (~10%) but ensures determinism.

### Expected Results

**Kaggle 140k (99.88% val AUC):**
- Converges in ~20 epochs
- Training time: ~2 hours on A40
- Test AUC: 99.91%+

**Per-Module Performance (Ablations):**
- BADM only: ~98.5% AUC
- AADM only: ~98.2% AUC
- Both (no reasoning): ~99.3% AUC
- Full RADAR: ~99.9% AUC

## Common Issues & Solutions

### 1. Out of Memory
- Reduce `batch_size` (128 → 64 → 32)
- Increase `gradient_accumulation_steps`
- Reduce `num_workers` (frees RAM)

### 2. Slow Data Loading
- Enable `persistent_workers=True`
- Increase `num_workers` (up to CPU count - 1)
- Use SSD for data directory

### 3. Non-finite Loss
- Check input data ranges (should be normalized)
- Reduce learning rate
- Enable gradient clipping (already at 1.0)

### 4. Poor Convergence
- Increase `warmup_ratio` (0.1 → 0.2)
- Reduce learning rate
- Check label distribution (should be balanced)

### 5. Checkpoint Loading
```python
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
```
Always specify `map_location` for CPU inference.

## Research Extensions

### Adding New Baselines
1. Create file in `src/baselines/`
2. Inherit from `nn.Module`
3. Implement `forward(x, freq_cached, sobel_cached) -> Dict`
4. Return `{"logit": ..., "prob": ...}`
5. Add to `src/baselines/__init__.py`

### Modifying Loss Function
Edit `src/method/loss.py`:
- Add new loss component to `forward()`
- Update `LossConfig` dataclass
- Add lambda weight to config YAML

### New Evidence Sources
1. Add detector in `src/method/`
2. Update `RADAR.forward()` to call detector
3. Increase `max_evidence_sources` in ERM
4. Update loss function branch supervision

### Cross-Dataset Evaluation
```python
# Train on domain A
python 2_train_model.py --data_dir ./data/kaggle_140k_prepared

# Evaluate on domain B (modify evaluate.py)
test_images, test_labels = load_presplit_data("./data/faceforensics", "test")
```

## File Naming Conventions

- `0_*.py`: Data download/setup
- `1_*.py`: Data preparation
- `2_*.py`: Training variants
- `3_*.py`: Ablation experiments
- `4_*.py`: Analysis/comparison
- `5_*.py`: Visualization

Numbers indicate typical execution order for a full pipeline run.
