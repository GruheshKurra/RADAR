# RunPod Setup Guide for RADAR Training

Complete guide for training RADAR on RunPod with A40 GPU.

## 🖥️ Pod Configuration

### Recommended Specs
- **GPU:** A40 (48GB VRAM)
- **RAM:** 48GB
- **vCPUs:** 9
- **Storage:** **400GB** (minimum 300GB)
- **Template:** PyTorch 2.0+

### Storage Breakdown
```
FaceForensics++ compressed:  ~130GB
Extracted frames:            ~50GB
Model checkpoints:           ~5GB
Cache (torch/HF):           ~10GB
Training outputs:            ~5GB
Buffer:                     ~50GB
─────────────────────────────────
Recommended:                 400GB
```

---

## 🚀 One-Time Setup

### 1. Start Pod & Connect
```bash
# Connect via SSH (get details from RunPod UI)
ssh root@<pod-ip> -p <port>
```

### 2. Clone Repository
```bash
cd /workspace
git clone https://github.com/GruheshKurra/RADAR.git
cd RADAR
```

### 3. Install Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm albumentations datasets pillow tqdm scikit-learn pyyaml opencv-python
```

### 4. Verify GPU
```bash
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 📥 Download & Prepare Dataset

### Option 1: Sample Test (30 minutes)
```bash
# Download sample
python 0_download_faceforensics.py --sample_only --skip_test

# Extract frames
python src/data/prepare_faceforensics.py \
  --video_dir ./data/faceforensics \
  --output_dir ./data/faceforensics_frames

# Quick train test (5 epochs)
python 2_train_model_fast.py \
  --data_dir ./data/faceforensics_frames \
  --batch_size 128 \
  --num_epochs 5
```

### Option 2: Full Dataset (Overnight)
```bash
# Download full FF++ (run in tmux/screen)
tmux new -s download
python 0_download_faceforensics.py
# Ctrl+B, D to detach

# Extract all frames
python src/data/prepare_faceforensics.py \
  --video_dir ./data/faceforensics \
  --output_dir ./data/faceforensics_frames
```

---

## ⚡ Optimized Training Commands

### Fast Mode (Testing - 2 hours)
```bash
# Uses 50% data, batch_size=128, 10 epochs
python 2_train_model_fast.py \
  --data_dir ./data/faceforensics_frames \
  --output_dir ./outputs \
  --batch_size 128 \
  --num_epochs 10
```

**A40 will automatically use:**
- Batch size: 128 (if you pass default 64, auto-upgrades to 128)
- Workers: 8 (uses 8 of 9 vCPUs)
- No gradient accumulation needed
- **Time: ~2 hours**

### Optimal Mode (Research - 6-8 hours)
```bash
# Full data, batch_size=128, 30 epochs
python 2_train_model.py \
  --data_dir ./data/faceforensics_frames \
  --output_dir ./outputs \
  --experiment_name radar_faceforensics \
  --num_epochs 30
```

**What A40 uses:**
- Batch size: 128 (auto-detected)
- Workers: 8
- Epochs: 30 (standard)
- Data: 100%
- **Time: ~6-8 hours**

---

## 🎯 A40-Specific Optimizations

The code automatically detects A40 (>40GB VRAM) and applies:

| Parameter | A40 Value | Regular GPU |
|-----------|-----------|-------------|
| `batch_size` | 128 | 64 |
| `num_workers` | 8 | 4 |
| `gradient_accumulation` | 1 | 2 |

**Memory utilization:**
- Batch 128: ~35GB VRAM (70% utilization)
- Batch 192: ~45GB VRAM (90% utilization)

---

## 📊 Monitoring Training

### In Another Terminal
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check training output
tail -f outputs/radar_*/training.log
```

### Check Progress
```bash
# List checkpoints
ls -lh outputs/radar_*/

# View metrics
cat outputs/radar_*/metrics.json | python -m json.tool
```

---

## 💾 Save & Download Results

### Export Results
```bash
python 3_export_results.py \
  --results_dir ./outputs/radar_faceforensics \
  --output_dir ./exports
```

### Download to Local
```bash
# Copy to /workspace (accessible via RunPod UI)
cp exports/*.zip /workspace/

# Or use SCP from local machine
scp -P <port> root@<ip>:/workspace/RADAR/exports/*.zip ~/Downloads/
```

---

## 🛠️ Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python 2_train_model.py --batch_size 96
```

### Slow Data Loading
```bash
# Check if 8 workers are running
ps aux | grep "python.*train" | wc -l
# Should show ~9-10 processes (1 main + 8 workers)
```

### Pod Interruption
```bash
# Always run in tmux
tmux new -s training
python 2_train_model.py --data_dir ./data/faceforensics_frames

# Detach: Ctrl+B, D
# Reattach: tmux attach -t training
```

---

## ⏱️ Expected Timings (A40)

| Task | Time |
|------|------|
| Download sample | 10 min |
| Extract frames (sample) | 5 min |
| Fast training (sample) | 30 min |
| **Full download** | **2-4 hours** |
| **Full frame extraction** | **30 min** |
| **Full training (30 epochs)** | **6-8 hours** |

---

## 📈 Expected Results

With A40 optimal settings:

| Metric | Value |
|--------|-------|
| Training AUC | 0.99+ |
| Validation AUC | 0.93-0.95 |
| Per-manipulation | 90-95% |
| Training speed | ~180 samples/sec |

---

## 🔄 Typical Workflow

```bash
# Day 1: Setup & Test (1 hour)
cd /workspace && git clone https://github.com/GruheshKurra/RADAR.git
cd RADAR
pip install torch torchvision timm albumentations datasets pillow tqdm scikit-learn pyyaml opencv-python
python 0_download_faceforensics.py --sample_only --skip_test
python src/data/prepare_faceforensics.py --video_dir ./data/faceforensics
python 2_train_model_fast.py --data_dir ./data/faceforensics_frames --num_epochs 5

# Day 1 Evening: Full Download (overnight)
tmux new -s download
python 0_download_faceforensics.py
# Ctrl+B, D to detach

# Day 2: Extract & Train (overnight)
python src/data/prepare_faceforensics.py --video_dir ./data/faceforensics
tmux new -s training
python 2_train_model.py --data_dir ./data/faceforensics_frames --num_epochs 30
# Ctrl+B, D

# Day 3: Export Results
python 3_export_results.py --results_dir ./outputs/radar_*
cp exports/*.zip /workspace/
```

---

## 💰 Cost Estimation

A40 Pod (~$0.79/hour on RunPod):
- Sample test: $0.50
- Full training: $6-8
- **Total with setup: ~$10-12**

---

## 🎓 Tips for Research

1. **Always use tmux** - pods can disconnect
2. **Download sample first** - validate pipeline
3. **Monitor GPU** - should be at 90%+ utilization with batch_size=128
4. **Save checkpoints** - enabled by default every epoch
5. **Export immediately** - when training completes

Happy training! 🚀
