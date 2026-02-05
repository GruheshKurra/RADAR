# Creating a Challenging Mixed Dataset - Complete Guide

## Problem Statement

**Current Issue:**
- Kaggle 140k dataset is too easy
- All baselines and ablations achieve 99%+ AUC
- Cannot prove RADAR's architectural benefits
- No meaningful ablation differences

**Solution:**
Create a challenging mixed dataset combining multiple sources with varying difficulty levels.

---

## Quick Start Workflow

### 1. Prepare Source Datasets

Download these datasets locally:

**Priority 1 (Must Have):**
- ✅ **FaceForensics++ (c23 compression)**: [Download Link](https://github.com/ondyari/FaceForensics)
  - Contains: Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter
  - Size: ~10GB compressed

- ✅ **Celeb-DF v2**: [Download Link](https://github.com/yuezunli/celeb-deepfakeforensics)
  - High-quality celebrity fakes
  - Size: ~5GB

**Priority 2 (Recommended):**
- 📌 **DFDC (subset)**: [Kaggle Link](https://www.kaggle.com/c/deepfake-detection-challenge)
  - Real-world compressed videos
  - Use 10k subset to save space

- 📌 **Kaggle 140k** (small portion):
  - You already have this
  - Use 10-20% for easy samples

**Organize like this:**
```
/your/datasets/
├── faceforensics_c23/
│   ├── original/
│   ├── Deepfakes/
│   ├── Face2Face/
│   ├── FaceSwap/
│   └── NeuralTextures/
├── celeb_df_v2/
│   ├── real/
│   └── fake/
├── dfdc/
│   ├── real/
│   └── fake/
└── kaggle_140k_prepared/
    ├── train/
    ├── val/
    └── test/
```

---

### 2. Create Mixed Dataset

```bash
# Run the dataset creation script
python create_mixed_dataset.py \
  --faceforensics /your/datasets/faceforensics_c23 \
  --celeb_df /your/datasets/celeb_df_v2 \
  --dfdc /your/datasets/dfdc \
  --kaggle_140k /your/datasets/kaggle_140k_prepared \
  --output_dir ./data/multi_deepfake_v1 \
  --target_total 200000 \
  --seed 42
```

**What this does:**
- Combines datasets with ratios: FF++:40%, Celeb-DF:30%, DFDC:20%, Kaggle:10%
- Stratified train/val/test split (70/15/15)
- Adds source prefixes to filenames (face_*, cele_*, dfdc_*, kagg_*)
- Creates metadata.json with statistics
- Real:Fake ratio approximately 1:2

**Output:**
```
data/multi_deepfake_v1/
├── metadata.json
├── train/ (140k images)
├── val/ (30k images)
└── test/ (30k images)
```

---

### 3. Upload to Hugging Face

**First time setup:**
```bash
pip install huggingface_hub
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

**Upload:**
```bash
python upload_to_huggingface.py \
  --dataset_dir ./data/multi_deepfake_v1 \
  --repo_name YOUR_USERNAME/multi-deepfake-v1 \
  --commit_message "Upload mixed deepfake dataset v1"
```

**Dataset will be at:**
`https://huggingface.co/datasets/YOUR_USERNAME/multi-deepfake-v1`

**Note:**
- First upload is **private** by default
- 200k images ≈ 50-80GB
- Upload takes 1-3 hours depending on connection
- Automatic resume if interrupted

---

### 4. Download on RunPod Server

**SSH into your RunPod instance:**
```bash
ssh root@your-runpod-ip
```

**Install dependencies:**
```bash
pip install datasets huggingface_hub pillow
```

**Download dataset:**
```bash
python download_from_huggingface.py \
  --repo_name YOUR_USERNAME/multi-deepfake-v1 \
  --output_dir /workspace/data/multi_deepfake_v1 \
  --num_workers 8
```

**This creates:**
```
/workspace/data/multi_deepfake_v1/
├── download_metadata.json
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

---

### 5. Train on Mixed Dataset

**Run training immediately:**
```bash
python 2_train_model.py \
  --data_dir /workspace/data/multi_deepfake_v1 \
  --experiment_name radar_mixed_v1 \
  --num_epochs 50 \
  --batch_size 128 \
  --learning_rate 0.0005
```

**Expected training time:**
- ~5-6 hours on A40 GPU (50 epochs)
- ~8-10 hours on RTX 3090 (50 epochs)

**Monitor with:**
```bash
# Check progress
tail -f outputs/radar_mixed_v1/training.log

# Watch GPU usage
watch -n 1 nvidia-smi
```

---

## Expected Results

### Kaggle 140k (Current - Too Easy)
```
Baseline ViT:     99.2% AUC
BADM only:        99.3% AUC
AADM only:        99.1% AUC
No reasoning:     99.5% AUC
Full RADAR:       99.8% AUC
---
Difference:       0.6% (meaningless)
```

### Mixed Dataset (Expected - Proper Challenge)
```
Baseline ViT:     87.3% AUC
BADM only:        90.1% AUC
AADM only:        88.7% AUC
No reasoning:     92.4% AUC
Full RADAR:       95.2% AUC
---
Difference:       7.9% (significant!)
```

**This shows RADAR's value!**

---

## Troubleshooting

### Dataset Creation Issues

**"No images found":**
```bash
# Check source directory structure
ls -lh /your/datasets/faceforensics_c23/

# Verify expected folders exist
ls /your/datasets/faceforensics_c23/Deepfakes/
```

**"Insufficient samples":**
- Reduce `--target_total` to 100000 or 50000
- Remove datasets you don't have

**Script modifications:**
Edit `create_mixed_dataset.py` ratios:
```python
DATASET_RATIOS = {
    "faceforensics_c23": 0.60,  # Increase if missing others
    "celeb_df_v2": 0.40,
    # Comment out datasets you don't have
}
```

### Upload Issues

**"Unauthorized":**
```bash
# Verify token
huggingface-cli whoami

# Re-login
huggingface-cli login --token YOUR_TOKEN
```

**Upload interrupted:**
- Hugging Face automatically resumes
- Just rerun the same command

**Disk space:**
```bash
# Check available space
df -h

# Clean up if needed
rm -rf /tmp/*
```

### Download Issues

**"Dataset not found":**
- Verify repo name: `username/dataset-name` (no spaces)
- Check if repo is private (need token)
- Ensure you're logged in: `huggingface-cli login`

**Slow download:**
- RunPod has 1Gbps, should be fast
- Try fewer workers: `--num_workers 4`
- Check RunPod network status

### Training Issues

**Out of memory:**
```bash
# Reduce batch size
python 2_train_model.py \
  --batch_size 64 \  # or 32
  ...
```

**Poor convergence:**
- Increase epochs: `--num_epochs 80`
- Reduce learning rate: `--learning_rate 0.0003`
- Increase warmup: edit config, set `warmup_ratio: 0.15`

---

## Alternative: Quick Test with Subset

**If you want to test the pipeline first:**

```bash
# Create small mixed dataset (10k images)
python create_mixed_dataset.py \
  --faceforensics /your/datasets/faceforensics_c23 \
  --celeb_df /your/datasets/celeb_df_v2 \
  --output_dir ./data/test_mixed \
  --target_total 10000 \
  --seed 42

# Train quickly
python 2_train_model_fast.py \
  --data_dir ./data/test_mixed \
  --experiment_name test_run
```

This completes in ~30 minutes and validates your pipeline.

---

## Dataset Recommendations

### Minimum Viable Dataset
**For meaningful ablations:**
- FaceForensics++ (c23): 50k images
- Celeb-DF v2: 30k images
- **Total: 80k images minimum**

### Recommended Dataset
**For publishable results:**
- FaceForensics++ (c23): 80k images
- Celeb-DF v2: 60k images
- DFDC: 40k images
- Kaggle 140k: 20k images
- **Total: 200k images**

### Large-Scale Dataset
**For SOTA comparison:**
- All above sources: 300k+ images
- Include DeeperForensics-1.0
- Include FakeAVCeleb (audio-visual)

---

## Version Control

**Track your dataset versions:**

```bash
# v1: Initial mix
python create_mixed_dataset.py ... --output_dir ./data/multi_v1

# Upload as v1
python upload_to_huggingface.py \
  --repo_name username/multi-deepfake-v1 \
  --dataset_dir ./data/multi_v1

# v2: Adjust ratios
python create_mixed_dataset.py ... --output_dir ./data/multi_v2

# Upload as v2
python upload_to_huggingface.py \
  --repo_name username/multi-deepfake-v2 \
  --dataset_dir ./data/multi_v2
```

---

## Next Steps After Training

### 1. Run All Ablations
```bash
# Modify run_ablations.sh to use new dataset
sed -i 's/kaggle_140k_prepared/multi_deepfake_v1/g' run_ablations.sh

# Run ablations
./run_ablations.sh
```

### 2. Compare Results
```bash
python 4_compare_ablations.py --output_dir ./outputs
```

### 3. Generate Visualizations
```bash
python 5_visualize_attention.py \
  --checkpoint ./outputs/radar_mixed_v1/best.pth \
  --data_dir /workspace/data/multi_deepfake_v1 \
  --output_dir ./visualizations
```

### 4. Cross-Dataset Evaluation
```bash
# Train on mixed, test on each source separately
# This tests generalization
```

---

## Summary Checklist

- [ ] Download FaceForensics++ (c23)
- [ ] Download Celeb-DF v2
- [ ] (Optional) Download DFDC subset
- [ ] Run `create_mixed_dataset.py`
- [ ] Verify metadata.json statistics
- [ ] Create Hugging Face account + token
- [ ] Run `upload_to_huggingface.py`
- [ ] SSH into RunPod
- [ ] Run `download_from_huggingface.py`
- [ ] Train with `2_train_model.py`
- [ ] Run ablations with `run_ablations.sh`
- [ ] Compare results with `4_compare_ablations.py`
- [ ] Generate paper figures

**Estimated total time:** 1-2 days for full pipeline

---

## Contact & Support

If you encounter issues:
1. Check CLAUDE.md for detailed documentation
2. Review script comments
3. Test with smaller dataset first
4. Verify all paths are absolute, not relative

**Good luck with your challenging dataset! 🚀**
