# RunPod Quick Start Guide (Memory-Efficient)

## 🚨 CRITICAL FIX - Memory Overflow Solved

**Previous Issue**: Pod crashed at ~18% due to loading all 1.4M images into RAM
**Solution**: Stream processing with configurable limits (default: 300k images)

---

## Quick Commands

### Fast Mode (230k images - 2-3 hours)
```bash
cd /workspace/RADAR-Clean
bash run_fast_pipeline.sh
```

### Complete Mode (300k images - 6-10 hours)
```bash
cd /workspace/RADAR-Clean
bash run_complete_pipeline.sh
```

### Custom Limit
```bash
cd /workspace/RADAR-Clean
python3 1_prepare_dataset.py --output_dir ./data --max_images 200000
python3 2_train_model.py --data_dir ./data --output_dir ./outputs
```

---

## What Changed

### ✅ Memory-Efficient Processing

**Before (BROKEN)**:
- Loaded all 1.4M images into list
- RAM usage: 100% → Pod crash
- Failed at 18% processing

**After (FIXED)**:
- Stream processing in batches of 1000
- Saves as it goes (not after loading all)
- Configurable limit with `--max_images`
- Default: 300k images (safe for A40)

### ✅ Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--max_images` | 300,000 | Prevent memory overflow |
| `--batch_size` | 128 | GPU memory usage |
| `--num_epochs` | 30 | Training duration |

---

## Pod Configuration

**Recommended Setup**:
- GPU: A40 (48GB VRAM)
- RAM: 48GB+
- Disk: 30GB
- Template: PyTorch 2.0+ with CUDA 11.8+

---

## Step-by-Step Instructions

### 1. Upload Project to RunPod

```bash
# Option A: Git clone
cd /workspace
git clone https://github.com/your-repo/RADAR-Clean.git
cd RADAR-Clean

# Option B: Upload zip via RunPod web UI
# Then unzip in /workspace
```

### 2. Run Fast Pipeline (RECOMMENDED FIRST)

```bash
cd /workspace/RADAR-Clean
bash run_fast_pipeline.sh
```

**What it does**:
- Downloads 230k images (10-20 min)
- Trains model (1.5-2.5 hours)
- Exports results (1-2 min)
- **Total: 2-3 hours**

### 3. Monitor Progress

Open second terminal:
```bash
# Watch GPU
watch -n 1 nvidia-smi

# Check progress
ls -lh /workspace/RADAR-Clean/data/wilddeepfake/real/
ls -lh /workspace/RADAR-Clean/data/wilddeepfake/fake/

# View training logs
tail -f /workspace/RADAR-Clean/outputs/*/metrics.json
```

### 4. Download Results

Results will be in:
```
/workspace/RADAR-Clean/exports/radar_wilddeepfake_*.zip
```

**Download via RunPod UI**:
1. Go to Files tab
2. Navigate to `/workspace/RADAR-Clean/exports/`
3. Right-click .zip file → Download

---

## Troubleshooting

### Memory Still High?

Reduce images further:
```bash
python3 1_prepare_dataset.py --output_dir ./data --max_images 150000
```

### GPU Out of Memory?

Reduce batch size:
```bash
python3 2_train_model_fast.py --batch_size 128  # instead of 256
```

### Download Interrupted?

Resume from checkpoint:
```bash
python3 1_prepare_dataset.py --output_dir ./data --check_only
# If images exist, skip to training
python3 2_train_model.py --data_dir ./data --output_dir ./outputs
```

### Check Dataset Status

```bash
python3 1_prepare_dataset.py --output_dir ./data --check_only
```

---

## Cost Optimization

1. **Start with Fast Mode** (230k images)
   - Validates setup works
   - Only 2-3 hours = lower cost

2. **Use Complete Mode** only if needed (300k images)
   - 6-10 hours
   - Better accuracy

3. **Stop Pod Immediately** after export
   - Don't keep running for storage
   - Download results quickly

4. **Use Spot Instances**
   - 80% cheaper
   - May be interrupted (save checkpoints)

---

## File Sizes

| Item | Size |
|------|------|
| 230k images (fast) | ~2-3 GB |
| 300k images (complete) | ~3-4 GB |
| Model checkpoint | ~150 MB |
| Results .zip | ~200 MB |

---

## Expected Timeline

### Fast Mode (230k images)
```
Dataset download:    10-20 min
Training (20 epochs): 1.5-2.5 hrs
Export:               1-2 min
─────────────────────────────
Total:                2-3 hours
```

### Complete Mode (300k images)
```
Dataset download:    15-30 min
Training (30 epochs): 5-9 hrs
Export:               1-2 min
─────────────────────────────
Total:                6-10 hours
```

---

## Important Notes

1. **Default limit is 300k images** - safe for most RunPod instances
2. **Images save during processing** - not after loading all
3. **You can stop and resume** - already downloaded images are kept
4. **Check progress with --check_only** before retrying
5. **Use --force to re-download** if dataset corrupt

---

## Advanced Usage

### Change Image Limit
```bash
python3 1_prepare_dataset.py \
    --output_dir ./data \
    --max_images 500000  # Custom limit
```

### Skip Download (if already done)
```bash
# Check status first
python3 1_prepare_dataset.py --output_dir ./data --check_only

# If images exist, go straight to training
python3 2_train_model.py --data_dir ./data --output_dir ./outputs
```

### Force Re-download
```bash
python3 1_prepare_dataset.py \
    --output_dir ./data \
    --max_images 300000 \
    --force
```

---

## Support

If you still encounter memory issues:
1. Reduce `--max_images` to 150000 or 200000
2. Close other processes on pod
3. Use pod with more RAM (64GB+)
4. Report issue with error logs

---

**Last Updated**: 2026-02-04
**Status**: Memory overflow bug FIXED ✅
