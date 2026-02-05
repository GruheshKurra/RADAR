# RADAR: Reasoning and Artifact Detection for Deepfake Recognition

Deep learning architecture for deepfake detection using dual artifact detection modules with iterative evidence refinement.

## Setup

```bash
pip install torch torchvision timm albumentations pillow tqdm scikit-learn pyyaml opencv-python tabulate matplotlib
```

## Dataset

Download Kaggle 140k Real and Fake Faces:

```bash
python 0_download_kaggle_140k.py --output_dir ./data/kaggle_140k
python src/data/prepare_kaggle.py \
  --input_dir ./data/kaggle_140k \
  --output_dir ./data/kaggle_140k_prepared
```

## Training

```bash
python 2_train_model.py \
  --data_dir ./data/kaggle_140k_prepared \
  --experiment_name radar_training \
  --num_epochs 30 \
  --batch_size 128
```

## Ablation Studies

```bash
./run_ablations.sh

python 4_compare_ablations.py
```

## Visualization

```bash
python 5_visualize_attention.py \
  --checkpoint ./outputs/radar_training/best.pth \
  --data_dir ./data/kaggle_140k_prepared
```

## Architecture

- **BADM**: Boundary Artifact Detection Module
- **AADM**: Frequency Artifact Detection Module
- **ERM**: Evidence Refinement Module with iterative reasoning

## Results

Training on Kaggle 140k dataset achieves 99.88% validation AUC with convergence in approximately 2 hours on A40 GPU.
