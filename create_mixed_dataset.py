#!/usr/bin/env python3
"""
Create a mixed deepfake dataset from multiple sources.
Combines FaceForensics++, Celeb-DF, and other datasets into a unified structure.
"""

import argparse
import shutil
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict

# Dataset mixing ratios
DATASET_RATIOS = {
    "faceforensics_c23": 0.40,
    "celeb_df_v2": 0.30,
    "dfdc": 0.20,
    "kaggle_140k": 0.10,
}

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_dataset_paths(source_dir: Path, dataset_name: str) -> Tuple[List[Path], List[Path]]:
    """Load real and fake image paths from a dataset source."""
    real_paths = []
    fake_paths = []

    # Adapt based on dataset structure
    if dataset_name == "faceforensics_c23":
        real_dir = source_dir / "real"
        fake_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

        if real_dir.exists():
            real_paths = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))

        for method in fake_methods:
            fake_dir = source_dir / method
            if fake_dir.exists():
                fake_paths.extend(list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png")))

    elif dataset_name == "celeb_df_v2":
        real_dir = source_dir / "real"
        fake_dir = source_dir / "fake"

        if real_dir.exists():
            real_paths = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
        if fake_dir.exists():
            fake_paths = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png"))

    elif dataset_name == "dfdc":
        # DFDC has different structure - adapt as needed
        real_dir = source_dir / "real"
        fake_dir = source_dir / "fake"

        if real_dir.exists():
            real_paths = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
        if fake_dir.exists():
            fake_paths = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png"))

    elif dataset_name == "kaggle_140k":
        # Already in real/fake structure
        real_dir = source_dir / "train" / "real"
        fake_dir = source_dir / "train" / "fake"

        if real_dir.exists():
            real_paths = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
        if fake_dir.exists():
            fake_paths = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png"))

    return real_paths, fake_paths


def stratified_split(
    paths: List[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split paths into train/val/test."""
    random.seed(seed)
    paths = list(paths)
    random.shuffle(paths)

    n = len(paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = paths[:n_train]
    val = paths[n_train:n_train + n_val]
    test = paths[n_train + n_val:]

    return train, val, test


def copy_with_prefix(src_paths: List[Path], dst_dir: Path, prefix: str, description: str):
    """Copy files with dataset prefix."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src_path in tqdm(src_paths, desc=description):
        # Add prefix to filename
        dst_name = f"{prefix}_{src_path.name}"
        dst_path = dst_dir / dst_name

        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)


def create_mixed_dataset(
    source_dirs: Dict[str, Path],
    output_dir: Path,
    dataset_ratios: Dict[str, float],
    target_total: int = 200000,
    seed: int = 42
):
    """Create mixed dataset from multiple sources."""

    print("\n" + "="*70)
    print("CREATING MIXED DEEPFAKE DATASET")
    print("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate target counts per dataset
    target_counts = {name: int(target_total * ratio) for name, ratio in dataset_ratios.items()}

    print("\nTarget composition:")
    for name, count in target_counts.items():
        print(f"  {name}: {count:,} images ({dataset_ratios[name]*100:.1f}%)")

    # Storage for all paths
    all_splits = defaultdict(lambda: {"real": [], "fake": []})
    stats = defaultdict(lambda: defaultdict(int))

    # Process each dataset
    for dataset_name, source_dir in source_dirs.items():
        if dataset_name not in dataset_ratios:
            continue

        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*70}")

        if not source_dir.exists():
            print(f"⚠️  Warning: {source_dir} not found, skipping...")
            continue

        # Load all images
        real_paths, fake_paths = load_dataset_paths(source_dir, dataset_name)

        print(f"Found: {len(real_paths):,} real, {len(fake_paths):,} fake")

        # Sample to target count
        target = target_counts[dataset_name]
        target_real = target // 3  # 1:2 ratio (real:fake)
        target_fake = target - target_real

        if len(real_paths) > target_real:
            real_paths = random.sample(real_paths, target_real)
        if len(fake_paths) > target_fake:
            fake_paths = random.sample(fake_paths, target_fake)

        # Split into train/val/test
        real_train, real_val, real_test = stratified_split(real_paths, TRAIN_RATIO, VAL_RATIO, seed)
        fake_train, fake_val, fake_test = stratified_split(fake_paths, TRAIN_RATIO, VAL_RATIO, seed)

        # Use short prefix
        prefix = dataset_name.split('_')[0][:4]  # e.g., "face", "cele", "dfdc"

        # Copy files
        copy_with_prefix(real_train, output_dir / "train" / "real", prefix, f"{dataset_name} train/real")
        copy_with_prefix(fake_train, output_dir / "train" / "fake", prefix, f"{dataset_name} train/fake")
        copy_with_prefix(real_val, output_dir / "val" / "real", prefix, f"{dataset_name} val/real")
        copy_with_prefix(fake_val, output_dir / "val" / "fake", prefix, f"{dataset_name} val/fake")
        copy_with_prefix(real_test, output_dir / "test" / "real", prefix, f"{dataset_name} test/real")
        copy_with_prefix(fake_test, output_dir / "test" / "fake", prefix, f"{dataset_name} test/fake")

        # Track stats
        stats[dataset_name]["train_real"] = len(real_train)
        stats[dataset_name]["train_fake"] = len(fake_train)
        stats[dataset_name]["val_real"] = len(real_val)
        stats[dataset_name]["val_fake"] = len(fake_val)
        stats[dataset_name]["test_real"] = len(real_test)
        stats[dataset_name]["test_fake"] = len(fake_test)

    # Create metadata
    metadata = {
        "dataset_name": "multi_deepfake_v1",
        "creation_date": str(Path.cwd()),
        "target_total": target_total,
        "sources": dict(stats),
        "ratios": dataset_ratios,
        "splits": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO
        },
        "actual_totals": {
            "train": {
                "real": sum(s["train_real"] for s in stats.values()),
                "fake": sum(s["train_fake"] for s in stats.values()),
            },
            "val": {
                "real": sum(s["val_real"] for s in stats.values()),
                "fake": sum(s["val_fake"] for s in stats.values()),
            },
            "test": {
                "real": sum(s["test_real"] for s in stats.values()),
                "fake": sum(s["test_fake"] for s in stats.values()),
            }
        }
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("✓ DATASET CREATION COMPLETE")
    print("="*70)

    print("\nFinal Statistics:")
    print(f"  Train: {metadata['actual_totals']['train']['real']:,} real, {metadata['actual_totals']['train']['fake']:,} fake")
    print(f"  Val:   {metadata['actual_totals']['val']['real']:,} real, {metadata['actual_totals']['val']['fake']:,} fake")
    print(f"  Test:  {metadata['actual_totals']['test']['real']:,} real, {metadata['actual_totals']['test']['fake']:,} fake")
    print(f"\nTotal: {sum(sum(split.values()) for split in metadata['actual_totals'].values()):,} images")
    print(f"Location: {output_dir}")
    print(f"Metadata: {output_dir / 'metadata.json'}")


def main():
    parser = argparse.ArgumentParser(description="Create mixed deepfake dataset")
    parser.add_argument("--faceforensics", type=str, help="FaceForensics++ c23 directory")
    parser.add_argument("--celeb_df", type=str, help="Celeb-DF v2 directory")
    parser.add_argument("--dfdc", type=str, help="DFDC directory")
    parser.add_argument("--kaggle_140k", type=str, help="Kaggle 140k directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--target_total", type=int, default=200000, help="Target total images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Build source_dirs dict
    source_dirs = {}
    if args.faceforensics:
        source_dirs["faceforensics_c23"] = Path(args.faceforensics)
    if args.celeb_df:
        source_dirs["celeb_df_v2"] = Path(args.celeb_df)
    if args.dfdc:
        source_dirs["dfdc"] = Path(args.dfdc)
    if args.kaggle_140k:
        source_dirs["kaggle_140k"] = Path(args.kaggle_140k)

    if not source_dirs:
        print("Error: Provide at least one source dataset")
        return

    create_mixed_dataset(
        source_dirs=source_dirs,
        output_dir=Path(args.output_dir),
        dataset_ratios=DATASET_RATIOS,
        target_total=args.target_total,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
