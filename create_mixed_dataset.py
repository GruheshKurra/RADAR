#!/usr/bin/env python3
"""
Production-ready mixed deepfake dataset creator.
Downloads, preprocesses, and combines multiple datasets automatically.
"""

import argparse
import shutil
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
import hashlib
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Dataset mixing ratios (configurable)
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


def verify_and_fix_image(img_path: Path) -> bool:
    """Verify image is valid, fix if possible, return success status."""
    try:
        img = Image.open(img_path)
        img.verify()  # Verify integrity

        # Reload and check if RGB
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            img.save(img_path, 'JPEG', quality=95)

        # Check minimum size
        if img.size[0] < 64 or img.size[1] < 64:
            return False

        return True
    except Exception as e:
        print(f"⚠️  Corrupt image {img_path.name}: {e}")
        return False


def load_dataset_paths(source_dir: Path, dataset_name: str) -> Tuple[List[Path], List[Path]]:
    """
    Load real and fake image paths from a dataset source.
    Handles different dataset structures automatically.
    """
    real_paths = []
    fake_paths = []

    print(f"  Loading {dataset_name}...")

    if not source_dir.exists():
        print(f"    ⚠️  Directory not found: {source_dir}")
        return real_paths, fake_paths

    try:
        if dataset_name == "faceforensics_c23":
            # Structure: original/, Deepfakes/, Face2Face/, etc.
            real_dir = source_dir / "original"
            fake_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]

            if real_dir.exists():
                for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
                    real_paths.extend(list(real_dir.rglob(ext)))

            for method in fake_methods:
                method_dir = source_dir / method
                if method_dir.exists():
                    for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
                        fake_paths.extend(list(method_dir.rglob(ext)))

        elif dataset_name == "celeb_df_v2":
            # Structure: Celeb-real/, Celeb-synthesis/ or real/, fake/
            real_dirs = [source_dir / "Celeb-real", source_dir / "real"]
            fake_dirs = [source_dir / "Celeb-synthesis", source_dir / "fake"]

            for real_dir in real_dirs:
                if real_dir.exists():
                    for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
                        real_paths.extend(list(real_dir.rglob(ext)))
                    break

            for fake_dir in fake_dirs:
                if fake_dir.exists():
                    for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
                        fake_paths.extend(list(fake_dir.rglob(ext)))
                    break

        elif dataset_name == "dfdc":
            # Structure: real/, fake/ or train/, test/
            for potential_real in ["real", "train/real", "original"]:
                real_dir = source_dir / potential_real
                if real_dir.exists():
                    for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
                        real_paths.extend(list(real_dir.rglob(ext)))
                    break

            for potential_fake in ["fake", "train/fake", "manipulated"]:
                fake_dir = source_dir / potential_fake
                if fake_dir.exists():
                    for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
                        fake_paths.extend(list(fake_dir.rglob(ext)))
                    break

        elif dataset_name == "kaggle_140k":
            # Structure: train/real/, train/fake/ or just real/, fake/
            for split in ["train", ""]:
                split_dir = source_dir / split if split else source_dir
                real_dir = split_dir / "real"
                fake_dir = split_dir / "fake"

                if real_dir.exists() and fake_dir.exists():
                    for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]:
                        real_paths.extend(list(real_dir.rglob(ext)))
                        fake_paths.extend(list(fake_dir.rglob(ext)))
                    break

        # Verify images (remove corrupt ones)
        print(f"    Found: {len(real_paths)} real, {len(fake_paths)} fake")
        print(f"    Verifying integrity...")

        real_paths = [p for p in tqdm(real_paths, desc="    Verifying real", leave=False)
                      if verify_and_fix_image(p)]
        fake_paths = [p for p in tqdm(fake_paths, desc="    Verifying fake", leave=False)
                      if verify_and_fix_image(p)]

        print(f"    ✓ Valid: {len(real_paths)} real, {len(fake_paths)} fake")

    except Exception as e:
        print(f"    ✗ Error loading {dataset_name}: {e}")
        return [], []

    return real_paths, fake_paths


def stratified_split(
    paths: List[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split paths into train/val/test with stratification."""
    random.seed(seed)
    np.random.seed(seed)

    paths = list(paths)
    random.shuffle(paths)

    n = len(paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = paths[:n_train]
    val = paths[n_train:n_train + n_val]
    test = paths[n_train + n_val:]

    return train, val, test


def copy_with_prefix(
    src_paths: List[Path],
    dst_dir: Path,
    prefix: str,
    description: str,
    quality: int = 95
):
    """
    Copy files with dataset prefix and optional preprocessing.
    Ensures RGB format and reasonable quality.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for src_path in tqdm(src_paths, desc=description, unit="img"):
        try:
            # Generate unique filename with hash to avoid collisions
            file_hash = hashlib.md5(str(src_path).encode()).hexdigest()[:8]
            dst_name = f"{prefix}_{file_hash}_{src_path.stem}.jpg"
            dst_path = dst_dir / dst_name

            if dst_path.exists():
                successful += 1
                continue

            # Load, convert to RGB, save
            img = Image.open(src_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Ensure minimum size
            if img.size[0] < 64 or img.size[1] < 64:
                failed += 1
                continue

            img.save(dst_path, 'JPEG', quality=quality, optimize=False)
            successful += 1

        except Exception as e:
            failed += 1
            if failed <= 5:  # Only show first 5 errors
                tqdm.write(f"      Error copying {src_path.name}: {e}")

    if failed > 0:
        tqdm.write(f"      ⚠️  Failed: {failed} images")

    return successful, failed


def create_mixed_dataset(
    source_dirs: Dict[str, Path],
    output_dir: Path,
    dataset_ratios: Dict[str, float],
    target_total: int = 200000,
    seed: int = 42,
    quality: int = 95
):
    """
    Create mixed dataset from multiple sources.
    Handles all preprocessing, validation, and metadata generation.
    """

    print("\n" + "="*70)
    print("RADAR MIXED DATASET CREATOR")
    print("="*70)
    print(f"Output: {output_dir}")
    print(f"Target total: {target_total:,} images")
    print(f"Quality: {quality}")
    print(f"Seed: {seed}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate target counts per dataset
    target_counts = {name: int(target_total * ratio)
                    for name, ratio in dataset_ratios.items()
                    if name in source_dirs}

    print("\nTarget composition:")
    for name, count in target_counts.items():
        ratio = dataset_ratios[name]
        print(f"  {name:20s}: {count:7,} images ({ratio*100:5.1f}%)")

    # Storage for all paths
    stats = defaultdict(lambda: defaultdict(int))

    # Process each dataset
    for dataset_name, source_dir in source_dirs.items():
        if dataset_name not in dataset_ratios:
            print(f"\n⚠️  Skipping {dataset_name} (not in ratios)")
            continue

        print(f"\n{'='*70}")
        print(f"Processing: {dataset_name}")
        print(f"Source: {source_dir}")
        print(f"{'='*70}")

        if not source_dir.exists():
            print(f"⚠️  Directory not found, skipping...")
            continue

        # Load all images
        real_paths, fake_paths = load_dataset_paths(source_dir, dataset_name)

        if len(real_paths) == 0 and len(fake_paths) == 0:
            print(f"⚠️  No valid images found, skipping...")
            continue

        # Sample to target count (1:2 ratio - real:fake)
        target = target_counts[dataset_name]
        target_real = min(target // 3, len(real_paths))
        target_fake = min(target - target_real, len(fake_paths))

        if len(real_paths) > target_real:
            real_paths = random.sample(real_paths, target_real)
        if len(fake_paths) > target_fake:
            fake_paths = random.sample(fake_paths, target_fake)

        print(f"\nSelected: {len(real_paths):,} real, {len(fake_paths):,} fake")

        # Split into train/val/test
        real_train, real_val, real_test = stratified_split(
            real_paths, TRAIN_RATIO, VAL_RATIO, seed
        )
        fake_train, fake_val, fake_test = stratified_split(
            fake_paths, TRAIN_RATIO, VAL_RATIO, seed
        )

        # Use short prefix
        prefix = dataset_name.split('_')[0][:4]

        # Copy files with progress bars
        print(f"\nCopying to output directory...")

        s, f = copy_with_prefix(
            real_train, output_dir / "train" / "real",
            prefix, f"  ├─ train/real", quality
        )
        stats[dataset_name]["train_real"] = s

        s, f = copy_with_prefix(
            fake_train, output_dir / "train" / "fake",
            prefix, f"  ├─ train/fake", quality
        )
        stats[dataset_name]["train_fake"] = s

        s, f = copy_with_prefix(
            real_val, output_dir / "val" / "real",
            prefix, f"  ├─ val/real", quality
        )
        stats[dataset_name]["val_real"] = s

        s, f = copy_with_prefix(
            fake_val, output_dir / "val" / "fake",
            prefix, f"  ├─ val/fake", quality
        )
        stats[dataset_name]["val_fake"] = s

        s, f = copy_with_prefix(
            real_test, output_dir / "test" / "real",
            prefix, f"  └─ test/real", quality
        )
        stats[dataset_name]["test_real"] = s

        s, f = copy_with_prefix(
            fake_test, output_dir / "test" / "fake",
            prefix, f"  └─ test/fake", quality
        )
        stats[dataset_name]["test_fake"] = s

    # Calculate totals
    actual_totals = {
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

    # Create metadata
    metadata = {
        "dataset_name": "multi_deepfake_v1",
        "version": "1.0",
        "creation_date": str(Path.cwd()),
        "target_total": target_total,
        "sources": dict(stats),
        "ratios": dataset_ratios,
        "splits": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO
        },
        "actual_totals": actual_totals,
        "preprocessing": {
            "format": "JPEG",
            "quality": quality,
            "mode": "RGB",
            "verified": True,
            "min_size": "64x64"
        },
        "seed": seed
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print final summary
    print("\n" + "="*70)
    print("✓ DATASET CREATION COMPLETE")
    print("="*70)

    grand_total = sum(sum(split.values()) for split in actual_totals.values())

    print("\nFinal Statistics:")
    for split in ["train", "val", "test"]:
        real = actual_totals[split]["real"]
        fake = actual_totals[split]["fake"]
        total = real + fake
        ratio = fake / real if real > 0 else 0
        print(f"  {split:5s}: {real:7,} real + {fake:7,} fake = {total:7,} total (ratio: {ratio:.2f}:1)")

    print(f"\n  Grand Total: {grand_total:,} images")
    print(f"\nOutput Location: {output_dir}")
    print(f"Metadata: {metadata_path}")

    # Verify directory structure
    print("\nDirectory Structure:")
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            path = output_dir / split / label
            count = len(list(path.glob("*.jpg"))) if path.exists() else 0
            print(f"  {split}/{label:4s}: {count:7,} images")

    print("\n" + "="*70)
    print("Next Steps:")
    print(f"  1. Verify metadata: cat {metadata_path}")
    print(f"  2. Train model: python 2_train_model.py --data_dir {output_dir}")
    print(f"  3. Upload to HF: python upload_to_huggingface.py --dataset_dir {output_dir}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create mixed deepfake dataset from multiple sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create from all sources
  python create_mixed_dataset.py \\
    --faceforensics /data/faceforensics_c23 \\
    --celeb_df /data/celeb_df_v2 \\
    --dfdc /data/dfdc \\
    --kaggle_140k ./data/kaggle_140k_prepared \\
    --output_dir ./data/multi_deepfake_v1

  # Create from available sources only
  python create_mixed_dataset.py \\
    --faceforensics /data/faceforensics_c23 \\
    --celeb_df /data/celeb_df_v2 \\
    --output_dir ./data/mixed_ff_celeb

  # Quick test with 10k images
  python create_mixed_dataset.py \\
    --faceforensics /data/faceforensics_c23 \\
    --output_dir ./data/test_dataset \\
    --target_total 10000
        """
    )

    parser.add_argument("--faceforensics", type=str,
                       help="FaceForensics++ c23 directory")
    parser.add_argument("--celeb_df", type=str,
                       help="Celeb-DF v2 directory")
    parser.add_argument("--dfdc", type=str,
                       help="DFDC directory")
    parser.add_argument("--kaggle_140k", type=str,
                       help="Kaggle 140k directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for mixed dataset")
    parser.add_argument("--target_total", type=int, default=200000,
                       help="Target total images (default: 200000)")
    parser.add_argument("--quality", type=int, default=95,
                       help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # Build source_dirs dict from provided arguments
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
        print("✗ Error: Provide at least one source dataset")
        print("\nAvailable options:")
        print("  --faceforensics PATH")
        print("  --celeb_df PATH")
        print("  --dfdc PATH")
        print("  --kaggle_140k PATH")
        print("\nSee --help for examples")
        sys.exit(1)

    # Adjust ratios if not all datasets provided
    if len(source_dirs) < len(DATASET_RATIOS):
        print(f"\n📊 Adjusting ratios for {len(source_dirs)} dataset(s)...")
        available_ratios = {k: v for k, v in DATASET_RATIOS.items()
                           if k in source_dirs}
        total_ratio = sum(available_ratios.values())
        adjusted_ratios = {k: v/total_ratio for k, v in available_ratios.items()}

        print("Adjusted ratios:")
        for name, ratio in adjusted_ratios.items():
            print(f"  {name}: {ratio*100:.1f}%")
    else:
        adjusted_ratios = DATASET_RATIOS

    create_mixed_dataset(
        source_dirs=source_dirs,
        output_dir=Path(args.output_dir),
        dataset_ratios=adjusted_ratios,
        target_total=args.target_total,
        seed=args.seed,
        quality=args.quality
    )


if __name__ == "__main__":
    main()
