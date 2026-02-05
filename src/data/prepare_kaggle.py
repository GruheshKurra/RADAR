#!/usr/bin/env python3
"""Prepare Kaggle 140k dataset for training."""

import argparse
import shutil
import sys
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging import print_section, print_complete, print_result


def reorganize_dataset(input_dir: Path, output_dir: Path):
    """Reorganize Kaggle structure to training structure."""

    # Actual Kaggle structure: real_vs_fake/real-vs-fake/train/, valid/, test/
    kaggle_root = input_dir / "real_vs_fake" / "real-vs-fake"

    if not kaggle_root.exists():
        # Fallback: try alternative structures
        kaggle_root = input_dir / "real_and_fake_face"
        if not kaggle_root.exists():
            kaggle_root = input_dir

    # Check if already in train/real, train/fake format
    if (kaggle_root / "train" / "real").exists() and (kaggle_root / "train" / "fake").exists():
        print(f"Dataset already in correct format at {kaggle_root}")
        mappings = [
            ("train/real", "train/real"),
            ("train/fake", "train/fake"),
            ("valid/real", "val/real"),
            ("valid/fake", "val/fake"),
            ("test/real", "test/real"),
            ("test/fake", "test/fake"),
        ]
    else:
        # Old format
        mappings = [
            ("training_real", "train/real"),
            ("training_fake", "train/fake"),
            ("validation_real", "val/real"),
            ("validation_fake", "val/fake"),
        ]

    stats = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}, "test": {"real": 0, "fake": 0}}

    for src_name, dst_path in mappings:
        src_dir = kaggle_root / src_name
        if not src_dir.exists():
            print(f"⚠️  Warning: {src_dir} not found, skipping...")
            continue

        dst_dir = output_dir / dst_path
        dst_dir.mkdir(parents=True, exist_ok=True)

        images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))

        parts = dst_path.split("/")
        split = parts[0] if parts[0] in ["train", "val", "test"] else "train"
        class_name = parts[1] if len(parts) > 1 else "fake"

        print(f"\nCopying {src_name} ({len(images)} images)...")

        for img_path in tqdm(images, desc=f"  {src_name}"):
            dst_file = dst_dir / img_path.name
            if not dst_file.exists():
                shutil.copy2(img_path, dst_file)
            stats[split][class_name] += 1

    return stats


def create_metadata(output_dir: Path, stats: dict):
    """Save dataset metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": "Kaggle 140k Real and Fake Faces",
        "source": "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces",
        "total_images": sum(sum(s.values()) for s in stats.values()),
        "statistics": stats
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare Kaggle 140k dataset")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to downloaded Kaggle dataset")
    parser.add_argument("--output_dir", type=str, default="./data/kaggle_140k_prepared",
                        help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"✗ Error: {input_dir} not found")
        sys.exit(1)

    print_section(
        "KAGGLE 140K PREPARATION",
        f"Input: {input_dir}\nOutput: {output_dir}"
    )

    stats = reorganize_dataset(input_dir, output_dir)
    create_metadata(output_dir, stats)

    total_real = sum(s["real"] for s in stats.values())
    total_fake = sum(s["fake"] for s in stats.values())

    print_complete(
        "PREPARATION COMPLETE",
        {
            "Train real": stats["train"]["real"],
            "Train fake": stats["train"]["fake"],
            "Val real": stats["val"]["real"],
            "Val fake": stats["val"]["fake"],
            "Total": total_real + total_fake,
            "Location": str(output_dir)
        }
    )

    print("\nNext step:")
    print(f"  python 2_train_model.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()
