#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import sys
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import shutil


def download_wilddeepfake(output_dir: Path):
    print("\n[1] Downloading WildDeepfake from HuggingFace...")

    try:
        ds = load_dataset("xingjunm/WildDeepfake")

        wild_dir = output_dir / "wilddeepfake"
        real_dir = wild_dir / "real"
        fake_dir = wild_dir / "fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        print("Organizing WildDeepfake dataset...")

        if "train" in ds:
            for idx, sample in enumerate(tqdm(ds["train"], desc="Train split")):
                img = sample["image"]
                label = sample["label"]

                target_dir = real_dir if label == 0 else fake_dir
                img_path = target_dir / f"train_{idx:06d}.jpg"
                img.save(img_path)

        if "test" in ds:
            for idx, sample in enumerate(tqdm(ds["test"], desc="Test split")):
                img = sample["image"]
                label = sample["label"]

                target_dir = real_dir if label == 0 else fake_dir
                img_path = target_dir / f"test_{idx:06d}.jpg"
                img.save(img_path)

        print(f"✓ WildDeepfake saved to: {wild_dir}")
        print(f"  Real images: {len(list(real_dir.glob('*.jpg')))}")
        print(f"  Fake images: {len(list(fake_dir.glob('*.jpg')))}")

    except Exception as e:
        print(f"✗ Error downloading WildDeepfake: {e}")
        print("You may need to: pip install datasets")


def download_faceforensics(output_dir: Path):
    print("\n[2] FaceForensics++ Download Instructions")
    print("="*60)
    print("FaceForensics++ requires manual download and agreement to TOS.")
    print("\nSteps:")
    print("1. Download from Kaggle:")
    print("   curl -L -o ~/Downloads/ff-c23.zip \\")
    print("     https://www.kaggle.com/api/v1/datasets/download/xdxd003/ff-c23")
    print("\n2. Extract the zip file")
    print("\n3. Run frame extraction:")
    print(f"   python src/data/extract_frames.py \\")
    print(f"     --ff_root /path/to/extracted/ff-c23 \\")
    print(f"     --output_root {output_dir} \\")
    print(f"     --num_frames 10 \\")
    print(f"     --num_workers 4")
    print("\nThis will create: {}/ff_c23/{{real,fake}}/".format(output_dir))
    print("="*60)


def check_dataset_status(output_dir: Path):
    print("\n" + "="*60)
    print("DATASET STATUS")
    print("="*60)

    datasets = {
        "wilddeepfake": "WildDeepfake",
        "ff_c23": "FaceForensics++ (c23)",
        "stylegan": "StyleGAN",
        "cifake": "CIFAKE",
    }

    for folder_name, display_name in datasets.items():
        dataset_path = output_dir / folder_name
        if dataset_path.exists():
            real_count = len(list((dataset_path / "real").glob("*.jpg"))) if (dataset_path / "real").exists() else 0
            fake_count = len(list((dataset_path / "fake").glob("*.jpg"))) if (dataset_path / "fake").exists() else 0
            total = real_count + fake_count

            if total > 0:
                print(f"✓ {display_name:25} {total:8,} images (R:{real_count:,} F:{fake_count:,})")
            else:
                print(f"○ {display_name:25} Folder exists but empty")
        else:
            print(f"✗ {display_name:25} Not found")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for RADAR"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for datasets (default: ./data)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["wilddeepfake", "faceforensics", "all", "check"],
        default=["check"],
        help="Which datasets to download (default: check)"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "check" in args.datasets:
        check_dataset_status(output_dir)
        return

    if "all" in args.datasets or "wilddeepfake" in args.datasets:
        download_wilddeepfake(output_dir)

    if "all" in args.datasets or "faceforensics" in args.datasets:
        download_faceforensics(output_dir)

    check_dataset_status(output_dir)

    print("\n✓ Dataset preparation complete!")
    print(f"Data directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
