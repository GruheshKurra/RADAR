#!/usr/bin/env python3
"""
Download dataset from Hugging Face Hub to local directory.
Optimized for RunPod/SSH servers with proper structure.
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import io
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def save_image_worker(args):
    """Worker function to save a single image."""
    idx, sample, output_dir, split, label = args
    try:
        img = sample['image']

        # Handle different image formats
        if isinstance(img, Image.Image):
            pass  # Already PIL Image
        elif isinstance(img, dict) and 'bytes' in img:
            img = Image.open(io.BytesIO(img['bytes']))
        else:
            return False

        # Create output path
        class_name = "real" if label == 0 else "fake"
        output_subdir = output_dir / split / class_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Save with index in filename
        output_path = output_subdir / f"{idx:08d}.jpg"
        img.save(output_path, quality=95)
        return True

    except Exception as e:
        print(f"Error saving image {idx}: {e}")
        return False


def download_dataset_from_hf(
    repo_name: str,
    output_dir: Path,
    hf_token: str = None,
    num_workers: int = 4
):
    """
    Download dataset from Hugging Face and save in standard structure.

    Args:
        repo_name: HF repo name (e.g., "username/dataset-name")
        output_dir: Local output directory
        hf_token: HF API token (if private repo)
        num_workers: Parallel workers for saving
    """

    print("\n" + "="*70)
    print("DOWNLOADING FROM HUGGING FACE HUB")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRepo: {repo_name}")
    print(f"Output: {output_dir}")
    print(f"Workers: {num_workers}")

    # Load dataset
    print("\nLoading dataset from Hugging Face...")
    print("(This may take a few minutes for large datasets)")

    try:
        dataset = load_dataset(
            repo_name,
            token=hf_token,
            cache_dir=str(output_dir / ".cache")
        )
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check repo name format: username/dataset-name")
        print("2. For private repos: huggingface-cli login")
        print("3. Verify internet connection")
        return

    print(f"✓ Dataset loaded")

    # Process each split
    stats = {}
    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            continue

        split = dataset[split_name]
        print(f"\n{'='*70}")
        print(f"Processing split: {split_name}")
        print(f"{'='*70}")
        print(f"Total samples: {len(split):,}")

        # Map split name
        output_split_name = "val" if split_name == "validation" else split_name

        # Count by class
        labels = split['label'] if 'label' in split.features else [0] * len(split)
        real_count = sum(1 for l in labels if l == 0)
        fake_count = sum(1 for l in labels if l == 1)

        print(f"  Real: {real_count:,}")
        print(f"  Fake: {fake_count:,}")

        # Save images with progress bar
        print(f"\nSaving images to {output_dir / output_split_name}/...")

        # Prepare arguments for workers
        save_args = []
        for idx, sample in enumerate(split):
            label = sample.get('label', 0)
            save_args.append((idx, sample, output_dir, output_split_name, label))

        # Parallel processing
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(save_image_worker, save_args),
                total=len(save_args),
                desc=f"  {split_name}"
            ))

        saved_count = sum(results)
        print(f"✓ Saved {saved_count:,} / {len(split):,} images")

        stats[output_split_name] = {
            "real": real_count,
            "fake": fake_count,
            "total": len(split),
            "saved": saved_count
        }

    # Save metadata
    metadata = {
        "source": repo_name,
        "output_dir": str(output_dir),
        "splits": stats
    }

    metadata_path = output_dir / "download_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*70)
    print("✓ DOWNLOAD COMPLETE")
    print("="*70)

    print("\nFinal Statistics:")
    for split_name, split_stats in stats.items():
        print(f"  {split_name}: {split_stats['saved']:,} images "
              f"({split_stats['real']:,} real, {split_stats['fake']:,} fake)")

    print(f"\nDataset ready for training:")
    print(f"  python 2_train_model.py --data_dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download dataset from Hugging Face")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="HF repo name (e.g., username/dataset-name)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Local output directory")
    parser.add_argument("--token", type=str, default=None,
                        help="HF API token (for private repos)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()

    download_dataset_from_hf(
        repo_name=args.repo_name,
        output_dir=Path(args.output_dir),
        hf_token=args.token,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
