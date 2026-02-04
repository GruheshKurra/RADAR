#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import sys
import os
import io
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

os.environ['HF_HOME'] = str(Path.cwd() / 'data' / 'hf_cache')
os.environ['HF_DATASETS_CACHE'] = str(Path.cwd() / 'data' / 'hf_cache')

def save_image(args):
    img, img_path = args
    try:
        if hasattr(img, 'save'):
            img.save(str(img_path), optimize=False, quality=95)
        elif isinstance(img, dict) and 'bytes' in img:
            Image.open(io.BytesIO(img['bytes'])).save(str(img_path), optimize=False, quality=95)
        return img_path
    except:
        return None

def download_wilddeepfake(output_dir: Path, max_images: int = None):
    print("\n" + "="*70)
    print("[1/3] DOWNLOADING WILDDEEPFAKE FROM HUGGINGFACE")
    print("="*70)
    if max_images:
        print(f"Dataset: Limited to {max_images:,} images (memory-efficient mode)")
    else:
        print("Dataset: ~1.16M images (994k train + 165k test)")
    print("Size: Varies based on limit")
    print("="*70)

    try:
        num_proc = max(1, mp.cpu_count() - 2)
        print(f"\nLoading dataset with {num_proc} processes...")
        print("This may take several minutes for large datasets...\n")
        ds = load_dataset("xingjunm/WildDeepfake", num_proc=num_proc, streaming=False)

        wild_dir = output_dir / "wilddeepfake"
        real_dir = wild_dir / "real"
        fake_dir = wild_dir / "fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("[2/3] STREAMING DATASET PROCESSING (BALANCED SAMPLING)")
        print("="*70)
        if max_images:
            print(f"Target: {max_images:,} images with balanced real/fake ratio (35/65)")
        print("="*70)

        stats = {"train_real": 0, "train_fake": 0, "test_real": 0, "test_fake": 0}
        total_saved = 0
        save_batch = []
        batch_size = 1000

        for split_name in ["train", "test"]:
            if split_name in ds:
                split_size = len(ds[split_name])
                print(f"\nProcessing {split_name} split ({split_size:,} images)...")

                if max_images and total_saved >= max_images:
                    print(f"Reached limit of {max_images:,} images, stopping.")
                    break

                target_real = int(max_images * 0.35) if max_images else float('inf')
                target_fake = int(max_images * 0.65) if max_images else float('inf')
                total_real = stats["train_real"] + stats["test_real"]
                total_fake = stats["train_fake"] + stats["test_fake"]

                pbar = tqdm(total=min(split_size, max_images - total_saved if max_images else split_size),
                           desc=f"  Saving {split_name}", unit="img")

                for idx, sample in enumerate(ds[split_name]):
                    if max_images and total_saved >= max_images:
                        break

                    try:
                        # WildDeepfake uses 'png' column for images
                        img = sample.get("png") or sample.get("image") or sample.get("img")

                        # Extract label from __key__ path (e.g., "./1/fake/131/1057" or "./1/real/131/1057")
                        key = sample.get("__key__", "")
                        is_fake = "fake" in key.lower()
                        is_real = "real" in key.lower()

                        # Fallback to explicit label if available
                        if not is_fake and not is_real:
                            label = sample.get("label", "")
                            if isinstance(label, str):
                                is_fake = "fake" in label.lower()
                                is_real = "real" in label.lower()
                            else:
                                is_fake = label == 1
                                is_real = label == 0

                        if img is not None and (is_fake or is_real):
                            total_real = stats["train_real"] + stats["test_real"]
                            total_fake = stats["train_fake"] + stats["test_fake"]

                            # Skip if we've reached the target for this class
                            if is_real and total_real >= target_real:
                                continue
                            if is_fake and total_fake >= target_fake:
                                continue

                            target_dir = real_dir if is_real else fake_dir
                            img_path = target_dir / f"{split_name}_{idx:07d}.jpg"
                            save_batch.append((img, img_path))

                            # Track statistics
                            if is_real:
                                stats[f"{split_name}_real"] += 1
                            else:
                                stats[f"{split_name}_fake"] += 1

                            if len(save_batch) >= batch_size:
                                max_workers = min(16, mp.cpu_count())
                                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                    results = list(executor.map(save_image, save_batch))
                                successful = sum(1 for r in results if r is not None)
                                total_saved += successful
                                pbar.update(successful)
                                save_batch = []

                                total_real = stats["train_real"] + stats["test_real"]
                                total_fake = stats["train_fake"] + stats["test_fake"]
                                if max_images and (total_real >= target_real and total_fake >= target_fake):
                                    break
                    except Exception as e:
                        continue

                if save_batch:
                    max_workers = min(16, mp.cpu_count())
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        results = list(executor.map(save_image, save_batch))
                    successful = sum(1 for r in results if r is not None)
                    total_saved += successful
                    pbar.update(successful)
                    save_batch = []

                pbar.close()

        real_count = len(list(real_dir.glob('*.jpg')))
        fake_count = len(list(fake_dir.glob('*.jpg')))

        print("\n" + "="*70)
        print("✓ WILDDEEPFAKE DOWNLOAD COMPLETE!")
        print("="*70)
        print(f"Location: {wild_dir}")
        print(f"Real images: {real_count:,}")
        print(f"Fake images: {fake_count:,}")
        print(f"Total: {real_count + fake_count:,}")
        if max_images:
            print(f"Requested limit: {max_images:,}")
        print(f"Train: {stats['train_real']:,} real + {stats['train_fake']:,} fake")
        print(f"Test:  {stats['test_real']:,} real + {stats['test_fake']:,} fake")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Error downloading WildDeepfake: {e}")
        import traceback
        traceback.print_exc()


def download_faceforensics(output_dir: Path, workspace_root: Path = None):
    print("\n[2] Downloading FaceForensics++ from Kaggle...")

    if workspace_root is None:
        workspace_root = Path("/workspace") if Path("/workspace").exists() else Path.cwd()

    ff_zip = workspace_root / "ff-c23.zip"
    ff_dir = workspace_root / "ff-c23"

    if not ff_zip.exists() and not ff_dir.exists():
        print("Downloading FF-c23 from Kaggle (this will take several minutes)...")
        try:
            subprocess.run([
                "curl", "-L", "-o", str(ff_zip),
                "https://www.kaggle.com/api/v1/datasets/download/xdxd003/ff-c23"
            ], check=True)
            print(f"✓ Downloaded to {ff_zip}")
        except subprocess.CalledProcessError:
            print("✗ Failed to download FF-c23. Continuing without it...")
            return

    if ff_zip.exists() and not ff_dir.exists():
        print("Unzipping FaceForensics++ (this may take several minutes)...")
        ff_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(["unzip", "-q", str(ff_zip), "-d", str(ff_dir)], check=True)
            print(f"✓ Extracted to {ff_dir}")
        except subprocess.CalledProcessError:
            print("✗ Failed to extract FF-c23")
            return

    if ff_dir.exists():
        print("Extracting frames from videos...")
        from extract_frames import extract_ff_dataset
        num_workers = max(1, mp.cpu_count())
        extract_ff_dataset(ff_dir, output_dir, num_frames=10, num_workers=num_workers)

        print("\nCleaning up to save space...")
        if ff_zip.exists():
            ff_zip.unlink()
            print(f"✓ Removed {ff_zip}")
        if ff_dir.exists():
            shutil.rmtree(ff_dir)
            print(f"✓ Removed {ff_dir}")


def check_dataset_status(output_dir: Path):
    print("\n" + "="*70)
    print("DATASET STATUS CHECK")
    print("="*70)

    datasets = {
        "wilddeepfake": "WildDeepfake",
    }

    total_images = 0
    for folder_name, display_name in datasets.items():
        dataset_path = output_dir / folder_name
        if dataset_path.exists():
            print(f"\nChecking {display_name}...")
            real_count = len(list((dataset_path / "real").glob("*.jpg"))) if (dataset_path / "real").exists() else 0
            fake_count = len(list((dataset_path / "fake").glob("*.jpg"))) if (dataset_path / "fake").exists() else 0
            total = real_count + fake_count
            total_images += total

            if total > 0:
                print(f"  ✓ Status: Ready")
                print(f"  ✓ Location: {dataset_path}")
                print(f"  ✓ Real images: {real_count:,}")
                print(f"  ✓ Fake images: {fake_count:,}")
                print(f"  ✓ Total: {total:,}")
            else:
                print(f"  ○ Status: Folder exists but empty")
        else:
            print(f"\n{display_name}:")
            print(f"  ✗ Status: Not found")
            print(f"  ⓘ Run: python src/data/download_datasets.py --datasets wilddeepfake")

    print("\n" + "="*70)
    if total_images > 0:
        print(f"Total images available: {total_images:,}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for RADAR (memory-efficient)"
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
        default=["wilddeepfake"],
        help="Which datasets to download (default: wilddeepfake)"
    )
    parser.add_argument(
        "--workspace_root",
        type=str,
        default=None,
        help="Workspace root for temporary downloads"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=300000,
        help="Maximum images to download (default: 300000, prevents memory overflow)"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace_root = Path(args.workspace_root) if args.workspace_root else None

    if "check" in args.datasets:
        check_dataset_status(output_dir)
        return

    print("\n" + "="*70)
    print("RADAR DATASET DOWNLOAD (MEMORY-EFFICIENT)")
    print("="*70)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Max images: {args.max_images:,}")
    print("="*70)

    if "all" in args.datasets or "wilddeepfake" in args.datasets:
        download_wilddeepfake(output_dir, max_images=args.max_images)

    if "all" in args.datasets or "faceforensics" in args.datasets:
        download_faceforensics(output_dir, workspace_root)

    check_dataset_status(output_dir)

    print("\n" + "="*70)
    print("✓ DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"Data directory: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  cd src/experiments")
    print("  python run.py --config configs/wilddeepfake.yaml --output ../../outputs")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
