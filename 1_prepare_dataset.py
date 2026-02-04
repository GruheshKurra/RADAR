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
from concurrent.futures import ThreadPoolExecutor
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
            print(f"Target: {max_images:,} images with balanced real/fake ratio")
        print("="*70)

        stats = {"train_real": 0, "train_fake": 0, "test_real": 0, "test_fake": 0}
        total_saved = 0
        save_batch = []
        batch_size = 1000

        debug_printed = False
        for split_name in ["train", "test"]:
            if split_name in ds:
                split_size = len(ds[split_name])
                print(f"\nProcessing {split_name} split ({split_size:,} images)...")

                if not debug_printed:
                    try:
                        sample = ds[split_name][0]
                        print(f"\n  Sample structure debug:")
                        print(f"    Available keys: {list(sample.keys())}")
                        for k, v in sample.items():
                            if k not in ["png", "image", "img"]:
                                print(f"    {k}: {repr(v)[:100]}")
                        debug_printed = True
                    except:
                        pass

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
                        img = sample.get("png") or sample.get("image") or sample.get("img")

                        is_fake = False
                        is_real = False

                        key = sample.get("__key__", "")
                        if key:
                            key_lower = key.lower()
                            key_parts = key_lower.replace("\\", "/").split("/")
                            for part in key_parts:
                                if part == "fake":
                                    is_fake = True
                                    break
                                elif part == "real":
                                    is_real = True
                                    break

                        if not is_fake and not is_real:
                            label = sample.get("label", sample.get("cls", sample.get("class", None)))
                            if label is not None:
                                if isinstance(label, str):
                                    label_lower = label.lower().strip()
                                    is_fake = label_lower in ("fake", "1", "deepfake", "manipulated")
                                    is_real = label_lower in ("real", "0", "original", "authentic")
                                elif isinstance(label, (int, float)):
                                    is_fake = int(label) == 1
                                    is_real = int(label) == 0

                        if img is not None and (is_fake or is_real):
                            total_real = stats["train_real"] + stats["test_real"]
                            total_fake = stats["train_fake"] + stats["test_fake"]

                            if is_real and total_real >= target_real:
                                continue
                            if is_fake and total_fake >= target_fake:
                                continue

                            target_dir = real_dir if is_real else fake_dir
                            img_path = target_dir / f"{split_name}_{idx:07d}.jpg"
                            save_batch.append((img, img_path))

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
                    except:
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

        return True

    except Exception as e:
        print(f"\n✗ Error downloading WildDeepfake: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset_status(output_dir: Path):
    print("\n" + "="*70)
    print("DATASET STATUS CHECK")
    print("="*70)

    wild_dir = output_dir / "wilddeepfake"

    if wild_dir.exists():
        print(f"\n✓ WildDeepfake found at: {wild_dir}")
        real_count = len(list((wild_dir / "real").glob("*.jpg"))) if (wild_dir / "real").exists() else 0
        fake_count = len(list((wild_dir / "fake").glob("*.jpg"))) if (wild_dir / "fake").exists() else 0
        total = real_count + fake_count

        if total > 0:
            print(f"  Real images: {real_count:,}")
            print(f"  Fake images: {fake_count:,}")
            print(f"  Total: {total:,}")
            print("="*70 + "\n")
            return True
        else:
            print("  ✗ Folder exists but empty")
            print("="*70 + "\n")
            return False
    else:
        print(f"\n✗ WildDeepfake not found")
        print("="*70 + "\n")
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare WildDeepfake dataset for RADAR training")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for datasets")
    parser.add_argument("--check_only", action="store_true", help="Only check dataset status without downloading")
    parser.add_argument("--max_images", type=int, default=300000, help="Maximum images to download (default: 300000)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if dataset exists")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("RADAR DATASET PREPARATION (MEMORY-EFFICIENT MODE)")
    print("="*70)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Max images: {args.max_images:,} (prevents memory overflow)")
    print("="*70)

    if args.check_only:
        exists = check_dataset_status(output_dir)
        sys.exit(0 if exists else 1)

    existing = check_dataset_status(output_dir)
    if existing and not args.force:
        print("Dataset already exists. Use --force to re-download.\n")
        sys.exit(0)

    success = download_wilddeepfake(output_dir, max_images=args.max_images)

    if success:
        check_dataset_status(output_dir)
        print("\n" + "="*70)
        print("✓ DATASET PREPARATION COMPLETE!")
        print("="*70)
        print("\nNext step:")
        print("  python 2_train_model.py --data_dir ./data --output_dir ./outputs")
        print("="*70 + "\n")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
