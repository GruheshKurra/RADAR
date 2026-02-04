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


def download_wilddeepfake(output_dir: Path):
    print("\n" + "="*70)
    print("[1/3] DOWNLOADING WILDDEEPFAKE FROM HUGGINGFACE")
    print("="*70)
    print("Dataset: ~1.16M images (994k train + 165k test)")
    print("Size: ~10GB")
    print("="*70)

    try:
        num_proc = max(1, mp.cpu_count() - 2)
        print(f"\nLoading dataset with {num_proc} processes...")
        print("This may take several minutes for large datasets...\n")
        ds = load_dataset("xingjunm/WildDeepfake", num_proc=num_proc)

        wild_dir = output_dir / "wilddeepfake"
        real_dir = wild_dir / "real"
        fake_dir = wild_dir / "fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("[2/3] ORGANIZING DATASET INTO REAL/FAKE FOLDERS")
        print("="*70)

        save_tasks = []
        stats = {"train_real": 0, "train_fake": 0, "test_real": 0, "test_fake": 0}

        for split_name in ["train", "test"]:
            if split_name in ds:
                split_size = len(ds[split_name])
                print(f"\nProcessing {split_name} split ({split_size:,} images)...")

                for idx, sample in enumerate(tqdm(ds[split_name], desc=f"  Organizing {split_name}", unit="img")):
                    try:
                        img = sample.get("png") or sample.get("image") or sample.get("img")

                        key = sample.get("__key__", "")
                        is_fake = "fake" in key.lower()
                        is_real = "real" in key.lower()

                        if not is_fake and not is_real:
                            label = sample.get("label", "")
                            if isinstance(label, str):
                                is_fake = "fake" in label.lower()
                                is_real = "real" in label.lower()
                            else:
                                is_fake = label == 1
                                is_real = label == 0

                        if img is not None and (is_fake or is_real):
                            target_dir = real_dir if is_real else fake_dir
                            img_path = target_dir / f"{split_name}_{idx:07d}.jpg"
                            save_tasks.append((img, img_path))

                            if is_real:
                                stats[f"{split_name}_real"] += 1
                            else:
                                stats[f"{split_name}_fake"] += 1
                    except:
                        continue

        print("\n" + "="*70)
        print(f"[3/3] SAVING {len(save_tasks):,} IMAGES TO DISK")
        print("="*70)
        print(f"Train: {stats['train_real']:,} real + {stats['train_fake']:,} fake")
        print(f"Test:  {stats['test_real']:,} real + {stats['test_fake']:,} fake")

        max_workers = min(32, mp.cpu_count() * 2)
        print(f"\nUsing {max_workers} parallel workers for maximum speed...")
        print("Estimated time: 5-15 minutes depending on your system\n")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(save_image, save_tasks),
                total=len(save_tasks),
                desc="  Saving images",
                unit="img",
                ncols=80,
                smoothing=0.1
            ))

        successful = sum(1 for r in results if r is not None)
        real_count = len(list(real_dir.glob('*.jpg')))
        fake_count = len(list(fake_dir.glob('*.jpg')))

        print("\n" + "="*70)
        print("✓ WILDDEEPFAKE DOWNLOAD COMPLETE!")
        print("="*70)
        print(f"Location: {wild_dir}")
        print(f"Real images: {real_count:,}")
        print(f"Fake images: {fake_count:,}")
        print(f"Total: {real_count + fake_count:,}")
        print(f"Success rate: {successful}/{len(save_tasks)} ({100*successful/len(save_tasks):.1f}%)")
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("RADAR DATASET PREPARATION")
    print("="*70)
    print(f"Output directory: {output_dir.absolute()}")
    print("="*70)

    if args.check_only:
        exists = check_dataset_status(output_dir)
        sys.exit(0 if exists else 1)

    existing = check_dataset_status(output_dir)
    if existing:
        print("Dataset already exists. Use --force to re-download.\n")
        sys.exit(0)

    success = download_wilddeepfake(output_dir)

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
