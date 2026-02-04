#!/usr/bin/env python3
"""
DEPRECATED: Use FaceForensics++ instead (0_download_faceforensics.py)

This script is kept for reference only. WildDeepfake has labeling issues.
For research, use FaceForensics++ which is the standard benchmark.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import os
import io
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import shutil

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from utils.logging import print_section, print_complete, print_result

os.environ['HF_HOME'] = str(Path.cwd() / 'data' / 'hf_cache')
os.environ['HF_DATASETS_CACHE'] = str(Path.cwd() / 'data' / 'hf_cache')

REAL_RATIO = 0.35
FAKE_RATIO = 0.65
SAVE_BATCH_SIZE = 1000
IMAGE_QUALITY = 95
MAX_WORKERS = 2
HEARTBEAT_INTERVAL = 500


def save_image(args):
    img, img_path = args
    try:
        if hasattr(img, 'save'):
            img.save(str(img_path), optimize=False, quality=IMAGE_QUALITY)
        elif isinstance(img, dict) and 'bytes' in img:
            Image.open(io.BytesIO(img['bytes'])).save(str(img_path), optimize=False, quality=IMAGE_QUALITY)
        return img_path
    except Exception:
        return None


def detect_label_from_key(key: str) -> tuple[bool, bool]:
    if not key:
        return False, False

    key_lower = key.lower().replace("\\", "/")
    key_parts = key_lower.split("/")

    for part in key_parts:
        if part == "fake":
            return True, False
        if part == "real":
            return False, True

    return False, False


def detect_label_from_field(label) -> tuple[bool, bool]:
    if label is None:
        return False, False

    if isinstance(label, str):
        label_lower = label.lower().strip()
        is_fake = label_lower in ("fake", "1", "deepfake", "manipulated")
        is_real = label_lower in ("real", "0", "original", "authentic")
        return is_fake, is_real

    if isinstance(label, (int, float)):
        return int(label) == 1, int(label) == 0

    return False, False


def should_skip_sample(is_real: bool, is_fake: bool, total_real: int, total_fake: int,
                       target_real: float, target_fake: float) -> bool:
    if is_real and total_real >= target_real:
        return True
    if is_fake and total_fake >= target_fake:
        return True
    return False


def save_progress(progress_file: Path, stats: dict, split_name: str, idx: int):
    progress = {
        "train_real": stats["train_real"],
        "train_fake": stats["train_fake"],
        "test_real": stats["test_real"],
        "test_fake": stats["test_fake"],
        "last_split": split_name,
        "last_idx": idx,
    }
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def load_progress(progress_file: Path) -> dict:
    if not progress_file.exists():
        return {"train_real": 0, "train_fake": 0, "test_real": 0, "test_fake": 0, "last_split": None, "last_idx": 0}
    with open(progress_file, 'r') as f:
        return json.load(f)


def download_wilddeepfake(output_dir: Path, max_images: int = None):
    limit_msg = f"Limited to {max_images:,} images (pod-safe mode)" if max_images else "~1.16M images (994k train + 165k test)"
    print_section(
        "[1/3] DOWNLOADING WILDDEEPFAKE FROM HUGGINGFACE",
        f"Dataset: {limit_msg}\nSize: Varies based on limit"
    )

    try:
        print(f"\nLoading dataset (streaming mode for pod safety)...")
        print("This prevents memory spikes and freeze issues...\n")
        ds = load_dataset("xingjunm/WildDeepfake", streaming=True)

        wild_dir = output_dir / "wilddeepfake"
        real_dir = wild_dir / "real"
        fake_dir = wild_dir / "fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        progress_file = wild_dir / "progress.json"
        progress = load_progress(progress_file)

        body_lines = []
        if max_images:
            body_lines.append(f"Target: {max_images:,} images with balanced real/fake ratio")
        if progress["last_split"]:
            body_lines.append(f"Resuming from: {progress['last_split']} index {progress['last_idx']}")

        print_section("[2/3] STREAMING DATASET PROCESSING (BALANCED SAMPLING)", "\n".join(body_lines))

        stats = {
            "train_real": progress["train_real"],
            "train_fake": progress["train_fake"],
            "test_real": progress["test_real"],
            "test_fake": progress["test_fake"]
        }
        total_saved = sum(stats.values())
        save_batch = []

        debug_printed = False
        for split_name in ["train", "test"]:
            if progress["last_split"] and split_name < progress["last_split"]:
                continue

            if split_name in ds:
                print(f"\nProcessing {split_name} split...")

                if not debug_printed:
                    try:
                        sample = ds[split_name][0]
                        print(f"\n  Sample structure debug:")
                        print(f"    Available keys: {list(sample.keys())}")
                        for k, v in sample.items():
                            if k not in ["png", "image", "img"]:
                                print(f"    {k}: {repr(v)[:100]}")
                        debug_printed = True
                    except Exception:
                        pass

                if max_images and total_saved >= max_images:
                    print(f"Reached limit of {max_images:,} images, stopping.")
                    break

                target_real = int(max_images * REAL_RATIO) if max_images else float('inf')
                target_fake = int(max_images * FAKE_RATIO) if max_images else float('inf')
                total_real = stats["train_real"] + stats["test_real"]
                total_fake = stats["train_fake"] + stats["test_fake"]

                pbar = tqdm(desc=f"  Saving {split_name}", unit="img")

                for idx, sample in enumerate(ds[split_name]):
                    if progress["last_split"] == split_name and idx <= progress["last_idx"]:
                        continue

                    if idx % HEARTBEAT_INTERVAL == 0:
                        print(f"Heartbeat: split={split_name}, idx={idx}, saved={total_saved}")

                    if max_images and total_saved >= max_images:
                        break

                    try:
                        img = sample.get("png") or sample.get("image") or sample.get("img")

                        is_fake, is_real = detect_label_from_key(sample.get("__key__", ""))

                        if not is_fake and not is_real:
                            label = sample.get("label", sample.get("cls", sample.get("class", None)))
                            is_fake, is_real = detect_label_from_field(label)

                        if img is not None and (is_fake or is_real):
                            total_real = stats["train_real"] + stats["test_real"]
                            total_fake = stats["train_fake"] + stats["test_fake"]

                            if should_skip_sample(is_real, is_fake, total_real, total_fake, target_real, target_fake):
                                continue

                            target_dir = real_dir if is_real else fake_dir
                            img_path = target_dir / f"{split_name}_{idx:07d}.jpg"
                            save_batch.append((img, img_path))

                            if is_real:
                                stats[f"{split_name}_real"] += 1
                            else:
                                stats[f"{split_name}_fake"] += 1

                            if len(save_batch) >= SAVE_BATCH_SIZE:
                                for img, img_path in save_batch:
                                    result = save_image((img, img_path))
                                    if result is not None:
                                        total_saved += 1
                                        pbar.update(1)

                                save_progress(progress_file, stats, split_name, idx)
                                save_batch = []

                                total_real = stats["train_real"] + stats["test_real"]
                                total_fake = stats["train_fake"] + stats["test_fake"]
                                if max_images and (total_real >= target_real and total_fake >= target_fake):
                                    break
                    except Exception:
                        continue

                if save_batch:
                    for img, img_path in save_batch:
                        result = save_image((img, img_path))
                        if result is not None:
                            total_saved += 1
                            pbar.update(1)
                    save_progress(progress_file, stats, split_name, idx)
                    save_batch = []

                pbar.close()

        real_count = len(list(real_dir.glob('*.jpg')))
        fake_count = len(list(fake_dir.glob('*.jpg')))

        summary = {
            "Location": str(wild_dir),
            "Real images": real_count,
            "Fake images": fake_count,
            "Total": real_count + fake_count,
            "Train": f"{stats['train_real']:,} real + {stats['train_fake']:,} fake",
            "Test": f"{stats['test_real']:,} real + {stats['test_fake']:,} fake",
        }
        if max_images:
            summary["Requested limit"] = max_images

        print_complete("WILDDEEPFAKE DOWNLOAD COMPLETE", summary)

        return True

    except Exception as e:
        print(f"\n✗ Error downloading WildDeepfake: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataset_status(output_dir: Path):
    print_section("DATASET STATUS CHECK")

    wild_dir = output_dir / "wilddeepfake"

    if wild_dir.exists():
        print(f"\n✓ WildDeepfake found at: {wild_dir}")
        real_count = len(list((wild_dir / "real").glob("*.jpg"))) if (wild_dir / "real").exists() else 0
        fake_count = len(list((wild_dir / "fake").glob("*.jpg"))) if (wild_dir / "fake").exists() else 0
        total = real_count + fake_count

        if total > 0:
            print_result({"Real images": real_count, "Fake images": fake_count, "Total": total}, "  ")
            print()
            return True
        else:
            print("  ✗ Folder exists but empty\n")
            return False
    else:
        print(f"\n✗ WildDeepfake not found\n")
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

    print_section(
        "RADAR DATASET PREPARATION (MEMORY-EFFICIENT MODE)",
        f"Output directory: {output_dir.absolute()}\nMax images: {args.max_images:,} (prevents memory overflow)"
    )

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
        print_complete(
            "DATASET PREPARATION COMPLETE",
            {"Next step": "python 2_train_model.py --data_dir ./data --output_dir ./outputs"}
        )
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
