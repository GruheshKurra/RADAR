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

def download_wilddeepfake(output_dir: Path):
    print("\n[1] Downloading WildDeepfake from HuggingFace...")

    try:
        num_proc = max(1, mp.cpu_count() - 2)
        ds = load_dataset("xingjunm/WildDeepfake", num_proc=num_proc)

        wild_dir = output_dir / "wilddeepfake"
        real_dir = wild_dir / "real"
        fake_dir = wild_dir / "fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        print("Organizing WildDeepfake dataset...")

        save_tasks = []

        for split_name in ["train", "test"]:
            if split_name in ds:
                for idx, sample in enumerate(ds[split_name]):
                    try:
                        img = sample.get("image") or sample.get("img") or sample.get("picture")
                        label = sample.get("label") or sample.get("target") or 0
                        if img is not None:
                            target_dir = real_dir if label == 0 else fake_dir
                            img_path = target_dir / f"{split_name}_{idx:06d}.jpg"
                            save_tasks.append((img, img_path))
                    except:
                        continue

        max_workers = min(32, mp.cpu_count() * 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(save_image, save_tasks), total=len(save_tasks), desc="Saving images"))

        successful = sum(1 for r in results if r is not None)
        print(f"✓ WildDeepfake saved to: {wild_dir}")
        print(f"  Real images: {len(list(real_dir.glob('*.jpg')))}")
        print(f"  Fake images: {len(list(fake_dir.glob('*.jpg')))}")
        print(f"  Success rate: {successful}/{len(save_tasks)}")

    except Exception as e:
        print(f"✗ Error downloading WildDeepfake: {e}")
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
    print("\n" + "="*60)
    print("DATASET STATUS")
    print("="*60)

    datasets = {
        "wilddeepfake": "WildDeepfake",
        "ff_c23": "FaceForensics++ (c23)",
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
    parser.add_argument(
        "--workspace_root",
        type=str,
        default=None,
        help="Workspace root for temporary downloads"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace_root = Path(args.workspace_root) if args.workspace_root else None

    if "check" in args.datasets:
        check_dataset_status(output_dir)
        return

    if "all" in args.datasets or "wilddeepfake" in args.datasets:
        download_wilddeepfake(output_dir)

    if "all" in args.datasets or "faceforensics" in args.datasets:
        download_faceforensics(output_dir, workspace_root)

    check_dataset_status(output_dir)

    print("\n✓ Dataset preparation complete!")
    print(f"Data directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
