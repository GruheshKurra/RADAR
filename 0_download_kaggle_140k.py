#!/usr/bin/env python3
"""Download 140k Real and Fake Faces from Kaggle."""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from utils.logging import print_section, print_complete


KAGGLE_DATASET = "xhlulu/140k-real-and-fake-faces"


def check_kaggle_setup():
    """Verify Kaggle API is configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("✗ Kaggle API not configured!")
        print("\nSetup steps:")
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def download_dataset(output_dir: Path):
    """Download dataset using Kaggle API."""
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(output_dir)]
    subprocess.run(cmd, check=True)


def extract_file(args):
    """Extract a single file from zip (worker function)."""
    zip_path, member, output_dir = args
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(member, output_dir)
    return member


def unzip_dataset(output_dir: Path, max_workers: int = 16):
    """Unzip dataset using parallel extraction for speed."""
    zip_file = output_dir / "140k-real-and-fake-faces.zip"
    if not zip_file.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_file}")

    print(f"Extracting with {max_workers} parallel workers...")

    # Get list of all files in zip
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        members = zip_ref.namelist()

    total_files = len(members)
    print(f"Total files: {total_files:,}")

    # Parallel extraction
    tasks = [(zip_file, member, output_dir) for member in members]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_file, task) for task in tasks]

        # Progress bar
        with tqdm(total=total_files, desc="Extracting", unit="files") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    zip_file.unlink()
    print("✓ Extraction complete!")


def main():
    parser = argparse.ArgumentParser(description="Download 140k Real and Fake Faces")
    parser.add_argument("--output_dir", type=str, default="./data/kaggle_140k",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section(
        "KAGGLE 140K DOWNLOAD",
        f"Dataset: 140k Real and Fake Faces\nSize: ~2GB\nOutput: {output_dir.absolute()}"
    )

    if not check_kaggle_setup():
        sys.exit(1)

    try:
        print("\nDownloading from Kaggle...")
        download_dataset(output_dir)

        print("\nUnzipping (parallel extraction)...")
        unzip_dataset(output_dir, max_workers=16)

        print_complete(
            "DOWNLOAD COMPLETE",
            {"Location": str(output_dir), "Size": "140,000 images"}
        )

        print("\nNext step:")
        print(f"  python src/data/prepare_kaggle.py --input_dir {output_dir}")

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Install Kaggle: pip install kaggle")
        print("2. Setup API token (see above)")
        print("3. Accept dataset terms: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
        sys.exit(1)


if __name__ == "__main__":
    main()
