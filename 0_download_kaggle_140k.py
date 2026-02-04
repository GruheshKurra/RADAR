#!/usr/bin/env python3
"""Download 140k Real and Fake Faces from Kaggle."""

import argparse
import subprocess
import sys
from pathlib import Path

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


def unzip_dataset(output_dir: Path):
    """Unzip downloaded dataset."""
    zip_file = output_dir / "140k-real-and-fake-faces.zip"
    if not zip_file.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_file}")

    cmd = ["unzip", "-q", str(zip_file), "-d", str(output_dir)]
    subprocess.run(cmd, check=True)
    zip_file.unlink()


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

        print("Unzipping...")
        unzip_dataset(output_dir)

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
