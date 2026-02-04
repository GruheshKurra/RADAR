#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from utils.logging import print_section, print_complete


DOWNLOAD_SCRIPT_URL = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/download.py"


def download_script(output_path: Path):
    import urllib.request

    script_path = output_path / "download_ff.py"
    if script_path.exists():
        return script_path

    print("Downloading FaceForensics++ download script...")
    urllib.request.urlretrieve(DOWNLOAD_SCRIPT_URL, script_path)
    return script_path


def run_download(script_path: Path, output_dir: Path, sample_only: bool, skip_test: bool):
    cmd = [
        sys.executable, str(script_path),
        "-d", "compressed",
        str(output_dir),
        "--not_mask"
    ]

    if skip_test:
        cmd.append("--not_test")
    if sample_only:
        cmd.append("--sample_only")

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download FaceForensics++ dataset")
    parser.add_argument("--output_dir", type=str, default="./data/faceforensics",
                        help="Output directory")
    parser.add_argument("--sample_only", action="store_true",
                        help="Download only 5 videos per folder (for testing)")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip test split (download only train/val)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "SAMPLE MODE" if args.sample_only else "FULL DOWNLOAD"
    splits = "train/val only" if args.skip_test else "train/val/test"

    print_section(
        "FACEFORENSICS++ DOWNLOAD",
        f"Mode: {mode}\nSplits: {splits}\nOutput: {output_dir.absolute()}\nSize: ~130GB (compressed)"
    )

    script_path = download_script(output_dir)

    try:
        run_download(script_path, output_dir, args.sample_only, args.skip_test)
        print_complete("DOWNLOAD COMPLETE", {"Location": str(output_dir)})

        print("\nNext step:")
        print(f"  python src/data/prepare_faceforensics.py --video_dir {output_dir}")

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
