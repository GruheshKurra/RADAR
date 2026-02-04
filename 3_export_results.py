#!/usr/bin/env python3

import argparse
import shutil
import json
from pathlib import Path
from datetime import datetime
import zipfile
from tqdm import tqdm
import sys


def create_results_package(results_dir: Path, output_path: Path):
    print("\n" + "="*70)
    print("CREATING RESULTS PACKAGE")
    print("="*70)
    print(f"Source: {results_dir}")
    print(f"Output: {output_path}")
    print("="*70 + "\n")

    if not results_dir.exists():
        print(f"✗ Error: Results directory not found: {results_dir}")
        return False

    required_files = ["best.pth", "config.json", "metrics.json"]
    missing_files = [f for f in required_files if not (results_dir / f).exists()]

    if missing_files:
        print(f"✗ Error: Missing required files: {', '.join(missing_files)}")
        return False

    with open(results_dir / "metrics.json", 'r') as f:
        metrics = json.load(f)

    print("Collecting files...")
    files_to_zip = []

    for item in results_dir.rglob("*"):
        if item.is_file():
            files_to_zip.append(item)

    print(f"Found {len(files_to_zip)} files to package\n")

    print("Creating zip archive...")
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for file_path in tqdm(files_to_zip, desc="Compressing", unit="file"):
            arcname = file_path.relative_to(results_dir.parent)
            zipf.write(file_path, arcname)

        readme_content = f"""RADAR Training Results
{'='*70}

Experiment: {results_dir.name}
Timestamp: {metrics.get('timestamp', 'N/A')}

Results:
--------
Best Validation AUC: {metrics.get('best_val_auc', 'N/A'):.4f}
Test AUC: {metrics.get('test_metrics', {}).get('auc', 'N/A'):.4f}
Test Accuracy: {metrics.get('test_metrics', {}).get('accuracy', 'N/A'):.4f}

Contents:
---------
- best.pth: Best model checkpoint
- config.json: Training configuration
- metrics.json: Training history and test results

To load the model:
------------------
import torch
checkpoint = torch.load('best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
"""
        zipf.writestr(f"{results_dir.name}/README.txt", readme_content)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print("\n" + "="*70)
    print("✓ RESULTS PACKAGE CREATED SUCCESSFULLY")
    print("="*70)
    print(f"File: {output_path}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Files packaged: {len(files_to_zip)}")
    print("\nSummary:")
    print(f"  Best Val AUC: {metrics.get('best_val_auc', 'N/A'):.4f}")
    print(f"  Test AUC: {metrics.get('test_metrics', {}).get('auc', 'N/A'):.4f}")
    print(f"  Test Accuracy: {metrics.get('test_metrics', {}).get('accuracy', 'N/A'):.4f}")
    print("="*70 + "\n")

    return True


def create_download_script(zip_path: Path, output_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = f"download_results_{timestamp}.sh"
    script_path = output_dir / script_name

    script_content = f"""#!/bin/bash

ZIP_FILE="{zip_path.name}"
DOWNLOAD_DIR="$HOME/Downloads"

echo "========================================"
echo "RADAR Results Download Script"
echo "========================================"
echo "This script will download the results to: $DOWNLOAD_DIR"
echo ""

if [ ! -f "$ZIP_FILE" ]; then
    echo "✗ Error: Results file not found: $ZIP_FILE"
    exit 1
fi

mkdir -p "$DOWNLOAD_DIR"

echo "Copying results package..."
cp "$ZIP_FILE" "$DOWNLOAD_DIR/"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ DOWNLOAD COMPLETE"
    echo "========================================"
    echo "Location: $DOWNLOAD_DIR/$ZIP_FILE"
    echo "Size: $(du -h "$DOWNLOAD_DIR/$ZIP_FILE" | cut -f1)"
    echo ""
    echo "To extract:"
    echo "  cd $DOWNLOAD_DIR"
    echo "  unzip $ZIP_FILE"
    echo "========================================"
else
    echo "✗ Error: Failed to copy file"
    exit 1
fi
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    script_path.chmod(0o755)

    print(f"\n✓ Download script created: {script_path}")
    print(f"  Run: bash {script_path.name}")

    return script_path


def main():
    parser = argparse.ArgumentParser(description="Export and package RADAR training results")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to results directory")
    parser.add_argument("--output_dir", type=str, default="./exports", help="Output directory for zip file")
    parser.add_argument("--create_download_script", action="store_true", help="Create download helper script")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"{results_dir.name}_{timestamp}.zip"
    zip_path = output_dir / zip_filename

    success = create_results_package(results_dir, zip_path)

    if success:
        if args.create_download_script:
            create_download_script(zip_path, output_dir)

        print("Results are ready for download!")
        print("\nOptions to download:")
        print(f"  1. Direct copy: cp {zip_path} ~/Downloads/")
        print(f"  2. SCP: scp {zip_path} user@local:~/Downloads/")
        print(f"  3. Use download script (if created)")
        print("")
        sys.exit(0)
    else:
        print("\n✗ Failed to create results package")
        sys.exit(1)


if __name__ == "__main__":
    main()
