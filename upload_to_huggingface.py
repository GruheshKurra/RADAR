#!/usr/bin/env python3
"""
Upload mixed deepfake dataset to Hugging Face Hub.
Supports large datasets with proper chunking and metadata.
"""

import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from tqdm import tqdm


def upload_dataset_to_hf(
    dataset_dir: Path,
    repo_name: str,
    hf_token: str = None,
    private: bool = True,
    commit_message: str = "Upload mixed deepfake dataset"
):
    """
    Upload dataset to Hugging Face Hub.

    Args:
        dataset_dir: Local dataset directory (with train/val/test structure)
        repo_name: HF repo name (e.g., "username/multi-deepfake-v1")
        hf_token: HF API token (or set HF_TOKEN env var)
        private: Whether to make repo private
        commit_message: Commit message
    """

    print("\n" + "="*70)
    print("UPLOADING TO HUGGING FACE HUB")
    print("="*70)

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    # Load metadata
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"\nDataset: {metadata.get('dataset_name', 'unknown')}")
        print(f"Total images: {sum(sum(split.values()) for split in metadata['actual_totals'].values()):,}")

    # Create API instance
    api = HfApi(token=hf_token)

    # Create repo (if doesn't exist)
    print(f"\nCreating/accessing repo: {repo_name}")
    try:
        create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            private=private,
            token=hf_token,
            exist_ok=True
        )
        print(f"✓ Repo ready: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        print(f"⚠️  Repo creation note: {e}")

    # Create README
    readme_content = f"""---
license: cc-by-nc-4.0
task_categories:
- image-classification
- zero-shot-image-classification
tags:
- deepfake-detection
- face-forensics
- synthetic-media
pretty_name: {metadata.get('dataset_name', 'Multi-Source Deepfake Dataset')}
size_categories:
- 100K<n<1M
---

# {metadata.get('dataset_name', 'Multi-Source Deepfake Dataset')}

## Dataset Description

Mixed deepfake detection dataset combining multiple high-quality sources for challenging evaluation.

### Dataset Statistics

```json
{json.dumps(metadata.get('actual_totals', {}), indent=2)}
```

### Sources

{chr(10).join(f"- **{name}**: {info['train_real'] + info['train_fake']:,} train, {info['val_real'] + info['val_fake']:,} val, {info['test_real'] + info['test_fake']:,} test"
              for name, info in metadata.get('sources', {}).items())}

### Structure

```
dataset/
├── train/
│   ├── real/  # Real faces
│   └── fake/  # Synthetic faces
├── val/
│   ├── real/
│   └── fake/
├── test/
│   ├── real/
│   └── fake/
└── metadata.json
```

### Usage

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("{repo_name}")

# Load specific split
train_data = load_dataset("{repo_name}", split="train")
```

### Citation

If you use this dataset, please cite the original sources:

- FaceForensics++: [citation]
- Celeb-DF: [citation]
- DFDC: [citation]

### License

CC-BY-NC 4.0 (Non-commercial use only)
"""

    readme_path = dataset_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"✓ Created README.md")

    # Upload folder (handles large files automatically)
    print(f"\nUploading dataset folder...")
    print(f"This may take a while depending on dataset size...")

    try:
        api.upload_folder(
            folder_path=str(dataset_dir),
            repo_id=repo_name,
            repo_type="dataset",
            commit_message=commit_message,
            token=hf_token,
        )

        print("\n" + "="*70)
        print("✓ UPLOAD COMPLETE")
        print("="*70)
        print(f"\nDataset URL: https://huggingface.co/datasets/{repo_name}")
        print(f"\nTo download on remote server:")
        print(f"  from datasets import load_dataset")
        print(f"  dataset = load_dataset('{repo_name}')")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check HF token: huggingface-cli login")
        print("2. Verify repo permissions")
        print("3. For large datasets, ensure stable connection")


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Local dataset directory")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="HF repo name (e.g., username/dataset-name)")
    parser.add_argument("--token", type=str, default=None,
                        help="HF API token (or use HF_TOKEN env var)")
    parser.add_argument("--public", action="store_true",
                        help="Make repo public (default: private)")
    parser.add_argument("--commit_message", type=str,
                        default="Upload mixed deepfake dataset",
                        help="Commit message")
    args = parser.parse_args()

    upload_dataset_to_hf(
        dataset_dir=Path(args.dataset_dir),
        repo_name=args.repo_name,
        hf_token=args.token,
        private=not args.public,
        commit_message=args.commit_message
    )


if __name__ == "__main__":
    main()
