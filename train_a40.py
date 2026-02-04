#!/usr/bin/env python3
"""
Optimized training script for A40 GPU (48GB VRAM)
Automatically detects and uses optimal settings for available hardware.
"""

import argparse
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / 'src'))


def get_optimal_config(config_path: Path = None):
    """Load config or use A40 optimal defaults."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    # A40 optimal defaults
    return {
        "experiment_name": "radar_a40_optimal",
        "img_size": 224,
        "batch_size": 128,
        "num_epochs": 30,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "num_workers": 8,
        "prefetch_factor": 4,
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.1,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "seed": 42,
        "early_stopping_patience": 10,
        "lambda_main": 1.0,
        "lambda_branch": 0.3,
        "lambda_orthogonal": 0.1,
        "lambda_deep_supervision": 0.05,
        "label_smoothing": 0.1,
        "orthogonality_margin": 0.1,
        "source_domain": "faceforensics"
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train RADAR on A40 GPU with optimal settings"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--config", type=str, default="./configs/a40_optimal.yaml",
                        help="Config file (optional)")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast config (50%% data, 10 epochs)")
    args = parser.parse_args()

    # Load config
    if args.fast:
        config_path = Path("./configs/a40_fast.yaml")
        print("🚀 Fast mode: 50% data, 10 epochs, batch_size=256")
    else:
        config_path = Path(args.config)
        print("⚡ Optimal mode: Full data, 30 epochs, batch_size=128")

    config = get_optimal_config(config_path if config_path.exists() else None)

    # Update paths from CLI
    config["data_dir"] = args.data_dir

    # Import and run training
    from pathlib import Path as P
    import subprocess

    script = "2_train_model_fast.py" if args.fast else "2_train_model.py"

    cmd = [
        sys.executable, script,
        "--data_dir", config["data_dir"],
        "--output_dir", args.output_dir,
        "--batch_size", str(config["batch_size"]),
        "--num_epochs", str(config["num_epochs"]),
        "--learning_rate", str(config["learning_rate"])
    ]

    print(f"\n🎯 Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
