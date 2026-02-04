import torch
from torch.utils.data import DataLoader
import yaml
import json
from pathlib import Path
import argparse
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))

os.environ['TORCH_HOME'] = str(Path(__file__).parent.parent.parent / 'data' / 'torch_cache')
os.environ['HF_HOME'] = str(Path(__file__).parent.parent.parent / 'data' / 'hf_cache')
os.environ['TIMM_CACHE_DIR'] = str(Path(__file__).parent.parent.parent / 'data' / 'torch_cache' / 'hub')

from method import RADAR, RADARConfig
from data.dataset import DeepfakeDataset, get_train_transforms, get_val_transforms
from data.splits import load_domain_data, create_stratified_split
from experiments.train import train_model, evaluate
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="./outputs")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    data_dir = Path(config["data_dir"])
    images, labels = load_domain_data(data_dir, config["source_domain"])

    train_images, train_labels, val_images, val_labels, test_images, test_labels = create_stratified_split(
        images, labels, config["train_ratio"], config["val_ratio"], config["seed"]
    )

    train_dataset = DeepfakeDataset(
        train_images, train_labels,
        get_train_transforms(config["img_size"]),
        config.get("preprocess_dir"),
        validate_cache=True
    )

    val_dataset = DeepfakeDataset(
        val_images, val_labels,
        get_val_transforms(config["img_size"]),
        config.get("preprocess_dir"),
        validate_cache=False
    )

    test_dataset = DeepfakeDataset(
        test_images, test_labels,
        get_val_transforms(config["img_size"]),
        config.get("preprocess_dir"),
        validate_cache=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                             shuffle=True, num_workers=config["num_workers"],
                             pin_memory=True, drop_last=True,
                             persistent_workers=config["num_workers"] > 0,
                             prefetch_factor=4 if config["num_workers"] > 0 else None,
                             worker_init_fn=lambda x: np.random.seed(config["seed"] + x))

    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            shuffle=False, num_workers=config["num_workers"],
                            pin_memory=True,
                            persistent_workers=config["num_workers"] > 0,
                            prefetch_factor=4 if config["num_workers"] > 0 else None,
                            worker_init_fn=lambda x: np.random.seed(config["seed"] + x))

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=config["num_workers"],
                             pin_memory=True,
                             persistent_workers=config["num_workers"] > 0,
                             prefetch_factor=4 if config["num_workers"] > 0 else None,
                             worker_init_fn=lambda x: np.random.seed(config["seed"] + x))

    model_config = RADARConfig(**{k: v for k, v in config.items() if k in RADARConfig.__annotations__})
    model = RADAR(model_config).to(device)

    print(f"Training {config['experiment_name']}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    history, best_auc = train_model(model, train_loader, val_loader, config, device, output_dir)

    checkpoint_path = output_dir / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics = evaluate(model, test_loader, device,
                           use_badm=config.get("use_badm", True),
                           use_aadm=config.get("use_aadm", True))
    print(f"\nTest Set Results:")
    print(f"AUC (gated): {test_metrics['auc']:.4f}")
    print(f"AUC (reasoning-only): {test_metrics['reasoning_auc']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Gating α: {test_metrics['gating_alpha']:.4f}")

    results = {
        "history": history,
        "best_val_auc": best_auc,
        "test_metrics": {
            "auc": float(test_metrics["auc"]),
            "reasoning_auc": float(test_metrics["reasoning_auc"]),
            "accuracy": float(test_metrics["accuracy"]),
            "reasoning_accuracy": float(test_metrics["reasoning_accuracy"]),
            "gating_alpha": float(test_metrics["gating_alpha"]),
            "convergence_delta": float(test_metrics["convergence_delta"]),
        }
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}, Test AUC: {test_metrics['auc']:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
