#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader, Subset
import argparse
import sys
import os
from pathlib import Path
import random
import numpy as np
import json
import yaml
import time
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp

sys.path.append(str(Path(__file__).parent / 'src'))

os.environ['TORCH_HOME'] = str(Path(__file__).parent / 'data' / 'torch_cache')
os.environ['HF_HOME'] = str(Path(__file__).parent / 'data' / 'hf_cache')
os.environ['TIMM_CACHE_DIR'] = str(Path(__file__).parent / 'data' / 'torch_cache' / 'hub')

from method import RADAR, RADARConfig, RADARLoss, LossConfig
from data.dataset import DeepfakeDataset, get_train_transforms, get_val_transforms
from data.splits import load_domain_data, create_stratified_split, load_presplit_data, is_presplit_dataset
from experiments.train import train_epoch as _train_epoch, evaluate as _evaluate
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, accuracy_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, scheduler, loss_fn, scaler, device, gradient_accumulation_steps):
    config = {"use_badm": True, "use_aadm": True}
    losses, skipped, _ = _train_epoch(model, loader, optimizer, scheduler, loss_fn, scaler, device, gradient_accumulation_steps, config)
    return losses, skipped


def evaluate(model, loader, device):
    metrics = _evaluate(model, loader, device, use_badm=True, use_aadm=True)
    return {"auc": metrics["auc"], "accuracy": metrics["accuracy"]}


def train_model(model, train_loader, val_loader, config, device, output_dir):
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    total_steps = len(train_loader) * config["num_epochs"] // config["gradient_accumulation_steps"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=total_steps,
        pct_start=config["warmup_ratio"],
        anneal_strategy='cos'
    )

    loss_config = LossConfig(
        lambda_main=config["lambda_main"],
        lambda_branch=config["lambda_branch"],
        lambda_orthogonal=config["lambda_orthogonal"],
        lambda_deep_supervision=config["lambda_deep_supervision"],
        label_smoothing=config["label_smoothing"],
        orthogonality_margin=config["orthogonality_margin"]
    )
    loss_fn = RADARLoss(loss_config)
    scaler = GradScaler(device)

    history = {"train_loss": [], "val_auc": [], "val_acc": []}
    best_auc = 0.0
    patience_counter = 0

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    for epoch in range(config["num_epochs"]):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 70)

        train_losses, skipped = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, scaler, device, config["gradient_accumulation_steps"]
        )

        val_metrics = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"  Val AUC: {val_metrics['auc']:.4f}")
        print(f"  Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Skipped batches: {skipped}")

        history["train_loss"].append(train_losses['total'])
        history["val_auc"].append(val_metrics['auc'])
        history["val_acc"].append(val_metrics['accuracy'])

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "config": config
            }
            torch.save(checkpoint, output_dir / "best.pth")
            print(f"  ✓ New best model saved (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config['early_stopping_patience']}")

        if patience_counter >= config["early_stopping_patience"]:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    return history, best_auc


def main():
    parser = argparse.ArgumentParser(description="Train RADAR model (FAST mode with dataset subset)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="radar_wilddeepfake_fast", help="Experiment name")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset_ratio", type=float, default=0.2, help="Use subset of data (0.2 = 20%)")
    args = parser.parse_args()

    gpu_count = torch.cuda.device_count()
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        optimal_batch = args.batch_size
        optimal_workers = min(8, mp.cpu_count() - 1)
    else:
        optimal_batch = 64
        optimal_workers = 4

    config = {
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "source_domain": "wilddeepfake",
        "subset_ratio": args.subset_ratio,
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 384,
        "evidence_dim": 64,
        "reasoning_iterations": 3,
        "reasoning_heads": 4,
        "fft_size": 112,
        "dropout": 0.1,
        "batch_size": optimal_batch,
        "num_epochs": args.num_epochs,
        "early_stopping_patience": 7,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 1,
        "num_workers": optimal_workers,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "lambda_main": 1.0,
        "lambda_branch": 0.3,
        "lambda_orthogonal": 0.1,
        "lambda_deep_supervision": 0.05,
        "label_smoothing": 0.1,
        "orthogonality_margin": 0.1,
    }

    print("\n" + "="*70)
    print("RADAR FAST TRAINING PIPELINE")
    print("="*70)
    print(f"Mode: FAST (using {config['subset_ratio']*100:.0f}% of dataset)")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Workers: {config['num_workers']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['num_epochs']}")
    print("="*70)

    set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {gpu_mem_gb:.1f} GB")

    output_dir = Path(args.output_dir) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)

    data_dir = Path(config["data_dir"])

    if is_presplit_dataset(data_dir):
        print(f"Detected pre-split dataset structure")
        train_images, train_labels = load_presplit_data(data_dir, "train")
        val_images, val_labels = load_presplit_data(data_dir, "val")

        if (data_dir / "test").exists():
            test_images, test_labels = load_presplit_data(data_dir, "test")
        else:
            test_images, test_labels = val_images, val_labels

        total_images = len(train_images) + len(val_images)
        subset_size = int(total_images * config["subset_ratio"])

        if config["subset_ratio"] < 1.0:
            indices = list(range(len(train_images)))
            random.Random(config["seed"]).shuffle(indices)
            train_subset_size = int(len(train_images) * config["subset_ratio"])
            train_images = [train_images[i] for i in indices[:train_subset_size]]
            train_labels = [train_labels[i] for i in indices[:train_subset_size]]

            val_subset_size = int(len(val_images) * config["subset_ratio"])
            val_images = val_images[:val_subset_size]
            val_labels = val_labels[:val_subset_size]

            print(f"Using subset: train={len(train_images):,}, val={len(val_images):,}")
    else:
        print(f"Using single-domain structure with stratified split")
        images, labels = load_domain_data(data_dir, config["source_domain"])

        total_images = len(images)
        subset_size = int(total_images * config["subset_ratio"])

        indices = list(range(total_images))
        random.Random(config["seed"]).shuffle(indices)
        subset_indices = indices[:subset_size]

        images_subset = [images[i] for i in subset_indices]
        labels_subset = [labels[i] for i in subset_indices]

        print(f"Total dataset: {total_images:,} images")
        print(f"Using subset: {len(images_subset):,} images ({config['subset_ratio']*100:.0f}%)")

        train_images, train_labels, val_images, val_labels, test_images, test_labels = create_stratified_split(
            images_subset, labels_subset, config["train_ratio"], config["val_ratio"], config["seed"]
        )

    print(f"Train samples: {len(train_images):,}")
    print(f"Val samples: {len(val_images):,}")
    print(f"Test samples: {len(test_images):,}")

    train_dataset = DeepfakeDataset(
        train_images, train_labels,
        get_train_transforms(config["img_size"]),
        preprocess_dir=None,
        validate_cache=False
    )

    val_dataset = DeepfakeDataset(
        val_images, val_labels,
        get_val_transforms(config["img_size"]),
        preprocess_dir=None,
        validate_cache=False
    )

    test_dataset = DeepfakeDataset(
        test_images, test_labels,
        get_val_transforms(config["img_size"]),
        preprocess_dir=None,
        validate_cache=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=config["num_workers"],
        pin_memory=True, drop_last=True,
        persistent_workers=True,
        prefetch_factor=8
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8
    )

    model_config = RADARConfig(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        evidence_dim=config["evidence_dim"],
        reasoning_iterations=config["reasoning_iterations"],
        reasoning_heads=config["reasoning_heads"],
        fft_size=config["fft_size"],
        dropout=config["dropout"]
    )
    model = RADAR(model_config).to(device)

    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='max-autotune')
            print("✓ Model compiled with torch.compile")
        except:
            print("⚠ torch.compile not available, using eager mode")

    history, best_auc = train_model(model, train_loader, val_loader, config, device, output_dir)

    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)

    checkpoint = torch.load(output_dir / "best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics = evaluate(model, test_loader, device)

    print(f"\nTest Results:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    results = {
        "history": history,
        "best_val_auc": float(best_auc),
        "test_metrics": {
            "auc": float(test_metrics["auc"]),
            "accuracy": float(test_metrics["accuracy"])
        },
        "config": config,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Val AUC: {best_auc:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Results saved to: {output_dir}")
    print("\nNext step:")
    print(f"  python 3_export_results.py --results_dir {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
