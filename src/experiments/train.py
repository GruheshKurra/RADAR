import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast
from pathlib import Path
import time
import json
from typing import Dict
import sys
sys.path.append(str(Path(__file__).parent.parent))

from method import RADAR, RADARConfig, RADARLoss
from method.loss import LossConfig


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, scheduler,
                loss_fn, scaler, device: str, gradient_accumulation_steps: int) -> Dict:
    model.train()
    total_losses = {"total": 0, "main": 0, "branch": 0, "orthogonal": 0,
                   "deep_supervision": 0}
    num_batches = 0
    accumulated_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch_data in enumerate(loader):
        if len(batch_data) == 4:
            images, labels, _, extras = batch_data
            freq_cached = extras.get("freq_cached").to(device, non_blocking=True) if "freq_cached" in extras else None
            sobel_cached = extras.get("sobel_cached").to(device, non_blocking=True) if "sobel_cached" in extras else None
        else:
            images, labels, _ = batch_data
            freq_cached = sobel_cached = None

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device, enabled=(device=="cuda")):
            outputs = model(images, freq_cached=freq_cached, sobel_cached=sobel_cached)
            losses = loss_fn(outputs, labels)
            loss = losses["total"] / gradient_accumulation_steps

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()

        for k, v in losses.items():
            total_losses[k] += v.item()
        num_batches += 1
        accumulated_batches += 1

        is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
        is_last_batch = (batch_idx + 1) == len(loader)

        if is_accumulation_step or is_last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accumulated_batches = 0

    return {k: v / num_batches for k, v in total_losses.items()}


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    model.eval()

    total_samples = len(loader.dataset)
    all_probs = torch.zeros(total_samples, device=device)
    all_labels = torch.zeros(total_samples, dtype=torch.long, device=device)
    current_idx = 0

    for batch_data in loader:
        if len(batch_data) == 4:
            images, labels, _, extras = batch_data
            freq_cached = extras.get("freq_cached").to(device) if "freq_cached" in extras else None
            sobel_cached = extras.get("sobel_cached").to(device) if "sobel_cached" in extras else None
        else:
            images, labels, _ = batch_data
            freq_cached = sobel_cached = None

        images = images.to(device)

        with autocast(device_type=device, enabled=(device=="cuda")):
            outputs = model(images, freq_cached=freq_cached, sobel_cached=sobel_cached)

        batch_size = outputs["prob"].shape[0]
        all_probs[current_idx:current_idx + batch_size] = outputs["prob"].squeeze()
        all_labels[current_idx:current_idx + batch_size] = labels
        current_idx += batch_size

    all_probs = all_probs[:current_idx].cpu().numpy()
    all_labels = all_labels[:current_idx].cpu().numpy()
    all_preds = (all_probs > 0.5).astype(int)

    from sklearn.metrics import accuracy_score, roc_auc_score

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_probs),
        "probs": all_probs,
        "labels": all_labels,
    }


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: dict, device: str, checkpoint_dir: Path):
    loss_config = LossConfig(**{k: v for k, v in config.items() if k in LossConfig.__annotations__})
    loss_fn = RADARLoss(loss_config)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"],
                     weight_decay=config["weight_decay"])

    total_steps = len(train_loader) * config["num_epochs"] // config["gradient_accumulation_steps"]
    scheduler = OneCycleLR(optimizer, max_lr=config["learning_rate"],
                          total_steps=total_steps, pct_start=config["warmup_ratio"])

    scaler = GradScaler(enabled=(device=="cuda"))

    best_auc = 0
    history = {"train_loss": [], "val_auc": [], "val_acc": []}

    for epoch in range(config["num_epochs"]):
        epoch_start = time.time()
        loss_fn.set_epoch(epoch)

        train_losses = train_epoch(model, train_loader, optimizer, scheduler,
                                  loss_fn, scaler, device, config["gradient_accumulation_steps"])
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_losses["total"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"Epoch {epoch+1}/{config['num_epochs']}: "
              f"Loss={train_losses['total']:.4f}, AUC={val_metrics['auc']:.4f}, "
              f"Acc={val_metrics['accuracy']:.4f}, Time={time.time()-epoch_start:.1f}s")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            checkpoint_path = checkpoint_dir / "best.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "auc": best_auc,
            }, checkpoint_path)

    return history, best_auc
