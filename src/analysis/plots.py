import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import roc_curve


def plot_training_curves(histories: Dict[str, Dict], output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=name)
        axes[1].plot(epochs, history["val_auc"], label=name)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation AUC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(results: Dict[str, Dict], output_path: Path):
    plt.figure(figsize=(8, 8))

    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result["labels"], result["probs"])
        auc = result["auc"]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_results(results: Dict[str, float], output_path: Path):
    names = list(results.keys())
    aucs = list(results.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if 'RADAR' in n else 'steelblue' for n in names]
    bars = ax.bar(range(len(names)), aucs, color=colors, edgecolor='black')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('AUC')
    ax.set_title('Ablation Study Results')
    ax.set_ylim([min(aucs) - 0.05, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(aucs):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
