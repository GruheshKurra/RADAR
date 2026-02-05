#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).parent / 'src'))

from method import RADAR, RADARConfig
from data.dataset import DeepfakeDataset, get_val_transforms
from data.splits import load_presplit_data
from torch.utils.data import DataLoader


def visualize_attention_evolution(model, test_loader, device, save_dir, num_samples=10):
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0

    with torch.no_grad():
        for images, labels, _ in test_loader:
            if sample_count >= num_samples:
                break

            images = images.to(device)
            outputs = model(images, use_badm=True, use_aadm=True)

            attention_history = outputs["attention_history"].cpu().numpy()
            gating_alpha = outputs["gating_alpha"].cpu().numpy()
            probs = outputs["prob"].cpu().numpy()

            for i in range(min(images.size(0), num_samples - sample_count)):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                iterations = attention_history.shape[1]
                x = np.arange(iterations)

                badm_attention = attention_history[i, :, 0]
                aadm_attention = attention_history[i, :, 1]

                axes[0].plot(x, badm_attention, 'o-', label='BADM (Boundary)', linewidth=2, markersize=8)
                axes[0].plot(x, aadm_attention, 's-', label='AADM (Frequency)', linewidth=2, markersize=8)
                axes[0].set_xlabel('Reasoning Iteration', fontsize=12)
                axes[0].set_ylabel('Attention Weight', fontsize=12)
                axes[0].set_title(f'Attention Evolution\nTrue: {"Fake" if labels[i] else "Real"}, Pred: {probs[i][0]:.3f}', fontsize=12)
                axes[0].legend(fontsize=10)
                axes[0].grid(True, alpha=0.3)
                axes[0].set_ylim([0, 1])

                axes[1].text(0.5, 0.7, f'Gating Alpha: {gating_alpha[i][0]:.3f}',
                           ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
                axes[1].text(0.5, 0.5, f'Reasoning Weight: {gating_alpha[i][0]:.1%}',
                           ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
                axes[1].text(0.5, 0.3, f'External Weight: {(1-gating_alpha[i][0]):.1%}',
                           ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
                axes[1].axis('off')
                axes[1].set_title('Gating Mechanism', fontsize=12)

                plt.tight_layout()
                plt.savefig(save_dir / f'attention_sample_{sample_count}.png', dpi=150, bbox_inches='tight')
                plt.close()

                sample_count += 1
                if sample_count >= num_samples:
                    break

    print(f"\nSaved {sample_count} attention visualizations to {save_dir}")


def visualize_attention_statistics(model, test_loader, device, save_dir):
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_badm_attention = []
    all_aadm_attention = []
    all_gating_alpha = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images, use_badm=True, use_aadm=True)

            attention_history = outputs["attention_history"].cpu().numpy()
            gating_alpha = outputs["gating_alpha"].cpu().numpy()

            final_badm = attention_history[:, -1, 0]
            final_aadm = attention_history[:, -1, 1]

            all_badm_attention.extend(final_badm)
            all_aadm_attention.extend(final_aadm)
            all_gating_alpha.extend(gating_alpha[:, 0])
            all_labels.extend(labels.numpy())

    all_badm_attention = np.array(all_badm_attention)
    all_aadm_attention = np.array(all_aadm_attention)
    all_gating_alpha = np.array(all_gating_alpha)
    all_labels = np.array(all_labels)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for label, name, color in [(0, 'Real', 'blue'), (1, 'Fake', 'red')]:
        mask = all_labels == label
        axes[0, 0].hist(all_badm_attention[mask], bins=30, alpha=0.5, label=name, color=color)
    axes[0, 0].set_xlabel('BADM Attention Weight', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Final BADM Attention Distribution', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for label, name, color in [(0, 'Real', 'blue'), (1, 'Fake', 'red')]:
        mask = all_labels == label
        axes[0, 1].hist(all_aadm_attention[mask], bins=30, alpha=0.5, label=name, color=color)
    axes[0, 1].set_xlabel('AADM Attention Weight', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Final AADM Attention Distribution', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    for label, name, color in [(0, 'Real', 'blue'), (1, 'Fake', 'red')]:
        mask = all_labels == label
        axes[1, 0].hist(all_gating_alpha[mask], bins=30, alpha=0.5, label=name, color=color)
    axes[1, 0].set_xlabel('Gating Alpha', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Gating Alpha Distribution', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(all_badm_attention[all_labels==0], all_aadm_attention[all_labels==0],
                      alpha=0.3, s=10, c='blue', label='Real')
    axes[1, 1].scatter(all_badm_attention[all_labels==1], all_aadm_attention[all_labels==1],
                      alpha=0.3, s=10, c='red', label='Fake')
    axes[1, 1].set_xlabel('BADM Attention', fontsize=12)
    axes[1, 1].set_ylabel('AADM Attention', fontsize=12)
    axes[1, 1].set_title('BADM vs AADM Attention', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'attention_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved attention statistics to {save_dir}/attention_statistics.png")

    print(f"\nAttention Statistics:")
    print(f"  BADM (Real): {all_badm_attention[all_labels==0].mean():.3f} ± {all_badm_attention[all_labels==0].std():.3f}")
    print(f"  BADM (Fake): {all_badm_attention[all_labels==1].mean():.3f} ± {all_badm_attention[all_labels==1].std():.3f}")
    print(f"  AADM (Real): {all_aadm_attention[all_labels==0].mean():.3f} ± {all_aadm_attention[all_labels==0].std():.3f}")
    print(f"  AADM (Fake): {all_aadm_attention[all_labels==1].mean():.3f} ± {all_aadm_attention[all_labels==1].std():.3f}")
    print(f"  Gating Alpha (Real): {all_gating_alpha[all_labels==0].mean():.3f} ± {all_gating_alpha[all_labels==0].std():.3f}")
    print(f"  Gating Alpha (Fake): {all_gating_alpha[all_labels==1].mean():.3f} ± {all_gating_alpha[all_labels==1].std():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize RADAR attention mechanisms")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./visualizations")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_config = RADARConfig(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        evidence_dim=64,
        reasoning_iterations=3,
        reasoning_heads=4,
        fft_size=112,
        dropout=0.1
    )
    model = RADAR(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loading test data...")
    data_dir = Path(args.data_dir)
    test_images, test_labels = load_presplit_data(data_dir, "test")

    test_dataset = DeepfakeDataset(
        test_images, test_labels,
        get_val_transforms(224),
        preprocess_dir=None,
        validate_cache=False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=32,
        shuffle=False, num_workers=4,
        pin_memory=True
    )

    print(f"\nGenerating visualizations...")
    print(f"Output directory: {args.output_dir}")

    visualize_attention_evolution(model, test_loader, device, args.output_dir, args.num_samples)
    visualize_attention_statistics(model, test_loader, device, args.output_dir)

    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)
    print("\nUse these figures in your paper:")
    print(f"  - Individual samples: {args.output_dir}/attention_sample_*.png")
    print(f"  - Statistics: {args.output_dir}/attention_statistics.png")


if __name__ == "__main__":
    main()
