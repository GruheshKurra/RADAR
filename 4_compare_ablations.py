#!/usr/bin/env python3

import json
from pathlib import Path
import argparse
from tabulate import tabulate


def load_results(output_dir, experiment_names):
    results = {}
    for name in experiment_names:
        metrics_file = Path(output_dir) / name / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                results[name] = json.load(f)
        else:
            print(f"Warning: {metrics_file} not found")
    return results


def generate_ablation_table(results):
    table_data = []

    configs = {
        "radar_kaggle": ("Full RADAR", True, True, 3),
        "ablation_badm_only": ("BADM Only", True, False, 3),
        "ablation_aadm_only": ("AADM Only", False, True, 3),
        "ablation_no_reasoning": ("Both (No Reasoning)", True, True, 1),
    }

    for exp_name, (display_name, badm, aadm, iters) in configs.items():
        if exp_name in results:
            data = results[exp_name]
            val_auc = data.get("best_val_auc", 0) * 100
            test_auc = data["test_metrics"]["auc"] * 100
            test_acc = data["test_metrics"]["accuracy"] * 100

            table_data.append([
                display_name,
                f"{val_auc:.2f}%",
                f"{test_auc:.2f}%",
                f"{test_acc:.2f}%",
                "✓" if badm else "✗",
                "✓" if aadm else "✗",
                iters
            ])

    headers = ["Configuration", "Val AUC", "Test AUC", "Test Acc", "BADM", "AADM", "Iters"]
    return tabulate(table_data, headers=headers, tablefmt="grid")


def generate_iterations_table(results):
    table_data = []

    for iters in [1, 2, 3, 4, 5]:
        exp_name = f"ablation_iters_{iters}"
        if exp_name in results:
            data = results[exp_name]
            val_auc = data.get("best_val_auc", 0) * 100
            test_auc = data["test_metrics"]["auc"] * 100
            test_acc = data["test_metrics"]["accuracy"] * 100

            table_data.append([
                iters,
                f"{val_auc:.2f}%",
                f"{test_auc:.2f}%",
                f"{test_acc:.2f}%",
            ])

    headers = ["Iterations", "Val AUC", "Test AUC", "Test Acc"]
    return tabulate(table_data, headers=headers, tablefmt="grid")


def generate_latex_table(results):
    print("\n" + "="*70)
    print("LATEX TABLE (Copy to paper)")
    print("="*70)

    configs = {
        "radar_kaggle": ("Full RADAR", True, True, 3),
        "ablation_badm_only": ("BADM Only", True, False, 3),
        "ablation_aadm_only": ("AADM Only", False, True, 3),
        "ablation_no_reasoning": ("Both (No Reasoning)", True, True, 1),
    }

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Ablation Study Results}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Configuration & Val AUC & Test AUC & Test Acc \\")
    print(r"\midrule")

    for exp_name, (display_name, _, _, _) in configs.items():
        if exp_name in results:
            data = results[exp_name]
            val_auc = data.get("best_val_auc", 0) * 100
            test_auc = data["test_metrics"]["auc"] * 100
            test_acc = data["test_metrics"]["accuracy"] * 100

            if exp_name == "radar_kaggle":
                print(f"\\textbf{{{display_name}}} & \\textbf{{{val_auc:.2f}\\%}} & \\textbf{{{test_auc:.2f}\\%}} & \\textbf{{{test_acc:.2f}\\%}} \\\\")
            else:
                print(f"{display_name} & {val_auc:.2f}\\% & {test_auc:.2f}\\% & {test_acc:.2f}\\% \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:ablation}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Compare ablation results")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    all_experiments = [
        "radar_kaggle",
        "ablation_badm_only",
        "ablation_aadm_only",
        "ablation_no_reasoning",
        "ablation_iters_1",
        "ablation_iters_2",
        "ablation_iters_3",
        "ablation_iters_4",
        "ablation_iters_5",
    ]

    print("\n" + "="*70)
    print("RADAR ABLATION STUDY RESULTS")
    print("="*70)

    results = load_results(args.output_dir, all_experiments)

    if not results:
        print("No results found!")
        return

    print("\n" + "="*70)
    print("TABLE 1: Component Ablation")
    print("="*70)
    print(generate_ablation_table(results))

    print("\n" + "="*70)
    print("TABLE 2: Reasoning Iterations Ablation")
    print("="*70)
    print(generate_iterations_table(results))

    generate_latex_table(results)

    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)

    if "radar_kaggle" in results and "ablation_badm_only" in results:
        full_auc = results["radar_kaggle"]["test_metrics"]["auc"] * 100
        badm_auc = results["ablation_badm_only"]["test_metrics"]["auc"] * 100
        improvement = full_auc - badm_auc
        print(f"1. Full RADAR improves over BADM-only by {improvement:.2f}% AUC")

    if "radar_kaggle" in results and "ablation_aadm_only" in results:
        full_auc = results["radar_kaggle"]["test_metrics"]["auc"] * 100
        aadm_auc = results["ablation_aadm_only"]["test_metrics"]["auc"] * 100
        improvement = full_auc - aadm_auc
        print(f"2. Full RADAR improves over AADM-only by {improvement:.2f}% AUC")

    if "radar_kaggle" in results and "ablation_no_reasoning" in results:
        full_auc = results["radar_kaggle"]["test_metrics"]["auc"] * 100
        no_reason_auc = results["ablation_no_reasoning"]["test_metrics"]["auc"] * 100
        improvement = full_auc - no_reason_auc
        print(f"3. Reasoning module adds {improvement:.2f}% AUC improvement")

    print("\n" + "="*70)
    print("Export these tables to your paper!")
    print("="*70)


if __name__ == "__main__":
    main()
