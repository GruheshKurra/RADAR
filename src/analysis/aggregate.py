import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_experiment_results(output_dir: Path) -> Dict:
    import yaml

    metrics_path = output_dir / "metrics.json"
    config_yaml = output_dir / "config.yaml"
    config_json = output_dir / "config.json"

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    if config_yaml.exists():
        with open(config_yaml, 'r') as f:
            config = yaml.safe_load(f)
    elif config_json.exists():
        with open(config_json, 'r') as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"No config file found in {output_dir}")

    return {"metrics": metrics, "config": config}


def aggregate_multi_seed_results(experiment_dirs: List[Path]) -> Dict:
    all_results = [load_experiment_results(d) for d in experiment_dirs]

    metrics_keys = ["val_auc", "val_acc", "train_loss"]
    aggregated = {}

    for key in metrics_keys:
        values = []
        for result in all_results:
            if "history" in result["metrics"] and key in result["metrics"]["history"]:
                final_value = result["metrics"]["history"][key][-1]
                values.append(final_value)

        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return aggregated


def create_results_table(experiments: Dict[str, List[Path]]) -> pd.DataFrame:
    rows = []

    for exp_name, exp_dirs in experiments.items():
        agg = aggregate_multi_seed_results(exp_dirs)

        row = {
            "Method": exp_name,
            "AUC": f"{agg['val_auc']['mean']:.3f} ± {agg['val_auc']['std']:.3f}",
            "Accuracy": f"{agg['val_acc']['mean']:.3f} ± {agg['val_acc']['std']:.3f}",
        }
        rows.append(row)

    return pd.DataFrame(rows)
