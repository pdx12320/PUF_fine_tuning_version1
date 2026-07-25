#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ranking_common import ranking_key, write_frame_new, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    output_dir = Path(args.output_dir)
    rows = []
    by_model = {}
    for job in manifest["jobs"]:
        summary_path = Path(job["output_dir"]) / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary["status"] != "SUCCESS":
            raise RuntimeError(summary_path)
        metrics = summary["best_dev_fixed_metrics"]
        config = summary["config"]
        row = {
            "experiment_id": job["experiment_id"],
            "model_type": job["model_type"],
            "learning_rate": config["learning_rate"],
            "negative_sampling": config["negative_sampling"],
            "loss": config.get("loss", "pair_difference_logistic"),
            "lambda_rank": config.get("lambda_rank"),
            "margin": config.get("margin"),
            "checkpoint": summary["best_checkpoint"],
            "epochs": summary["epochs_completed"],
            "training_seconds": summary["training_seconds"],
            "peak_gpu_bytes": summary["peak_gpu_bytes"],
            "trainable_parameters": summary["trainable_parameters"],
            **metrics,
        }
        rows.append(row)
        by_model.setdefault(job["model_type"], []).append(
            (ranking_key(metrics), row, summary_path)
        )
    selected = {}
    for model_type, candidates in by_model.items():
        _, row, summary_path = max(
            candidates, key=lambda value: value[0]
        )
        selected[model_type] = {
            "experiment_id": row["experiment_id"],
            "learning_rate": row["learning_rate"],
            "checkpoint": row["checkpoint"],
            "config": str(
                next(
                    job["config"]
                    for job in manifest["jobs"]
                    if job["experiment_id"] == row["experiment_id"]
                )
            ),
            "summary": str(summary_path),
            "dev_fixed_metrics": {
                key: row[key]
                for key in row
                if key.startswith(
                    (
                        "precision_at_",
                        "recall_at_",
                        "enrichment_at_",
                        "discovered_at_",
                    )
                )
                or key
                in {
                    "average_precision",
                    "pr_auc",
                    "roc_auc",
                    "ndcg",
                    "selection_mean_precision_at_k",
                }
            },
        }
    write_frame_new(
        pd.DataFrame(rows),
        output_dir / "lr_grid_dev_fixed.csv",
    )
    result = {
        "status": "PASS",
        "selection_universe": "immutable dev_1to10 fixed set",
        "selection_rule": (
            "mean Precision@K then P@100, P@500, AP"
        ),
        "selected": selected,
        "test_access": False,
    }
    write_json_new(output_dir / "lr_selection.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
