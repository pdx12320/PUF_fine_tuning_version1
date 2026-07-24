#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ranking_common import ranking_key, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-summary", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seed_summary = pd.read_csv(args.seed_summary)
    baseline = pd.read_csv(args.baseline_metrics)
    rows = []
    for record in seed_summary.to_dict("records"):
        metrics = {
            "model_id": record["model_type"],
            "metric_source": "three_seed_mean",
        }
        for name in (
            "selection_mean_precision_at_k",
            "precision_at_100",
            "precision_at_500",
            "average_precision",
        ):
            metrics[name] = record[f"{name}_mean"]
        rows.append(metrics)
    for model_id in ("random", "existing_binary_lamar"):
        record = baseline.loc[
            baseline["model_id"] == model_id
        ].iloc[0].to_dict()
        rows.append(
            {
                "model_id": model_id,
                "metric_source": "fixed_model_dev",
                **{
                    name: record[name]
                    for name in (
                        "selection_mean_precision_at_k",
                        "precision_at_100",
                        "precision_at_500",
                        "average_precision",
                    )
                },
            }
        )
    selected = max(rows, key=ranking_key)
    result = {
        "status": "PASS",
        "selection_universe": "complete immutable dev split",
        "selection_rule": (
            "mean Precision@K then P@100, P@500, AP"
        ),
        "selected_model": selected["model_id"],
        "selected_metrics": selected,
        "all_models": rows,
        "test_access": False,
    }
    write_json_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
