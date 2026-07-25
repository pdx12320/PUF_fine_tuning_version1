#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ranking_common import write_frame_new, write_json_new


METRICS = [
    "average_precision",
    "pr_auc",
    "roc_auc",
    "ndcg",
    "selection_mean_precision_at_k",
    "precision_at_10",
    "precision_at_50",
    "precision_at_100",
    "precision_at_500",
    "precision_at_1000",
    "recall_at_10",
    "recall_at_50",
    "recall_at_100",
    "recall_at_500",
    "recall_at_1000",
    "enrichment_at_10",
    "enrichment_at_50",
    "enrichment_at_100",
    "enrichment_at_500",
    "enrichment_at_1000",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.metrics)
    pattern = re.compile(r"^final_(.+)_seed(42|43|44)$")
    parsed = frame["model_id"].astype(str).str.extract(pattern)
    if parsed.isna().any().any():
        raise ValueError(frame.loc[parsed.isna().any(axis=1), "model_id"])
    frame["model_type"] = parsed[0]
    frame["seed"] = parsed[1].astype(int)
    if set(frame["seed"]) != {42, 43, 44}:
        raise AssertionError(frame["seed"].value_counts().to_dict())
    rows = []
    for model_type, group in frame.groupby("model_type"):
        if set(group["seed"]) != {42, 43, 44}:
            raise AssertionError((model_type, group["seed"].tolist()))
        row = {"model_type": model_type, "seed_count": len(group)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(
                group[metric].std(ddof=1)
            )
            row[f"{metric}_seed42"] = float(
                group.loc[group["seed"] == 42, metric].iloc[0]
            )
        rows.append(row)
    result_frame = pd.DataFrame(rows).sort_values("model_type")
    write_frame_new(result_frame, args.output_csv)
    result = {
        "status": "PASS",
        "seeds": [42, 43, 44],
        "deployment_seed": 42,
        "selection_uses_seed_mean": True,
        "seed_summary": result_frame.to_dict("records"),
        "test_access": False,
    }
    write_json_new(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
