#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ranking_common import (
    META_COLUMNS,
    ranking_metrics,
    sha256_file,
    write_frame_new,
    write_json_new,
)


FINAL_SEED_COLUMNS = {
    "kmer": "final_kmer_seed42",
    "cnn": "final_cnn_seed42",
    "frozen_lamar": "final_frozen_lamar_seed42",
    "lora_lamar": "final_lora_lamar_seed42",
    "hybrid_lamar": "final_hybrid_lamar_seed42",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--final-seed-predictions", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    baseline_path = Path(args.baseline_predictions)
    seed_path = Path(args.final_seed_predictions)
    baseline = pd.read_parquet(baseline_path)
    seeds = pd.read_parquet(seed_path)

    for name, frame in (("baseline", baseline), ("final_seed", seeds)):
        missing = set(META_COLUMNS) - set(frame.columns)
        if missing:
            raise AssertionError(f"{name} missing metadata columns: {sorted(missing)}")
        if len(frame) != 282_325:
            raise AssertionError(f"{name} row count: {len(frame)}")
        if set(frame["split"].astype(str)) != {"dev"}:
            raise AssertionError(f"{name} is not dev-only")
        if int(frame["label"].sum()) != 159:
            raise AssertionError(f"{name} positive count: {int(frame['label'].sum())}")

    if not baseline["sequence_id"].astype(str).equals(
        seeds["sequence_id"].astype(str)
    ):
        raise AssertionError("Sequence ID order differs between cached prediction sets")
    if not np.array_equal(
        baseline["label"].to_numpy(), seeds["label"].to_numpy()
    ):
        raise AssertionError("Labels differ between cached prediction sets")

    required_baseline = {"random", "existing_binary_lamar"}
    missing_baseline = required_baseline - set(baseline.columns)
    if missing_baseline:
        raise AssertionError(
            f"Baseline predictions missing columns: {sorted(missing_baseline)}"
        )
    missing_seed = set(FINAL_SEED_COLUMNS.values()) - set(seeds.columns)
    if missing_seed:
        raise AssertionError(
            f"Final-seed predictions missing columns: {sorted(missing_seed)}"
        )

    combined = baseline[META_COLUMNS].copy()
    combined["random"] = baseline["random"].to_numpy(dtype=np.float64)
    combined["existing_binary_lamar"] = baseline[
        "existing_binary_lamar"
    ].to_numpy(dtype=np.float64)
    for model_id, source_column in FINAL_SEED_COLUMNS.items():
        combined[model_id] = seeds[source_column].to_numpy(dtype=np.float64)

    ids = combined["sequence_id"].astype(str)
    labels = combined["label"].to_numpy(dtype=np.int64)
    model_ids = [
        "random",
        "kmer",
        "cnn",
        "existing_binary_lamar",
        "frozen_lamar",
        "lora_lamar",
        "hybrid_lamar",
    ]
    rows = [
        {
            "split": "dev",
            "model_id": model_id,
            **ranking_metrics(labels, combined[model_id].to_numpy(), ids),
        }
        for model_id in model_ids
    ]

    write_frame_new(combined, args.output_predictions)
    write_frame_new(pd.DataFrame(rows), args.output_metrics)
    result = {
        "status": "PASS",
        "split": "dev",
        "rows": int(len(combined)),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "model_ids": model_ids,
        "sources": {
            "baseline_predictions": {
                "path": str(baseline_path),
                "sha256": sha256_file(baseline_path),
            },
            "final_seed_predictions": {
                "path": str(seed_path),
                "sha256": sha256_file(seed_path),
            },
        },
        "cached_inference_reused": True,
        "metrics": rows,
        "test_access": False,
    }
    write_json_new(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
