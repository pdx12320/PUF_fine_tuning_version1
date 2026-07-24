#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ranking_common import (
    META_COLUMNS,
    ranking_metrics,
    sha256_file,
    write_frame_new,
    write_json_new,
)


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Expected MODEL_ID=PATH")
    model_id, path = value.split("=", 1)
    return model_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction", action="append", required=True
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-metrics-csv", required=True)
    parser.add_argument("--output-metrics-json", required=True)
    args = parser.parse_args()

    items = [parse_prediction(value) for value in args.prediction]
    combined = None
    metrics_rows = []
    manifests = {}
    for model_id, path in items:
        frame = pd.read_parquet(path)
        if set(frame["model_id"].unique()) != {model_id}:
            raise AssertionError(
                (model_id, frame["model_id"].unique().tolist())
            )
        if combined is None:
            combined = frame[META_COLUMNS].copy()
            expected_ids = combined["sequence_id"].astype(str)
            expected_labels = combined["label"].to_numpy()
        else:
            if not expected_ids.equals(
                frame["sequence_id"].astype(str)
            ):
                raise AssertionError(
                    f"Row order/id mismatch for {model_id}"
                )
            if not (
                expected_labels == frame["label"].to_numpy()
            ).all():
                raise AssertionError(
                    f"Label mismatch for {model_id}"
                )
        score = frame["ranking_score"].to_numpy()
        combined[model_id] = score
        metrics = ranking_metrics(
            expected_labels, score, expected_ids
        )
        metrics_rows.append(
            {"split": args.split, "model_id": model_id, **metrics}
        )
        manifests[model_id] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    if combined is None:
        raise RuntimeError("No predictions")
    write_frame_new(combined, args.output_predictions)
    metrics_frame = pd.DataFrame(metrics_rows)
    write_frame_new(metrics_frame, args.output_metrics_csv)
    result = {
        "status": "PASS",
        "split": args.split,
        "rows": len(combined),
        "models": [model_id for model_id, _ in items],
        "input_predictions": manifests,
        "combined_predictions": args.output_predictions,
        "combined_predictions_sha256": sha256_file(
            args.output_predictions
        ),
        "metrics": metrics_rows,
    }
    write_json_new(args.output_metrics_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
