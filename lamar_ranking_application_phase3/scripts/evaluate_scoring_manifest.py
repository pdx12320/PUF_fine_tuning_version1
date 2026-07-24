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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    combined = None
    rows = []
    sources = {}
    for job in manifest["jobs"]:
        model_id = job["model_id"]
        path = Path(job["output"])
        frame = pd.read_parquet(path)
        if set(frame["model_id"].unique()) != {model_id}:
            raise AssertionError(model_id)
        if combined is None:
            combined = frame[META_COLUMNS].copy()
            ids = combined["sequence_id"].astype(str)
            labels = combined["label"].to_numpy()
        else:
            if not ids.equals(frame["sequence_id"].astype(str)):
                raise AssertionError(f"ID mismatch: {model_id}")
            if not (labels == frame["label"].to_numpy()).all():
                raise AssertionError(f"Label mismatch: {model_id}")
        score = frame["ranking_score"].to_numpy()
        combined[model_id] = score
        rows.append(
            {
                "split": args.split,
                "model_id": model_id,
                **ranking_metrics(labels, score, ids),
            }
        )
        sources[model_id] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    if combined is None:
        raise RuntimeError("No scoring jobs")
    write_frame_new(combined, args.output_predictions)
    write_frame_new(pd.DataFrame(rows), args.output_metrics)
    result = {
        "status": "PASS",
        "split": args.split,
        "rows": len(combined),
        "sources": sources,
        "metrics": rows,
        "test_access": args.split == "test",
    }
    write_json_new(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
