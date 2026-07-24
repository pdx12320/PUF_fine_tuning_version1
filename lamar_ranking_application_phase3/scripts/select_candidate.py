#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ranking_common import ranking_key, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-name", required=True)
    args = parser.parse_args()
    candidates = json.loads(Path(args.candidates).read_text())
    metrics = pd.read_csv(args.metrics)
    by_id = {
        row["candidate_id"]: row
        for row in candidates["candidates"]
    }
    choices = []
    for record in metrics.to_dict("records"):
        model_id = record["model_id"]
        if model_id not in by_id:
            raise KeyError(model_id)
        choices.append(
            (ranking_key(record), by_id[model_id], record)
        )
    _, selected, selected_metrics = max(
        choices, key=lambda value: value[0]
    )
    result = {
        "status": "PASS",
        "selection_name": args.selection_name,
        "selection_universe": "complete immutable dev split",
        "selection_rule": (
            "mean Precision@K then P@100, P@500, AP"
        ),
        "selected": selected,
        "selected_metrics": selected_metrics,
        "all_candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "metrics": record,
            }
            for _, candidate, record in choices
        ],
        "test_access": False,
    }
    write_json_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
