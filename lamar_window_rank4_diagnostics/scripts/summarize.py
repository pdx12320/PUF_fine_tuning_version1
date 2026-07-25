#!/usr/bin/env python3
"""Combine four completed formal window runs after invariant checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open() as handle:
        config = yaml.safe_load(handle)
    output = Path(config["output_dir"])
    frozen = config["frozen_conditions"]
    rows = []
    negative_hashes = None
    for width in [int(value) for value in config["windows"]]:
        path = output / "runs" / f"window_{width}bp_formal" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text())
        if summary["status"] != "SUCCESS":
            raise RuntimeError(f"Window {width} status: {summary['status']}")
        if summary["optimizer_steps"] != int(frozen["optimizer_steps"]):
            raise RuntimeError(f"Window {width} optimizer-step mismatch")
        if summary["dev_rows"] != int(frozen["dev_total_rows"]):
            raise RuntimeError(f"Window {width} dev-universe mismatch")
        hashes = [
            entry["negative_selection_sha256"] for entry in summary["history"]
        ]
        if negative_hashes is None:
            negative_hashes = hashes
        elif hashes != negative_hashes:
            raise RuntimeError(f"Window {width} negative selections differ")
        metrics = summary["dev_metrics"]
        rows.append(
            {
                "window_bp": width,
                "definition": summary["definition"],
                "optimizer_steps": summary["optimizer_steps"],
                "dev_rows": summary["dev_rows"],
                "average_precision": metrics["average_precision"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "P@10": metrics["P@10"],
                "P@50": metrics["P@50"],
                "P@100": metrics["P@100"],
                "P@500": metrics["P@500"],
                "P@1000": metrics["P@1000"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "predictions_sha256": summary["predictions_sha256"],
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["average_precision", "P@100", "window_bp"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    results = output / "results"
    target = results / "window_ablation_summary.tsv"
    if target.exists():
        raise FileExistsError(target)
    frame.to_csv(target, sep="\t", index=False)
    combined = {
        "status": "SUMMARY_OK",
        "selection_metric": "complete_dev_average_precision",
        "tie_breakers": ["P@100", "shorter_window"],
        "selected_window_bp": int(frame.iloc[0]["window_bp"]),
        "same_negative_selections_verified": True,
        "same_optimizer_steps_verified": True,
        "same_complete_dev_universe_verified": True,
        "calibration_or_test_accessed": False,
        "rows": frame.to_dict("records"),
    }
    json_path = results / "window_ablation_summary.json"
    if json_path.exists():
        raise FileExistsError(json_path)
    json_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    print(json.dumps(combined, sort_keys=True))


if __name__ == "__main__":
    main()
