#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import subprocess
from pathlib import Path

import numpy
import pandas
import pyarrow
import sklearn
import torch
import transformers

from ranking_common import load_yaml, sha256_file, write_json_new


def marker_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text().strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    output = Path(config["run_dir"])
    dataset = Path(config["dataset_dir"])
    binary = Path(config["binary_model_dir"])
    zero_shot = Path(config["zero_shot_dir"])

    dataset_manifest = json.loads(
        (dataset / "dataset_manifest.json").read_text()
    )
    binary_audit = json.loads(
        Path(config["binary_data_audit"]).read_text()
    )
    zero_audit = json.loads(
        (zero_shot / "results/input_audit.json").read_text()
    )
    checkpoint = Path(config["binary_checkpoint"])
    with sqlite3.connect(
        f"file:{Path(config['negative_pool_sqlite']).resolve()}?mode=ro",
        uri=True,
    ) as connection:
        negative_rows = int(
            connection.execute(
                "select count(*) from negatives"
            ).fetchone()[0]
        )
        matched_rows = int(
            connection.execute(
                "select count(*) from negatives where matched=1"
            ).fetchone()[0]
        )
        hard_rows = int(
            connection.execute(
                "select count(*) from negatives where hard=1"
            ).fetchone()[0]
        )
    failures = []
    if dataset_manifest["status"] != "complete":
        failures.append("dataset manifest not complete")
    if binary_audit["status"] != "PASS":
        failures.append("binary data audit did not pass")
    if zero_audit["status"] != "PASS":
        failures.append("zero-shot input audit did not pass")
    if binary_audit["sequence_hash_cross_split_violations"] != 0:
        failures.append("sequence hash cross-split violations")
    if binary_audit["leakage_group_cross_split_violations"] != 0:
        failures.append("leakage-group cross-split violations")
    if negative_rows != 1975244:
        failures.append(f"unexpected negative pool rows {negative_rows}")
    if not checkpoint.is_file():
        failures.append("binary checkpoint missing")
    forbidden = [
        output / "predictions/test_rank_predictions.parquet",
        output / "TEST_EVALUATION_STARTED",
        output / "TEST_EVALUATION_COMPLETE",
    ]
    if any(path.exists() for path in forbidden):
        failures.append("test output/marker exists before Phase 3 freeze")

    try:
        nvidia = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip().splitlines()
    except Exception as error:
        nvidia = [f"unavailable: {error}"]
    result = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "scientific_scope": (
            "sequence-only candidate prioritization; labels remain the "
            "immutable computational labels from the binary dataset"
        ),
        "immutable_inputs": {
            "dataset": str(dataset),
            "dataset_SUCCESS": marker_text(dataset / "SUCCESS"),
            "binary_model_dir": str(binary),
            "binary_SUCCESS": marker_text(binary / "SUCCESS"),
            "zero_shot_dir": str(zero_shot),
            "zero_shot_SUCCESS": marker_text(zero_shot / "SUCCESS"),
            "dataset_manifest_sha256": sha256_file(
                dataset / "dataset_manifest.json"
            ),
            "binary_checkpoint": str(checkpoint),
            "binary_checkpoint_sha256": sha256_file(checkpoint),
            "negative_pool_sqlite": config["negative_pool_sqlite"],
        },
        "counts": {
            "train_positive": 1028,
            "train_negative_pool": negative_rows,
            "train_matched_negative": matched_rows,
            "train_rule_hard_negative": hard_rows,
            "dev_positive": 159,
            "dev_negative_full": 282166,
            "calibration_positive": 165,
            "calibration_negative": 165000,
            "test_positive_expected_but_not_read": 161,
            "test_negative_expected_but_not_read": 161000,
        },
        "leakage": {
            "sequence_hash_cross_split_violations": binary_audit[
                "sequence_hash_cross_split_violations"
            ],
            "leakage_group_cross_split_violations": binary_audit[
                "leakage_group_cross_split_violations"
            ],
        },
        "test_access": (
            "No Phase 3 test rows, labels, embeddings, predictions, "
            "or metrics read; only inherited expected counts recorded."
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "gpus": nvidia,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "git": (
            "lamar7.21 is not a Git repository; script and output "
            "SHA-256 manifests are required instead"
        ),
    }
    write_json_new(output / "results/input_audit.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(result)
    marker = output / "INPUT_AUDIT_OK"
    if marker.exists():
        raise FileExistsError(marker)
    marker.write_text("PASS\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
