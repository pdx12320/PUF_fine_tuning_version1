#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy
import pandas
import pyarrow
import sklearn
import torch
import transformers

from ablation_common import load_config, sha256_file, write_json


def main():
    cfg = load_config(Path(__file__).parents[1] / "configs/config.json")
    run = Path(cfg["run_dir"])
    data = Path(cfg["dataset_dir"])
    model = Path(cfg["existing_model_dir"])
    if not (data / "SUCCESS").exists() or not (model / "SUCCESS").exists():
        raise RuntimeError("Immutable input SUCCESS marker missing")
    data_audit = json.loads((model / "data_audit.json").read_text())
    if data_audit["status"] != "PASS":
        raise RuntimeError(data_audit)
    required_counts = {
        "train_positive": 1028, "dev_positive": 159, "dev_negative": 1590,
        "calibration_positive": 165, "calibration_negative": 165000,
        "test_positive": 161, "test_negative": 161000,
    }
    observed = {
        "train_positive": data_audit["file_counts"]["train"]["label_1"],
        "dev_positive": data_audit["file_counts"]["dev"]["label_1"],
        "dev_negative": data_audit["file_counts"]["dev"]["label_0"],
        "calibration_positive": data_audit["file_counts"]["calibration"]["label_1"],
        "calibration_negative": data_audit["file_counts"]["calibration"]["label_0"],
        "test_positive": data_audit["file_counts"]["test"]["label_1"],
        "test_negative": data_audit["file_counts"]["test"]["label_0"],
    }
    if observed != required_counts:
        raise AssertionError((observed, required_counts))
    checkpoint = Path(cfg["pretrained_checkpoint"])
    manifest = {
        "status": "PASS", "immutable_dataset": str(data), "immutable_model_results": str(model),
        "dataset_SUCCESS": (data / "SUCCESS").read_text().strip(),
        "model_SUCCESS": (model / "SUCCESS").read_text().strip(),
        "counts": observed,
        "sequence_hash_cross_split_violations": data_audit["sequence_hash_cross_split_violations"],
        "leakage_group_cross_split_violations": data_audit["leakage_group_cross_split_violations"],
        "dataset_input_checksums": data_audit["input_checksums"],
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_checkpoint_sha256": sha256_file(checkpoint),
        "existing_model_checksums_sha256": sha256_file(model / "checksums.sha256"),
        "test_access_at_preflight": "integrity metadata inherited from prior PASS audit; no labels, embeddings, predictions, or metrics read",
        "software": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": numpy.__version__,
            "pandas": pandas.__version__, "pyarrow": pyarrow.__version__,
            "sklearn": sklearn.__version__,
            "nvidia_smi": subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            ).stdout.strip().splitlines(),
        },
    }
    write_json(run / "results/input_audit.json", manifest)
    (run / "INPUT_AUDIT_OK").write_text("PASS\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
