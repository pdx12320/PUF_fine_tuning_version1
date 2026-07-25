#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ranking_common import load_yaml, sha256_file, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument("--deployment-selection", required=True)
    parser.add_argument("--overall-selection", required=True)
    parser.add_argument("--seed-summary", required=True)
    parser.add_argument("--threshold-policy", required=True)
    parser.add_argument("--composition-summary", required=True)
    args = parser.parse_args()
    master = load_yaml(args.master)
    run_dir = Path(master["run_dir"])
    freeze_path = run_dir / "PRETEST_FROZEN.json"
    if freeze_path.exists():
        raise FileExistsError(freeze_path)
    forbidden = [
        run_dir / "predictions/test_rank_predictions.parquet",
        run_dir / "TEST_EVALUATION_STARTED",
        run_dir / "TEST_EVALUATION_COMPLETE",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("Test artifact exists before freeze")
    deployment = json.loads(
        Path(args.deployment_selection).read_text()
    )
    overall = json.loads(Path(args.overall_selection).read_text())
    seed_summary = json.loads(Path(args.seed_summary).read_text())
    thresholds = json.loads(Path(args.threshold_policy).read_text())
    composition = json.loads(
        Path(args.composition_summary).read_text()
    )
    if any(
        value["status"] != "PASS"
        for value in (
            deployment,
            overall,
            seed_summary,
            thresholds,
            composition,
        )
    ):
        raise RuntimeError("A required dev/calibration artifact failed")
    checkpoint_dir = run_dir / "ranking_checkpoints"
    suffix = {
        "kmer": ".joblib",
        "cnn": ".pt",
        "frozen_lamar": ".safetensors",
        "lora_lamar": ".safetensors",
        "hybrid_lamar": ".safetensors",
    }
    copied = {}
    for model_type, selected in deployment["selected"].items():
        source = Path(selected["checkpoint"])
        target = (
            checkpoint_dir
            / f"{model_type}_seed42{suffix[model_type]}"
        )
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(source, target)
        copied[model_type] = {
            "source": str(source),
            "source_sha256": sha256_file(source),
            "frozen_copy": str(target),
            "frozen_copy_sha256": sha256_file(target),
            "config": selected["config"],
            "config_sha256": sha256_file(selected["config"]),
        }
        if (
            copied[model_type]["source_sha256"]
            != copied[model_type]["frozen_copy_sha256"]
        ):
            raise AssertionError(model_type)
    result = {
        "status": "FROZEN",
        "scientific_question": (
            "Does sequence-only Lamar ranking improve finite-budget "
            "C-editing candidate discovery over the existing binary score?"
        ),
        "overall_dev_selected_model": overall["selected_model"],
        "overall_dev_selection": args.overall_selection,
        "overall_dev_selection_sha256": sha256_file(
            args.overall_selection
        ),
        "deployment_seed": 42,
        "selected_checkpoints": copied,
        "existing_binary_checkpoint": {
            "path": master["binary_checkpoint"],
            "sha256": sha256_file(master["binary_checkpoint"]),
        },
        "composition_audit_model": {
            "path": str(run_dir / "models/composition_only.joblib"),
            "sha256": sha256_file(
                run_dir / "models/composition_only.joblib"
            ),
        },
        "dev_seed_summary": args.seed_summary,
        "dev_seed_summary_sha256": sha256_file(args.seed_summary),
        "threshold_policy": args.threshold_policy,
        "threshold_policy_sha256": sha256_file(
            args.threshold_policy
        ),
        "test_plan": {
            "input": str(
                Path(master["zero_shot_dir"])
                / "embeddings/test.parquet"
            ),
            "expected_rows": 161161,
            "expected_positive": 161,
            "expected_negative": 161000,
            "models": [
                "random",
                "existing_binary_lamar",
                "kmer",
                "cnn",
                "frozen_lamar",
                "lora_lamar",
                "hybrid_lamar",
                "composition_only_audit",
            ],
            "evaluation_passes": 1,
        },
        "test_access_before_freeze": False,
    }
    write_json_new(freeze_path, result)
    for marker_name in ("DEV_SELECTION_COMPLETE", "CALIBRATION_COMPLETE"):
        marker = run_dir / marker_name
        if marker.exists():
            raise FileExistsError(marker)
        marker.write_text("PASS\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
