#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ranking_common import load_yaml, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument("--deployment-selection", required=True)
    args = parser.parse_args()
    master = load_yaml(args.master)
    run_dir = Path(master["run_dir"])
    freeze_path = run_dir / "PRETEST_FROZEN.json"
    freeze = json.loads(freeze_path.read_text())
    if freeze["status"] != "FROZEN":
        raise RuntimeError(freeze["status"])
    started_marker = run_dir / "TEST_EVALUATION_STARTED"
    if started_marker.exists():
        raise FileExistsError(started_marker)
    if (run_dir / "TEST_EVALUATION_COMPLETE").exists():
        raise RuntimeError("Test already evaluated")
    input_path = Path(master["zero_shot_dir"]) / "embeddings/test.parquet"
    if str(input_path) != freeze["test_plan"]["input"]:
        raise AssertionError((str(input_path), freeze["test_plan"]["input"]))
    deployment = json.loads(
        Path(args.deployment_selection).read_text()
    )
    output_dir = run_dir / "work/locked_test_scores"
    jobs = [
        {
            "model_id": "random",
            "model_type": "random",
            "input": str(input_path),
            "output": str(output_dir / "random.parquet"),
            "log": str(run_dir / "logs/test_score_random.log"),
            "seed": 42,
        },
        {
            "model_id": "existing_binary_lamar",
            "model_type": "binary_lamar",
            "input": str(input_path),
            "output": str(
                output_dir / "existing_binary_lamar.parquet"
            ),
            "log": str(
                run_dir / "logs/test_score_existing_binary_lamar.log"
            ),
            "seed": 42,
        },
    ]
    frozen = freeze["selected_checkpoints"]
    for model_type in (
        "kmer",
        "cnn",
        "frozen_lamar",
        "lora_lamar",
        "hybrid_lamar",
    ):
        jobs.append(
            {
                "model_id": model_type,
                "model_type": model_type,
                "model_config": deployment["selected"][model_type][
                    "config"
                ],
                "checkpoint": frozen[model_type]["frozen_copy"],
                "input": str(input_path),
                "output": str(output_dir / f"{model_type}.parquet"),
                "log": str(
                    run_dir / "logs" / f"test_score_{model_type}.log"
                ),
                "seed": 42,
            }
        )
    result = {
        "status": "READY",
        "stage": "locked_test",
        "pretest_freeze": str(freeze_path),
        "input": str(input_path),
        "jobs": jobs,
        "test_access": True,
        "one_scoring_execution_per_model": True,
    }
    write_json_new(
        run_dir / "configs/locked_test_scoring_manifest.json",
        result,
    )
    started_marker.write_text(
        datetime.now(timezone.utc).isoformat() + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
