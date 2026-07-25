#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_common import write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    selection = json.loads(Path(args.selection).read_text())["selected"]
    stage = args.stage
    output_dir = run_dir / "work" / f"{stage}_scores"
    jobs = [
        {
            "model_id": "random",
            "model_type": "random",
            "input": args.input,
            "output": str(output_dir / "random.parquet"),
            "log": str(run_dir / "logs" / f"{stage}_score_random.log"),
            "seed": 42,
        },
        {
            "model_id": "existing_binary_lamar",
            "model_type": "binary_lamar",
            "input": args.input,
            "output": str(output_dir / "existing_binary_lamar.parquet"),
            "log": str(
                run_dir
                / "logs"
                / f"{stage}_score_existing_binary_lamar.log"
            ),
            "seed": 42,
        },
    ]
    for model_type in (
        "kmer",
        "cnn",
        "frozen_lamar",
        "lora_lamar",
        "hybrid_lamar",
    ):
        chosen = selection[model_type]
        jobs.append(
            {
                "model_id": model_type,
                "model_type": model_type,
                "model_config": chosen["config"],
                "checkpoint": chosen["checkpoint"],
                "input": args.input,
                "output": str(output_dir / f"{model_type}.parquet"),
                "log": str(
                    run_dir
                    / "logs"
                    / f"{stage}_score_{model_type}.log"
                ),
                "seed": 42,
            }
        )
    manifest = {
        "status": "READY",
        "stage": stage,
        "selection": args.selection,
        "input": args.input,
        "jobs": jobs,
        "test_access": False,
    }
    write_json_new(
        run_dir / "configs" / f"{stage}_scoring_manifest.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
