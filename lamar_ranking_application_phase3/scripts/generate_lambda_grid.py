#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ranking_common import write_json_new


def lambda_label(value: float) -> str:
    return str(value).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--lr-selection", required=True)
    parser.add_argument("--sampling-selection", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    hybrid = json.loads(Path(args.lr_selection).read_text())[
        "selected"
    ]["hybrid_lamar"]
    sampling = json.loads(
        Path(args.sampling_selection).read_text()
    )["selected"]
    sampling_config = json.loads(
        Path(sampling["config"]).read_text()
    )
    base = json.loads(Path(hybrid["config"]).read_text())
    base["negative_sampling"] = sampling_config[
        "negative_sampling"
    ]
    if "guided_negative_paths" in sampling_config:
        base["guided_negative_paths"] = sampling_config[
            "guided_negative_paths"
        ]
    else:
        base.pop("guided_negative_paths", None)
    config_dir = run_dir / "configs/formal_lambda"
    if config_dir.exists():
        raise FileExistsError(config_dir)
    config_dir.mkdir(parents=True)
    jobs = []
    for value in (0.1, 0.5, 1.0):
        config = copy.deepcopy(base)
        config["lambda_rank"] = value
        experiment_id = (
            f"lambda_hybrid_{lambda_label(value)}_seed42"
        )
        config_path = config_dir / f"{experiment_id}.json"
        write_json_new(config_path, config)
        jobs.append(
            {
                "experiment_id": experiment_id,
                "model_type": "hybrid_lamar",
                "config": str(config_path),
                "output_dir": str(
                    run_dir / "models/lambda_grid" / experiment_id
                ),
                "log": str(
                    run_dir / "logs" / f"{experiment_id}.log"
                ),
            }
        )
    manifest = {
        "status": "READY",
        "stage": "hybrid_lambda_grid",
        "jobs": jobs,
        "external_candidates": [],
        "negative_sampling_source": args.sampling_selection,
        "test_access": False,
    }
    write_json_new(
        run_dir / "configs/lambda_grid_manifest.json", manifest
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
