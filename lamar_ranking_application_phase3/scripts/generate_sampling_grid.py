#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ranking_common import write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--lr-selection", required=True)
    parser.add_argument("--hard-round1", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    selected = json.loads(Path(args.lr_selection).read_text())[
        "selected"
    ]["lora_lamar"]
    base = json.loads(Path(selected["config"]).read_text())
    config_dir = run_dir / "configs/formal_sampling"
    if config_dir.exists():
        raise FileExistsError(config_dir)
    config_dir.mkdir(parents=True)
    jobs = []
    for strategy in ("matched", "hard", "mixed", "guided"):
        config = copy.deepcopy(base)
        config["negative_sampling"] = strategy
        if strategy == "guided":
            config["guided_negative_paths"] = [args.hard_round1]
        experiment_id = f"sampling_lora_{strategy}_seed42"
        config_path = config_dir / f"{experiment_id}.json"
        write_json_new(config_path, config)
        jobs.append(
            {
                "experiment_id": experiment_id,
                "model_type": "lora_lamar",
                "config": str(config_path),
                "output_dir": str(
                    run_dir / "models/sampling_grid" / experiment_id
                ),
                "log": str(
                    run_dir / "logs" / f"{experiment_id}.log"
                ),
            }
        )
    manifest = {
        "status": "READY",
        "stage": "negative_sampling_grid",
        "jobs": jobs,
        "external_candidates": [
            {
                "candidate_id": "sampling_lora_random_seed42",
                "model_type": "lora_lamar",
                "config": selected["config"],
                "checkpoint": selected["checkpoint"],
                "summary": selected["summary"],
                "source": "reused_lr_selected_random",
            }
        ],
        "hard_round1": args.hard_round1,
        "test_access": False,
    }
    write_json_new(
        run_dir / "configs/sampling_grid_manifest.json", manifest
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
