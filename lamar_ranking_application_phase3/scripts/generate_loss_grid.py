#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ranking_common import write_json_new


def margin_label(value: float) -> str:
    return str(value).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sampling-selection", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    selected = json.loads(
        Path(args.sampling_selection).read_text()
    )["selected"]
    base = json.loads(Path(selected["config"]).read_text())
    config_dir = run_dir / "configs/formal_loss"
    if config_dir.exists():
        raise FileExistsError(config_dir)
    config_dir.mkdir(parents=True)
    jobs = []
    for margin in (0.1, 0.5, 1.0):
        config = copy.deepcopy(base)
        config["loss"] = "margin"
        config["margin"] = margin
        experiment_id = (
            f"loss_lora_margin_{margin_label(margin)}_seed42"
        )
        config_path = config_dir / f"{experiment_id}.json"
        write_json_new(config_path, config)
        jobs.append(
            {
                "experiment_id": experiment_id,
                "model_type": "lora_lamar",
                "config": str(config_path),
                "output_dir": str(
                    run_dir / "models/loss_grid" / experiment_id
                ),
                "log": str(
                    run_dir / "logs" / f"{experiment_id}.log"
                ),
            }
        )
    manifest = {
        "status": "READY",
        "stage": "ranking_loss_grid",
        "jobs": jobs,
        "external_candidates": [
            {
                **selected,
                "candidate_id": "loss_lora_pairwise_logistic_seed42",
                "source": "reused_sampling_selected_logistic",
            }
        ],
        "test_access": False,
    }
    write_json_new(
        run_dir / "configs/loss_grid_manifest.json", manifest
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
