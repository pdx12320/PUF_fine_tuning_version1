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
    parser.add_argument("--loss-selection", required=True)
    parser.add_argument("--hard-round1", required=True)
    parser.add_argument("--hard-round2", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    selected = json.loads(Path(args.loss_selection).read_text())[
        "selected"
    ]
    config = copy.deepcopy(
        json.loads(Path(selected["config"]).read_text())
    )
    config["negative_sampling"] = "guided"
    config["guided_negative_paths"] = [
        args.hard_round1,
        args.hard_round2,
    ]
    config_dir = run_dir / "configs/formal_guided_round2"
    if config_dir.exists():
        raise FileExistsError(config_dir)
    config_dir.mkdir(parents=True)
    experiment_id = "guided_round1plus2_lora_seed42"
    config_path = config_dir / f"{experiment_id}.json"
    write_json_new(config_path, config)
    job = {
        "experiment_id": experiment_id,
        "model_type": "lora_lamar",
        "config": str(config_path),
        "output_dir": str(
            run_dir / "models/guided_round2" / experiment_id
        ),
        "log": str(run_dir / "logs" / f"{experiment_id}.log"),
    }
    manifest = {
        "status": "READY",
        "stage": "guided_round2_comparison",
        "jobs": [job],
        "external_candidates": [
            {
                **selected,
                "candidate_id": "best_before_round2_lora_seed42",
                "source": "reused_loss_selected",
            }
        ],
        "test_access": False,
    }
    write_json_new(
        run_dir / "configs/guided_round2_grid_manifest.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
