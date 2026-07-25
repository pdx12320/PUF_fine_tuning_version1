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
    parser.add_argument("--lora-selection", required=True)
    parser.add_argument("--lambda-selection", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    lr = json.loads(Path(args.lr_selection).read_text())["selected"]
    lora = json.loads(Path(args.lora_selection).read_text())["selected"]
    hybrid = json.loads(Path(args.lambda_selection).read_text())["selected"]
    seed42 = {
        "kmer": lr["kmer"],
        "cnn": lr["cnn"],
        "frozen_lamar": lr["frozen_lamar"],
        "lora_lamar": lora,
        "hybrid_lamar": hybrid,
    }
    config_dir = run_dir / "configs/formal_final_seeds"
    if config_dir.exists():
        raise FileExistsError(config_dir)
    config_dir.mkdir(parents=True)
    jobs = []
    external = []
    for model_type, selected in seed42.items():
        external.append(
            {
                **selected,
                "candidate_id": f"final_{model_type}_seed42",
                "model_type": model_type,
                "source": "reused_dev_selected_seed42",
            }
        )
        base = json.loads(Path(selected["config"]).read_text())
        for seed in (43, 44):
            config = copy.deepcopy(base)
            config["seed"] = seed
            experiment_id = f"final_{model_type}_seed{seed}"
            config_path = config_dir / f"{experiment_id}.json"
            write_json_new(config_path, config)
            jobs.append(
                {
                    "experiment_id": experiment_id,
                    "model_type": model_type,
                    "config": str(config_path),
                    "output_dir": str(
                        run_dir
                        / "models/final_seeds"
                        / experiment_id
                    ),
                    "log": str(
                        run_dir / "logs" / f"{experiment_id}.log"
                    ),
                }
            )
    manifest = {
        "status": "READY",
        "stage": "final_three_seeds",
        "jobs": jobs,
        "external_candidates": external,
        "seeds": [42, 43, 44],
        "deployment_seed": 42,
        "test_access": False,
    }
    write_json_new(
        run_dir / "configs/final_seed_grid_manifest.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
