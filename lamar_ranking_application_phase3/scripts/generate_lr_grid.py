#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_common import write_json_new


LEARNING_RATES = (1e-5, 3e-5, 1e-4)
MODEL_TYPES = (
    "kmer",
    "cnn",
    "frozen_lamar",
    "lora_lamar",
    "hybrid_lamar",
)


def label(value: float) -> str:
    return f"{value:.0e}".replace("-", "m")


def config_for(model_type: str, learning_rate: float) -> dict:
    config = {
        "model_type": model_type,
        "seed": 42,
        "negative_sampling": "random",
        "learning_rate": learning_rate,
        "pairs_per_epoch": 10280,
        "epochs": 20,
        "patience": 3,
    }
    if model_type == "kmer":
        config["alpha"] = 1e-5
        return config
    config.update(
        {
            "loss": "pairwise_logistic",
            "margin": 0.5,
            "weight_decay": 0.0,
            "warmup_ratio": 0.03,
            "fp16": True,
            "eval_batch_size": 256,
            "accumulation_steps": 1,
        }
    )
    if model_type == "cnn":
        config["pair_batch_size"] = 128
    elif model_type == "frozen_lamar":
        config.update(
            {
                "pair_batch_size": 64,
                "head_dropout": 0.1,
            }
        )
    else:
        config.update(
            {
                "pair_batch_size": 32,
                "head_dropout": 0.1,
                "lora_rank": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.05,
            }
        )
        if model_type == "hybrid_lamar":
            config["lambda_rank"] = 0.5
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config_dir = run_dir / "configs/formal_lr"
    if config_dir.exists():
        raise FileExistsError(config_dir)
    config_dir.mkdir(parents=True)
    jobs = []
    for model_type in MODEL_TYPES:
        for learning_rate in LEARNING_RATES:
            experiment_id = (
                f"lr_{model_type}_{label(learning_rate)}_seed42"
            )
            config_path = config_dir / f"{experiment_id}.json"
            write_json_new(
                config_path,
                config_for(model_type, learning_rate),
            )
            jobs.append(
                {
                    "experiment_id": experiment_id,
                    "model_type": model_type,
                    "config": str(config_path),
                    "output_dir": str(
                        run_dir / "models/dev_grid" / experiment_id
                    ),
                    "log": str(
                        run_dir / "logs" / f"{experiment_id}.log"
                    ),
                }
            )
    manifest = {
        "status": "READY",
        "stage": "learning_rate_grid",
        "jobs": jobs,
        "learning_rates": list(LEARNING_RATES),
        "model_types": list(MODEL_TYPES),
        "seed": 42,
        "test_access": False,
    }
    write_json_new(run_dir / "configs/lr_grid_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
