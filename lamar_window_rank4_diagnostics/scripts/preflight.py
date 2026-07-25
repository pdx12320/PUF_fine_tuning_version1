#!/usr/bin/env python3
"""Read-only preflight for the centered-window Frozen + Head ablation."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import pandas as pd
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_ids(values: list[int]) -> str:
    payload = ",".join(str(value) for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def check_forbidden(config: dict, paths: list[Path]) -> None:
    forbidden = tuple(str(value).lower() for value in config["forbidden_path_tokens"])
    for path in paths:
        lowered = str(path).lower()
        if any(token in lowered for token in forbidden):
            raise RuntimeError(f"Forbidden calibration/test path: {path}")


def centered(sequence: str, width: int) -> str:
    flank = (width - 1) // 2
    return sequence[50 - flank : 51 + flank]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    output = Path(config["output_dir"]).resolve()
    dataset = Path(config["dataset_dir"]).resolve()
    model_run = Path(config["model_run_dir"]).resolve()
    dev_path = Path(config["dev_universe"]).resolve()
    frozen = config["frozen_conditions"]
    windows = [int(value) for value in config["windows"]]

    required = {
        "train_positives": dataset / "train_positives.tsv.gz",
        "dataset_success": dataset / "SUCCESS",
        "negative_pool": model_run / "work/train_pool.sqlite",
        "model_success": model_run / "SUCCESS",
        "model_master": model_run / "configs/master.yaml",
        "source_frozen_config": model_run / "configs/s1_frozen_center.json",
        "source_modeling": model_run / "scripts/modeling_binary.py",
        "source_common": model_run / "scripts/common.py",
        "dev_universe": dev_path,
    }
    check_forbidden(config, list(required.values()))
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    source_config = json.loads(required["source_frozen_config"].read_text())
    expected_source = {
        "seed": frozen["seed"],
        "mode": frozen["mode"],
        "pooling": frozen["pooling"],
        "negative_strategy": frozen["negative_strategy"],
        "sampling_ratio": frozen["sampling_ratio"],
        "batch_size": frozen["batch_size"],
        "accumulation_steps": frozen["accumulation_steps"],
        "head_lr": frozen["head_lr"],
        "head_dropout": frozen["head_dropout"],
        "loss": frozen["loss"],
        "weight_decay": frozen["weight_decay"],
        "warmup_ratio": frozen["warmup_ratio"],
        "fp16": frozen["fp16"],
    }
    mismatches = {
        key: {"expected": value, "source": source_config.get(key)}
        for key, value in expected_source.items()
        if source_config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Frozen source configuration mismatch: {mismatches}")

    train = pd.read_csv(required["train_positives"], sep="\t", compression="gzip")
    dev = pd.read_parquet(dev_path)
    with sqlite3.connect(required["negative_pool"]) as connection:
        negative_count = int(
            connection.execute("select count(*) from negatives").fetchone()[0]
        )
        negative_sequence_lengths = dict(
            connection.execute(
                "select length(seq), count(*) from negatives group by length(seq)"
            ).fetchall()
        )
        negative_center_bases = dict(
            connection.execute(
                "select substr(seq,51,1), count(*) from negatives "
                "group by substr(seq,51,1)"
            ).fetchall()
        )

    assertions: list[dict] = []

    def check(name: str, passed: bool, observed: object) -> None:
        assertions.append(
            {
                "assertion": name,
                "status": "PASS" if passed else "FAIL",
                "observed": str(observed),
            }
        )

    check("windows_exact", windows == [21, 41, 61, 101], windows)
    check(
        "windows_odd_and_centered",
        all(width > 0 and width % 2 == 1 for width in windows),
        windows,
    )
    check("train_positive_rows", len(train) == frozen["train_positive_rows"], len(train))
    check(
        "train_positive_split",
        set(train["split"].astype(str)) == {"train"},
        train["split"].value_counts().to_dict(),
    )
    check(
        "train_sequence_length_101",
        train["sequence_context"].astype(str).str.len().eq(101).all(),
        train["sequence_context"].astype(str).str.len().value_counts().to_dict(),
    )
    check(
        "train_center_C",
        train["sequence_context"].astype(str).str[50].eq("C").all(),
        train["sequence_context"].astype(str).str[50].value_counts().to_dict(),
    )
    check("negative_pool_rows", negative_count == frozen["train_negative_pool_rows"], negative_count)
    check("negative_sequence_length_101", negative_sequence_lengths == {101: negative_count}, negative_sequence_lengths)
    check("negative_center_C", negative_center_bases == {"C": negative_count}, negative_center_bases)
    check("dev_total_rows", len(dev) == frozen["dev_total_rows"], len(dev))
    check(
        "dev_label_counts",
        dev["label"].value_counts().to_dict()
        == {0: frozen["dev_negative_rows"], 1: frozen["dev_positive_rows"]},
        dev["label"].value_counts().to_dict(),
    )
    check(
        "dev_split",
        set(dev["split"].astype(str)) == {"dev"},
        dev["split"].value_counts().to_dict(),
    )
    check("dev_genomic_key_unique", dev["genomic_key"].is_unique, dev["genomic_key"].nunique())
    check("dev_sequence_id_unique", dev["sequence_id"].is_unique, dev["sequence_id"].nunique())
    check(
        "dev_sequence_length_101",
        dev["sequence_context"].astype(str).str.len().eq(101).all(),
        dev["sequence_context"].astype(str).str.len().value_counts().to_dict(),
    )
    check(
        "dev_center_C",
        dev["sequence_context"].astype(str).str[50].eq("C").all(),
        dev["sequence_context"].astype(str).str[50].value_counts().to_dict(),
    )
    check(
        "train_dev_leakage_group_disjoint",
        not (
            set(train["leakage_group"].astype(str))
            & set(dev["leakage_group"].astype(str))
        ),
        len(
            set(train["leakage_group"].astype(str))
            & set(dev["leakage_group"].astype(str))
        ),
    )
    for width in windows:
        cropped_train = train["sequence_context"].astype(str).map(
            lambda sequence: centered(sequence, width)
        )
        cropped_dev = dev["sequence_context"].astype(str).map(
            lambda sequence: centered(sequence, width)
        )
        check(
            f"window_{width}_train_definition",
            cropped_train.str.len().eq(width).all()
            and cropped_train.str[(width - 1) // 2].eq("C").all(),
            {
                "lengths": cropped_train.str.len().value_counts().to_dict(),
                "centers": cropped_train.str[(width - 1) // 2].value_counts().to_dict(),
            },
        )
        check(
            f"window_{width}_dev_definition",
            cropped_dev.str.len().eq(width).all()
            and cropped_dev.str[(width - 1) // 2].eq("C").all(),
            {
                "lengths": cropped_dev.str.len().value_counts().to_dict(),
                "centers": cropped_dev.str[(width - 1) // 2].value_counts().to_dict(),
            },
        )

    expected_steps = math.ceil(
        math.ceil(
            (
                frozen["train_positive_rows"]
                * (1 + frozen["sampling_ratio"])
            )
            / frozen["batch_size"]
        )
        / frozen["accumulation_steps"]
    )
    check("steps_per_epoch", expected_steps == frozen["steps_per_epoch"], expected_steps)
    check(
        "optimizer_steps",
        frozen["steps_per_epoch"] * frozen["epochs"] == frozen["optimizer_steps"],
        frozen["steps_per_epoch"] * frozen["epochs"],
    )

    # Exact epoch-level negative selections are deterministic and identical
    # across windows. Store their hashes without copying the immutable pool.
    import sys
    sys.path.insert(0, str(model_run / "scripts"))
    from common import NegativePool

    pool = NegativePool(required["negative_pool"], int(frozen["seed"]))
    negative_per_epoch = int(
        frozen["train_positive_rows"] * frozen["sampling_ratio"]
    )
    epoch_negative_hashes = {}
    for epoch in range(int(frozen["epochs"])):
        ids = pool.ids_for_epoch(
            negative_per_epoch, str(frozen["negative_strategy"]), epoch
        )
        epoch_negative_hashes[str(epoch + 1)] = digest_ids(
            sorted(int(value) for value in ids)
        )

    failed = [row for row in assertions if row["status"] != "PASS"]
    report = {
        "status": "PREFLIGHT_OK" if not failed else "PREFLIGHT_FAILED",
        "config": str(config_path),
        "windows": windows,
        "window_definitions": {
            str(width): {
                "left_flank": (width - 1) // 2,
                "center": "C",
                "right_flank": (width - 1) // 2,
            }
            for width in windows
        },
        "frozen_conditions": frozen,
        "source_paths": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in required.items()
            if path.is_file() and path.suffix != ".sqlite"
        },
        "negative_pool": {
            "path": str(required["negative_pool"]),
            "size_bytes": required["negative_pool"].stat().st_size,
            "provenance_note": (
                "Immutable source pool; exact selected IDs are locked below. "
                "A full 2 GB checksum was intentionally avoided during preflight."
            ),
            "rows": negative_count,
            "negative_ids_per_epoch": negative_per_epoch,
            "epoch_selection_sha256": epoch_negative_hashes,
        },
        "assertions": assertions,
        "failed_assertions": failed,
        "calibration_or_test_accessed": False,
    }
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    report_path = results / "preflight_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(assertions).to_csv(
        results / "preflight_assertions.tsv", sep="\t", index=False
    )
    print(json.dumps({"status": report["status"], "assertions": len(assertions), "failed": len(failed)}))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
