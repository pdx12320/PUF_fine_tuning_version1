#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TEXT = (
    "/run/media/",
    "/data/ydx/",
    "/Users/",
    "10.20.",
    "ZHANGLAB",
)


def require_files() -> list[Path]:
    relative = [
        "README.md",
        "EXPERIMENT_DESIGN.md",
        "REPRODUCIBILITY.md",
        "ARTIFACT_POLICY.md",
        "configs/ranking_training_config.yaml",
        "reports/final_ranking_report.md",
        "results/ranking_leaderboard.csv",
        "results/precision_recall_at_k.csv",
        "results/enrichment_results.csv",
        "results/test_ranking_metrics.json",
        "results/final_seed_summary.csv",
        "results/shortcut_bias_analysis.csv",
        "results/threshold_strategy_results.csv",
        "application/budget_simulation.csv",
        "error_analysis/top100_false_positives.csv",
        "error_analysis/top100_false_negatives.csv",
        "figures/precision_at_k.png",
        "figures/recall_at_k.png",
        "figures/budget_curve.png",
        "figures/ranking_distribution.png",
        "provenance/RUN_SUMMARY.json",
    ]
    paths = [ROOT / item for item in relative]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise AssertionError({"missing": missing})
    return paths


def validate_leaderboard() -> None:
    path = ROOT / "results/ranking_leaderboard.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 7:
        raise AssertionError({"leaderboard_rows": len(rows)})
    by_model = {row["Model"]: row for row in rows}
    expected = {
        "Existing Binary Lamar": {
            "Test AP": 0.17195756066788312,
            "Precision@100": 0.32,
            "Precision@500": 0.124,
            "Precision@1000": 0.085,
        },
        "LoRA Lamar ranking": {
            "Test AP": 0.09217794483351867,
            "Precision@100": 0.20,
            "Precision@500": 0.10,
            "Precision@1000": 0.06,
        },
    }
    for model, values in expected.items():
        if model not in by_model:
            raise AssertionError({"missing_model": model})
        for column, target in values.items():
            observed = float(by_model[model][column])
            if abs(observed - target) > 1e-12:
                raise AssertionError(
                    {
                        "model": model,
                        "column": column,
                        "observed": observed,
                        "expected": target,
                    }
                )


def validate_test_json() -> None:
    value = json.loads(
        (ROOT / "results/test_ranking_metrics.json").read_text()
    )
    if value["status"] != "PASS" or value["rows"] != 161_161:
        raise AssertionError(
            {"status": value.get("status"), "rows": value.get("rows")}
        )
    binary = value["metrics"]["existing_binary_lamar"]
    counts = [
        binary[f"discovered_at_{budget}"]
        for budget in (10, 50, 100, 500, 1000)
    ]
    if counts != [7, 23, 32, 62, 85]:
        raise AssertionError({"binary_discoveries": counts})


def validate_error_tables() -> None:
    for name in (
        "top100_false_positives.csv",
        "top100_false_negatives.csv",
    ):
        path = ROOT / "error_analysis" / name
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 100:
            raise AssertionError({name: len(rows)})
        required = {
            "sequence",
            "score",
            "rank",
            "gene",
            "negative_type",
            "true_efficiency",
            "category",
        }
        if not required.issubset(rows[0]):
            raise AssertionError(
                {name: sorted(required - set(rows[0]))}
            )


def validate_pngs() -> dict[str, list[int]]:
    dimensions = {}
    for path in sorted((ROOT / "figures").glob("*.png")):
        data = path.read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            raise AssertionError({"invalid_png": path.name})
        if data[12:16] != b"IHDR":
            raise AssertionError({"missing_ihdr": path.name})
        width, height = struct.unpack(">II", data[16:24])
        if width < 100 or height < 100:
            raise AssertionError(
                {"small_png": path.name, "width": width, "height": height}
            )
        dimensions[path.name] = [width, height]
    if len(dimensions) != 4:
        raise AssertionError({"png_count": len(dimensions)})
    return dimensions


def validate_public_paths() -> int:
    suffixes = {
        ".csv",
        ".json",
        ".md",
        ".py",
        ".sha256",
        ".txt",
        ".yaml",
        ".yml",
    }
    checked = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(errors="replace")
        hits = [item for item in FORBIDDEN_TEXT if item in text]
        if hits:
            raise AssertionError(
                {
                    "non_public_path": str(path.relative_to(ROOT)),
                    "hits": hits,
                }
            )
        checked += 1
    return checked


def main() -> None:
    required = require_files()
    validate_leaderboard()
    validate_test_json()
    validate_error_tables()
    dimensions = validate_pngs()
    checked_text = validate_public_paths()
    result = {
        "status": "PASS",
        "root": str(ROOT),
        "required_files": len(required),
        "text_files_sanitized": checked_text,
        "png_dimensions": dimensions,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
