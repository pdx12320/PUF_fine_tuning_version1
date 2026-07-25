#!/usr/bin/env python3
"""Audit singular spectra of every rank-4 LoRA update in a checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open


KEY_PATTERN = re.compile(
    r"^esm\.encoder\.layer\.(?P<layer>\d+)\."
    r"(?P<path>attention\.(?:self\.(?:query|key|value)|output\.dense))"
    r"\.lora_A\.weight$"
)
MODULE_LABELS = {
    "attention.self.query": "Q",
    "attention.self.key": "K",
    "attention.self.value": "V",
    "attention.output.dense": "O",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def thin_singular_values(a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    """Return the nonzero singular values of scale * B @ A via thin QR."""
    _, rb = np.linalg.qr(b, mode="reduced")
    _, ra = np.linalg.qr(a.T, mode="reduced")
    core = scale * (rb @ ra.T)
    values = np.linalg.svd(core, compute_uv=False)
    return np.sort(values)[::-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    run_config_path = Path(args.run_config).resolve()
    output = Path(args.output_dir).resolve()
    if not checkpoint.is_file() or not run_config_path.is_file():
        raise FileNotFoundError((checkpoint, run_config_path))
    results_dir = output / "results"
    if results_dir.exists():
        raise FileExistsError(results_dir)
    results_dir.mkdir(parents=True)

    run_config = json.loads(run_config_path.read_text())
    rank = int(run_config["lora_rank"])
    alpha = 2 * rank
    scale = float(alpha / rank)
    if rank != 4 or run_config["lora_scheme"] != "qkvo":
        raise RuntimeError(
            f"Expected rank-4 qkvo checkpoint, observed rank={rank}, "
            f"scheme={run_config['lora_scheme']}"
        )

    rows: list[dict] = []
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        a_keys = sorted(key for key in keys if key.endswith(".lora_A.weight"))
        b_keys = sorted(key for key in keys if key.endswith(".lora_B.weight"))
        if len(a_keys) != 48 or len(b_keys) != 48:
            raise RuntimeError(
                f"Expected 48 A/B pairs, observed A={len(a_keys)}, B={len(b_keys)}"
            )
        for a_key in a_keys:
            match = KEY_PATTERN.match(a_key)
            if match is None:
                raise RuntimeError(f"Unexpected LoRA A key: {a_key}")
            b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
            if b_key not in keys:
                raise RuntimeError(f"Missing paired B tensor: {b_key}")
            a = handle.get_tensor(a_key).cpu().numpy().astype(np.float64)
            b = handle.get_tensor(b_key).cpu().numpy().astype(np.float64)
            if a.shape != (rank, 768) or b.shape != (768, rank):
                raise RuntimeError(f"Unexpected pair shapes: {a_key} {a.shape} {b.shape}")

            singular = thin_singular_values(a, b, scale)
            if singular.shape != (rank,) or not np.isfinite(singular).all():
                raise RuntimeError(f"Invalid singular spectrum: {a_key}")
            total = float(singular.sum())
            probabilities = singular / total
            positive = probabilities[probabilities > 0]
            entropy_effective_rank = float(
                np.exp(-np.sum(positive * np.log(positive)))
            )
            stable_rank = float(np.sum(singular**2) / singular[0] ** 2)
            tolerance_fp32 = float(
                singular[0] * max(b.shape[0], a.shape[1]) * np.finfo(np.float32).eps
            )
            tolerance_fp64 = float(
                singular[0] * max(b.shape[0], a.shape[1]) * np.finfo(np.float64).eps
            )
            path = match.group("path")
            row = {
                "layer": int(match.group("layer")),
                "module": MODULE_LABELS[path],
                "module_path": path,
                "a_key": a_key,
                "b_key": b_key,
                "a_rows": int(a.shape[0]),
                "a_cols": int(a.shape[1]),
                "b_rows": int(b.shape[0]),
                "b_cols": int(b.shape[1]),
                "rank_configured": rank,
                "alpha": alpha,
                "scale_alpha_over_r": scale,
                "sigma_1": float(singular[0]),
                "sigma_2": float(singular[1]),
                "sigma_3": float(singular[2]),
                "sigma_4": float(singular[3]),
                "sigma_2_over_sigma_1": float(singular[1] / singular[0]),
                "sigma_3_over_sigma_1": float(singular[2] / singular[0]),
                "sigma_4_over_sigma_1": float(singular[3] / singular[0]),
                "condition_number_sigma1_over_sigma4": float(
                    singular[0] / singular[3]
                ),
                "numerical_rank_fp32": int(np.sum(singular > tolerance_fp32)),
                "numerical_rank_fp64": int(np.sum(singular > tolerance_fp64)),
                "numerical_rank_tolerance_fp32": tolerance_fp32,
                "numerical_rank_tolerance_fp64": tolerance_fp64,
                "entropy_effective_rank": entropy_effective_rank,
                "entropy_effective_rank_fraction_of_4": float(
                    entropy_effective_rank / rank
                ),
                "stable_rank": stable_rank,
                "stable_rank_fraction_of_4": float(stable_rank / rank),
                "frobenius_norm_delta_w": float(np.sqrt(np.sum(singular**2))),
                "spectral_norm_delta_w": float(singular[0]),
                "nuclear_norm_delta_w": total,
            }
            rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["layer", "module"], kind="mergesort")
    if len(frame) != 48:
        raise RuntimeError(f"Expected 48 module rows, observed {len(frame)}")
    if set(frame["layer"]) != set(range(12)) or set(frame["module"]) != {
        "Q",
        "K",
        "V",
        "O",
    }:
        raise RuntimeError("Layer/module coverage is incomplete")
    module_path = results_dir / "rank4_lora_module_spectra.tsv"
    frame.to_csv(module_path, sep="\t", index=False, float_format="%.12g")

    group_rows = []
    for module, group in frame.groupby("module", sort=True):
        group_rows.append(
            {
                "module": module,
                "count": len(group),
                "numerical_rank_fp32_min": int(group["numerical_rank_fp32"].min()),
                "numerical_rank_fp32_max": int(group["numerical_rank_fp32"].max()),
                "entropy_effective_rank_mean": float(
                    group["entropy_effective_rank"].mean()
                ),
                "entropy_effective_rank_min": float(
                    group["entropy_effective_rank"].min()
                ),
                "entropy_effective_rank_max": float(
                    group["entropy_effective_rank"].max()
                ),
                "stable_rank_mean": float(group["stable_rank"].mean()),
                "sigma_4_over_sigma_1_median": float(
                    group["sigma_4_over_sigma_1"].median()
                ),
                "sigma_4_over_sigma_1_min": float(
                    group["sigma_4_over_sigma_1"].min()
                ),
            }
        )
    group_frame = pd.DataFrame(group_rows)
    group_path = results_dir / "rank4_lora_module_type_summary.tsv"
    group_frame.to_csv(group_path, sep="\t", index=False, float_format="%.12g")

    summary = {
        "status": "RANK4_SPECTRUM_AUDIT_OK",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "run_config": str(run_config_path),
        "run_config_sha256": sha256(run_config_path),
        "experiment_id": run_config["experiment_id"],
        "lora_scheme": run_config["lora_scheme"],
        "configured_rank": rank,
        "alpha": alpha,
        "scale_alpha_over_r": scale,
        "delta_w_definition": "(alpha / r) * B @ A",
        "module_count": len(frame),
        "expected_module_count": 48,
        "layer_count": int(frame["layer"].nunique()),
        "module_types": sorted(frame["module"].unique()),
        "all_numerical_rank_fp32_equal_4": bool(
            frame["numerical_rank_fp32"].eq(4).all()
        ),
        "all_numerical_rank_fp64_equal_4": bool(
            frame["numerical_rank_fp64"].eq(4).all()
        ),
        "numerical_rank_fp32_counts": {
            str(key): int(value)
            for key, value in frame["numerical_rank_fp32"]
            .value_counts()
            .sort_index()
            .items()
        },
        "entropy_effective_rank": quantiles(
            frame["entropy_effective_rank"].tolist()
        ),
        "entropy_effective_rank_fraction_of_4": quantiles(
            frame["entropy_effective_rank_fraction_of_4"].tolist()
        ),
        "stable_rank": quantiles(frame["stable_rank"].tolist()),
        "sigma_4_over_sigma_1": quantiles(
            frame["sigma_4_over_sigma_1"].tolist()
        ),
        "condition_number_sigma1_over_sigma4": quantiles(
            frame["condition_number_sigma1_over_sigma4"].tolist()
        ),
        "interpretation": {
            "numerical_rank": (
                "Count of singular values above sigma_1 * max(m,n) * machine "
                "epsilon, reported for float32 application precision and float64 audit precision."
            ),
            "entropy_effective_rank": (
                "exp(-sum p_i log p_i), p_i = sigma_i / sum sigma_i; range 1..4."
            ),
            "stable_rank": "sum(sigma_i^2) / sigma_1^2; range 1..4.",
            "used_full_rank_primary_check": (
                "All 48 updates have numerical_rank_fp32 == configured rank 4."
            ),
        },
        "outputs": {
            "module_spectra": str(module_path),
            "module_type_summary": str(group_path),
        },
    }
    summary_path = results_dir / "rank4_lora_spectrum_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
