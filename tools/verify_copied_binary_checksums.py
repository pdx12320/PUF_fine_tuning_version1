#!/usr/bin/env python3
"""Verify copied binary artifacts against successful-run SHA-256 manifests."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARY_SUFFIXES = {
    ".gz",
    ".joblib",
    ".parquet",
    ".png",
    ".pt",
    ".safetensors",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(None, 1)
        result[relative.strip()] = digest
    return result


def original_path(public: Path) -> tuple[str, str] | None:
    relative = public.relative_to(ROOT).as_posix()
    dataset_prefix = "lamar_binary_project/dataset_build/"
    model_prefix = "lamar_binary_project/model_training/"
    zero_prefix = "lamar_binary_project/zero_shot_ablation/"

    if relative.startswith(dataset_prefix + "results/"):
        return "dataset", relative.removeprefix(dataset_prefix + "results/")

    if relative.startswith(model_prefix + "results/"):
        return "model", relative.removeprefix(model_prefix + "results/")
    if relative.startswith(model_prefix + "subgroup_analysis/"):
        return "model", relative.removeprefix(model_prefix)
    if relative.startswith(model_prefix + "models/baselines/"):
        path = relative.removeprefix(model_prefix + "models/")
        return "model", "checkpoints/" + path
    if relative.startswith(model_prefix + "models/final_seeds/"):
        path = relative.removeprefix(model_prefix + "models/final_seeds/")
        return "model", "checkpoints/runs/" + path
    if relative == model_prefix + "models/calibration_model.joblib":
        return "model", "calibration_model.joblib"

    for folder in ("results/", "figures/", "error_analysis/", "models/"):
        if relative.startswith(zero_prefix + folder):
            return "zero", relative.removeprefix(zero_prefix)
    if relative.startswith(zero_prefix + "predictions/dev/"):
        path = relative.removeprefix(zero_prefix + "predictions/dev/")
        return "zero", "predictions/" + path

    return None


def main() -> int:
    manifests = {
        "dataset": read_manifest(
            ROOT
            / "lamar_binary_project/dataset_build/provenance/checksums.server.sha256"
        ),
        "model": read_manifest(
            ROOT
            / "lamar_binary_project/model_training/provenance/checksums.server.sha256"
        ),
        "zero": read_manifest(
            ROOT
            / "lamar_binary_project/zero_shot_ablation/provenance/checksums.server.sha256"
        ),
    }
    checked = 0
    failures: list[str] = []
    for public in sorted((ROOT / "lamar_binary_project").rglob("*")):
        if not public.is_file() or public.suffix not in BINARY_SUFFIXES:
            continue
        mapping = original_path(public)
        if mapping is None:
            failures.append(f"no checksum mapping: {public.relative_to(ROOT)}")
            continue
        stage, relative = mapping
        expected = manifests[stage].get(relative)
        if expected is None:
            failures.append(f"not in server manifest: {stage}:{relative}")
            continue
        observed = sha256(public)
        if observed != expected:
            failures.append(
                f"checksum mismatch: {public.relative_to(ROOT)} "
                f"expected={expected} observed={observed}"
            )
        checked += 1

    wiki = ROOT / "igem_drylab_wiki/assets"
    source = ROOT / "lamar_binary_project/zero_shot_ablation/figures"
    for name in (
        "PR_curves.png",
        "ROC_curves.png",
        "calibration_curve.png",
        "embedding_PCA.png",
        "metadata_correlation.png",
    ):
        if sha256(wiki / name) != sha256(source / name):
            failures.append(f"Wiki figure copy mismatch: {name}")
        checked += 1

    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"BINARY_CHECKSUMS_PASS files={checked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
