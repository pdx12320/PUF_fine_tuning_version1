#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_common import write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.training_manifest)
    manifest = json.loads(manifest_path.read_text())
    candidates = list(manifest.get("external_candidates", []))
    for job in manifest["jobs"]:
        summary_path = Path(job["output_dir"]) / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary["status"] != "SUCCESS":
            raise RuntimeError(summary_path)
        candidates.append(
            {
                "candidate_id": job["experiment_id"],
                "model_type": job["model_type"],
                "config": job["config"],
                "checkpoint": summary["best_checkpoint"],
                "summary": str(summary_path),
                "source": "trained_in_stage",
            }
        )
    identifiers = [row["candidate_id"] for row in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError(identifiers)
    result = {
        "status": "PASS",
        "stage": manifest["stage"],
        "training_manifest": str(manifest_path),
        "candidates": candidates,
        "test_access": False,
    }
    write_json_new(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
