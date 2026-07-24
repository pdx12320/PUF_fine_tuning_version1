#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_common import write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    candidate_manifest = json.loads(
        Path(args.candidates).read_text()
    )
    output_dir = run_dir / "work" / f"{args.stage}_scores"
    jobs = []
    for candidate in candidate_manifest["candidates"]:
        candidate_id = candidate["candidate_id"]
        jobs.append(
            {
                "model_id": candidate_id,
                "model_type": candidate["model_type"],
                "model_config": candidate["config"],
                "checkpoint": candidate["checkpoint"],
                "input": args.input,
                "output": str(output_dir / f"{candidate_id}.parquet"),
                "log": str(
                    run_dir
                    / "logs"
                    / f"{args.stage}_score_{candidate_id}.log"
                ),
            }
        )
    result = {
        "status": "READY",
        "stage": args.stage,
        "candidate_manifest": args.candidates,
        "input": args.input,
        "jobs": jobs,
        "test_access": False,
    }
    write_json_new(
        run_dir / "configs" / f"{args.stage}_scoring_manifest.json",
        result,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
