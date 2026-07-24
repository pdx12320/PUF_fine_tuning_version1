#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ranking_common import sha256_file, write_frame_new, write_json_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-count", type=int, default=50000)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    frames = []
    manifests = {}
    for value in args.shard:
        path = Path(value)
        manifest = json.loads(
            path.with_suffix(".manifest.json").read_text()
        )
        if manifest["status"] != "PASS":
            raise RuntimeError(path)
        frames.append(pd.read_parquet(path))
        manifests[str(path)] = {
            "sha256": sha256_file(path),
            "rows": len(frames[-1]),
        }
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["source_score", "id"], ascending=[False, True]
    )
    combined = combined.drop_duplicates(
        "sequence_hash", keep="first"
    ).head(args.top_count).copy()
    if len(combined) != args.top_count:
        raise AssertionError((len(combined), args.top_count))
    combined["source_rank"] = range(1, len(combined) + 1)
    combined["mining_round"] = args.round
    write_frame_new(combined, args.output)
    result = {
        "status": "PASS",
        "mining_round": args.round,
        "input_shards": manifests,
        "selected_rows": len(combined),
        "unique_sequence_hashes": int(
            combined["sequence_hash"].nunique()
        ),
        "output": args.output,
        "output_sha256": sha256_file(args.output),
    }
    write_json_new(Path(args.output).with_suffix(".manifest.json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
