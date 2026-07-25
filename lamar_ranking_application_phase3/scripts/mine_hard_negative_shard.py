#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import torch

from mine_hard_negatives import (
    COLUMNS,
    build_binary_scorer,
    build_ranking_scorer,
)
from ranking_common import load_yaml, sha256_file, write_frame_new, write_json_new


def fetch_range(connection, start_id, end_id, count):
    connection.row_factory = sqlite3.Row
    result = connection.execute(
        f"select {COLUMNS} from negatives "
        "where id>? and id<=? order by id limit ?",
        (int(start_id), int(end_id), int(count)),
    ).fetchall()
    rows = []
    for raw in result:
        row = dict(raw)
        row.update(
            {
                "sequence_context": row.pop("seq"),
                "gc_fraction": row.pop("gc"),
                "label": 0,
                "split": "train",
                "true_efficiency": 0.0,
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument(
        "--source-model",
        choices=("binary_lamar", "lora_lamar", "hybrid_lamar"),
        required=True,
    )
    parser.add_argument("--model-config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--round", type=int, choices=(1, 2), required=True)
    parser.add_argument("--start-id", type=int, required=True)
    parser.add_argument("--end-id", type=int, required=True)
    parser.add_argument("--top-count", type=int, default=50000)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--exclude")
    args = parser.parse_args()
    if args.start_id < 0 or args.end_id <= args.start_id:
        raise ValueError((args.start_id, args.end_id))
    master = load_yaml(args.master)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads(Path(master["binary_data_audit"]).read_text())
    if not (
        audit["status"] == "PASS"
        and audit["sequence_hash_cross_split_violations"] == 0
        and audit["leakage_group_cross_split_violations"] == 0
    ):
        raise RuntimeError("Leakage audit failed")
    excluded = set()
    if args.exclude:
        excluded = set(
            pd.read_parquet(args.exclude, columns=["id"])["id"].astype(int)
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.90, 0)
        torch.cuda.reset_peak_memory_stats(device)
    if args.source_model == "binary_lamar":
        scorer, details, checkpoint = build_binary_scorer(
            master, device, args.batch_size
        )
    else:
        scorer, details, checkpoint = build_ranking_scorer(
            master,
            args.model_config,
            args.checkpoint,
            device,
            args.batch_size,
        )
    sqlite_path = Path(master["negative_pool_sqlite"]).resolve()
    connection = sqlite3.connect(
        f"file:{sqlite_path}?mode=ro", uri=True
    )
    expected = int(
        connection.execute(
            "select count(*) from negatives where id>? and id<=?",
            (args.start_id, args.end_id),
        ).fetchone()[0]
    )
    heap_size = min(expected, args.top_count + 5000)
    heap = []
    scanned = 0
    cursor = args.start_id
    started = time.time()
    while cursor < args.end_id:
        rows = fetch_range(
            connection, cursor, args.end_id, args.chunk_size
        )
        if not rows:
            break
        cursor = int(rows[-1]["id"])
        scores = scorer(rows)
        for row, score in zip(rows, scores):
            identifier = int(row["id"])
            if identifier in excluded:
                continue
            item = (float(score), identifier, row)
            if len(heap) < heap_size:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
        scanned += len(rows)
        if scanned % 200000 < len(rows):
            print(
                json.dumps(
                    {
                        "range": [args.start_id, args.end_id],
                        "scanned": scanned,
                        "expected": expected,
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )
    connection.close()
    if scanned != expected:
        raise AssertionError((scanned, expected))
    candidates = sorted(heap, key=lambda item: (-item[0], item[1]))
    selected = []
    seen = set()
    for score, identifier, row in candidates:
        if row["sequence_hash"] in seen:
            continue
        seen.add(row["sequence_hash"])
        value = dict(row)
        value.update(
            {
                "source_score": float(score),
                "source_model": args.source_model,
                "mining_round": args.round,
                "shard_start_id": args.start_id,
                "shard_end_id": args.end_id,
            }
        )
        selected.append(value)
        if len(selected) >= min(args.top_count, expected):
            break
    for rank, row in enumerate(selected, 1):
        row["shard_source_rank"] = rank
    write_frame_new(pd.DataFrame(selected), output)
    result = {
        "status": "PASS",
        "range": [args.start_id, args.end_id],
        "rows_scanned": scanned,
        "selected_rows": len(selected),
        "unique_sequence_hashes": len(seen),
        "source_model": args.source_model,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "excluded_ids": len(excluded),
        "details": details,
        "seconds": time.time() - started,
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    write_json_new(output.with_suffix(".manifest.json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
