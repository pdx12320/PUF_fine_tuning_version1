#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from modeling_ranking import load_trainable, make_lamar_ranker
from ranking_common import load_yaml, sha256_file, write_frame_new, write_json_new
from train_pairwise import score_rows


COLUMNS = (
    "id,chrom,position,genomic_key,gene_id,gene_name,transcript_ids,"
    "seq,sequence_hash,leakage_group,gc,entropy,c_count,median_depth,"
    "gene_coverage,negative_type,matched,hard"
)


def sqlite_rows(connection, start_id: int, count: int) -> list[dict]:
    connection.row_factory = sqlite3.Row
    result = connection.execute(
        f"select {COLUMNS} from negatives where id>? order by id limit ?",
        (int(start_id), int(count)),
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


def build_binary_scorer(master, device, batch_size):
    scripts = str(master["binary_scripts_dir"])
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from modeling_binary import load_trainable as load_binary_trainable
    from modeling_binary import make_model as make_binary_model
    from train_lamar import predict

    run_config = json.loads(Path(master["binary_run_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        master["tokenizer"],
        local_files_only=True,
        model_max_length=103,
    )
    binary_master = {
        "lamar_repo": master["lamar_repo"],
        "tokenizer": master["tokenizer"],
        "architecture_config": master["architecture_config"],
        "base_state": master["pretrained_checkpoint"],
    }
    model, details = make_binary_model(
        binary_master, run_config, tokenizer
    )
    checkpoint = Path(master["binary_checkpoint"])
    load_binary_trainable(model, checkpoint)
    model.to(device).eval()

    def apply(rows):
        return predict(
            model, rows, tokenizer, batch_size, device
        ).astype(np.float64)

    return apply, details, checkpoint


def build_ranking_scorer(
    master, config_path, checkpoint, device, batch_size
):
    run_config = json.loads(Path(config_path).read_text())
    if run_config["model_type"] not in {"lora_lamar", "hybrid_lamar"}:
        raise ValueError(run_config["model_type"])
    tokenizer = AutoTokenizer.from_pretrained(
        master["tokenizer"],
        local_files_only=True,
        model_max_length=103,
    )
    model, details = make_lamar_ranker(master, run_config, tokenizer)
    load_trainable(model, checkpoint)
    model.to(device).eval()

    def apply(rows):
        return score_rows(
            model,
            rows,
            run_config["model_type"],
            device,
            batch_size,
            tokenizer,
        )

    return apply, details, Path(checkpoint)


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
    parser.add_argument("--top-count", type=int, default=50000)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--exclude")
    parser.add_argument("--max-rows", type=int, default=-1)
    args = parser.parse_args()

    master = load_yaml(args.master)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads(Path(master["binary_data_audit"]).read_text())
    required_pass = (
        audit["status"] == "PASS"
        and audit["sequence_hash_cross_split_violations"] == 0
        and audit["leakage_group_cross_split_violations"] == 0
    )
    if not required_pass:
        raise RuntimeError("Inherited binary leakage audit did not pass")
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
        if not args.model_config or not args.checkpoint:
            raise ValueError(
                "Round-2 ranking mining requires config and checkpoint"
            )
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
    total_pool = int(
        connection.execute("select count(*) from negatives").fetchone()[0]
    )
    scan_limit = (
        min(total_pool, args.max_rows)
        if args.max_rows > 0
        else total_pool
    )
    heap_size = min(scan_limit, args.top_count + 5000)
    heap = []
    scanned = 0
    last_id = 0
    started = time.time()
    while scanned < scan_limit:
        count = min(args.chunk_size, scan_limit - scanned)
        rows = sqlite_rows(connection, last_id, count)
        if not rows:
            break
        last_id = int(rows[-1]["id"])
        scores = scorer(rows)
        for row, score in zip(rows, scores):
            identifier = int(row["id"])
            if identifier in excluded:
                continue
            if not row["leakage_group"] or not row["sequence_hash"]:
                raise AssertionError(identifier)
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
                        "scanned": scanned,
                        "scan_limit": scan_limit,
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )
    connection.close()
    candidates = sorted(heap, key=lambda item: (-item[0], item[1]))
    selected = []
    seen_sequences = set()
    for score, identifier, row in candidates:
        seq_hash = row["sequence_hash"]
        if seq_hash in seen_sequences:
            continue
        seen_sequences.add(seq_hash)
        record = dict(row)
        record["source_score"] = float(score)
        record["source_model"] = args.source_model
        record["mining_round"] = args.round
        selected.append(record)
        if len(selected) >= args.top_count:
            break
    if len(selected) < min(args.top_count, scan_limit - len(excluded)):
        raise RuntimeError(
            f"Only {len(selected)} unique hard negatives selected"
        )
    for rank, row in enumerate(selected, 1):
        row["source_rank"] = rank
    frame = pd.DataFrame(selected)
    write_frame_new(frame, output)
    manifest = {
        "status": "PASS",
        "mining_round": args.round,
        "source_model": args.source_model,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "train_pool_rows": total_pool,
        "rows_scanned": scanned,
        "selected_rows": len(frame),
        "unique_sequence_hashes": int(frame["sequence_hash"].nunique()),
        "excluded_prior_round_ids": len(excluded),
        "filters": [
            "immutable train-only strict-negative SQLite source",
            "inherited zero cross-split leakage-group violations",
            "unique sequence_hash",
            "exclude prior mining round IDs when supplied",
        ],
        "dev_calibration_test_scored_for_mining": False,
        "details": details,
        "seconds": time.time() - started,
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    write_json_new(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
