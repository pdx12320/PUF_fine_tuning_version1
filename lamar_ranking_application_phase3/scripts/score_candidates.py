#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from modeling_ranking import (
    CNNRanker,
    load_trainable,
    make_lamar_ranker,
)
from ranking_common import (
    META_COLUMNS,
    deterministic_random_scores,
    load_yaml,
    sha256_file,
    write_frame_new,
    write_json_new,
)
from train_kmer_ranker import feature_matrix
from train_pairwise import record_collator, forward_batch


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def load_metadata(path: Path) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [column for column in META_COLUMNS if column in available]
    frame = pd.read_parquet(path, columns=columns)
    missing = set(META_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing metadata columns {sorted(missing)}")
    return frame


@torch.inference_mode()
def neural_scores(model, rows, model_type, tokenizer, device, batch_size):
    model.eval()
    loader = DataLoader(
        Rows(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=record_collator(model_type, tokenizer),
    )
    output = []
    for batch in loader:
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            score = forward_batch(model, batch, model_type, device)
        output.append(score.float().cpu().numpy())
    return np.concatenate(output)


def score_binary(master, frame, device, batch_size):
    binary_scripts = str(master["binary_scripts_dir"])
    if binary_scripts not in sys.path:
        sys.path.insert(0, binary_scripts)
    from modeling_binary import load_trainable as load_binary_trainable
    from modeling_binary import make_model as make_binary_model
    from train_lamar import predict as binary_predict

    run_config = json.loads(
        Path(master["binary_run_config"]).read_text()
    )
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
    rows = frame[["sequence_context", "label"]].to_dict("records")
    probability = binary_predict(
        model, rows, tokenizer, batch_size, device
    )
    return probability.astype(np.float64), details, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument(
        "--model-type",
        choices=(
            "random",
            "kmer",
            "cnn",
            "frozen_lamar",
            "lora_lamar",
            "hybrid_lamar",
            "binary_lamar",
        ),
        required=True,
    )
    parser.add_argument("--model-config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    master = load_yaml(args.master)
    source = Path(args.input)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    frame = load_metadata(source)
    rows = frame.to_dict("records")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.90, 0)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    details = {}
    checkpoint = None

    if args.model_type == "random":
        scores = deterministic_random_scores(
            frame["sequence_id"].astype(str), args.seed
        )
        details = {"random_seed": args.seed, "method": "SHA256 uniform score"}
    elif args.model_type == "kmer":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required")
        checkpoint = Path(args.checkpoint)
        saved = joblib.load(checkpoint)
        scores = saved["model"].decision_function(
            feature_matrix(
                rows, saved["vectorizer"], saved["scaler"]
            )
        )
        details = {
            "trainable_parameters": int(saved["model"].coef_.size),
            "numeric_features": saved["numeric_features"],
        }
    elif args.model_type == "binary_lamar":
        scores, details, checkpoint = score_binary(
            master, frame, device, args.batch_size
        )
    else:
        if not args.checkpoint or not args.model_config:
            raise ValueError(
                "--checkpoint and --model-config are required"
            )
        checkpoint = Path(args.checkpoint)
        run_config = json.loads(Path(args.model_config).read_text())
        if run_config["model_type"] != args.model_type:
            raise ValueError(
                (run_config["model_type"], args.model_type)
            )
        if args.model_type == "cnn":
            model = CNNRanker()
            model.load_state_dict(
                torch.load(checkpoint, map_location=device)
            )
            tokenizer = None
            details = {
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                )
            }
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                master["tokenizer"],
                local_files_only=True,
                model_max_length=103,
            )
            model, details = make_lamar_ranker(
                master, run_config, tokenizer
            )
            load_trainable(model, checkpoint)
        model.to(device).eval()
        scores = neural_scores(
            model,
            rows,
            args.model_type,
            tokenizer,
            device,
            args.batch_size,
        )

    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise AssertionError((len(scores), len(frame)))
    prediction = frame[META_COLUMNS].copy()
    prediction["model_id"] = args.model_id
    prediction["ranking_score"] = scores
    write_frame_new(prediction, output)
    manifest = {
        "status": "PASS",
        "model_id": args.model_id,
        "model_type": args.model_type,
        "rows": len(frame),
        "positive_count": int(frame["label"].sum()),
        "negative_count": int((frame["label"] == 0).sum()),
        "source": str(source),
        "source_sha256": sha256_file(source),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": (
            sha256_file(checkpoint) if checkpoint else None
        ),
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
