#!/usr/bin/env python3
"""Train or smoke-test one Frozen + Head centered-window condition."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


def load_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def centered(sequence: str, width: int) -> str:
    flank = (width - 1) // 2
    cropped = sequence[50 - flank : 51 + flank]
    if len(cropped) != width or cropped[flank] != "C":
        raise RuntimeError(f"Invalid centered crop for width {width}")
    return cropped


class Records(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def make_collator(tokenizer, width: int):
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    base_ids = {base: tokenizer.convert_tokens_to_ids(base) for base in "ATCGN"}
    token_length = width + 2
    center_position = (width - 1) // 2 + 1

    def apply(rows: list[dict]) -> dict[str, torch.Tensor]:
        sequences = [centered(str(row["sequence_context"]), width) for row in rows]
        encoded = [
            [cls_id]
            + [base_ids.get(base, base_ids["N"]) for base in sequence]
            + [eos_id]
            for sequence in sequences
        ]
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), token_length), dtype=torch.long),
            "center_positions": torch.full(
                (len(rows),), center_position, dtype=torch.long
            ),
            "labels": torch.tensor(
                [float(row["label"]) for row in rows], dtype=torch.float32
            ),
        }

    return apply


def encode_dev(dev: pd.DataFrame, tokenizer, width: int) -> np.ndarray:
    cls_id = int(tokenizer.cls_token_id)
    eos_id = int(tokenizer.eos_token_id)
    base_ids = {
        ord(base): int(tokenizer.convert_tokens_to_ids(base)) for base in "ATCGN"
    }
    lookup = np.full(256, base_ids[ord("N")], dtype=np.int16)
    for key, value in base_ids.items():
        lookup[key] = value
    encoded = np.empty((len(dev), width + 2), dtype=np.int16)
    encoded[:, 0] = cls_id
    encoded[:, -1] = eos_id
    for index, source in enumerate(dev["sequence_context"].astype(str)):
        sequence = centered(source, width)
        raw = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        encoded[index, 1:-1] = lookup[raw]
    center_position = (width - 1) // 2 + 1
    if not np.all(encoded[:, center_position] == tokenizer.convert_tokens_to_ids("C")):
        raise RuntimeError("Encoded dev center token is not C")
    return encoded


@torch.no_grad()
def predict_encoded(model, encoded, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    output = np.empty(len(encoded), dtype=np.float32)
    center_position = (encoded.shape[1] - 2 - 1) // 2 + 1
    for start in range(0, len(encoded), batch_size):
        stop = min(start + batch_size, len(encoded))
        ids = torch.as_tensor(
            np.asarray(encoded[start:stop], dtype=np.int64), device=device
        )
        mask = torch.ones_like(ids)
        centers = torch.full(
            (stop - start,), center_position, dtype=torch.long, device=device
        )
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(ids, mask, centers)
        output[start:stop] = logits.float().cpu().numpy()
    return output


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -50, 50)))
    precision, recall, _ = precision_recall_curve(labels, probability)
    result = {
        "average_precision": float(average_precision_score(labels, probability)),
        "pr_auc": float(auc(recall, precision)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
    }
    order = np.argsort(-probability, kind="mergesort")
    for k in (10, 50, 100, 500, 1000):
        use = min(k, len(labels))
        hits = int(labels[order[:use]].sum())
        result[f"P@{k}"] = float(hits / use)
        result[f"Recall@{k}"] = float(hits / max(1, labels.sum()))
        result[f"positive_hits@{k}"] = hits
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--window", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    width = int(args.window)
    if width not in [int(value) for value in config["windows"]]:
        raise ValueError(f"Window {width} is not declared")
    output_root = Path(config["output_dir"]).resolve()
    preflight_path = output_root / "results/preflight_report.json"
    if not preflight_path.is_file():
        raise RuntimeError("PREFLIGHT report is missing")
    preflight = json.loads(preflight_path.read_text())
    if preflight["status"] != "PREFLIGHT_OK":
        raise RuntimeError(f"Preflight status is {preflight['status']}")

    frozen = config["frozen_conditions"]
    seed = int(frozen["seed"])
    seed_everything(seed)
    model_run = Path(config["model_run_dir"]).resolve()
    dataset = Path(config["dataset_dir"]).resolve()
    model_master = load_yaml(model_run / "configs/master.yaml")
    sys.path.insert(0, str(model_run / "scripts"))
    from common import NegativePool, read_tsv_records
    from modeling_binary import make_model, save_trainable

    tokenizer = AutoTokenizer.from_pretrained(
        model_master["tokenizer"], local_files_only=True, model_max_length=103
    )
    run_config = {
        "mode": "frozen",
        "pooling": "center",
        "head_dropout": float(frozen["head_dropout"]),
        "head_lr": float(frozen["head_lr"]),
        "seed": seed,
    }
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            float(frozen["cuda_memory_fraction"]), device=device
        )
    model, details = make_model(model_master, run_config, tokenizer)
    model.to(device)
    positives = read_tsv_records(dataset / "train_positives.tsv.gz")
    pool = NegativePool(model_run / "work/train_pool.sqlite", seed)
    collator = make_collator(tokenizer, width)

    mode_name = "smoke" if args.smoke else "formal"
    run_dir = output_root / "runs" / f"window_{width}bp_{mode_name}"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "window": width,
                "left_flank": (width - 1) // 2,
                "center": "C",
                "right_flank": (width - 1) // 2,
                "frozen_conditions": frozen,
                "model_details": details,
                "device": str(device),
                "smoke": args.smoke,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(frozen["head_lr"]),
        weight_decay=float(frozen["weight_decay"]),
    )
    formal_total_steps = int(frozen["optimizer_steps"])
    total_steps = 1 if args.smoke else formal_total_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(formal_total_steps * float(frozen["warmup_ratio"])),
        formal_total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(frozen["fp16"] and device.type == "cuda")
    )
    batch_size = int(frozen["batch_size"])
    accumulation = int(frozen["accumulation_steps"])
    n_negative = len(positives) * int(frozen["sampling_ratio"])
    global_step = 0
    history = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    for epoch in range(int(frozen["epochs"])):
        negative_ids = pool.ids_for_epoch(
            n_negative, str(frozen["negative_strategy"]), epoch
        )
        rows = positives + pool.fetch(negative_ids)
        random.Random(seed + epoch).shuffle(rows)
        if args.smoke:
            rows = rows[:batch_size]
        loader = DataLoader(
            Records(rows),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batches = 0
        for batch_index, batch in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                    batch["center_positions"].to(device, non_blocking=True),
                )
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, batch["labels"].to(device, non_blocking=True)
                ) / accumulation
            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * accumulation
            batches += 1
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable, float(frozen["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step >= total_steps:
                    break
        history.append(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "mean_batch_loss": running_loss / max(1, batches),
                "negative_selection_sha256": hashlib.sha256(
                    ",".join(
                        str(value) for value in sorted(int(x) for x in negative_ids)
                    ).encode("ascii")
                ).hexdigest(),
            }
        )
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if global_step >= total_steps:
            break
    if global_step != total_steps:
        raise RuntimeError(f"Expected {total_steps} steps, observed {global_step}")

    if args.smoke:
        dev = pd.read_parquet(config["dev_universe"]).head(32).copy()
    else:
        dev = pd.read_parquet(config["dev_universe"]).copy()
    encoded = encode_dev(dev, tokenizer, width)
    eval_batch_size = min(
        int(frozen["eval_batch_size"]), 32 if device.type == "cpu" else 10**9
    )
    logits = predict_encoded(model, encoded, eval_batch_size, device)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -50, 50)))
    labels = dev["label"].to_numpy(dtype=np.int64)
    result_metrics = metrics(labels, logits) if len(np.unique(labels)) == 2 else {}
    predictions = dev[
        ["sequence_id", "genomic_key", "split", "label", "sequence_context"]
    ].copy()
    predictions["window"] = width
    predictions["raw_logit"] = logits
    predictions["probability"] = probability
    prediction_path = run_dir / "dev_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    checkpoint_path = run_dir / "final_trainable.safetensors"
    save_trainable(model, checkpoint_path)
    summary = {
        "status": "SMOKE_OK" if args.smoke else "SUCCESS",
        "window": width,
        "definition": f"{(width - 1) // 2} + C + {(width - 1) // 2}",
        "optimizer_steps": global_step,
        "history": history,
        "dev_rows": len(dev),
        "dev_metrics": result_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "elapsed_seconds": time.time() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "calibration_or_test_accessed": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
