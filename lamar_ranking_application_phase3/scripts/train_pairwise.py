#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from modeling_ranking import (
    CNNRanker,
    load_trainable,
    make_lamar_ranker,
    save_trainable,
)
from ranking_common import (
    NegativePool,
    load_yaml,
    positive_indices,
    ranking_key,
    ranking_metrics,
    read_tsv_records,
    seed_everything,
    write_frame_new,
    write_json_new,
)


class PairDataset(Dataset):
    def __init__(self, positives: list[dict], negatives: list[dict]):
        if len(positives) != len(negatives):
            raise ValueError((len(positives), len(negatives)))
        self.positives = positives
        self.negatives = negatives

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, index):
        return self.positives[index], self.negatives[index]


class RecordDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def nucleotide_ids(tokenizer) -> dict[str, int]:
    return {
        base: tokenizer.convert_tokens_to_ids(base)
        for base in "ACGTN"
    }


def encode_sequences(sequences: list[str], tokenizer) -> dict[str, torch.Tensor]:
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    ids = nucleotide_ids(tokenizer)
    encoded = [
        [cls_id] + [ids.get(base, ids["N"]) for base in sequence] + [eos_id]
        for sequence in sequences
    ]
    return {
        "input_ids": torch.tensor(encoded, dtype=torch.long),
        "attention_mask": torch.ones((len(encoded), 103), dtype=torch.long),
        "center_positions": torch.full(
            (len(encoded),), 51, dtype=torch.long
        ),
    }


def onehot(sequences: list[str]) -> torch.Tensor:
    index = {base: position for position, base in enumerate("ACGT")}
    array = np.zeros((len(sequences), 4, 101), dtype=np.float32)
    for row_index, sequence in enumerate(sequences):
        for column, base in enumerate(sequence):
            if base in index:
                array[row_index, index[base], column] = 1.0
    return torch.from_numpy(array)


def pair_collator(model_type: str, tokenizer=None):
    def apply(pairs):
        positive, negative = zip(*pairs)
        sequences = [row["sequence_context"] for row in positive]
        sequences += [row["sequence_context"] for row in negative]
        if model_type == "cnn":
            return onehot(sequences), len(positive)
        return encode_sequences(sequences, tokenizer), len(positive)

    return apply


def record_collator(model_type: str, tokenizer=None):
    def apply(rows):
        sequences = [row["sequence_context"] for row in rows]
        if model_type == "cnn":
            return onehot(sequences)
        return encode_sequences(sequences, tokenizer)

    return apply


def forward_batch(model, batch, model_type: str, device):
    if model_type == "cnn":
        return model(batch.to(device, non_blocking=True))
    return model(
        batch["input_ids"].to(device, non_blocking=True),
        batch["attention_mask"].to(device, non_blocking=True),
        batch["center_positions"].to(device, non_blocking=True),
    )


def rank_loss(
    positive_score: torch.Tensor,
    negative_score: torch.Tensor,
    loss_name: str,
    margin: float,
) -> torch.Tensor:
    difference = positive_score - negative_score
    if loss_name == "pairwise_logistic":
        return nn.functional.softplus(-difference).mean()
    if loss_name == "margin":
        return torch.relu(float(margin) - difference).mean()
    raise ValueError(loss_name)


def total_loss(
    positive_score: torch.Tensor,
    negative_score: torch.Tensor,
    run_config: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    ranking = rank_loss(
        positive_score,
        negative_score,
        run_config["loss"],
        float(run_config.get("margin", 0.5)),
    )
    if run_config["model_type"] != "hybrid_lamar":
        return ranking, {
            "ranking_loss": float(ranking.detach().item()),
            "bce_loss": 0.0,
        }
    positive_target = torch.ones_like(positive_score)
    negative_target = torch.zeros_like(negative_score)
    bce = 0.5 * (
        nn.functional.binary_cross_entropy_with_logits(
            positive_score, positive_target
        )
        + nn.functional.binary_cross_entropy_with_logits(
            negative_score, negative_target
        )
    )
    value = bce + float(run_config["lambda_rank"]) * ranking
    return value, {
        "ranking_loss": float(ranking.detach().item()),
        "bce_loss": float(bce.detach().item()),
    }


@torch.inference_mode()
def score_rows(
    model,
    rows: list[dict],
    model_type: str,
    device,
    batch_size: int,
    tokenizer=None,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        RecordDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=record_collator(model_type, tokenizer),
    )
    result = []
    for batch in loader:
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            score = forward_batch(model, batch, model_type, device)
        result.append(score.float().cpu().numpy())
    return np.concatenate(result)


def build_model(master: dict, run_config: dict, device):
    model_type = run_config["model_type"]
    if model_type == "cnn":
        model = CNNRanker()
        tokenizer = None
        details = {
            "mode": "cnn",
            "sequence_input": "one-hot A/C/G/T only",
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "total_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            master["tokenizer"],
            local_files_only=True,
            model_max_length=103,
        )
        model, details = make_lamar_ranker(master, run_config, tokenizer)
    model.to(device)
    return model, tokenizer, details


def save_checkpoint_new(model, model_type: str, path: Path) -> None:
    if model_type == "cnn":
        if path.exists():
            raise FileExistsError(path)
        torch.save(model.state_dict(), path)
    else:
        save_trainable(model, path)


def load_checkpoint(model, model_type: str, path: Path, device) -> dict:
    if model_type == "cnn":
        model.load_state_dict(torch.load(path, map_location=device))
        return {"loaded": str(path)}
    return load_trainable(model, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    master = load_yaml(args.master)
    run_config = json.loads(Path(args.run_config).read_text())
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json_new(output / "config.json", run_config)

    seed = int(run_config["seed"])
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(
            float(run_config.get("gpu_memory_fraction", 0.90)), 0
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset_dir = Path(master["dataset_dir"])
    positive_rows = read_tsv_records(
        dataset_dir / "train_positives.tsv.gz", "train"
    )
    dev_rows = read_tsv_records(dataset_dir / "dev_1to10.tsv.gz", "dev")
    guided_paths = [
        str(path)
        for path in run_config.get("guided_negative_paths", [])
    ]
    pool = NegativePool(
        master["negative_pool_sqlite"],
        seed=seed,
        guided_paths=guided_paths,
    )
    model, tokenizer, details = build_model(master, run_config, device)
    model_type = run_config["model_type"]

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(run_config["learning_rate"]),
        weight_decay=float(run_config.get("weight_decay", 0.0)),
    )
    epochs = int(run_config.get("epochs", 20))
    pairs_per_epoch = int(run_config.get("pairs_per_epoch", 10280))
    pair_batch_size = int(run_config.get("pair_batch_size", 16))
    accumulation = int(run_config.get("accumulation_steps", 1))
    optimizer_steps_per_epoch = math.ceil(
        math.ceil(pairs_per_epoch / pair_batch_size) / accumulation
    )
    total_steps = max(1, optimizer_steps_per_epoch * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * float(run_config.get("warmup_ratio", 0.03))),
        total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(run_config.get("fp16", True) and device.type == "cuda")
    )

    history = []
    best_key = None
    best_checkpoint = None
    patience_count = 0
    global_step = 0
    started = time.time()
    for epoch in range(epochs):
        negative_ids, sampling_manifest = pool.ids_for_epoch(
            pairs_per_epoch,
            run_config["negative_sampling"],
            epoch,
        )
        negatives = pool.fetch(negative_ids)
        indices = positive_indices(
            len(positive_rows), pairs_per_epoch, seed, epoch
        )
        positives = [positive_rows[int(index)] for index in indices]
        loader = DataLoader(
            PairDataset(positives, negatives),
            batch_size=pair_batch_size,
            shuffle=False,
            num_workers=int(run_config.get("workers", 0)),
            pin_memory=device.type == "cuda",
            collate_fn=pair_collator(model_type, tokenizer),
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        aggregate_loss = 0.0
        aggregate_rank = 0.0
        aggregate_bce = 0.0
        batch_count = 0
        stop_now = False
        for batch_index, (batch, positive_count) in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                scores = forward_batch(model, batch, model_type, device)
                positive_score = scores[:positive_count]
                negative_score = scores[positive_count:]
                loss, pieces = total_loss(
                    positive_score, negative_score, run_config
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            aggregate_loss += float(loss.detach().item())
            aggregate_rank += pieces["ranking_loss"]
            aggregate_bce += pieces["bce_loss"]
            batch_count += 1
            if (batch_index + 1) % accumulation == 0 or (
                batch_index + 1 == len(loader)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_now = True
                    break

        dev_score = score_rows(
            model,
            dev_rows,
            model_type,
            device,
            int(run_config.get("eval_batch_size", 256)),
            tokenizer,
        )
        metrics = ranking_metrics(
            [row["label"] for row in dev_rows],
            dev_score,
            [row["sequence_id"] for row in dev_rows],
        )
        current_key = ranking_key(metrics)
        record = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "pair_count": pairs_per_epoch,
            "actual_sequence_forwards": pairs_per_epoch * 2,
            "mean_total_loss": aggregate_loss / max(1, batch_count),
            "mean_ranking_loss": aggregate_rank / max(1, batch_count),
            "mean_bce_loss": aggregate_bce / max(1, batch_count),
            "sampling": sampling_manifest,
            "dev_metrics": metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if best_key is None or current_key > best_key:
            best_key = current_key
            patience_count = 0
            suffix = ".pt" if model_type == "cnn" else ".safetensors"
            best_checkpoint = output / f"checkpoint_epoch{epoch + 1:02d}{suffix}"
            save_checkpoint_new(model, model_type, best_checkpoint)
        else:
            patience_count += 1
        if stop_now:
            break
        if patience_count >= int(run_config.get("patience", 3)):
            break

    if best_checkpoint is None:
        raise RuntimeError("No checkpoint was created")
    load_report = load_checkpoint(
        model, model_type, best_checkpoint, device
    )
    best_dev_score = score_rows(
        model,
        dev_rows,
        model_type,
        device,
        int(run_config.get("eval_batch_size", 256)),
        tokenizer,
    )
    best_metrics = ranking_metrics(
        [row["label"] for row in dev_rows],
        best_dev_score,
        [row["sequence_id"] for row in dev_rows],
    )
    prediction = pd.DataFrame(dev_rows)
    prediction["ranking_score"] = best_dev_score
    write_frame_new(prediction, output / "dev_fixed_predictions.parquet")
    summary = {
        "status": "SUCCESS",
        "model_type": model_type,
        "config": run_config,
        "details": details,
        "checkpoint_load": load_report,
        "best_checkpoint": str(best_checkpoint),
        "best_dev_fixed_metrics": best_metrics,
        "history": history,
        "epochs_completed": len(history),
        "pairs_per_epoch": pairs_per_epoch,
        "actual_pairs_all_epochs": pairs_per_epoch * len(history),
        "actual_sequence_forwards_all_epochs": (
            pairs_per_epoch * 2 * len(history)
        ),
        "training_seconds": time.time() - started,
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "trainable_parameters": details["trainable_parameters"],
        "total_parameters": details["total_parameters"],
        "device": str(device),
    }
    write_json_new(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
