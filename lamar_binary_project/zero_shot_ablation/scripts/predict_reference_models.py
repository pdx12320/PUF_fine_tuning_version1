#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from ablation_common import embedding_matrix, embedding_metadata, load_config, write_json


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 64, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 7, padding=3), nn.ReLU(), nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(128, 1))
    def forward(self, x):
        return self.head(self.net(x).squeeze(-1)).squeeze(-1)


def onehot(sequences):
    index = {base: i for i, base in enumerate("ACGT")}
    array = np.zeros((len(sequences), 4, 101), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for column, base in enumerate(sequence):
            if base in index:
                array[row, index[base], column] = 1
    return torch.from_numpy(array)


@torch.inference_mode()
def cnn_probability(model, sequences, device):
    model.eval()
    result = []
    tensor = onehot(sequences)
    for (batch,) in DataLoader(TensorDataset(tensor), batch_size=1024):
        result.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(result)


def kmer_probability(existing, metadata):
    saved = joblib.load(existing / "checkpoints/baselines/kmer_logistic.joblib")
    numeric = metadata[["gc_fraction", "c_count", "entropy"]].to_numpy(dtype=np.float64)
    x = saved["vectorizer"].transform(metadata.sequence_context)
    x = hstack([x, csr_matrix(saved["scaler"].transform(numeric))])
    return saved["model"].predict_proba(x)[:, 1]


def frozen_head_probability(existing, embedding_path, device):
    state = load_file(str(existing / "checkpoints/runs/s1_frozen_center/best_trainable.safetensors"))
    required = {
        "classifier.0.weight", "classifier.0.bias",
        "classifier.2.weight", "classifier.2.bias",
    }
    if set(state) != required:
        raise AssertionError(sorted(state))
    embedding = torch.from_numpy(embedding_matrix(embedding_path, "center"))
    result = []
    weight = state["classifier.0.weight"].to(device)
    bias = state["classifier.0.bias"].to(device)
    linear_weight = state["classifier.2.weight"].to(device)
    linear_bias = state["classifier.2.bias"].to(device)
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(embedding), batch_size=4096):
            value = nn.functional.layer_norm(batch.to(device), (768,), weight, bias, 1e-5)
            logits = nn.functional.linear(value, linear_weight, linear_bias).squeeze(-1)
            result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result)


def adapted_probability(cfg, existing, name, metadata, device):
    sys.path.insert(0, str(existing / "scripts"))
    sys.path.insert(0, cfg["lamar_repo"])
    from modeling_binary import load_trainable, make_model
    from train_lamar import predict
    run_config = json.loads((existing / f"configs/{name}.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer"], local_files_only=True, model_max_length=103
    )
    master = {
        "lamar_repo": cfg["lamar_repo"], "tokenizer": cfg["tokenizer"],
        "architecture_config": cfg["architecture_config"],
        "base_state": cfg["pretrained_checkpoint"],
    }
    model, details = make_model(master, run_config, tokenizer)
    checkpoint = existing / f"checkpoints/runs/{name}/best_trainable.safetensors"
    load_trainable(model, checkpoint)
    model.to(device).eval()
    rows = metadata[["sequence_context", "label"]].to_dict("records")
    probability = predict(model, rows, tokenizer, 256, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probability, details


def existing_lora_probability(existing, split, metadata):
    source = existing / (
        "calibration_predictions.parquet" if split == "calibration"
        else "test_predictions.parquet" if split == "test"
        else "checkpoints/runs/final_seed42/dev_predictions.parquet"
    )
    frame = pd.read_parquet(source)
    probability_column = "raw_probability" if "raw_probability" in frame.columns else "probability"
    indexed = frame.set_index("genomic_key")[probability_column]
    probability = indexed.loc[metadata.genomic_key].to_numpy(dtype=np.float64)
    return probability, str(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("dev", "calibration", "test"), required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    existing = Path(cfg["existing_model_dir"])
    embedding_path = Path(args.embedding)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    metadata = embedding_metadata(embedding_path)
    if set(metadata.split) != {args.split}:
        raise AssertionError(metadata.split.value_counts().to_dict())
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.80, 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = metadata[["sequence_id", "split", "label", "genomic_key"]].copy()
    details = {}

    frame["kmer_logistic"] = kmer_probability(existing, metadata)
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load(existing / "checkpoints/baselines/cnn.pt", map_location=device))
    frame["cnn"] = cnn_probability(cnn, metadata.sequence_context.tolist(), device)
    del cnn
    if device.type == "cuda":
        torch.cuda.empty_cache()
    frame["frozen_lamar_head"] = frozen_head_probability(existing, embedding_path, device)
    frame["partial_lamar_2blocks"], details["partial_lamar_2blocks"] = adapted_probability(
        cfg, existing, "s1_partial_u2", metadata, device
    )
    frame["full_lamar"], details["full_lamar"] = adapted_probability(
        cfg, existing, "s1_full", metadata, device
    )
    frame["lora_best"], lora_source = existing_lora_probability(existing, args.split, metadata)
    details["lora_source"] = lora_source
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    manifest = {
        "status": "PASS", "split": args.split, "rows": len(frame),
        "models": [column for column in frame.columns if column not in {"sequence_id", "split", "label", "genomic_key"}],
        "no_training_performed": True, "details": details,
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
