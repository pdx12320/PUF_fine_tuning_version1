#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer

from ablation_common import META_COLUMNS, load_config, load_split_rows, sha256_file, write_json


class RowDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def collator(tokenizer):
    cls, eos = tokenizer.cls_token_id, tokenizer.eos_token_id
    ids = {base: tokenizer.convert_tokens_to_ids(base) for base in "ATCGN"}
    def apply(rows):
        encoded = [[cls] + [ids.get(base, ids["N"]) for base in row["sequence_context"]] + [eos] for row in rows]
        return rows, torch.tensor(encoded, dtype=torch.long), torch.ones((len(rows), 103), dtype=torch.long)
    return apply


def fixed_list(array: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(array.reshape(-1), type=pa.float16())
    return pa.FixedSizeListArray.from_arrays(flat, array.shape[1])


def arrow_table(rows, center, mean, masked_mean, cls):
    values = {column: pa.array([row[column] for row in rows]) for column in META_COLUMNS}
    values.update({
        "embedding_center": fixed_list(center),
        "embedding_mean": fixed_list(mean),
        "embedding_masked_mean": fixed_list(masked_mean),
        "embedding_cls": fixed_list(cls),
    })
    return pa.table(values)


def build_model(cfg, tokenizer):
    sys.path.insert(0, cfg["lamar_repo"])
    from LAMAR.modeling_nucESM2 import EsmForMaskedLM
    config = AutoConfig.from_pretrained(
        cfg["architecture_config"], vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id,
        token_dropout=False, positional_embedding_type="rotary", hidden_size=768,
        intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12,
    )
    model = EsmForMaskedLM(config)
    state = load_file(cfg["pretrained_checkpoint"])
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(incompatible)
    for parameter in model.parameters():
        parameter.requires_grad = False
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Pretrained model was not fully frozen")
    return model, len(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "dev", "calibration", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int, default=-1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows, source_manifest = load_split_rows(cfg, args.split)
    if args.max_rows > 0:
        rows = rows[:args.max_rows]
    for row in rows:
        if len(row["sequence_context"]) != 101 or row["sequence_context"][50] != "C":
            raise AssertionError((row["genomic_key"], len(row["sequence_context"]), row["sequence_context"][50]))
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], local_files_only=True, model_max_length=103)
    model, tensor_count = build_model(cfg, tokenizer)
    backbone_parameters = sum(p.numel() for p in model.esm.parameters())
    pretrained_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_parameters != 0:
        raise AssertionError(trainable_parameters)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.80, 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(
        RowDataset(rows), batch_size=int(cfg["embedding_batch_size"]), shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collator(tokenizer),
    )
    writer = None
    max_mean_difference = 0.0
    started = time.time()
    with torch.inference_mode():
        for metadata, input_ids, attention_mask in loader:
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                hidden = model.esm(
                    input_ids=input_ids.to(device, non_blocking=True),
                    attention_mask=attention_mask.to(device, non_blocking=True),
                    return_dict=True,
                ).last_hidden_state
            if hidden.shape[1:] != (103, 768):
                raise AssertionError(tuple(hidden.shape))
            center = hidden[:, 51]
            nucleotide = hidden[:, 1:102]
            mean = nucleotide.mean(1)
            sequence_mask = attention_mask.to(device).bool()
            sequence_mask[:, 0] = False
            sequence_mask[:, -1] = False
            masked_mean = (hidden * sequence_mask.unsqueeze(-1)).sum(1) / sequence_mask.sum(1, keepdim=True)
            cls = hidden[:, 0]
            max_mean_difference = max(max_mean_difference, float((mean - masked_mean).abs().max().item()))
            arrays = [x.detach().cpu().to(torch.float16).numpy() for x in (center, mean, masked_mean, cls)]
            table = arrow_table(metadata, *arrays)
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output, table.schema, compression="zstd", use_dictionary=True)
            writer.write_table(table, row_group_size=len(metadata))
    if writer is not None:
        writer.close()
    if max_mean_difference > 1e-5:
        raise AssertionError(max_mean_difference)
    elapsed = time.time() - started
    manifest = {
        "status": "PASS", "split": args.split, "rows": len(rows),
        "label_1": sum(row["label"] for row in rows),
        "label_0": len(rows) - sum(row["label"] for row in rows),
        "sequence_length": 101, "raw_center_index": 50,
        "model_input_shape": [len(rows), 103],
        "hidden_state_shape_per_batch": ["batch", 103, 768],
        "checkpoint_path": cfg["pretrained_checkpoint"],
        "checkpoint_sha256": sha256_file(cfg["pretrained_checkpoint"]),
        "checkpoint_tensor_count": tensor_count,
        "pretrained_model_parameters_including_MLM_head": pretrained_parameters,
        "backbone_parameters": backbone_parameters,
        "trainable_parameters": trainable_parameters,
        "requires_grad_all_false": True,
        "pooling": {
            "center": "hidden[:, 51] corresponding to raw nucleotide index 50",
            "mean": "mean of 101 nucleotide tokens hidden[:,1:102]",
            "masked_mean": "attention-mask mean excluding CLS/EOS/padding",
            "mean_masked_mean_max_abs_difference": max_mean_difference,
            "cls": "hidden[:,0]; architecture and tokenizer support CLS",
        },
        "source": source_manifest, "seconds": elapsed,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "output": str(output), "output_sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
