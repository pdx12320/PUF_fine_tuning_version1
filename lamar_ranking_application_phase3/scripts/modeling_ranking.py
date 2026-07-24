#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig


def build_config(config_path: str | Path, tokenizer):
    return AutoConfig.from_pretrained(
        str(config_path),
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        num_labels=1,
        problem_type="single_label_classification",
        token_dropout=False,
        positional_embedding_type="rotary",
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
    )


class LoRALinear(nn.Module):
    def __init__(
        self, base: nn.Linear, rank: int, alpha: int, dropout: float
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, value):
        delta = self.lora_B(self.lora_A(self.dropout(value)))
        return self.base(value) + delta * self.scaling


class RankingMLP(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        bottleneck = 256
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.02)
                nn.init.zeros_(module.bias)

    def forward(self, value):
        return self.net(value).squeeze(-1)


class LamarRanker(nn.Module):
    def __init__(
        self,
        config,
        esm_model_cls,
        pooling: str = "center",
        dropout: float = 0.1,
        frozen_backbone: bool = False,
    ) -> None:
        super().__init__()
        if pooling != "center":
            raise ValueError("Phase 3 preregisters center pooling only")
        self.config = config
        self.pooling = pooling
        self.frozen_backbone = frozen_backbone
        self.esm = esm_model_cls(config, add_pooling_layer=False)
        self.ranking_head = RankingMLP(config.hidden_size, dropout)

    def encode(self, input_ids, attention_mask, center_positions):
        hidden = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, center_positions.long()]

    def forward(self, input_ids, attention_mask, center_positions):
        if self.frozen_backbone:
            self.esm.eval()
            with torch.no_grad():
                representation = self.encode(
                    input_ids, attention_mask, center_positions
                )
        else:
            representation = self.encode(
                input_ids, attention_mask, center_positions
            )
        return self.ranking_head(representation)


class CNNRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 64, 9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 7, padding=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.ranking_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, value):
        embedding = self.encoder(value).squeeze(-1)
        return self.ranking_head(embedding).squeeze(-1)


def load_pretrained(model: LamarRanker, state_path: str | Path) -> dict:
    state = load_file(str(state_path))
    incompatible = model.load_state_dict(state, strict=False)
    missing_encoder = [
        key for key in incompatible.missing_keys if key.startswith("esm.")
    ]
    if missing_encoder:
        raise RuntimeError(missing_encoder[:20])
    return {
        "tensor_count": len(state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def inject_qkvo_lora(
    model: LamarRanker, rank: int, alpha: int, dropout: float
) -> list[str]:
    suffixes = (
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
    )
    replacements = []
    for full_name, module in list(model.esm.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(full_name.endswith(suffix) for suffix in suffixes):
            continue
        parent_name, child = full_name.rsplit(".", 1)
        parent = model.esm.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(module, rank, alpha, dropout))
        replacements.append("esm." + full_name)
    if not replacements:
        raise RuntimeError("No q/k/v/o Lamar linear modules found")
    return replacements


def make_lamar_ranker(master: dict, run_config: dict, tokenizer):
    repository = Path(master["lamar_repo"])
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from LAMAR.modeling_nucESM2 import EsmModel

    mode = str(run_config["model_type"])
    if mode not in {"frozen_lamar", "lora_lamar", "hybrid_lamar"}:
        raise ValueError(mode)
    config = build_config(master["architecture_config"], tokenizer)
    model = LamarRanker(
        config,
        EsmModel,
        pooling="center",
        dropout=float(run_config.get("head_dropout", 0.1)),
        frozen_backbone=mode == "frozen_lamar",
    )
    load_report = load_pretrained(model, master["pretrained_checkpoint"])
    for parameter in model.esm.parameters():
        parameter.requires_grad = False
    details = {
        "mode": mode,
        "pooling": "center",
        "head": "LayerNorm(768)-Linear(256)-GELU-Linear(1)",
        "load_report": load_report,
    }
    if mode in {"lora_lamar", "hybrid_lamar"}:
        rank = int(run_config.get("lora_rank", 4))
        alpha = int(run_config.get("lora_alpha", 8))
        dropout = float(run_config.get("lora_dropout", 0.05))
        if (rank, alpha, dropout) != (4, 8, 0.05):
            raise ValueError(
                "Phase 3 fixes LoRA rank=4 alpha=8 dropout=0.05"
            )
        details.update(
            {
                "lora_modules": inject_qkvo_lora(
                    model, rank, alpha, dropout
                ),
                "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_rank": rank,
                "lora_alpha": alpha,
                "lora_dropout": dropout,
            }
        )
    for parameter in model.ranking_head.parameters():
        parameter.requires_grad = True
    details["trainable_parameters"] = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    details["total_parameters"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    return model, details


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.state_dict().items()
        if name in names
    }


def save_trainable(model: nn.Module, path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state(model), str(target))


def load_trainable(model: nn.Module, path: str | Path) -> dict:
    state = load_file(str(path))
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(result.unexpected_keys)
    return {
        "loaded_tensors": len(state),
        "missing_tensors": len(result.missing_keys),
    }
