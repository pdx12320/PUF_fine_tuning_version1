#!/usr/bin/env python3
from __future__ import annotations

import json, math, sys
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig


def build_config(config_path, tokenizer):
    return AutoConfig.from_pretrained(str(config_path),vocab_size=len(tokenizer),pad_token_id=tokenizer.pad_token_id,mask_token_id=tokenizer.mask_token_id,num_labels=1,problem_type="single_label_classification",token_dropout=False,positional_embedding_type="rotary",hidden_size=768,intermediate_size=3072,num_attention_heads=12,num_hidden_layers=12)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__(); self.base=base; self.rank=rank; self.scaling=float(alpha)/rank; self.dropout=nn.Dropout(dropout)
        for p in self.base.parameters(): p.requires_grad=False
        self.lora_A=nn.Linear(base.in_features,rank,bias=False); self.lora_B=nn.Linear(rank,base.out_features,bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight,a=math.sqrt(5)); nn.init.zeros_(self.lora_B.weight)
    def forward(self,x): return self.base(x)+self.lora_B(self.lora_A(self.dropout(x)))*self.scaling


class LamarBinaryClassifier(nn.Module):
    def __init__(self, config, esm_model_cls, pooling="mean", dropout=0.1):
        super().__init__(); self.config=config; self.pooling=pooling; self.esm=esm_model_cls(config,add_pooling_layer=False); h=config.hidden_size
        self.attention_score=nn.Linear(h,1,bias=False) if pooling=="attention" else None
        self.classifier=nn.Sequential(nn.LayerNorm(h),nn.Dropout(dropout),nn.Linear(h,1))
        for m in self.classifier.modules():
            if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,config.initializer_range); nn.init.zeros_(m.bias)
        if self.attention_score is not None: nn.init.normal_(self.attention_score.weight,0,config.initializer_range)
    def pooled(self,hidden,mask,center_positions):
        batch=torch.arange(hidden.shape[0],device=hidden.device); sequence_mask=mask.bool().clone(); sequence_mask[:,0]=False; last=mask.sum(1).long()-1; sequence_mask[batch,last]=False
        if self.pooling=="center": return hidden[batch,center_positions.long()]
        if self.pooling=="mean": return (hidden*sequence_mask.unsqueeze(-1)).sum(1)/sequence_mask.sum(1,keepdim=True).clamp_min(1)
        score=self.attention_score(hidden).squeeze(-1).masked_fill(~sequence_mask,-1e4); weight=torch.softmax(score,dim=1); return (hidden*weight.unsqueeze(-1)).sum(1)
    def forward(self,input_ids,attention_mask,center_positions):
        hidden=self.esm(input_ids=input_ids,attention_mask=attention_mask,return_dict=True).last_hidden_state
        return self.classifier(self.pooled(hidden,attention_mask,center_positions)).squeeze(-1)


def load_pretrained(model,state_path):
    state=load_file(str(state_path)); incompatible=model.load_state_dict(state,strict=False)
    missing_encoder=[k for k in incompatible.missing_keys if k.startswith("esm.")]
    if missing_encoder: raise RuntimeError(missing_encoder[:20])
    return {"tensor_count":len(state),"missing_keys":list(incompatible.missing_keys),"unexpected_keys":list(incompatible.unexpected_keys)}


def _replace(parent,name,new): setattr(parent,name,new)


def inject_lora(model, scheme, rank, dropout):
    suffixes={
      "qv":("attention.self.query","attention.self.value"),
      "qkvo":("attention.self.query","attention.self.key","attention.self.value","attention.output.dense"),
      "all_attention":("attention.self.query","attention.self.key","attention.self.value","attention.output.dense"),
      "attention_mlp":("attention.self.query","attention.self.key","attention.self.value","attention.output.dense","intermediate.dense","output.dense"),
    }[scheme]
    replacements=[]
    for full_name,module in list(model.esm.named_modules()):
        if isinstance(module,nn.Linear) and any(full_name.endswith(s) for s in suffixes):
            parent_name,child=full_name.rsplit(".",1); parent=model.esm.get_submodule(parent_name); _replace(parent,child,LoRALinear(module,rank,2*rank,dropout)); replacements.append("esm."+full_name)
    if not replacements: raise RuntimeError((scheme,suffixes))
    return replacements


def configure_trainable(model, mode, unfreeze=0, lora_scheme="qv", lora_rank=8, lora_dropout=0.05):
    for p in model.esm.parameters(): p.requires_grad=False
    details={"mode":mode}
    if mode=="partial":
        for layer in model.esm.encoder.layer[-unfreeze:]:
            for p in layer.parameters(): p.requires_grad=True
        for p in model.esm.encoder.emb_layer_norm_after.parameters(): p.requires_grad=True
        details["unfreeze_last_n"]=unfreeze
    elif mode=="lora": details.update({"lora_modules":inject_lora(model,lora_scheme,lora_rank,lora_dropout),"lora_scheme":lora_scheme,"lora_rank":lora_rank,"lora_alpha":2*lora_rank,"lora_dropout":lora_dropout})
    elif mode=="full":
        for p in model.esm.parameters(): p.requires_grad=True
    elif mode!="frozen": raise ValueError(mode)
    for p in model.classifier.parameters(): p.requires_grad=True
    if model.attention_score is not None:
        for p in model.attention_score.parameters(): p.requires_grad=True
    details["trainable_parameters"]=sum(p.numel() for p in model.parameters() if p.requires_grad); details["total_parameters"]=sum(p.numel() for p in model.parameters())
    return details


def trainable_state(model): return {name:p.detach().cpu().contiguous() for name,p in model.state_dict().items() if name in {n for n,q in model.named_parameters() if q.requires_grad}}


def save_trainable(model,path): save_file(trainable_state(model),str(path))


def load_trainable(model,path):
    state=load_file(str(path)); result=model.load_state_dict(state,strict=False); return {"loaded":len(state),"missing":len(result.missing_keys),"unexpected":list(result.unexpected_keys)}


def make_model(cfg, run_config, tokenizer):
    repo=Path(cfg["lamar_repo"]); sys.path.insert(0,str(repo)) if str(repo) not in sys.path else None
    from LAMAR.modeling_nucESM2 import EsmModel
    config=build_config(cfg["architecture_config"],tokenizer); model=LamarBinaryClassifier(config,EsmModel,run_config["pooling"],run_config.get("head_dropout",0.1)); load=load_pretrained(model,cfg["base_state"])
    details=configure_trainable(model,run_config["mode"],run_config.get("unfreeze",0),run_config.get("lora_scheme","qv"),run_config.get("lora_rank",8),run_config.get("lora_dropout",0.05)); details["load_report"]=load
    return model,details
