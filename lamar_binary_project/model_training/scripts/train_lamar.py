#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from common import NegativePool, binary_metrics, load_yaml, read_tsv_records, seed_everything, write_json
from modeling_binary import load_trainable, make_model, save_trainable


class Records(Dataset):
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,index): return self.rows[index]


def collator(tokenizer):
    cls,eos=tokenizer.cls_token_id,tokenizer.eos_token_id
    ids={base:tokenizer.convert_tokens_to_ids(base) for base in "ATCGN"}
    def apply(rows):
        encoded=[[cls]+[ids.get(base,ids["N"]) for base in row["sequence_context"]]+[eos] for row in rows]
        return {"input_ids":torch.tensor(encoded,dtype=torch.long),"attention_mask":torch.ones((len(rows),103),dtype=torch.long),"center_positions":torch.full((len(rows),),51,dtype=torch.long),"labels":torch.tensor([row["label"] for row in rows],dtype=torch.float32)}
    return apply


def loss_function(logits,labels,kind,gamma,pos_weight):
    if kind=="bce": return nn.functional.binary_cross_entropy_with_logits(logits,labels)
    if kind=="weighted_bce": return nn.functional.binary_cross_entropy_with_logits(logits,labels,pos_weight=torch.tensor(pos_weight,device=logits.device))
    if kind=="focal":
        bce=nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction="none"); pt=torch.exp(-bce); return (((1-pt)**gamma)*bce).mean()
    raise ValueError(kind)


@torch.no_grad()
def predict(model,rows,tokenizer,batch_size,device):
    model.eval(); output=[]
    loader=DataLoader(Records(rows),batch_size=batch_size,shuffle=False,collate_fn=collator(tokenizer),num_workers=0)
    for batch in loader:
        logits=model(batch["input_ids"].to(device),batch["attention_mask"].to(device),batch["center_positions"].to(device)); output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--master",required=True); p.add_argument("--run-config",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--max-steps",type=int,default=-1); args=p.parse_args()
    master=load_yaml(args.master); rc=json.loads(Path(args.run_config).read_text()); out=Path(args.output_dir)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); (out/"config.json").write_text(json.dumps(rc,indent=2,sort_keys=True)+"\n")
    seed=int(rc["seed"]); seed_everything(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    data=Path(master["dataset_dir"]); positives=read_tsv_records(data/"train_positives.tsv.gz"); dev=read_tsv_records(data/"dev_1to10.tsv.gz")
    pool=NegativePool(Path(master["run_dir"])/"work/train_pool.sqlite",seed)
    tokenizer=AutoTokenizer.from_pretrained(master["tokenizer"],local_files_only=True,model_max_length=103)
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(0.80,0)
    model,details=make_model(master,rc,tokenizer); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    if rc.get("gradient_checkpointing",False) and hasattr(model.esm,"gradient_checkpointing_enable"): model.esm.gradient_checkpointing_enable()
    head=[]; backbone=[]
    for name,param in model.named_parameters():
        if not param.requires_grad: continue
        (head if name.startswith("classifier") or name.startswith("attention_score") or "lora_" in name else backbone).append(param)
    groups=[]
    if head: groups.append({"params":head,"lr":float(rc["head_lr"])})
    if backbone: groups.append({"params":backbone,"lr":float(rc.get("backbone_lr",rc["head_lr"]*0.1))})
    optimizer=torch.optim.AdamW(groups,weight_decay=float(rc.get("weight_decay",0.01)))
    epochs=int(rc.get("epochs",20)); ratio=int(rc.get("sampling_ratio",10)); n_negative=len(positives)*ratio
    batch_size=int(rc.get("batch_size",16)); accumulation=int(rc.get("accumulation_steps",2)); steps_epoch=math.ceil(math.ceil((len(positives)+n_negative)/batch_size)/accumulation); total_steps=max(1,steps_epoch*epochs)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(total_steps*float(rc.get("warmup_ratio",0.03))),total_steps)
    scaler=torch.cuda.amp.GradScaler(enabled=bool(rc.get("fp16",True) and device.type=="cuda")); torch.cuda.reset_peak_memory_stats(device) if device.type=="cuda" else None
    pos_weight=min(5.0,math.sqrt(ratio)) if rc["loss"]=="weighted_bce" else 1.0
    history=[]; best=-1; patience=0; global_step=0; started=time.time(); checkpoint=out/"best_trainable.safetensors"
    for epoch in range(epochs):
        negative_ids=pool.ids_for_epoch(n_negative,rc["negative_strategy"],epoch); rows=positives+pool.fetch(negative_ids); random.Random(seed+epoch).shuffle(rows)
        loader=DataLoader(Records(rows),batch_size=batch_size,shuffle=False,collate_fn=collator(tokenizer),num_workers=int(rc.get("workers",0)),pin_memory=True)
        model.train(); optimizer.zero_grad(set_to_none=True); running=0.0; batches=0
        for index,batch in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits=model(batch["input_ids"].to(device,non_blocking=True),batch["attention_mask"].to(device,non_blocking=True),batch["center_positions"].to(device,non_blocking=True)); loss=loss_function(logits,batch["labels"].to(device),rc["loss"],float(rc.get("focal_gamma",2)),pos_weight)/accumulation
            scaler.scale(loss).backward(); running+=float(loss.item())*accumulation; batches+=1
            if (index+1)%accumulation==0 or index+1==len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]],1.0); scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step+=1
                if args.max_steps>0 and global_step>=args.max_steps: break
        probability=predict(model,dev,tokenizer,int(rc.get("eval_batch_size",128)),device); metrics=binary_metrics([r["label"] for r in dev],probability)
        record={"epoch":epoch+1,"global_step":global_step,"train_loss":running/max(1,batches),"actual_positive":len(positives),"actual_negative":len(negative_ids),**metrics}; history.append(record); print(json.dumps(record),flush=True)
        if metrics["average_precision"]>best+1e-6: best=metrics["average_precision"]; patience=0; save_trainable(model,checkpoint)
        else: patience+=1
        if args.max_steps>0 and global_step>=args.max_steps: break
        if patience>=int(rc.get("patience",3)): break
    load_trainable(model,checkpoint); probability=predict(model,dev,tokenizer,int(rc.get("eval_batch_size",128)),device); metrics=binary_metrics([r["label"] for r in dev],probability)
    frame=pd.DataFrame(dev); frame["probability"]=probability; frame.to_parquet(out/"dev_predictions.parquet",index=False)
    summary={"status":"SUCCESS","model":"lamar","config":rc,"details":details,"dev_metrics":metrics,"history":history,"training_time":time.time()-started,"peak_memory":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0,"trainable_parameters":details["trainable_parameters"],"total_parameters":details["total_parameters"],"actual_training_positive":len(positives),"actual_training_negative_per_epoch":n_negative,"pos_weight":pos_weight,"checkpoint":str(checkpoint)}
    write_json(out/"summary.json",summary); print(json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__": main()
