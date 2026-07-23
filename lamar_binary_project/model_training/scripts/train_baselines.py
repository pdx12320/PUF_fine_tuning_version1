#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from scipy.sparse import csr_matrix,hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from common import NegativePool,binary_metrics,load_yaml,read_tsv_records,write_json

def numeric(rows): return np.asarray([[r["gc_fraction"],r["c_count"],r["entropy"]] for r in rows],dtype=np.float64)
def onehot(rows):
    index={b:i for i,b in enumerate("ACGT")}; x=np.zeros((len(rows),4,101),dtype=np.float32)
    for i,r in enumerate(rows):
        for j,b in enumerate(r["sequence_context"]):
            if b in index:x[i,index[b],j]=1
    return torch.from_numpy(x)
class CNN(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Conv1d(4,64,9,padding=4),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(64,128,7,padding=3),nn.ReLU(),nn.AdaptiveMaxPool1d(1)); self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(128,1))
    def forward(self,x): return self.head(self.net(x).squeeze(-1)).squeeze(-1)
@torch.no_grad()
def cnn_predict(model,x,device):
    model.eval(); out=[]
    for (batch,) in DataLoader(TensorDataset(x),batch_size=512):out.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(out)
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);args=p.parse_args();cfg=load_yaml(args.config);run=Path(cfg["run_dir"]);data=Path(cfg["dataset_dir"]);seed=int(cfg["seed"])
    pos=read_tsv_records(data/"train_positives.tsv.gz");dev=read_tsv_records(data/"dev_1to10.tsv.gz");pool=NegativePool(run/"work/train_pool.sqlite",seed);neg=pool.fetch(pool.ids_for_epoch(len(pos)*10,"random",0));train=pos+neg;random.Random(seed).shuffle(train)
    y=np.asarray([r["label"] for r in train]);yd=np.asarray([r["label"] for r in dev]);base=run/"checkpoints/baselines";base.mkdir(parents=True,exist_ok=True); rows=[]
    t=time.time();vec=CountVectorizer(analyzer="char",ngram_range=(1,4),lowercase=False);x=vec.fit_transform([r["sequence_context"] for r in train]);xd=vec.transform([r["sequence_context"] for r in dev]);scale=StandardScaler().fit(numeric(train));x=hstack([x,csr_matrix(scale.transform(numeric(train)))]);xd=hstack([xd,csr_matrix(scale.transform(numeric(dev)))]);lr=LogisticRegression(max_iter=2000,solver="liblinear",random_state=seed);lr.fit(x,y);prob=lr.predict_proba(xd)[:,1];metrics=binary_metrics(yd,prob);joblib.dump({"vectorizer":vec,"scaler":scale,"model":lr},base/"kmer_logistic.joblib");f=pd.DataFrame(dev);f["probability"]=prob;f.to_parquet(base/"kmer_logistic_dev_predictions.parquet",index=False);s={"status":"SUCCESS","model":"kmer_logistic","features":"1-4mer+GC+C_count+entropy","dev_metrics":metrics,"training_time":time.time()-t,"trainable_parameters":int(x.shape[1]+1),"total_parameters":int(x.shape[1]+1),"seed":seed};write_json(base/"kmer_logistic_summary.json",s);rows.append(s)
    t=time.time();metadata_train=np.asarray([[r["gc_fraction"],r["c_count"],r["entropy"],r["median_depth"],r["gene_coverage"]] for r in train]);metadata_dev=np.asarray([[r["gc_fraction"],r["c_count"],r["entropy"],r["median_depth"],r["gene_coverage"]] for r in dev]);metadata_scaler=StandardScaler().fit(metadata_train);metadata_model=LogisticRegression(max_iter=2000,solver="liblinear",random_state=seed).fit(metadata_scaler.transform(metadata_train),y);prob=metadata_model.predict_proba(metadata_scaler.transform(metadata_dev))[:,1];metrics=binary_metrics(yd,prob);joblib.dump({"scaler":metadata_scaler,"model":metadata_model},base/"metadata_only_logistic.joblib");f=pd.DataFrame(dev);f["probability"]=prob;f.to_parquet(base/"metadata_only_dev_predictions.parquet",index=False);s={"status":"SUCCESS","model":"metadata_only_logistic","features":"GC+C_count+entropy+coverage+gene_coverage_proxy","dev_metrics":metrics,"training_time":time.time()-t,"trainable_parameters":6,"total_parameters":6,"seed":seed,"audit_only":True};write_json(base/"metadata_only_summary.json",s);rows.append(s)
    t=time.time();device=torch.device("cuda" if torch.cuda.is_available() else "cpu");torch.cuda.set_per_process_memory_fraction(0.80,0) if device.type=="cuda" else None;model=CNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01);xt=onehot(train);xdev=onehot(dev);yt=torch.tensor(y,dtype=torch.float32);best=-1;pat=0;hist=[]
    for epoch in range(20):
        model.train();losses=[]
        for xb,yb in DataLoader(TensorDataset(xt,yt),batch_size=128,shuffle=True):
            opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(xb.to(device)),yb.to(device));loss.backward();opt.step();losses.append(loss.item())
        pdev=cnn_predict(model,xdev,device);m=binary_metrics(yd,pdev);hist.append({"epoch":epoch+1,"loss":float(np.mean(losses)),**m})
        if m["average_precision"]>best:best=m["average_precision"];pat=0;torch.save(model.state_dict(),base/"cnn.pt")
        else:pat+=1
        if pat>=3:break
    model.load_state_dict(torch.load(base/"cnn.pt",map_location=device));prob=cnn_predict(model,xdev,device);metrics=binary_metrics(yd,prob);f=pd.DataFrame(dev);f["probability"]=prob;f.to_parquet(base/"cnn_dev_predictions.parquet",index=False);n=sum(p.numel() for p in model.parameters());s={"status":"SUCCESS","model":"cnn","architecture":"onehot-Conv64-Conv128-globalmax-linear","dev_metrics":metrics,"history":hist,"training_time":time.time()-t,"peak_memory":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0,"trainable_parameters":n,"total_parameters":n,"seed":seed};write_json(base/"cnn_summary.json",s);rows.append(s);pd.DataFrame([{**{k:v for k,v in r.items() if k!="dev_metrics"},**{f"dev_{k}":v for k,v in r["dev_metrics"].items()}} for r in rows]).to_csv(run/"baseline_results.csv",index=False);print(json.dumps(rows,indent=2))
if __name__=="__main__":main()
