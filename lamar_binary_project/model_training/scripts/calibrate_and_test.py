#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch,yaml
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer
from common import binary_metrics,expected_calibration_error,load_yaml,read_tsv_records,write_json
from modeling_binary import load_trainable,make_model
from train_lamar import predict

def logit(p): p=np.clip(np.asarray(p),1e-7,1-1e-7);return np.log(p/(1-p))
def fit_method(method,p,y):
 if method=="none":return None
 if method=="platt":m=LogisticRegression(C=1e6,solver="lbfgs",max_iter=1000);m.fit(logit(p).reshape(-1,1),y);return m
 m=IsotonicRegression(out_of_bounds="clip");m.fit(p,y);return m
def apply_method(method,m,p):
 if method=="none":return np.asarray(p)
 if method=="platt":return m.predict_proba(logit(p).reshape(-1,1))[:,1]
 return m.predict(p)
def write_reliability_diagram(curves,path):
 width,height=720,600;left,right,top,bottom=80,680,40,530
 def xy(x,y):return left+x*(right-left),bottom-y*(bottom-top)
 colors={"none":"#4C78A8","platt":"#F58518","isotonic":"#54A24B"}
 parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="white"/>',f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>',f'<line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="black"/>']
 x0,y0=xy(0,0);x1,y1=xy(1,1);parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="#777" stroke-dasharray="6 5"/>')
 for tick in (0,.2,.4,.6,.8,1):
  x,y=xy(tick,0);parts.append(f'<line x1="{x}" y1="{bottom}" x2="{x}" y2="{bottom+6}" stroke="black"/><text x="{x}" y="{bottom+24}" text-anchor="middle" font-size="13">{tick:g}</text>')
  x,y=xy(0,tick);parts.append(f'<line x1="{left-6}" y1="{y}" x2="{left}" y2="{y}" stroke="black"/><text x="{left-12}" y="{y+4}" text-anchor="end" font-size="13">{tick:g}</text>')
 for index,(method,pred,truth) in enumerate(curves):
  points=" ".join(f"{xy(float(x),float(y))[0]:.2f},{xy(float(x),float(y))[1]:.2f}" for x,y in zip(pred,truth));color=colors[method];parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
  for x,y in zip(pred,truth):
   px,py=xy(float(x),float(y));parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}"/>')
  ly=65+index*22;parts.append(f'<line x1="500" y1="{ly}" x2="530" y2="{ly}" stroke="{color}" stroke-width="3"/><text x="538" y="{ly+5}" font-size="14">{method}</text>')
 parts.extend([f'<text x="{(left+right)/2}" y="585" text-anchor="middle" font-size="16">Predicted probability</text>',f'<text x="20" y="{(top+bottom)/2}" text-anchor="middle" font-size="16" transform="rotate(-90 20 {(top+bottom)/2})">Observed frequency</text>',f'<text x="{(left+right)/2}" y="25" text-anchor="middle" font-size="18">Calibration reliability (OOF)</text>','</svg>'])
 Path(path).write_text("\n".join(parts)+"\n")
def workpoint(y,p,target):
 y=np.asarray(y);p=np.asarray(p);neg=np.sort(p[y==0])[::-1];allowed=int(math.floor(target*len(neg)/1_000_000));threshold=1.0 if allowed==0 else (np.nextafter(neg[allowed],1.0) if allowed<len(neg) else 0.0);m=binary_metrics(y,p,threshold);m.update({"target_fp_per_million":target,"allowed_fp":allowed});return m
def load_model(master,run,best):
 cfg=best["selected_seed42_summary"]["config"];tok=AutoTokenizer.from_pretrained(master["tokenizer"],local_files_only=True,model_max_length=103)
 if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(0.80,0)
 model,details=make_model(master,cfg,tok);ckpt=Path(best["selected_seed42_summary"]["checkpoint"]);load_trainable(model,ckpt);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model.to(device);return model,tok,device,cfg,ckpt,details
def calibration_stage(master,run,best):
 if (run/"CALIBRATION_COMPLETE").exists():raise FileExistsError("Calibration already complete")
 rows=read_tsv_records(Path(master["dataset_dir"])/"calibration_1to1000.tsv.gz");model,tok,device,cfg,ckpt,details=load_model(master,run,best);raw=predict(model,rows,tok,256,device);y=np.array([r["label"] for r in rows]);groups=np.array([r["leakage_group"] for r in rows]);methods=("none","platt","isotonic");oof={m:np.zeros(len(y)) for m in methods};folds=StratifiedGroupKFold(5,shuffle=True,random_state=20260722)
 for train_idx,valid_idx in folds.split(raw,y,groups):
  for method in methods:oof[method][valid_idx]=apply_method(method,fit_method(method,raw[train_idx],y[train_idx]),raw[valid_idx])
 comparison=[]
 for method in methods:
  metrics=binary_metrics(y,oof[method]);comparison.append({"method":method,**metrics})
 chosen=min(comparison,key=lambda r:(r["brier"],r["ece"]))["method"];model_cal=fit_method(chosen,raw,y);cal=apply_method(chosen,model_cal,raw);joblib.dump({"method":chosen,"model":model_cal},run/"calibration_model.joblib")
 frame=pd.DataFrame(rows);frame["raw_probability"]=raw
 for method in methods:frame[f"oof_{method}_probability"]=oof[method]
 frame["calibrated_probability"]=cal;frame.to_parquet(run/"calibration_predictions.parquet",index=False);pd.DataFrame(comparison).to_csv(run/"calibration_results.csv",index=False)
 analysis=[]
 for threshold in (.01,.05,.1,.2,.5):analysis.append({"kind":"fixed_threshold",**binary_metrics(y,cal,threshold)})
 for target in (10,50,100,500,1000):analysis.append({"kind":"fp_workpoint",**workpoint(y,cal,target)})
 pd.DataFrame(analysis).to_csv(run/"threshold_analysis.csv",index=False);default=[r for r in analysis if r["kind"]=="fp_workpoint" and r["target_fp_per_million"]==100][0]
 thresholds={name:[float(np.quantile(frame[name],q)) for q in (1/3,2/3)] for name in ("c_count","gc_fraction","entropy","median_depth","gene_coverage")}
 final={"calibration_method":chosen,"threshold":default["threshold"],"selection":"maximum recall subject to <=100 false positives per million calibration negatives","calibration_metrics":binary_metrics(y,cal,default["threshold"]),"workpoint":default,"subgroup_bin_boundaries":thresholds,"model_checkpoint":str(ckpt),"model_config":cfg};write_json(run/"final_threshold.json",final);(run/"final_model_config.yaml").write_text(yaml.safe_dump({"checkpoint":str(ckpt),"base_state":master["base_state"],"config":cfg,"calibration_method":chosen,"threshold":default["threshold"]},sort_keys=False))
 subgroup_metrics(frame,cal,default["threshold"],thresholds,run/"subgroup_analysis/calibration_subgroups.parquet")
 dev_frame=pd.read_parquet(ckpt.parent/"dev_predictions.parquet").rename(columns={"probability":"raw_probability"});dev_prob=apply_method(chosen,model_cal,dev_frame.raw_probability.to_numpy());dev_frame["probability"]=dev_prob;dev_frame.to_parquet(run/"dev_predictions.parquet",index=False);subgroup_metrics(dev_frame,dev_prob,default["threshold"],thresholds,run/"subgroup_analysis/dev_subgroups.parquet")
 hard_rows=read_tsv_records(Path(master["dataset_dir"])/"dev_hard_negatives.tsv.gz");hard_raw=predict(model,hard_rows,tok,256,device);hard_prob=apply_method(chosen,model_cal,hard_raw);hard_frame=pd.DataFrame(hard_rows);hard_frame["raw_probability"]=hard_raw;hard_frame["probability"]=hard_prob;hard_frame["is_predicted_positive"]=hard_prob>=default["threshold"];hard_frame.to_parquet(run/"dev_hard_negative_predictions.parquet",index=False);write_json(run/"dev_hard_negative_summary.json",{"n":len(hard_frame),"mean_probability":float(np.mean(hard_prob)),"median_probability":float(np.median(hard_prob)),"predicted_positive":int((hard_prob>=default["threshold"]).sum()),"fp_per_million_at_frozen_threshold":float((hard_prob>=default["threshold"]).mean()*1_000_000)})
 curves=[];curve_rows=[]
 for method in methods:
  truth,pred=calibration_curve(y,oof[method],n_bins=12,strategy="quantile");curves.append((method,pred,truth));curve_rows.extend({"method":method,"mean_predicted_probability":float(x),"observed_frequency":float(z)} for x,z in zip(pred,truth))
 pd.DataFrame(curve_rows).to_csv(run/"calibration_reliability_points.csv",index=False);write_reliability_diagram(curves,run/"calibration_reliability.svg");(run/"CALIBRATION_COMPLETE").write_text("PASS\n")
def subgroup_metrics(frame,prob,threshold,bounds,out):
 rows=[];y=frame.label.to_numpy()
 for field,(a,b) in bounds.items():
  values=frame[field].to_numpy();cats=np.where(values<=a,"low",np.where(values<=b,"medium","high"))
  for cat in ("low","medium","high"):
   mask=cats==cat
   if mask.sum() and len(np.unique(y[mask]))==2:rows.append({"field":field,"group":cat,"n":int(mask.sum()),"positive":int(y[mask].sum()),**binary_metrics(y[mask],prob[mask],threshold)})
 for field in ("gene_id","transcript_ids","negative_type"):
  for group,sub in frame.assign(probability=prob).groupby(field):
   if len(sub)<20:continue
   row={"field":field,"group":str(group),"n":len(sub),"positive":int(sub.label.sum()),"mean_probability":float(sub.probability.mean())}
   if sub.label.nunique()==2:row.update(binary_metrics(sub.label,sub.probability,threshold))
   rows.append(row)
 pd.DataFrame(rows).to_parquet(out,index=False)
def test_stage(master,run,best):
 if not (run/"CALIBRATION_COMPLETE").exists():raise RuntimeError("Calibration not frozen")
 if (run/"TEST_EVALUATION_STARTED").exists():raise RuntimeError("Locked test already consumed")
 (run/"TEST_EVALUATION_STARTED").write_text("single locked-test access started\n")
 final=json.loads((run/"final_threshold.json").read_text());calibrator=joblib.load(run/"calibration_model.joblib");rows=read_tsv_records(Path(master["dataset_dir"])/"test_1to1000.tsv.gz");model,tok,device,cfg,ckpt,details=load_model(master,run,best);raw=predict(model,rows,tok,256,device);prob=apply_method(calibrator["method"],calibrator["model"],raw);frame=pd.DataFrame(rows);frame["raw_probability"]=raw;frame["probability"]=prob;frame.to_parquet(run/"test_predictions.parquet",index=False);threshold=final["threshold"];metrics=binary_metrics(frame.label,prob,threshold)
 for target in (10,100,1000):metrics[f"recall_at_{target}_fp_per_million"]=workpoint(frame.label,prob,target)["recall"]
 write_json(run/"test_metrics.json",metrics);pred=prob>=threshold;actual_fp=frame[(frame.label==0)&pred].sort_values("probability",ascending=False).copy();top_negative=frame[frame.label==0].sort_values("probability",ascending=False).head(100).copy();top_negative["is_false_positive_at_frozen_threshold"]=top_negative.probability>=threshold;fn=frame[(frame.label==1)&~pred].sort_values("probability").copy();actual_fp.to_csv(run/"error_analysis/predicted_false_positives.csv",index=False);top_negative.to_csv(run/"error_analysis/false_positive_top100.csv",index=False);top_negative.to_csv(run/"error_analysis/highest_probability_negative_top100.csv",index=False)
 low_cov=final["subgroup_bin_boundaries"]["median_depth"][0];low_entropy=final["subgroup_bin_boundaries"]["entropy"][0]
 def category(r):
  if r.true_efficiency<=.15:return "low_efficiency_positive"
  if r.median_depth<=low_cov:return "coverage_related"
  if r.entropy<=low_entropy:return "hard_sequence"
  if r.sequence_context[48:53] in set(top_negative.sequence_context.str[48:53]):return "special_motif"
  return "other"
 if len(fn):fn["error_category"]=fn.apply(category,axis=1)
 fn.to_csv(run/"error_analysis/false_negative_all.csv",index=False);subgroup_metrics(frame,prob,threshold,final["subgroup_bin_boundaries"],run/"subgroup_analysis/test_subgroups.parquet");(run/"TEST_EVALUATION_COMPLETE").write_text("PASS\n")
def main():
 p=argparse.ArgumentParser();p.add_argument("--master",required=True);p.add_argument("--stage",choices=("calibration","test"),required=True);a=p.parse_args();master=load_yaml(a.master);run=Path(master["run_dir"]);best=json.loads((run/"BEST_DEV_CONFIG.json").read_text());calibration_stage(master,run,best) if a.stage=="calibration" else test_stage(master,run,best)
if __name__=="__main__":main()
