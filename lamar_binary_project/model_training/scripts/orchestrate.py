#!/usr/bin/env python3
from __future__ import annotations
import argparse,concurrent.futures,csv,json,os,subprocess,time
from pathlib import Path
import pandas as pd
from common import load_yaml,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument("--master",required=True);args=p.parse_args();master=load_yaml(args.master);run=Path(master["run_dir"]);py=master["python"];trainer=run/"scripts/train_lamar.py";results=[]
 def execute(item):
  name,stage,cfg,gpu=item; cfg={**cfg,"experiment_id":name,"stage":stage}; cp=run/f"configs/{name}.json";out=run/f"checkpoints/runs/{name}";log=run/f"logs/{name}.log";status=run/f"work/{name}.status.json";cp.write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n")
  if (out/"summary.json").exists(): return json.loads((out/"summary.json").read_text())
  env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);started=time.time();command=[py,str(trainer),"--master",args.master,"--run-config",str(cp),"--output-dir",str(out)]
  with log.open("w") as h: code=subprocess.run(command,stdout=h,stderr=subprocess.STDOUT,env=env).returncode
  value={"experiment_id":name,"stage":stage,"exit_code":code,"elapsed":time.time()-started,"status":"FAILED"}
  if code==0 and (out/"summary.json").exists(): value=json.loads((out/"summary.json").read_text());value.update({"experiment_id":name,"stage":stage,"exit_code":0})
  write_json(status,value);return value
 def stage(items):
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: out=list(pool.map(execute,[(n,s,c,i%2) for i,(n,s,c) in enumerate(items)]))
  results.extend(out);write_leaderboard();return [x for x in out if x.get("status")=="SUCCESS"]
 def row(r):
  c=r.get("config",{});m=r.get("dev_metrics",{});return {"model":"LAMAR","training_strategy":c.get("mode"),"pooling":c.get("pooling"),"sampling_ratio":c.get("sampling_ratio"),"negative_strategy":c.get("negative_strategy"),"loss":c.get("loss"),"learning_rate":c.get("head_lr"),"trainable_parameters":r.get("trainable_parameters"),"total_parameters":r.get("total_parameters"),"seed":c.get("seed"),"dev_PR_AUC":m.get("pr_auc"),"dev_average_precision":m.get("average_precision"),"calibration_PR_AUC":"","test_PR_AUC":"","test_precision":"","test_recall":"","test_F1":"","FP_per_million":"","training_time":r.get("training_time"),"peak_memory":r.get("peak_memory"),"experiment_id":r.get("experiment_id",c.get("experiment_id")),"stage":r.get("stage",c.get("stage")),"status":r.get("status"),"config_json":json.dumps(c,sort_keys=True)}
 def write_leaderboard(): pd.DataFrame([row(r) for r in results]).to_csv(run/"leaderboard.csv",index=False)
 def best(values):
  good=[x for x in values if x.get("status")=="SUCCESS" and x.get("dev_metrics",{}).get("average_precision") is not None]
  if not good: raise RuntimeError("No successful configurations")
  return max(good,key=lambda x:(x["dev_metrics"]["average_precision"],x["dev_metrics"]["pr_auc"]))
 base={"seed":42,"pooling":"mean","mode":"frozen","sampling_ratio":10,"negative_strategy":"random","loss":"bce","head_lr":1e-4,"backbone_lr":1e-5,"head_dropout":.1,"weight_decay":.01,"warmup_ratio":.03,"batch_size":16,"eval_batch_size":128,"accumulation_steps":2,"epochs":20,"patience":3,"fp16":True}
 frozen=stage([(f"s1_frozen_{pool}","stage1_frozen",{**base,"pooling":pool}) for pool in ("mean","center","attention")]);pooling=best(frozen)["config"]["pooling"]
 structure=[]
 structure+=stage([(f"s1_partial_u{n}","stage1_structure",{**base,"mode":"partial","pooling":pooling,"unfreeze":n}) for n in (1,2,4)])
 structure+=stage([(f"s1_lora_{scheme}","stage1_structure",{**base,"mode":"lora","pooling":pooling,"lora_scheme":scheme,"lora_rank":8,"lora_dropout":.05}) for scheme in ("qv","qkvo","attention_mlp")])
 structure+=stage([("s1_full","stage1_structure",{**base,"mode":"full","pooling":pooling,"batch_size":4,"accumulation_steps":8,"gradient_checkpointing":True})]);current=best(frozen+structure);template=current["config"]
 lr_runs=stage([(f"s2_lr_{str(lr).replace('.','p')}","stage2_lr",{**template,"seed":42,"head_lr":lr,"backbone_lr":lr*.1}) for lr in (1e-5,3e-5,1e-4)]);current=best(lr_runs);template=current["config"]
 sample_cfg=[]
 for ratio in (1,5,10,20):sample_cfg.append((f"s3_ratio_{ratio}","stage3_sampling",{**template,"seed":42,"sampling_ratio":ratio,"negative_strategy":"random"}))
 sample_cfg.append(("s3_dynamic_full","stage3_sampling",{**template,"seed":42,"sampling_ratio":20,"negative_strategy":"dynamic_full"}));sample_runs=stage(sample_cfg);current=best(sample_runs);template=current["config"]
 neg_runs=stage([(f"s4_negative_{kind}","stage4_negative",{**template,"seed":42,"sampling_ratio":10,"negative_strategy":kind}) for kind in ("random","matched","hard","mixed")]);current=best(neg_runs);template=current["config"]
 losses=[("bce",None),("weighted_bce",None),("focal",1),("focal",2),("focal",3)];loss_runs=stage([(f"s5_loss_{kind}{'' if gamma is None else gamma}","stage5_loss",{**template,"seed":42,"loss":kind,**({} if gamma is None else {"focal_gamma":gamma})}) for kind,gamma in losses]);current=best(loss_runs);template=current["config"]
 lora_runs=stage([(f"s6_lora_r{rank}_d{str(drop).replace('.','p')}","stage6_lora_rank",{**template,"seed":42,"mode":"lora","lora_scheme":"qkvo","lora_rank":rank,"lora_dropout":drop,"pooling":pooling}) for rank in (4,8,16) for drop in (0.0,.05)]);current=best([current]+lora_runs);template=current["config"]
 hp=[("wd0",{"weight_decay":0.0}),("wd01",{"weight_decay":.01}),("warm0",{"warmup_ratio":0.0}),("warm03",{"warmup_ratio":.03}),("batch8",{"batch_size":8,"accumulation_steps":4}),("batch16",{"batch_size":16,"accumulation_steps":2})]
 hp_runs=stage([(f"s7_{name}","stage7_hparams",{**template,"seed":42,**change}) for name,change in hp]);current=best([current]+hp_runs);template=current["config"]
 seed_runs=stage([(f"final_seed{seed}","final_seeds",{**template,"seed":seed,"epochs":20,"patience":3}) for seed in (42,43,44)]);selected=[x for x in seed_runs if x["config"]["seed"]==42][0]
 aggregate={"selection_policy":"all hyperparameters selected only by dev average precision; deployment seed fixed a priori to 42","best_hyperparameters":template,"seed_metrics":[{"seed":x["config"]["seed"],**x["dev_metrics"]} for x in seed_runs],"selected_seed42_summary":selected,"attention_scheme_equivalence":"qkvo and all_attention are identical in this LAMAR architecture; qkvo run represents both"};write_json(run/"BEST_DEV_CONFIG.json",aggregate);write_leaderboard();(run/"DEV_SELECTION_COMPLETE").write_text("PASS\n")
if __name__=="__main__":main()
