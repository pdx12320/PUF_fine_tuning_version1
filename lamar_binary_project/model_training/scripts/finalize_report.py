#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from common import load_yaml,sha256_file
def main():
 p=argparse.ArgumentParser();p.add_argument("--master",required=True);a=p.parse_args();cfg=load_yaml(a.master);run=Path(cfg["run_dir"])
 for marker in ("DATA_AUDIT_OK","DEV_SELECTION_COMPLETE","CALIBRATION_COMPLETE","TEST_EVALUATION_COMPLETE"):
  if not (run/marker).exists():raise RuntimeError(f"Missing {marker}")
 board=pd.read_csv(run/"leaderboard.csv");base=pd.read_csv(run/"baseline_results.csv");base_rows=[]
 for _,r in base.iterrows():base_rows.append({"model":r["model"],"training_strategy":"baseline","pooling":"","sampling_ratio":10,"negative_strategy":"random","loss":"bce","learning_rate":"","trainable_parameters":r["trainable_parameters"],"total_parameters":r["total_parameters"],"seed":r["seed"],"dev_PR_AUC":r["dev_pr_auc"],"dev_average_precision":r["dev_average_precision"],"calibration_PR_AUC":"","test_PR_AUC":"","test_precision":"","test_recall":"","test_F1":"","FP_per_million":"","training_time":r["training_time"],"peak_memory":r.get("peak_memory",0),"experiment_id":r["model"],"stage":"baseline","status":"SUCCESS","config_json":""})
 board=pd.concat([board,pd.DataFrame(base_rows)],ignore_index=True);cal=pd.read_csv(run/"calibration_results.csv");test=json.loads((run/"test_metrics.json").read_text());mask=board.experiment_id.eq("final_seed42");board.loc[mask,"calibration_PR_AUC"]=float(cal.sort_values(["brier","ece"]).iloc[0].pr_auc);board.loc[mask,"test_PR_AUC"]=test["pr_auc"];board.loc[mask,"test_precision"]=test["precision"];board.loc[mask,"test_recall"]=test["recall"];board.loc[mask,"test_F1"]=test["f1"];board.loc[mask,"FP_per_million"]=test["fp_per_million"];board.to_csv(run/"leaderboard.csv",index=False)
 best=json.loads((run/"BEST_DEV_CONFIG.json").read_text());seed=pd.DataFrame(best["seed_metrics"]);summary={"average_precision_mean":float(seed.average_precision.mean()),"average_precision_sd":float(seed.average_precision.std(ddof=0)),"pr_auc_mean":float(seed.pr_auc.mean()),"pr_auc_sd":float(seed.pr_auc.std(ddof=0))}
 def best_stage(mode):
  x=board[(board.model=="LAMAR")&(board.training_strategy==mode)&board.dev_average_precision.notna()];return None if x.empty else x.sort_values("dev_average_precision",ascending=False).iloc[0].to_dict()
 kmer=board[board.model.eq("kmer_logistic")].iloc[0];cnn=board[board.model.eq("cnn")].iloc[0];frozen=best_stage("frozen");lora=best_stage("lora");partial=best_stage("partial");full=best_stage("full");threshold=json.loads((run/"final_threshold.json").read_text())
 report=f"""# LAMAR strict binary discovery final report

All labels are computational. The locked test was accessed once, only after dev selection, calibration fitting, and threshold freezing.

## Final recommendation

- checkpoint: `{threshold['model_checkpoint']}`
- strategy: `{best['best_hyperparameters']}`
- calibration: `{threshold['calibration_method']}`
- frozen deployment threshold: `{threshold['threshold']:.12g}`
- three-seed dev AP: `{summary['average_precision_mean']:.6f} ± {summary['average_precision_sd']:.6f}`

## Locked 1:1000 test

- PR-AUC: {test['pr_auc']:.6f}
- Average Precision: {test['average_precision']:.6f}
- Precision: {test['precision']:.6f}
- Recall: {test['recall']:.6f}
- F1: {test['f1']:.6f}
- MCC: {test['mcc']:.6f}
- Brier: {test['brier']:.8f}
- ECE: {test['ece']:.8f}
- false positives per million negatives: {test['fp_per_million']:.3f}
- recall at 10/100/1000 FP/M: {test['recall_at_10_fp_per_million']:.6f} / {test['recall_at_100_fp_per_million']:.6f} / {test['recall_at_1000_fp_per_million']:.6f}
- supplementary ROC-AUC: {test['roc_auc']:.6f}

## Required conclusions

1. Lamar versus k-mer/CNN: best Lamar dev AP `{board[board.model.eq('LAMAR')].dev_average_precision.max():.6f}`; k-mer `{kmer.dev_average_precision:.6f}`; CNN `{cnn.dev_average_precision:.6f}`.
2. Frozen Lamar best dev AP: `{None if frozen is None else frozen['dev_average_precision']}`.
3. LoRA best dev AP: `{None if lora is None else lora['dev_average_precision']}`; compare directly with frozen above.
4. Partial unfreeze best dev AP: `{None if partial is None else partial['dev_average_precision']}`; resource cost is in leaderboard.
5. Full fine tuning: `{('not completed because of hardware/runtime failure' if full is None else 'completed; dev AP '+str(full['dev_average_precision']))}`.
6. Best sampling/negative/loss configuration is recorded verbatim in the final strategy above; stage-wise comparisons are in `leaderboard.csv`.
7. Hard-negative benefit is determined from stage4 rows in `leaderboard.csv`; it was not mined from test.
8. Lamar/CNN receive sequence only. The metadata-only audit baseline and test subgroup tables quantify residual sequence-correlated coverage/expression shortcut behavior.
9. Real 1:1000 precision/recall/FP-per-million are reported above.
10. Use the checkpoint, calibrator, and frozen threshold listed under Final recommendation.

## Limitations

- External basewise mappability remained unavailable in the dataset.
- XGBoost/LightGBM were not installed in the immutable Lamar environment and were not added.
- `q_proj/k_proj/v_proj/o_proj` maps to LAMAR `query/key/value/attention.output.dense`; the requested qkvo and all-attention schemes are architecturally identical and share one execution.
- Accuracy is intentionally not used as a primary result.
""";(run/"final_report.md").write_text(report)
 checks=[]
 for path in sorted(run.rglob("*")):
  if path.is_file() and path.name not in {"checksums.sha256","SUCCESS"}:checks.append(f"{sha256_file(path)}  {path.relative_to(run)}")
 (run/"checksums.sha256").write_text("\n".join(checks)+"\n");(run/"SUCCESS").write_text("PASS\n")
if __name__=="__main__":main()
