# LAMAR strict binary discovery final report

All labels are computational. The locked test was accessed once, only after dev selection, calibration fitting, and threshold freezing.

## Final recommendation

- checkpoint: `${LAMAR_WORK_ROOT}/lamar_binary_models/run_20260722T203752Z/checkpoints/runs/final_seed42/best_trainable.safetensors`
- strategy: `{'accumulation_steps': 2, 'backbone_lr': 1e-05, 'batch_size': 16, 'epochs': 20, 'eval_batch_size': 128, 'experiment_id': 's7_wd0', 'fp16': True, 'head_dropout': 0.1, 'head_lr': 0.0001, 'lora_dropout': 0.05, 'lora_rank': 4, 'lora_scheme': 'qkvo', 'loss': 'bce', 'mode': 'lora', 'negative_strategy': 'random', 'patience': 3, 'pooling': 'center', 'sampling_ratio': 10, 'seed': 42, 'stage': 'stage7_hparams', 'warmup_ratio': 0.03, 'weight_decay': 0.0}`
- calibration: `platt`
- frozen deployment threshold: `0.242983699736`
- three-seed dev AP: `0.808719 ± 0.013597`

## Locked 1:1000 test

- PR-AUC: 0.169754
- Average Precision: 0.171958
- Precision: 0.476190
- Recall: 0.124224
- F1: 0.197044
- MCC: 0.242859
- Brier: 0.00090393
- ECE: 0.00017023
- false positives per million negatives: 136.646
- recall at 10/100/1000 FP/M: 0.018634 / 0.099379 / 0.248447
- supplementary ROC-AUC: 0.957783

## Required conclusions

1. Lamar versus k-mer/CNN: best Lamar dev AP `0.825438`; k-mer `0.330512`; CNN `0.365052`.
2. Frozen Lamar best dev AP: `0.6859885628873351`.
3. LoRA best dev AP: `0.8254375974333445`; compare directly with frozen above.
4. Partial unfreeze best dev AP: `0.7759295749224606`; resource cost is in leaderboard.
5. Full fine tuning: `completed; dev AP 0.7990661424959914`.
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
