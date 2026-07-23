# Pretrained-representation ablation

This folder archives successful ablation `run_20260723T045155Z`.

The original pretrained LAMAR checkpoint was loaded with 85,851,793 frozen
backbone parameters and zero trainable backbone parameters. Hidden states had
shape `[batch, 103, 768]`: 101 nucleotide tokens plus model special tokens.

Four representations were compared: center, mean, masked mean, and CLS.
Center pooling was decisively strongest. The term “zero-shot” means no
gradient-based task adaptation; labeled train centroids were still estimated,
as required by the experimental design.

Key files:

- `results/comparison_leaderboard.csv`
- `reports/final_zero_shot_report.md`
- `results/calibration_results.csv`
- `results/threshold_analysis.csv`
- `error_analysis/`
- `figures/`
- `models/`

Full embeddings and calibration/test prediction matrices are referenced by
their original checksums but are not stored in Git.
