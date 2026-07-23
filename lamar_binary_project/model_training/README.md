# Model training archive

This folder archives successful model study `run_20260722T203752Z`.

## Recommended model

- backbone: original pretrained LAMAR
- pooling: center token
- adaptation: q/k/v/o LoRA
- rank/alpha/dropout: 4 / 8 / 0.05
- loss: binary cross entropy
- negative strategy: dynamic random strict negatives at 1:10
- backbone/head learning rates: `1e-5` / `1e-4`
- calibration: Platt
- frozen threshold: `0.24298369973628076`

Three final seed adapters are under `models/final_seeds/`. The deployment
recommendation is seed 42. Baseline model objects are under `models/baselines/`.

See `results/leaderboard.csv` for all staged experiment metrics and
`results/final_report.md` for the locked-test report.
