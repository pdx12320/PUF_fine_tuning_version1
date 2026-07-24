# Reproducibility

## Inputs

The scripts expect the immutable Phase 2 binary assets:

- `train_positives.tsv.gz`;
- read-only train-negative SQLite pool;
- complete dev/calibration/test embedding parquet files;
- existing Binary Lamar checkpoint and Platt calibrator;
- Lamar pretrained model, tokenizer, and architecture configuration.

Set the placeholders in `configs/ranking_training_config.yaml` to local paths. Do not change labels or split membership.

## Execution order

The high-level order used in the completed run was:

1. `preflight.py`
2. `prepare_dev_universe_v2.py`
3. `generate_lr_grid.py` and `run_experiment_grid.py`
4. full-dev scoring and `select_lr_grid.py`
5. sampling, loss, and lambda grids
6. train-only hard-negative mining, at most two rounds
7. final seed 42/43/44 training and full-dev scoring
8. `summarize_seed_metrics.py`
9. `select_overall_dev_model.py`
10. calibration scoring and `calibrate_and_freeze_thresholds.py`
11. `freeze_pretest_v2.py`
12. `generate_locked_test_manifest.py`
13. one `run_scoring_grid.py` pass on locked test
14. `finalize_locked_test.py`

Example command form:

```bash
python scripts/preflight.py \
  --master configs/ranking_training_config.yaml

python scripts/run_experiment_grid.py \
  --master configs/ranking_training_config.yaml \
  --manifest configs/lr_grid_manifest.json \
  --gpus 0,1
```

Individual scripts expose their exact arguments through `--help`.

## Scientific invariants

Before interpreting outputs, verify:

- train positives = 1,028;
- dev = 159 positives + 282,166 strict negatives;
- calibration = 165 positives + 165,000 strict negatives;
- locked test = 161 positives + 161,000 strict negatives;
- no dev/calibration/test examples are used for hard-negative mining;
- a pre-test freeze exists before locked-test scoring;
- locked test is scored once per frozen model;
- model selection is based on dev, not calibration or test.

## Published-result validation

From the repository root:

```bash
python -m py_compile lamar_ranking_application_phase3/scripts/*.py
python lamar_ranking_application_phase3/scripts/validate_public_export.py
cd lamar_ranking_application_phase3
shasum -a 256 -c provenance/PUBLIC_CHECKSUMS.sha256
```

The validation script checks schemas, expected model rows, primary Top-K values, PNG integrity, public-path sanitization, and required documentation.
