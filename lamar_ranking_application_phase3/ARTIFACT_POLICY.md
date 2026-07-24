# Artifact policy

## Included in Git

This public folder includes:

- path-sanitized experimental configuration;
- complete Phase 3 Python source;
- final dev, calibration, and locked-test summary metrics;
- learning-rate, sampling, loss, lambda, hard-mining, and three-seed tables;
- deployment-threshold and shortcut-bias analyses;
- Top-100 false-positive and false-negative analyses;
- final figures and scientific report;
- public checksums.

## Excluded from Git

The following executed artifacts remain in controlled run storage:

- pretrained and fine-tuned model checkpoints;
- full dev/calibration/test prediction parquet files;
- the immutable train-negative SQLite pool;
- 50,000-row hard-negative parquet files;
- the full 161,161-row ranked-candidate table;
- raw dataset and embedding files;
- logs and temporary work products.

They are excluded because of size, data-governance, and repository-portability concerns. No excluded artifact is required to read or verify the published summary conclusions.

## Reproducibility mapping

Logical artifact names used by the code:

| Logical artifact | Expected location under a private run |
|---|---|
| Binary dataset | `${BINARY_DATASET_DIR}` |
| Existing binary model | `${BINARY_MODEL_DIR}` |
| Zero-shot embeddings | `${ZERO_SHOT_DIR}` |
| Lamar pretrained checkpoint | `${LAMAR_PRETRAINED_CHECKPOINT}` |
| Ranking output | `${RUN_DIR}` |

The public config intentionally uses placeholders and contains no host-specific paths, credentials, or institutional endpoints.
