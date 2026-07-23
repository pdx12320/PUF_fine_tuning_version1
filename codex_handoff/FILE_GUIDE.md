# File guide

## Public archive

### Dataset

- `lamar_binary_project/dataset_build/results/dataset_summary.md`:
  authoritative compact count/limitation summary.
- `results/positive_filter_funnel.tsv` and
  `results/negative_filter_funnel.tsv`: filter attrition.
- `results/qc_assertions.tsv`: scientific assertions.
- `results/positives_main.tsv.gz`: main computational positives.
- `manifests/dataset_manifest.json`: parameters, counts, split protocol, and
  original logical inputs.
- `manifests/input_manifest.tsv`: input sizes, mtimes, and checksums where
  available.
- `provenance/checksums.server.sha256`: checksum index for the entire original
  server run, including omitted large shards.

### Model study

- `lamar_binary_project/model_training/results/leaderboard.csv`: one row per
  experiment.
- `results/final_report.md`: final conclusions.
- `results/final_threshold.json`: calibrator, threshold, and calibration
  workpoint.
- `results/test_metrics.json`: frozen locked-test metrics.
- `error_analysis/`: false-positive and false-negative cases.
- `subgroup_analysis/`: fixed subgroup results.
- `models/final_seeds/final_seed42/`: recommended adapter and summary.
- `configs/`: all staged experiment configurations.
- `scripts/`: training, calibration, and audit code.

### Zero-shot ablation

- `lamar_binary_project/zero_shot_ablation/results/comparison_leaderboard.csv`:
  complete representation/model comparison.
- `reports/final_zero_shot_report.md`: conclusions and caveats.
- `results/dev_selection.json`: dev-only selection record.
- `results/frozen_calibration.json`: pre-test calibration freeze.
- `provenance/run_markers/PRETEST_FROZEN.json`: hash freeze before test.
- `error_analysis/`: zero-shot and linear-probe errors.
- `models/`: centroid, probe, PCA, metadata, combined, and calibration objects.
- `figures/`: PR, ROC, calibration, PCA, and metadata-correlation figures.

## Full server-only artifacts

The following remain under the immutable run directories and are indexed by
the server checksum files:

- complete negative universes and train pool;
- fixed 1:1000 calibration/test tables;
- leakage maps;
- multi-gigabyte embedding matrices;
- calibration/test row-level predictions;
- full and large partial-fine-tuning checkpoints;
- SQLite train-pool indexes;
- runtime logs.

Do not copy these into ordinary Git history. Use an approved object store,
release archive, or institutional data repository if publication requires
distribution.

## Wiki

`igem_drylab_wiki/` contains six modular pages, a compact single-page version,
and reusable assets.

## Path placeholders

- `${LAMAR_WORK_ROOT}`: parent containing dataset/model/ablation directories.
- `${LAMAR_SOURCE_ROOT}`: pretrained LAMAR source/checkpoint tree.
- `${LAMAR_ENV}`: model environment.
- `${IGEM_DATA_ROOT}`: immutable input-data root.
- `${IGEM_ENV}`: dataset/pileup environment.
