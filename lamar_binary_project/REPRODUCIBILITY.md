# Reproducibility guide

## Public path variables

The public archive replaces user-specific paths with these placeholders:

```bash
export LAMAR_WORK_ROOT=/path/to/lamar7.21
export LAMAR_ENV=/path/to/lamar/conda_environment
export LAMAR_SOURCE_ROOT=/path/to/LAMAR/source
export IGEM_DATA_ROOT=/path/to/igem/data
export IGEM_ENV=/path/to/igem/conda_environment
```

JSON and YAML configuration snapshots contain literal `${VARIABLE}` tokens.
Render a working copy outside this repository before execution, for example:

```bash
envsubst < configs/master.yaml > /path/to/run/configs/master.rendered.yaml
```

Python scripts under the run snapshots are provenance code. Where a script
offers CLI path arguments, prefer them. Where a constant contains a literal
`${VARIABLE}`, render a temporary copy or update the constant in a new run
directory—never modify a successful frozen run.

## Environments recorded by the successful runs

The zero-shot ablation used:

- Python 3.11.15
- PyTorch 2.0.1+cu117
- transformers 4.32.1
- scikit-learn 1.5.2
- pandas 3.0.2
- NumPy 1.26.4
- pyarrow 20.0.0

The dataset run's full software record is
[`software_versions.txt`](dataset_build/manifests/software_versions.txt).

## Reproducing each stage

Dataset construction:

1. Render or provide input paths.
2. Run `dataset_build/scripts/preflight.py`.
3. Run the pileup smoke test.
4. Run `dataset_build/scripts/build_dataset.py` in a new UTC timestamped
   directory.
5. Require every assertion in `qc_assertions.tsv` to pass before writing
   `SUCCESS`.

Model study:

1. Treat the successful dataset split files as immutable.
2. Run `model_training/scripts/audit_and_index.py`.
3. Train baselines, then stage-wise LAMAR experiments using dev only.
4. Fit calibration and freeze the threshold on calibration.
5. Create the pre-test marker, then run the locked-test suite once.

Zero-shot ablation:

1. Load the original pretrained checkpoint, not a task adapter.
2. Confirm all backbone parameters have `requires_grad=False`.
3. Extract train and dev embeddings; select representation/probe using dev AP.
4. Extract calibration embeddings and freeze calibration/thresholds.
5. Access locked test only after `PRETEST_FROZEN.json`.

## Validation

The original server checksums are preserved as
`provenance/checksums.server.sha256`. They validate original frozen outputs,
not the path-sanitized public copies. The repository-level
`PUBLIC_CHECKSUMS.sha256` validates the public archive itself.
