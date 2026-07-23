# Reproducibility and open documentation

The public archive contains:

- dataset-construction and model-training source;
- all experiment configurations;
- QC assertions and filter funnels;
- aggregate dev, calibration, locked-test, subgroup, and ablation tables;
- false-positive and false-negative analyses;
- figures and Wiki-ready assets;
- compact baseline and final LoRA model artifacts;
- server and public-copy checksum manifests.

The multi-gigabyte derived arrays and full negative pools are not placed in
ordinary Git history. Their hashes remain in the successful-run checksum
manifests. This keeps the repository auditable without treating GitHub as a
raw sequencing or embedding store.

Three immutable run identifiers define the study:

- dataset: `run_20260721T231520Z`;
- model study: `run_20260722T203752Z`;
- pretrained-representation ablation: `run_20260723T045155Z`.

Environment-specific paths in the public snapshot use placeholders such as
`${LAMAR_WORK_ROOT}` and `${IGEM_DATA_ROOT}`. The full reproduction guide is
in `lamar_binary_project/REPRODUCIBILITY.md`.
