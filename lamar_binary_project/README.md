# Project archive

This directory contains three linked computational stages.

1. [`dataset_build/`](dataset_build/) recounts six MarkDuplicates RNA-seq BAMs,
   constructs computational positives and expression-supported strict
   computational negatives, applies sequence QC, and creates leakage-safe
   splits.
2. [`model_training/`](model_training/) compares k-mer, CNN, frozen,
   partial-unfreeze, LoRA, and full-fine-tuning models, then calibrates a
   frozen deployment threshold on a 1:1000 calibration set.
3. [`zero_shot_ablation/`](zero_shot_ablation/) evaluates whether the original
   pretrained LAMAR representation contains task-relevant sequence
   information before gradient-based task adaptation.

The end-to-end design is described in
[`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md). Reproduction and path
rendering are described in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Scientific language

Use the terms **computational positive** and **strict computational negative**.
The read-level Fisher/BH statistics are screening statistics and are not
biological-replicate experimental validation.

## Frozen invariants

- Every input sequence is 101 nt in transcript orientation.
- The modeled center is zero-based nucleotide index 50 and must be `C`.
- Dataset splits must never be randomized or regenerated for model comparison.
- Gene, leakage-group, exact-sequence, genomic-key, and overlapping-window
  leakage checks must remain zero.
- Calibration and test prevalence are both approximately 1:1000.
- The locked test is not a development resource.
