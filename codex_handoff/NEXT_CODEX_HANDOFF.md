# Next Codex handoff: LAMAR C-editing discovery

## Current state

The dataset, complete model study, locked-test evaluation, and pretrained
zero-shot/linear-probe ablation are finished. All three final run directories
contain `SUCCESS`. No training, extraction, or evaluation process remains
active.

Do not rebuild, resplit, or retrain merely to reproduce the current reports.
The public GitHub archive is a path-sanitized snapshot, while the frozen server
runs remain the authoritative full artifacts.

## Run identifiers

Under `${LAMAR_WORK_ROOT}`:

- dataset:
  `lamar_binary_dataset/run_20260721T231520Z`
- model study:
  `lamar_binary_models/run_20260722T203752Z`
- representation ablation:
  `lamar_zero_shot_ablation/run_20260723T045155Z`

Each parent directory also has a `LATEST_SUCCESSFUL_RUN.txt` pointer.

## Frozen dataset facts

- main computational positives: 1,513
- high-confidence computational positives: 1,457
- strict computational-negative universe: 2,821,734
- train positives: 1,028
- dev: 159 positives + 1,590 negatives
- calibration: 165 positives + 165,000 negatives
- locked test: 161 positives + 161,000 negatives

Every sequence is transcript-oriented, 101 nt long, and centered at
zero-based nucleotide index 50 on `C`.

All-six usable-depth-at-least-20 and all-six target-alt-zero violations among
strict negatives: 0.

Detected cross-split gene, leakage-group, exact-sequence, genomic-key, or
overlapping-window leakage: 0.

## Frozen-test rules

The locked test has already been used for the single pre-registered final
suite. It must not now be used to:

- choose models or hyperparameters;
- alter negative sampling or mining;
- fit a calibrator;
- adjust the threshold;
- decide new features;
- select among post-hoc variants.

New scientific development should use train/dev/calibration, or preferably a
new external dataset. Test metrics may be reported but not optimized.

## Final model

The recommended adapter is:

`${LAMAR_WORK_ROOT}/lamar_binary_models/run_20260722T203752Z/checkpoints/runs/final_seed42/best_trainable.safetensors`

It must be loaded with the original base checkpoint:

`${LAMAR_SOURCE_ROOT}/base_2k/mammalian80D_2048len1mer1sw_80M/checkpoint-250000/model.safetensors`

Configuration:

- center pooling;
- q/k/v/o LoRA;
- rank 4, alpha 8, dropout 0.05;
- BCE;
- dynamic random strict negatives at 1:10;
- backbone/head learning rates `1e-5` / `1e-4`;
- batch size 16, accumulation 2;
- warmup 0.03, weight decay 0;
- Platt calibration;
- frozen threshold `0.24298369973628076`.

Locked 1:1000 test:

- AP 0.171958; PR-AUC 0.169754;
- precision 0.476190; recall 0.124224;
- F1 0.197044; MCC 0.242859;
- 20 TP, 22 FP, 141 FN, 160,978 TN;
- 136.646 FP per million test negatives.

The calibration-set threshold target was at most 100 FP/M and achieved 96.97
FP/M. Test exceeded the target. Do not “correct” this using test labels.

## Representation-ablation conclusion

- pretrained center centroid, no gradient task adaptation: test AP 0.036180;
- center linear probe, 769 parameters: 0.066262;
- frozen LAMAR head: 0.064390;
- LoRA: 0.171958;
- CNN: 0.013912;
- k-mer logistic: 0.006598.

Pretraining contains useful signal, a simple classifier extracts much of the
fixed-representation signal, and LoRA contributes substantial additional
task adaptation. “Zero-shot” still used labeled train centroids; it is not
label-free.

## Shortcut warning

On dev:

- metadata-only AP: 0.759266;
- embedding-only linear AP: 0.698583;
- combined AP: 0.813666.

PC3 correlated with GC fraction at `r=0.637`. Coverage-PC correlations were
smaller, but sequence composition and dataset-generation bias cannot be
excluded. Do not claim that the model has isolated a causal editing motif.

## Known limitations

- External basewise mappability was unavailable and remains
  `NA_RESOURCE_MISSING`.
- Labels are computational and derived from one six-sample system.
- The strict negative definition cannot prove universal biological absence.
- Pooled Fisher/BH values are read-level screening, not biological-replicate
  experimental validation.
- XGBoost/LightGBM were unavailable and not installed.
- No independent prospective experiment has yet validated deployment
  precision.

## Safe next work

High-value continuations that do not invalidate the locked test:

1. package inference for a prospectively declared external experiment;
2. acquire an external basewise mappability resource and build a new versioned
   dataset rather than relabeling the frozen run in place;
3. design prospective wet-lab validation with threshold and sample size fixed
   before labels are observed;
4. analyze known editor motifs and structural features on train/dev only;
5. write a model card and external-deployment monitoring plan.

Any new dataset, threshold, reference, or scientific definition requires a new
UTC timestamped run and explicit approval.

## Before taking action

1. source an untracked path mapping based on `SERVER_PATHS.example.env`;
2. inspect every `SUCCESS` and checksum manifest;
3. read the relevant run's summary and configuration;
4. check GPU, CPU, memory, disk, and currently running jobs;
5. run a small fixture before any expensive computation;
6. obtain approval before launching a full-data or high-resource job.
