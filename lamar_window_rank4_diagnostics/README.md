# LAMAR centered-window and rank-4 LoRA diagnostics

This folder preserves two linked, dev-only diagnostics performed before any
new LoRA retraining:

1. a controlled Frozen + Head screen of 21, 41, 61, and 101 bp windows
   centered on the target C;
2. a singular-spectrum audit of every LoRA update in the existing rank-4
   `final_seed42` QKVO checkpoint.

The archive contains compact results and runnable analysis code. It
intentionally excludes source data, full prediction tables, model checkpoints,
server logs, caches, and machine-specific paths.

## Main conclusions

### Centered-window screen

The only experimental variable was the centered crop:

| Window | Definition | Complete-dev AP | ROC-AUC | P@10 | P@100 |
|---:|---|---:|---:|---:|---:|
| 61 bp | 30 + C + 30 | **0.038114** | 0.929425 | **0.40** | **0.12** |
| 101 bp | 50 + C + 50 | 0.036738 | 0.931075 | 0.30 | 0.07 |
| 21 bp | 10 + C + 10 | 0.031559 | **0.933560** | 0.10 | 0.10 |
| 41 bp | 20 + C + 20 | 0.025335 | 0.933403 | 0.10 | 0.09 |

The prespecified selection metric was average precision on the complete dev
universe, so the screen selected **61 bp**. Its AP was about 3.7% higher than
101 bp, but this is a single-seed screening result rather than a statistical
significance claim. The 101 bp condition remains the closest comparator.

Locked invariants:

- identical train/dev split;
- identical epoch-level negative IDs;
- seed 42;
- identical tokenizer and pretrained LAMAR state;
- Frozen backbone with center-pooled classification head;
- 7,080 optimizer steps for every window;
- identical optimizer and hyperparameters;
- identical complete dev universe: 159 positives and 282,166 negatives;
- no calibration or test access.

The 101 bp final trainable checkpoint was byte-identical to the original
Frozen + Head baseline, providing a direct reproduction control.

### Existing rank-4 LoRA checkpoint

The audited checkpoint used rank 4 QKVO LoRA in all 12 transformer layers:
48 update matrices in total. The implementation used `alpha = 8`, so:

```text
delta_W = (alpha / r) * B @ A = 2 * B @ A
```

All 48 updates had numerical rank 4 at float32 machine-precision tolerance.
However, their spectra were usually concentrated in fewer directions:

| Module | Mean entropy effective rank | Mean stable rank | Median sigma4/sigma1 |
|---|---:|---:|---:|
| K | **3.074** | **1.368** | **0.1683** |
| Q | 2.116 | 1.083 | 0.0444 |
| O | 2.020 | 1.052 | 0.0479 |
| V | 1.842 | 1.019 | 0.0375 |

Across all modules, median entropy effective rank was 2.021/4 and median
stable rank was 1.022. Thus rank 4 was algebraically active but not evenly
used in spectral-energy terms. The checkpoint does not provide evidence that
rank 4 was a capacity bottleneck or that rank 8 should be preferred. A
controlled rank 2 versus rank 4 experiment would be more informative than
expanding rank solely on the basis of this checkpoint.

## Folder guide

```text
configs/
  window_ablation.example.yaml
results/
  window_ablation/
    preflight_assertions.tsv
    window_ablation_summary.json
    window_ablation_summary.tsv
  rank4_spectrum/
    rank4_lora_module_spectra.tsv
    rank4_lora_module_type_summary.tsv
    rank4_lora_spectrum_summary.public.json
scripts/
  preflight.py
  run_window.py
  summarize.py
  analyze_rank4_spectrum.py
PROCESS.md
VALIDATION.md
software_versions.tsv
```

The 48-row module spectrum table is the primary detailed LoRA result. It
contains `sigma_1` through `sigma_4`, singular-value ratios, numerical rank,
entropy effective rank, stable rank, and update-matrix norms for every layer
and Q/K/V/O module.

## Reproduction outline

Copy `configs/window_ablation.example.yaml`, replace only the declared path
placeholders, and run:

```bash
python scripts/preflight.py --config configs/window_ablation.yaml

python scripts/run_window.py \
  --config configs/window_ablation.yaml \
  --window 21 \
  --device cuda:0

# Repeat for 41, 61, and 101 bp, one at a time.
python scripts/summarize.py --config configs/window_ablation.yaml
```

Audit an existing rank-4 checkpoint with:

```bash
python scripts/analyze_rank4_spectrum.py \
  --checkpoint /path/to/best_trainable.safetensors \
  --run-config /path/to/final_seed42.json \
  --output-dir /path/to/new/rank4_spectrum_audit
```

The scripts refuse to overwrite existing formal run or result directories.
