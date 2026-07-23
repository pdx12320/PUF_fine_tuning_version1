# LAMAR sequence-level C-editing discovery

This repository archives a reproducible computational study for ranking
101-nt RNA sequence contexts for experimental follow-up of C-to-U editing.
All labels in this repository are **computational labels**; they are not claims
of experimentally verified true positives or true negatives.

本仓库归档了一个可复现的 LAMAR 序列二分类项目：使用 101 nt
转录本方向序列判断候选中心 C 是否值得进入后续实验验证。仓库中的阳性和阴性均为
计算标签，不代表已经完成实验真阳性或真阴性验证。

## Headline results

- Computational positives: 1,513; high-confidence subset: 1,457.
- Strict computational-negative universe: 2,821,734 sites.
- Frozen evaluation: 161 positives and 161,000 strict negatives (1:1000).
- Best model: center-pooled LAMAR with q/k/v/o LoRA, rank 4.
- Locked-test average precision: 0.171958.
- Frozen operating point: precision 0.476190, recall 0.124224, and
  136.646 false positives per million test negatives.
- Pretrained center representation without gradient-based adaptation achieved
  test AP 0.036180; a linear probe achieved 0.066262.

## Repository layout

| Folder | Purpose |
| --- | --- |
| [`lamar_binary_project/`](lamar_binary_project/) | Dataset construction, training, ablation code, configurations, result tables, selected models, and provenance |
| [`igem_drylab_wiki/`](igem_drylab_wiki/) | Wiki-ready dry-lab narrative and figures |
| [`codex_handoff/`](codex_handoff/) | Frozen rules, run state, file guide, and next-agent handoff |

## Reproducibility boundary

The original successful runs are immutable. Their run identifiers are:

- dataset: `run_20260721T231520Z`
- model study: `run_20260722T203752Z`
- zero-shot ablation: `run_20260723T045155Z`

The locked test was opened once only after model selection, probability
calibration, and threshold selection were frozen. The test must not be reused
for hyperparameter, negative-mining, calibration, or threshold decisions.

## Large artifacts

The complete server outputs total approximately 8.5 GB. GitHub is used for
code, paper-level results, error analysis, figures, compact models, and
documentation—not for multi-gigabyte embeddings, negative universes, SQLite
training pools, or full prediction matrices. The original SHA-256 manifests
are preserved under each run's `provenance/` folder, and
[`ARTIFACT_POLICY.md`](lamar_binary_project/ARTIFACT_POLICY.md) documents every
omission class.

Public files use environment-variable placeholders rather than user-specific
server paths. See
[`REPRODUCIBILITY.md`](lamar_binary_project/REPRODUCIBILITY.md).
