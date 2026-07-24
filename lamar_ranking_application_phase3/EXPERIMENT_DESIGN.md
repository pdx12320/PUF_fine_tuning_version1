# Phase 3 experimental design

## 1. Scientific question

Phase 3 evaluates finite-budget candidate prioritization:

```text
sequence
  → Lamar representation
  → ranking score
  → sort all candidates
  → Top-K experimental candidates
```

The primary question is whether a sequence-only Lamar representation can discover more held-out C-editing positives when only K = 10, 50, 100, 500, or 1000 candidates can be validated.

The central comparison is:

1. use the existing Binary Lamar sigmoid/Platt probability directly as a ranking score;
2. explicitly optimize a new ranking model;
3. determine whether explicit ranking improves Top-K discovery.

This is not a further binary-classifier optimization benchmark.

## 2. Frozen scientific scope

The Phase 2 dataset and classifier were treated as immutable:

- no reconstruction of the binary dataset;
- no changes to positive or strict-negative definitions;
- no changes to train/dev/calibration/test split;
- no retraining of the existing Binary Lamar classifier;
- no test-guided model or threshold selection;
- no coverage, expression, RNA structure, or metadata model inputs;
- gene, coordinate, negative type, and efficiency were retained for reporting only.

All evaluated models are sequence-only.

## 3. Data partitions

| Split | Positive | Strict negative | Role |
|---|---:|---:|---|
| Train | 1,028 | 1,975,244 | Ranking training and train-only negative mining |
| Dev | 159 | 282,166 | Early stopping and final model selection |
| Calibration | 165 | 165,000 | Platt calibration and deployment thresholds only |
| Locked test | 161 | 161,000 | One final evaluation after pre-test freeze |

The full dev and locked-test universes preserve the highly imbalanced candidate-prioritization setting.

## 4. Online pair construction

The complete positive-negative Cartesian product was never materialized.

For each epoch:

1. use the 1,028 train positives as anchors;
2. repeat each positive 10 times;
3. sample one train-only negative for each anchor occurrence;
4. construct 10,280 online pairs;
5. optimize `score_positive > score_negative`.

Sampling is deterministic conditional on experiment seed and epoch:

```text
negative sampling seed = seed + 1009 × (epoch + 1)
positive schedule seed = seed + 7919 × (epoch + 1)
```

Within-epoch negative sampling is without replacement unless a selected source pool is smaller than the request.

## 5. Models

| ID | Model | Design |
|---|---|---|
| 0 | Random ranking | Deterministic SHA-256 uniform score, seed 42 |
| 1 | k-mer ranking | 3–6-mer TF-IDF plus GC/C-count/entropy; online pair-difference logistic SGD |
| 2 | CNN ranking | One-hot sequence → Conv1D(64, k=9) → Conv1D(128, k=7) → scalar head |
| 3 | Existing Binary Lamar | Frozen Phase 2 center-pooled LoRA sigmoid and Platt probability |
| 4 | Frozen Lamar ranking | Frozen backbone → center embedding → MLP scalar head |
| 5 | LoRA Lamar ranking | q/k/v/o LoRA, rank 4, alpha 8, dropout 0.05 → scalar head |
| 6 | Hybrid Binary + Ranking | BCE + lambda × ranking loss; lambda ∈ {0.1, 0.5, 1.0} |

The Lamar ranking head is:

```text
LayerNorm(768) → Linear(256) → GELU → Linear(1)
```

## 6. Ranking losses

Primary loss:

```text
L_pair = -log(sigmoid(score_positive - score_negative))
```

Margin-ranking ablation:

```text
L_margin = max(0, margin - score_positive + score_negative)
margin ∈ {0.1, 0.5, 1.0}
```

Listwise loss was preregistered but skipped because the data contain no natural query groups. Reliability was preferred over adding an artificial grouping definition.

## 7. Negative sampling

The following train-only sources were compared:

- Random strict negative.
- Dataset-defined matched strict negative.
- Dataset-defined rule-hard strict negative.
- Mixed: 20% easy + 40% matched + 40% hard.
- Binary-model-guided high-score strict negative.

Hard-negative mining used at most two rounds:

1. round 1 scanned only the train-negative pool with Existing Binary Lamar;
2. round 2 used the dev-selected round-1 ranking model;
3. each round retained 50,000 strict negatives;
4. dev, calibration, and test were forbidden mining sources;
5. each output round was unique by sequence hash and excluded prior-round candidate IDs.

Final audit found eight sequence hashes shared across the two 50,000-row files at different genomic sites. They belonged to the same leakage group. This is 0.008% of the combined 100,000 rows. The round-2 model was rejected on dev and did not influence the frozen final model or locked-test conclusion.

## 8. Optimization and model selection

- Learning rates: 1e-5, 3e-5, 1e-4.
- Maximum epochs: 20.
- Early-stopping patience: 3.
- Seeds: 42, 43, 44.
- FP16 training.
- LoRA targets: q_proj, k_proj, v_proj, o_proj.

Selection score:

```text
mean(Precision@10, Precision@50, Precision@100,
     Precision@500, Precision@1000)
```

Tie-break order:

1. Precision@100;
2. Precision@500;
3. Average Precision.

Three-seed dev means were used for model-family comparison. Seed 42 was the preregistered deployment checkpoint.

## 9. Locked-test protection

The evaluation sequence was:

```text
train
  → dev model selection
  → three-seed summary
  → calibration-only Platt/threshold analysis
  → PRETEST_FROZEN
  → one locked-test scoring pass
  → final report
```

`PRETEST_FROZEN.json` recorded selected configs, checkpoint hashes, threshold policy, overall dev winner, expected test counts, and one-pass evaluation policy before test access.

No model or threshold selection was performed after the test was opened.

## 10. Metrics

Primary candidate-prioritization metrics:

- Precision@K;
- Recall@K;
- discovered positives at K;
- Enrichment Factor at K:

```text
EF@K = (positive rate in Top K) / (background positive rate)
```

Secondary metrics:

- Average Precision;
- PR-AUC;
- ROC-AUC;
- NDCG.

The deployment budgets were K = 10, 50, 100, 500, and 1000. Ranking ties were broken deterministically by ascending sequence ID.

## 11. Threshold and deployment analysis

Calibration-only policies:

1. exact Top-K;
2. calibration score threshold;
3. calibration threshold followed by a Top-K cap.

Exact Top-K is the recommended primary policy because threshold-only selection can overfill or underfill a fixed experimental budget under distribution shift.

## 12. Shortcut-bias and error analysis

Sequence-composition audit features:

- GC fraction;
- C count;
- sequence entropy.

The composition-only audit had EF@100 = 0 on locked test, compared with EF@100 = 320.32 for Existing Binary Lamar. This supports sequence signal beyond simple composition, although nonzero score-composition correlations remain documented.

Error analysis includes:

- top 100 highest-ranked strict negatives;
- 100 lowest-ranked held-out positives;
- descriptive categories: motif similar, low-efficiency positive, boundary case, sequence-pattern failure, and other.

These categories are heuristic descriptions, not causal explanations of editing or non-editing.

## 13. Predefined interpretation

The project supports candidate prioritization if a sequence-only Lamar score produces strong Top-K enrichment over random. Explicit ranking is considered superior only if it exceeds the frozen Existing Binary Lamar ranker on dev-selected and locked-test Top-K metrics.

The observed result supports the first statement but not the second.
