# Sequence-only Lamar C-editing candidate prioritization

## Scope and interpretation

This Phase 3 benchmark evaluates finite-budget ranking, not further binary-classifier optimization. Positives and strict negatives remain the immutable computational labels from the binary dataset. “Positive” therefore means a held-out computational positive; this analysis does not replace prospective experimental validation.

The locked test contained 161 positives and 161,000 strict negatives. Model selection used dev only, deployment thresholds used calibration only, and test scoring occurred after the pre-test freeze.

## Direct answers

1. **Is Binary Lamar probability an effective ranker?** Yes. Locked-test AP was 0.171958, Precision@100 was 0.320, Precision@500 was 0.124, and EF@100 was 320.32.
2. **Did explicit ranking optimization exceed the binary classifier?** No. Existing Binary Lamar was selected on dev before test access and remained the strongest locked-test ranker.
3. **Was LoRA ranking better than binary LoRA?** No. LoRA ranking achieved Precision@100 = 0.200 and AP = 0.092178, versus 0.320 and 0.171958 for Existing Binary Lamar.
4. **Top-100 positive proportion:** 0.320, or 32/100.
5. **Top-500 positive proportion:** 0.124, or 62/500.
6. **Cost reduction from 1,000 to 100 validations:** 900 fewer validations, a 90% reduction under equal per-candidate cost. Top 100 retained 32/85 = 37.6% of positives found in Top 1000.
7. **Enrichment over random:** EF@100 = 320.32, EF@500 = 124.12, and EF@1000 = 85.09.
8. **Did model-guided hard-negative mining help?** No. Round-1 guided-minus-random Precision@K changes were negative at every budget: -0.10, -0.08, -0.13, -0.07, and -0.047 for K = 10, 50, 100, 500, and 1000. Round 2 also decreased all five values and was rejected on dev.
9. **Is sequence-only Lamar sufficient for prioritization?** It is sufficient to produce strong enrichment against these held-out computational labels. It is not sufficient to establish experimental editing, causal biology, transcript availability, or tissue-specific deployability.
10. **Sequence signal versus GC/C-count bias:** The composition-only audit achieved EF@100 = 0, versus 320.32 for Existing Binary Lamar. Performance beyond composition supports richer sequence signal, while nonzero score-composition correlations remain a documented shortcut risk.
11. **Final recommendation:** Existing Binary Lamar with Platt calibration and deterministic exact Top-K. Use calibration threshold + Top-K only when a minimum confidence floor is operationally required.

## Locked-test leaderboard

| Model | P@10 | P@50 | P@100 | P@500 | P@1000 | AP | NDCG |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing Binary Lamar | 0.700 | 0.460 | 0.320 | 0.124 | 0.085 | 0.171958 | 0.689140 |
| LoRA Lamar ranking | 0.500 | 0.280 | 0.200 | 0.100 | 0.060 | 0.092178 | 0.608652 |
| Hybrid Binary + Ranking | 0.500 | 0.220 | 0.170 | 0.082 | 0.055 | 0.081020 | 0.597272 |
| Frozen Lamar ranking | 0.100 | 0.260 | 0.190 | 0.072 | 0.046 | 0.060002 | 0.544523 |
| CNN ranking | 0.100 | 0.020 | 0.010 | 0.012 | 0.008 | 0.005480 | 0.401722 |
| k-mer ranking | 0.000 | 0.000 | 0.000 | 0.006 | 0.012 | 0.003659 | 0.390932 |
| Random ranking | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000980 | 0.344366 |

The full table is `results/ranking_leaderboard.csv`.

## Ranking, calibration, and deployment

Primary deployment metrics are Precision@K, Recall@K, discoveries, and enrichment for K = 10, 50, 100, 500, and 1000. AP, PR-AUC, ROC-AUC, and NDCG are secondary.

Raw and Platt-calibrated Binary Lamar scores have the same ordering because the fitted Platt slope is positive. Calibration therefore changes probability interpretation and score thresholds, not Top-K membership.

Exact Top-K is deterministic using ascending sequence ID as the tie-break. Threshold-only policies may underfill or overfill a fixed budget under distribution shift; `results/threshold_strategy_results.csv` quantifies this.

## Negative sampling and losses

Training generated 10,280 online anchor-positive/sampled-negative pairs per epoch and never materialized the multi-billion complete pair set. Sampling, loss, learning-rate, lambda, and three-seed results are preserved in the corresponding CSV files.

Each hard-negative round contained 50,000 unique sequence hashes and no repeated candidate IDs across rounds. A final audit found eight sequence hashes shared across rounds at different genomic sites within the same leakage group. The round-2 model was rejected on dev, so this did not influence the final model or test conclusion.

## Shortcut and error analysis

The shortcut audit used only GC fraction, C count, and sequence entropy. Gene, coordinate, negative type, and efficiency were retained for reporting and were never model inputs.

`error_analysis/top100_false_positives.csv` contains the highest-ranked strict negatives. Its categories are descriptive sequence-pattern heuristics, not causal explanations. `top100_false_negatives.csv` contains the 100 lowest-ranked positives, including low-efficiency positives.

## Reproducibility and limitations

- Public path configuration: `configs/ranking_training_config.yaml`.
- Executed Phase 3 scripts: `scripts/`.
- Three-seed variability: `results/final_seed_summary.csv`.
- Test labels were accessed only after the pre-test freeze.
- Full checkpoints and row-level prediction artifacts are not stored in Git; see `ARTIFACT_POLICY.md`.
- The labels are computational rather than independent prospective experimental truth.
