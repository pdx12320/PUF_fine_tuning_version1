# Result table guide

| File | Content |
|---|---|
| `ranking_leaderboard.csv` | Final locked-test comparison across seven ranking models |
| `precision_recall_at_k.csv` | Precision, recall, and discoveries at each experimental budget |
| `enrichment_results.csv` | Enrichment factor relative to split background rate |
| `test_ranking_metrics.json` | Complete locked-test ranking metrics, including AP/PR-AUC/ROC-AUC/NDCG |
| `dev_final_deployment_metrics.csv` | Complete dev metrics for frozen deployment checkpoints |
| `calibration_ranking_metrics.csv` | Calibration-universe ranking metrics |
| `calibration_analysis.csv` | Platt-calibration diagnostics |
| `threshold_strategy_results.csv` | Top-K, threshold-only, and threshold-plus-cap simulations |
| `final_seed_summary.csv` | Three-seed mean and variability on dev |
| `lr_grid_dev_fixed.csv` | Learning-rate search on the fixed dev selection universe |
| `sampling_full_dev_metrics.csv` | Negative-sampling ablation |
| `loss_full_dev_metrics.csv` | Pairwise-logistic and margin-loss ablation |
| `lambda_full_dev_metrics.csv` | Hybrid-loss lambda ablation |
| `guided_round2_full_dev_metrics.csv` | Pre/post second-round guided hard-mining comparison |
| `composition_audit_dev.json` | Dev shortcut-composition baseline |
| `shortcut_bias_analysis.csv` | Locked-test score correlations with GC, C count, and entropy |
| `overall_dev_selection.json` | Pre-test overall winner selected on complete dev |
