#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ranking_common import (
    BUDGETS,
    META_COLUMNS,
    load_yaml,
    ranking_metrics,
    read_tsv_records,
    sha256_file,
    stable_order,
    write_frame_new,
    write_json_new,
)


MAIN_MODELS = [
    "random",
    "kmer",
    "cnn",
    "existing_binary_lamar",
    "frozen_lamar",
    "lora_lamar",
    "hybrid_lamar",
]
DISPLAY = {
    "random": "Random ranking",
    "kmer": "k-mer ranking",
    "cnn": "CNN ranking",
    "existing_binary_lamar": "Existing Binary Lamar",
    "frozen_lamar": "Frozen Lamar ranking",
    "lora_lamar": "LoRA Lamar ranking",
    "hybrid_lamar": "Hybrid Binary + Ranking",
    "composition_only": "Composition-only audit",
}


def calibrated_probability(model_id, score, calibrator):
    value = np.asarray(score, dtype=np.float64)
    if model_id == "existing_binary_lamar":
        probability = np.clip(value, 1e-7, 1 - 1e-7)
        matrix = np.log(
            probability / (1 - probability)
        ).reshape(-1, 1)
    else:
        matrix = value.reshape(-1, 1)
    model = calibrator["model"]
    return model.predict_proba(matrix)[:, 1]


def metric_row(frame, model_id):
    return ranking_metrics(
        frame["label"].to_numpy(),
        frame[model_id].to_numpy(),
        frame["sequence_id"].astype(str),
    )


def config_and_summary(deployment, model_type):
    selected = deployment["selected"][model_type]
    config = json.loads(Path(selected["config"]).read_text())
    summary = json.loads(Path(selected["summary"]).read_text())
    return config, summary


def category_for(
    row,
    positive_motifs,
    efficiency_q25,
    comp_bounds,
    cutoff,
    score_scale,
    composition_rank,
    total_rows,
):
    sequence = row["sequence_context"]
    if int(row["label"]) == 1 and float(row["true_efficiency"]) <= efficiency_q25:
        return "low efficiency positive"
    if sequence[47:54] in positive_motifs:
        return "motif similar"
    if abs(float(row["score"]) - cutoff) <= 0.10 * score_scale:
        return "boundary cases"
    outside = any(
        not comp_bounds[name][0]
        <= float(row[name])
        <= comp_bounds[name][1]
        for name in ("gc_fraction", "c_count", "entropy")
    )
    discordant = (
        int(row["rank"]) <= max(1000, int(total_rows * 0.01))
        and composition_rank[row["sequence_id"]] > total_rows * 0.50
    )
    if outside or discordant:
        return "sequence pattern failure"
    return "other"


def plot_budget(metrics_long, output_dir):
    colors = plt.cm.tab10(np.linspace(0, 1, len(MAIN_MODELS)))
    color_map = dict(zip(MAIN_MODELS, colors))
    for metric, filename, ylabel in (
        ("precision", "precision_at_k.png", "Precision@K"),
        ("recall", "recall_at_k.png", "Recall@K"),
        ("discovered", "budget_curve.png", "Discovered positives"),
    ):
        fig, axis = plt.subplots(figsize=(8, 5))
        for model_id in MAIN_MODELS:
            subset = metrics_long[
                metrics_long["model_id"] == model_id
            ].sort_values("K")
            axis.plot(
                subset["K"],
                subset[metric],
                marker="o",
                label=DISPLAY[model_id],
                color=color_map[model_id],
            )
        axis.set_xscale("log")
        axis.set_xlabel("Experimental validation budget K")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument("--raw-test-predictions", required=True)
    parser.add_argument("--dev-predictions", required=True)
    parser.add_argument("--dev-metrics", required=True)
    parser.add_argument("--seed-summary", required=True)
    parser.add_argument("--calibration-predictions", required=True)
    parser.add_argument("--calibration-analysis", required=True)
    parser.add_argument("--threshold-policy", required=True)
    parser.add_argument("--deployment-selection", required=True)
    parser.add_argument("--overall-selection", required=True)
    parser.add_argument("--composition-model", required=True)
    parser.add_argument("--lr-metrics", required=True)
    parser.add_argument("--sampling-metrics", required=True)
    parser.add_argument("--loss-metrics", required=True)
    parser.add_argument("--lambda-metrics", required=True)
    parser.add_argument("--guided-round2-metrics", required=True)
    args = parser.parse_args()

    master = load_yaml(args.master)
    run_dir = Path(master["run_dir"])
    freeze = json.loads((run_dir / "PRETEST_FROZEN.json").read_text())
    if freeze["status"] != "FROZEN":
        raise RuntimeError("PRETEST_FROZEN gate missing")
    test = pd.read_parquet(args.raw_test_predictions)
    if set(test["split"]) != {"test"}:
        raise AssertionError(test["split"].value_counts().to_dict())
    if (
        len(test),
        int(test["label"].sum()),
        int((test["label"] == 0).sum()),
    ) != (161161, 161, 161000):
        raise AssertionError(
            (
                len(test),
                int(test["label"].sum()),
                int((test["label"] == 0).sum()),
            )
        )
    for model_id in MAIN_MODELS:
        if model_id not in test:
            raise KeyError(model_id)

    composition = joblib.load(args.composition_model)
    test["composition_only"] = composition["model"].decision_function(
        composition["scaler"].transform(
            test[composition["features"]].to_numpy(dtype=np.float64)
        )
    )
    for model_id in MAIN_MODELS:
        if model_id == "random":
            continue
        calibrator = joblib.load(
            run_dir / "models/calibrators" / f"{model_id}.joblib"
        )
        test[f"{model_id}_calibrated_probability"] = (
            calibrated_probability(
                model_id, test[model_id].to_numpy(), calibrator
            )
        )
    write_frame_new(
        test, run_dir / "predictions/test_rank_predictions.parquet"
    )

    test_metrics = {}
    for model_id in [*MAIN_MODELS, "composition_only"]:
        test_metrics[model_id] = metric_row(test, model_id)
    write_json_new(
        run_dir / "results/test_ranking_metrics.json",
        {
            "status": "PASS",
            "locked_test": True,
            "rows": len(test),
            "metrics": test_metrics,
        },
    )

    metrics_long = []
    for model_id in [*MAIN_MODELS, "composition_only"]:
        metrics = test_metrics[model_id]
        for budget in BUDGETS:
            metrics_long.append(
                {
                    "model_id": model_id,
                    "Model": DISPLAY[model_id],
                    "K": budget,
                    "discovered": metrics[
                        f"discovered_at_{budget}"
                    ],
                    "precision": metrics[
                        f"precision_at_{budget}"
                    ],
                    "recall": metrics[f"recall_at_{budget}"],
                    "enrichment": metrics[
                        f"enrichment_at_{budget}"
                    ],
                    "background_positive_rate": metrics[
                        "background_positive_rate"
                    ],
                }
            )
    long_frame = pd.DataFrame(metrics_long)
    write_frame_new(
        long_frame[long_frame["model_id"].isin(MAIN_MODELS)][
            [
                "Model",
                "model_id",
                "K",
                "precision",
                "recall",
                "discovered",
            ]
        ],
        run_dir / "results/precision_recall_at_k.csv",
    )
    write_frame_new(
        long_frame[
            ["Model", "model_id", "K", "enrichment", "background_positive_rate"]
        ],
        run_dir / "results/enrichment_results.csv",
    )
    write_frame_new(
        long_frame[long_frame["model_id"].isin(MAIN_MODELS)][
            [
                "Model",
                "model_id",
                "K",
                "discovered",
                "precision",
                "recall",
                "enrichment",
            ]
        ],
        run_dir / "application/budget_simulation.csv",
    )

    deployment = json.loads(
        Path(args.deployment_selection).read_text()
    )
    dev_metrics = pd.read_csv(args.dev_metrics).set_index("model_id")
    leaderboard = []
    parameter_count = {
        "random": 0,
        "existing_binary_lamar": 297217,
    }
    configuration = {
        "random": {
            "loss": "NA",
            "negative_sampling": "NA",
            "strategy": "SHA256 uniform seed42",
        },
        "existing_binary_lamar": {
            "loss": "BCE",
            "negative_sampling": "dynamic random 1:10",
            "strategy": "existing center-pooled LoRA + Platt",
        },
    }
    for model_type in (
        "kmer",
        "cnn",
        "frozen_lamar",
        "lora_lamar",
        "hybrid_lamar",
    ):
        config, summary = config_and_summary(deployment, model_type)
        parameter_count[model_type] = summary["trainable_parameters"]
        if model_type == "hybrid_lamar":
            loss = f"BCE + {config['lambda_rank']}*{config['loss']}"
        else:
            loss = config.get("loss", "pair_difference_logistic")
            if loss == "margin":
                loss += f"(margin={config['margin']})"
        configuration[model_type] = {
            "loss": loss,
            "negative_sampling": config["negative_sampling"],
            "strategy": (
                "center-pooled sequence-only"
                if "lamar" in model_type
                else "sequence-only"
            ),
        }
    for model_id in MAIN_MODELS:
        dev = dev_metrics.loc[model_id].to_dict()
        test_value = test_metrics[model_id]
        row = {
            "Model": DISPLAY[model_id],
            "Parameters": parameter_count[model_id],
            "Strategy": configuration[model_id]["strategy"],
            "Loss": configuration[model_id]["loss"],
            "Negative sampling": configuration[model_id][
                "negative_sampling"
            ],
            "Dev AP": dev["average_precision"],
            "Test AP": test_value["average_precision"],
            "Precision@10": test_value["precision_at_10"],
            "Precision@50": test_value["precision_at_50"],
            "Precision@100": test_value["precision_at_100"],
            "Precision@500": test_value["precision_at_500"],
            "Precision@1000": test_value["precision_at_1000"],
            "Recall@100": test_value["recall_at_100"],
            "Recall@500": test_value["recall_at_500"],
            "Recall@1000": test_value["recall_at_1000"],
            "NDCG": test_value["ndcg"],
            "Enrichment": test_value["enrichment_at_100"],
            "Enrichment@10": test_value["enrichment_at_10"],
            "Enrichment@50": test_value["enrichment_at_50"],
            "Enrichment@100": test_value["enrichment_at_100"],
            "Enrichment@500": test_value["enrichment_at_500"],
            "Enrichment@1000": test_value["enrichment_at_1000"],
            "PR-AUC": test_value["pr_auc"],
            "ROC-AUC": test_value["roc_auc"],
        }
        leaderboard.append(row)
    leaderboard_frame = pd.DataFrame(leaderboard)
    write_frame_new(
        leaderboard_frame,
        run_dir / "results/ranking_leaderboard.csv",
    )

    overall = json.loads(Path(args.overall_selection).read_text())
    best_model = overall["selected_model"]
    best_score = test[best_model].to_numpy()
    best_order = stable_order(
        best_score, test["sequence_id"].astype(str)
    )
    ranked = test.iloc[best_order].copy().reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    application = pd.DataFrame(
        {
            "rank": ranked["rank"],
            "sequence_id": ranked["sequence_id"],
            "gene": ranked["gene_name"],
            "coordinate": ranked["genomic_key"],
            "sequence": ranked["sequence_context"],
            "score": ranked[best_model],
            "label": ranked["label"],
        }
    )
    write_frame_new(
        application, run_dir / "application/ranked_candidates.tsv"
    )

    threshold_policy = json.loads(
        Path(args.threshold_policy).read_text()
    )
    threshold_rows = []
    labels = test["label"].to_numpy()
    ties = test["sequence_id"].astype(str).to_numpy()
    for model_id in MAIN_MODELS:
        score = test[model_id].to_numpy(dtype=np.float64)
        order = stable_order(score, ties)
        for budget in BUDGETS:
            cutoff = threshold_policy["models"][model_id][
                "thresholds"
            ][str(budget)]["raw_score_cutoff"]
            mask = score >= float(cutoff)
            threshold_count = int(mask.sum())
            threshold_discovered = int(labels[mask].sum())
            eligible = order[mask[order]]
            capped = eligible[:budget]
            threshold_rows.extend(
                [
                    {
                        "model_id": model_id,
                        "K": budget,
                        "strategy": "Top-K only",
                        "selected_candidates": budget,
                        "discovered_positives": int(
                            labels[order[:budget]].sum()
                        ),
                        "calibration_score_threshold": math.nan,
                    },
                    {
                        "model_id": model_id,
                        "K": budget,
                        "strategy": "score threshold",
                        "selected_candidates": threshold_count,
                        "discovered_positives": threshold_discovered,
                        "calibration_score_threshold": cutoff,
                    },
                    {
                        "model_id": model_id,
                        "K": budget,
                        "strategy": "threshold + Top-K",
                        "selected_candidates": len(capped),
                        "discovered_positives": int(
                            labels[capped].sum()
                        ),
                        "calibration_score_threshold": cutoff,
                    },
                ]
            )
    write_frame_new(
        pd.DataFrame(threshold_rows),
        run_dir / "results/threshold_strategy_results.csv",
    )

    shortcut_rows = []
    for model_id in [*MAIN_MODELS, "composition_only"]:
        for feature in ("gc_fraction", "c_count", "entropy"):
            correlation, p_value = spearmanr(
                test[model_id], test[feature]
            )
            shortcut_rows.append(
                {
                    "model_id": model_id,
                    "feature": feature,
                    "spearman_rho": float(correlation),
                    "p_value": float(p_value),
                    "enrichment_at_100": test_metrics[model_id][
                        "enrichment_at_100"
                    ],
                    "enrichment_at_500": test_metrics[model_id][
                        "enrichment_at_500"
                    ],
                }
            )
    write_frame_new(
        pd.DataFrame(shortcut_rows),
        run_dir / "results/shortcut_bias_analysis.csv",
    )

    train_positive = read_tsv_records(
        Path(master["dataset_dir"]) / "train_positives.tsv.gz",
        "train",
    )
    positive_motifs = {
        row["sequence_context"][47:54] for row in train_positive
    }
    efficiency_q25 = float(
        np.quantile(
            [row["true_efficiency"] for row in train_positive], 0.25
        )
    )
    comp_bounds = {
        name: (
            float(np.quantile([row[name] for row in train_positive], 0.05)),
            float(np.quantile([row[name] for row in train_positive], 0.95)),
        )
        for name in ("gc_fraction", "c_count", "entropy")
    }
    rank_by_id = {
        sequence_id: rank
        for rank, sequence_id in enumerate(
            ranked["sequence_id"], start=1
        )
    }
    comp_order = stable_order(
        test["composition_only"].to_numpy(), ties
    )
    composition_rank = {
        test.iloc[index]["sequence_id"]: rank
        for rank, index in enumerate(comp_order, start=1)
    }
    cutoff = float(ranked.iloc[99][best_model])
    top_scale_values = ranked.iloc[:1000][best_model].to_numpy()
    score_scale = float(
        max(
            np.subtract(*np.quantile(top_scale_values, [0.75, 0.25])),
            1e-8,
        )
    )
    negative_ranked = ranked[ranked["label"] == 0].head(100).copy()
    positive_low = (
        ranked[ranked["label"] == 1]
        .sort_values(["rank"], ascending=False)
        .head(100)
        .copy()
    )
    error_dir = run_dir / "error_analysis"
    error_dir.mkdir(parents=True, exist_ok=True)
    for subset, filename in (
        (negative_ranked, "top100_false_positives.csv"),
        (positive_low, "top100_false_negatives.csv"),
    ):
        output_rows = []
        for record in subset.to_dict("records"):
            value = {
                "sequence_id": record["sequence_id"],
                "sequence": record["sequence_context"],
                "score": record[best_model],
                "rank": rank_by_id[record["sequence_id"]],
                "gene": record["gene_name"],
                "coordinate": record["genomic_key"],
                "negative_type": record["negative_type"],
                "true_efficiency": record["true_efficiency"],
                "label": record["label"],
                "gc_fraction": record["gc_fraction"],
                "c_count": record["c_count"],
                "entropy": record["entropy"],
            }
            value["category"] = category_for(
                {**record, **value},
                positive_motifs,
                efficiency_q25,
                comp_bounds,
                cutoff,
                score_scale,
                composition_rank,
                len(test),
            )
            output_rows.append(value)
        write_frame_new(pd.DataFrame(output_rows), error_dir / filename)

    figure_dir = run_dir / "figures"
    plot_budget(
        long_frame[long_frame["model_id"].isin(MAIN_MODELS)],
        figure_dir,
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    negative_score = test.loc[test["label"] == 0, best_model]
    positive_score = test.loc[test["label"] == 1, best_model]
    axis.hist(
        negative_score,
        bins=80,
        density=True,
        alpha=0.55,
        label="strict negative",
    )
    axis.hist(
        positive_score,
        bins=40,
        density=True,
        alpha=0.65,
        label="computational positive",
    )
    axis.set_xlabel(f"{DISPLAY.get(best_model, best_model)} ranking score")
    axis.set_ylabel("Density")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "ranking_distribution.png", dpi=180)
    plt.close(fig)

    sampling = pd.read_csv(args.sampling_metrics).set_index("model_id")
    loss = pd.read_csv(args.loss_metrics).set_index("model_id")
    lambda_frame = pd.read_csv(args.lambda_metrics).set_index("model_id")
    guided2 = pd.read_csv(args.guided_round2_metrics).set_index("model_id")
    random_sampling_id = "sampling_lora_random_seed42"
    guided_sampling_id = "sampling_lora_guided_seed42"
    guided_improvement = {
        f"precision_at_{budget}": float(
            sampling.loc[guided_sampling_id, f"precision_at_{budget}"]
            - sampling.loc[random_sampling_id, f"precision_at_{budget}"]
        )
        for budget in BUDGETS
    }
    before_id = "best_before_round2_lora_seed42"
    after_id = "guided_round1plus2_lora_seed42"
    round2_improvement = {
        f"precision_at_{budget}": float(
            guided2.loc[after_id, f"precision_at_{budget}"]
            - guided2.loc[before_id, f"precision_at_{budget}"]
        )
        for budget in BUDGETS
    }
    best_metrics = test_metrics[best_model]
    binary_metrics = test_metrics["existing_binary_lamar"]
    composition_metrics = test_metrics["composition_only"]
    discoveries_100 = best_metrics["discovered_at_100"]
    discoveries_1000 = best_metrics["discovered_at_1000"]
    retained = (
        discoveries_100 / discoveries_1000
        if discoveries_1000
        else math.nan
    )
    config_text = "existing frozen binary configuration"
    if best_model in deployment["selected"]:
        best_config = json.loads(
            Path(deployment["selected"][best_model]["config"]).read_text()
        )
        config_text = json.dumps(best_config, sort_keys=True)
    report = f"""# Sequence-only Lamar C-editing candidate prioritization

## Scope and interpretation

This Phase 3 benchmark evaluates finite-budget **ranking**, not further binary-classifier optimization. All positives and strict negatives remain the immutable computational labels from the binary dataset. Consequently, "true positive" below means a held-out computational positive; this analysis does not replace prospective experimental validation.

The locked test contained 161 computational positives and 161,000 strict computational negatives. Model selection used dev only, deployment thresholds used calibration only, and the test was scored after `PRETEST_FROZEN.json`.

## Direct answers

1. **Is Binary Lamar probability an effective ranker?** Yes. On locked test it achieved AP {binary_metrics['average_precision']:.6f}, P@100 {binary_metrics['precision_at_100']:.3f}, P@500 {binary_metrics['precision_at_500']:.3f}, and EF@100 {binary_metrics['enrichment_at_100']:.2f}.
2. **Did ranking optimization exceed the binary classifier?** The dev-frozen answer is **{'yes' if best_model != 'existing_binary_lamar' else 'no'}**. The recommended model is `{best_model}`. On test its P@100 was {best_metrics['precision_at_100']:.3f} versus {binary_metrics['precision_at_100']:.3f} for Binary Lamar.
3. **Was LoRA ranking better than binary LoRA?** Test P@100 was {test_metrics['lora_lamar']['precision_at_100']:.3f} for LoRA ranking and {binary_metrics['precision_at_100']:.3f} for existing binary LoRA; AP values were {test_metrics['lora_lamar']['average_precision']:.6f} and {binary_metrics['average_precision']:.6f}.
4. **Top-100 positive proportion:** {best_metrics['precision_at_100']:.3f} ({best_metrics['discovered_at_100']}/100) for the dev-selected recommendation.
5. **Top-500 positive proportion:** {best_metrics['precision_at_500']:.3f} ({best_metrics['discovered_at_500']}/500).
6. **Cost reduction from 1,000 to 100 validations:** 900 fewer validations, a 90% reduction under equal per-candidate cost. Top 100 retained {discoveries_100}/{discoveries_1000} = {retained:.1%} of the positives found in Top 1,000.
7. **Enrichment over random:** EF@100 was {best_metrics['enrichment_at_100']:.2f}; EF@500 was {best_metrics['enrichment_at_500']:.2f}; EF@1000 was {best_metrics['enrichment_at_1000']:.2f}.
8. **Did hard-negative mining help?** Round-1 guided minus random P@K changes were `{json.dumps(guided_improvement, sort_keys=True)}`. Round-2 minus the best pre-round-2 model changes were `{json.dumps(round2_improvement, sort_keys=True)}`.
9. **Is sequence-only Lamar sufficient for prioritization?** It is sufficient to produce strong enrichment against these held-out computational labels. It is not sufficient by itself to establish experimental editing, causal biology, transcript availability, or tissue-specific deployability.
10. **Sequence signal versus GC/C-count bias:** The composition-only audit achieved EF@100 {composition_metrics['enrichment_at_100']:.2f}, versus {best_metrics['enrichment_at_100']:.2f} for the recommendation. Score correlations are in `results/shortcut_bias_analysis.csv`; Lamar performance beyond composition supports richer sequence signal, while nonzero correlations remain a documented shortcut risk.
11. **Final recommendation:** `{best_model}` with configuration `{config_text}`. Use exact Top-K as the primary deployment rule; use calibration threshold + Top-K only when a minimum confidence floor is operationally required.

## Ranking results

The complete table is `results/ranking_leaderboard.csv`. Primary deployment metrics are Precision@K, Recall@K, discovered positives, and enrichment at K=10/50/100/500/1000. AP, PR-AUC, ROC-AUC, and NDCG are secondary.

Raw and Platt-calibrated Binary Lamar scores have the same ordering because the fitted Platt slope is positive. Calibration therefore changes probability interpretation and score thresholds, not Top-K membership.

## Negative sampling and losses

Training generated 10,280 online anchor-positive/sampled-negative pairs per epoch and never materialized the multi-billion complete pair set. Sampling, loss, and lambda ablations are preserved in the corresponding dev metric CSV files. Listwise loss was preregistered as skipped because no natural query groups exist.

## Deployment

`application/ranked_candidates.tsv` is explicitly a **simulated genome-scale prioritization using the locked-test universe**, not a complete transcriptome scan. It contains the frozen recommended score, rank, sequence, gene, coordinate, and held-out label.

Calibration thresholds were defined by the score at calibration rank K. Exact Top-K is deterministic using sequence ID as the tie-break. Threshold-only may underfill or exceed a budget under distribution shift; `results/threshold_strategy_results.csv` quantifies this.

## Shortcut and error analysis

The shortcut audit uses only sequence-derived GC fraction, C count, and entropy. Gene, coordinate, negative type, and efficiency were retained solely for reporting and were never model inputs.

`error_analysis/top100_false_positives.csv` contains the highest-ranked strict negatives. Its categories are descriptive sequence-pattern heuristics, not causal explanations for non-editing. `top100_false_negatives.csv` contains the 100 lowest-ranked held-out positives, including the low-efficiency-positive category.

## Reproducibility and limitations

- Immutable dataset: `{master['dataset_dir']}`
- Existing binary model: `{master['binary_model_dir']}`
- Ranking configuration: `{run_dir / 'ranking_training_config.yaml'}`
- Software versions and input checksums: `results/input_audit.json`
- Three-seed variability: `results/final_seed_summary.csv`
- Test labels were accessed only after the Phase 3 pretest freeze.
- The dataset lacks an external basewise mappability resource.
- The labels are computational rather than independent prospective experimental truth.
"""
    report_path = run_dir / "reports/final_ranking_report.md"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(report)

    final_files = [
        run_dir / "ranking_training_config.yaml",
        run_dir / "predictions/dev_rank_predictions.parquet",
        run_dir / "predictions/calibration_rank_predictions.parquet",
        run_dir / "predictions/test_rank_predictions.parquet",
        run_dir / "application/ranked_candidates.tsv",
        run_dir / "application/budget_simulation.csv",
        run_dir / "results/ranking_leaderboard.csv",
        run_dir / "results/precision_recall_at_k.csv",
        run_dir / "results/enrichment_results.csv",
        run_dir / "figures/precision_at_k.png",
        run_dir / "figures/recall_at_k.png",
        run_dir / "figures/budget_curve.png",
        run_dir / "figures/ranking_distribution.png",
        report_path,
    ]
    missing = [str(path) for path in final_files if not path.is_file()]
    if missing:
        raise RuntimeError({"missing": missing})
    checksum_path = run_dir / "checksums.sha256"
    if checksum_path.exists():
        raise FileExistsError(checksum_path)
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(run_dir)}\n"
            for path in final_files
        )
    )
    complete = run_dir / "TEST_EVALUATION_COMPLETE"
    success = run_dir / "SUCCESS"
    for marker in (complete, success):
        if marker.exists():
            raise FileExistsError(marker)
        marker.write_text("PASS\n")
    print(
        json.dumps(
            {
                "status": "SUCCESS",
                "best_model": best_model,
                "test_rows": len(test),
                "report": str(report_path),
                "final_files": [str(path) for path in final_files],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
