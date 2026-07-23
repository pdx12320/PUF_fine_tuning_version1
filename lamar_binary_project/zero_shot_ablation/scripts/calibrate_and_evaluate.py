#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedGroupKFold

from ablation_common import (
    apply_calibrator,
    binary_metrics,
    embedding_matrix,
    embedding_metadata,
    fit_calibrator,
    fp_budget_threshold,
    hash_random_probability,
    load_config,
    sha256_file,
    write_json,
    zero_shot_probability,
)


def selected_candidates(run):
    selection = json.loads((run / "results/dev_selection.json").read_text())
    candidates = {
        "hash_random": {"family": "random", "representation": "none", "trainable_parameters": 0},
    }
    for representation, row in selection["selected_zero_by_representation"].items():
        candidates[f"lamar_zero_shot_{representation}"] = {
            "family": "zero_shot", "representation": representation,
            "artifact": str(run / "models" / f"{row['candidate']}.joblib"),
            "dev_average_precision": row["dev_average_precision"],
            "dev_pr_auc": row["dev_pr_auc"], "trainable_parameters": 0,
            "method": row["method"],
        }
    for representation, row in selection["selected_linear_by_representation"].items():
        candidates[f"lamar_linear_probe_{representation}"] = {
            "family": "linear_probe", "representation": representation,
            "artifact": str(run / "models/linear_probe" / f"{row['candidate']}.joblib"),
            "dev_average_precision": row["dev_average_precision"],
            "dev_pr_auc": row["dev_pr_auc"], "trainable_parameters": 769,
            "C": row["C"], "class_weight": row["class_weight"],
        }
    references = {
        "kmer_logistic": {"family": "reference", "representation": "1-4mer", "trainable_parameters": 344},
        "cnn": {"family": "reference", "representation": "onehot", "trainable_parameters": 59969},
        "frozen_lamar_head": {"family": "reference", "representation": "center", "trainable_parameters": 2305},
        "partial_lamar_2blocks": {"family": "reference", "representation": "center", "trainable_parameters": 14179585},
        "full_lamar": {"family": "reference", "representation": "center", "trainable_parameters": 85854098},
        "lora_best": {"family": "reference", "representation": "center", "trainable_parameters": 297217},
    }
    candidates.update(references)
    return candidates, selection


def score_frame(run, embedding_path, reference_path):
    metadata = embedding_metadata(embedding_path)
    references = pd.read_parquet(reference_path).set_index("sequence_id")
    frame = metadata.copy()
    candidates, _ = selected_candidates(run)
    frame["hash_random"] = hash_random_probability(frame.sequence_hash)
    loaded = {}
    for name, spec in candidates.items():
        if spec["family"] == "zero_shot":
            representation = spec["representation"]
            if representation not in loaded:
                loaded[representation] = embedding_matrix(embedding_path, representation)
            model = joblib.load(spec["artifact"])
            frame[name] = zero_shot_probability(model, loaded[representation])
        elif spec["family"] == "linear_probe":
            representation = spec["representation"]
            if representation not in loaded:
                loaded[representation] = embedding_matrix(embedding_path, representation)
            model = joblib.load(spec["artifact"])
            frame[name] = model["model"].predict_proba(
                model["scaler"].transform(loaded[representation])
            )[:, 1]
        elif spec["family"] == "reference":
            frame[name] = references.loc[frame.sequence_id, name].to_numpy()
    return frame, candidates


def dev_metrics(run, candidates):
    zero_predictions = pd.read_parquet(run / "predictions/zero_shot_dev_predictions.parquet")
    linear_predictions = pd.read_parquet(run / "predictions/linear_probe_dev_predictions.parquet")
    reference = pd.read_parquet(run / "predictions/reference_dev.parquet").set_index("sequence_id")
    selection = json.loads((run / "results/dev_selection.json").read_text())
    result = {}
    random_probability = hash_random_probability(zero_predictions.sequence_hash)
    result["hash_random"] = binary_metrics(zero_predictions.label, random_probability)
    for name, spec in candidates.items():
        if spec["family"] == "zero_shot":
            row = selection["selected_zero_by_representation"][spec["representation"]]
            result[name] = binary_metrics(zero_predictions.label, zero_predictions[row["candidate"]])
        elif spec["family"] == "linear_probe":
            row = selection["selected_linear_by_representation"][spec["representation"]]
            result[name] = binary_metrics(linear_predictions.label, linear_predictions[row["candidate"]])
        elif spec["family"] == "reference":
            result[name] = binary_metrics(reference.label, reference[name])
    return result


def calibration_stage(cfg):
    run = Path(cfg["run_dir"])
    if not (run / "DEV_SELECTION_COMPLETE").exists():
        raise RuntimeError("Dev selection is not frozen")
    if (run / "CALIBRATION_COMPLETE").exists():
        raise FileExistsError("CALIBRATION_COMPLETE")
    frame, candidates = score_frame(
        run, run / "embeddings/calibration.parquet",
        run / "predictions/reference_calibration.parquet",
    )
    labels = frame.label.to_numpy(dtype=np.int64)
    groups = frame.leakage_group.to_numpy()
    folds = list(StratifiedGroupKFold(
        int(cfg["calibration_folds"]), shuffle=True, random_state=int(cfg["seed"])
    ).split(np.zeros(len(labels)), labels, groups))
    calibration_rows, threshold_rows = [], []
    frozen = {}
    curve_rows = []
    calibrated_columns = frame[["sequence_id", "split", "label", "genomic_key"]].copy()
    for name, spec in candidates.items():
        raw = frame[name].to_numpy(dtype=np.float64)
        oof = {method: np.zeros(len(labels), dtype=np.float64) for method in ("none", "platt", "isotonic")}
        for train_index, valid_index in folds:
            for method in oof:
                fitted = fit_calibrator(method, raw[train_index], labels[train_index])
                oof[method][valid_index] = apply_calibrator(method, fitted, raw[valid_index])
        comparison = []
        for method, probability in oof.items():
            metrics = binary_metrics(labels, probability)
            row = {"model": name, "method": method, **metrics}
            comparison.append(row)
            calibration_rows.append(row)
        chosen = min(comparison, key=lambda row: (row["brier"], row["ece"]))["method"]
        fitted = fit_calibrator(chosen, raw, labels)
        calibrated = apply_calibrator(chosen, fitted, raw)
        artifact = run / f"models/calibrators/{name}.joblib"
        joblib.dump({"method": chosen, "model": fitted}, artifact)
        workpoints = {}
        for target in (10, 50, 100, 500, 1000):
            point = fp_budget_threshold(labels, calibrated, target)
            threshold_rows.append({"model": name, "kind": "fp_workpoint", **point})
            workpoints[str(target)] = point
        for threshold in (.01, .05, .1, .2, .5):
            threshold_rows.append({
                "model": name, "kind": "fixed_threshold",
                **binary_metrics(labels, calibrated, threshold),
            })
        default = workpoints[str(cfg["deployment_target_fp_per_million"])]
        frozen[name] = {
            **spec, "calibration_method": chosen, "calibrator": str(artifact),
            "threshold": default["threshold"], "calibration_metrics": binary_metrics(
                labels, calibrated, default["threshold"]
            ),
        }
        calibrated_columns[f"{name}_raw"] = raw
        calibrated_columns[f"{name}_probability"] = calibrated
        truth, predicted = calibration_curve(labels, oof[chosen], n_bins=12, strategy="quantile")
        curve_rows.extend({
            "model": name, "method": chosen,
            "mean_predicted_probability": float(x), "observed_frequency": float(y),
        } for x, y in zip(predicted, truth))
    calibrated_columns.to_parquet(run / "predictions/calibration_predictions.parquet", index=False)
    pd.DataFrame(calibration_rows).to_csv(run / "results/calibration_results.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(run / "results/threshold_analysis.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(run / "results/calibration_curve_points.csv", index=False)
    manifest = {
        "status": "PASS", "selection": "lowest 5-fold OOF Brier, ECE tie-break",
        "threshold_policy": f"maximum recall at <= {cfg['deployment_target_fp_per_million']} FP per million calibration negatives",
        "models": frozen,
    }
    write_json(run / "results/frozen_calibration.json", manifest)
    pretest_files = [
        run / "results/dev_selection.json", run / "results/frozen_calibration.json",
        *[Path(spec["calibrator"]) for spec in frozen.values()],
    ]
    write_json(run / "PRETEST_FROZEN.json", {
        "status": "FROZEN_BEFORE_TEST", "files": {
            str(path.relative_to(run)): sha256_file(path) for path in pretest_files
        },
    })
    (run / "CALIBRATION_COMPLETE").write_text("PASS\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def test_stage(cfg):
    run = Path(cfg["run_dir"])
    if not (run / "PRETEST_FROZEN.json").exists() or not (run / "TEST_EVALUATION_STARTED").exists():
        raise RuntimeError("Pre-test freeze/sentinel missing")
    if (run / "TEST_EVALUATION_COMPLETE").exists():
        raise FileExistsError("TEST_EVALUATION_COMPLETE")
    frame, candidates = score_frame(
        run, run / "embeddings/test.parquet", run / "predictions/reference_test.parquet"
    )
    labels = frame.label.to_numpy(dtype=np.int64)
    dev = dev_metrics(run, candidates)
    frozen = json.loads((run / "results/frozen_calibration.json").read_text())["models"]
    prediction = frame[[
        "sequence_id", "split", "label", "genomic_key", "gene_id", "gene_name",
        "sequence_context", "sequence_hash", "negative_type", "true_efficiency",
        "gc_fraction", "c_count", "entropy", "median_depth", "gene_coverage",
    ]].copy()
    leaderboard = []
    zero_metrics, linear_metrics = [], []
    calibration_predictions = pd.read_parquet(run / "predictions/calibration_predictions.parquet")
    for name, spec in candidates.items():
        raw = frame[name].to_numpy(dtype=np.float64)
        calibrator = joblib.load(frozen[name]["calibrator"])
        probability = apply_calibrator(calibrator["method"], calibrator["model"], raw)
        threshold = float(frozen[name]["threshold"])
        test_metrics = binary_metrics(labels, probability, threshold)
        calibration_probability = calibration_predictions[f"{name}_probability"].to_numpy()
        calibration_labels = calibration_predictions.label.to_numpy()
        calibration_metrics = binary_metrics(
            calibration_labels, calibration_probability, threshold
        )
        prediction[f"{name}_raw"] = raw
        prediction[f"{name}_probability"] = probability
        prediction[f"{name}_prediction"] = probability >= threshold
        row = {
            "model": name, "trainable_parameters": spec["trainable_parameters"],
            "representation": spec["representation"], "pooling": spec["representation"],
            "dev_AP": dev[name]["average_precision"],
            "calibration_AP": calibration_metrics["average_precision"],
            "test_AP": test_metrics["average_precision"],
            "test_PR_AUC": test_metrics["pr_auc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"], "test_F1": test_metrics["f1"],
            "test_MCC": test_metrics["mcc"], "ROC_AUC": test_metrics["roc_auc"],
            "Brier": test_metrics["brier"], "ECE": test_metrics["ece"],
            "FP_per_million": test_metrics["fp_per_million"],
            "threshold": threshold, "calibration_method": frozen[name]["calibration_method"],
            "test_tn": test_metrics["tn"], "test_fp": test_metrics["fp"],
            "test_fn": test_metrics["fn"], "test_tp": test_metrics["tp"],
        }
        leaderboard.append(row)
        family_row = {
            **row, "split": "test",
            "method": spec.get("method", spec.get("family")),
        }
        if spec["family"] == "zero_shot":
            zero_metrics.append(family_row)
        elif spec["family"] == "linear_probe":
            linear_metrics.append(family_row)
    prediction.to_parquet(run / "predictions/all_test_predictions.parquet", index=False)
    leaderboard_frame = pd.DataFrame(leaderboard).sort_values("test_AP", ascending=False)
    leaderboard_frame.to_csv(run / "results/comparison_leaderboard.csv", index=False)
    pd.DataFrame(zero_metrics).to_csv(run / "results/zero_shot_metrics.csv", index=False)
    pd.DataFrame(linear_metrics).to_csv(run / "results/linear_probe_metrics.csv", index=False)

    selection = json.loads((run / "results/dev_selection.json").read_text())
    best_zero_rep = selection["global_best_zero_shot"]["representation"]
    best_linear_rep = selection["global_best_linear_probe"]["representation"]
    selected_names = {
        "zero_shot": f"lamar_zero_shot_{best_zero_rep}",
        "linear_probe": f"lamar_linear_probe_{best_linear_rep}",
    }
    low_depth = float(calibration_predictions.merge(
        embedding_metadata(run / "embeddings/calibration.parquet")[["sequence_id", "median_depth"]],
        on="sequence_id"
    ).median_depth.quantile(1 / 3))
    low_entropy = float(embedding_metadata(
        run / "embeddings/calibration.parquet"
    ).entropy.quantile(1 / 3))
    for family, name in selected_names.items():
        threshold = float(frozen[name]["threshold"])
        probability = prediction[f"{name}_probability"]
        predicted = probability >= threshold
        actual_fp = prediction[(prediction.label == 0) & predicted].sort_values(
            f"{name}_probability", ascending=False
        ).copy()
        top_negative = prediction[prediction.label == 0].sort_values(
            f"{name}_probability", ascending=False
        ).head(100).copy()
        false_negative = prediction[(prediction.label == 1) & ~predicted].sort_values(
            f"{name}_probability"
        ).copy()
        top_motifs = set(top_negative.sequence_context.str[48:53])
        def category(row):
            if row.true_efficiency <= .15: return "low_efficiency_positive"
            if row.median_depth <= low_depth: return "coverage_related"
            if row.entropy <= low_entropy: return "hard_sequence"
            if row.sequence_context[48:53] in top_motifs: return "special_motif"
            return "other"
        if len(false_negative):
            false_negative["error_category"] = false_negative.apply(category, axis=1)
        directory = run / f"error_analysis/{family}"
        actual_fp.to_csv(directory / "predicted_false_positives.csv", index=False)
        top_negative.to_csv(directory / "false_positive_top100.csv", index=False)
        false_negative.to_csv(directory / "false_negative_all.csv", index=False)
    write_json(run / "results/test_evaluation_manifest.json", {
        "status": "PASS", "test_rows": len(frame), "label_1": int(labels.sum()),
        "label_0": int((labels == 0).sum()), "selected_error_models": selected_names,
        "test_was_not_used_for_selection_calibration_or_thresholds": True,
    })
    (run / "TEST_EVALUATION_COMPLETE").write_text("PASS\n")
    print(leaderboard_frame.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("calibration", "test"), required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    calibration_stage(cfg) if args.stage == "calibration" else test_stage(cfg)


if __name__ == "__main__":
    main()
