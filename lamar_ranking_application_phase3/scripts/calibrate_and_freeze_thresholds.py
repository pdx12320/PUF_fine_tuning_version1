#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from ranking_common import (
    BUDGETS,
    stable_order,
    write_frame_new,
    write_json_new,
)


NON_SCORE = {
    "sequence_id",
    "split",
    "label",
    "chrom",
    "position",
    "genomic_key",
    "gene_id",
    "gene_name",
    "transcript_ids",
    "sequence_context",
    "sequence_hash",
    "leakage_group",
    "negative_type",
    "true_efficiency",
    "gc_fraction",
    "c_count",
    "entropy",
}


def expected_calibration_error(labels, probability, bins=15):
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (
            p < right if right < 1.0 else p <= right
        )
        if mask.any():
            value += mask.mean() * abs(
                y[mask].mean() - p[mask].mean()
            )
    return float(value)


def transform_input(model_id: str, score: np.ndarray) -> np.ndarray:
    value = np.asarray(score, dtype=np.float64)
    if model_id == "existing_binary_lamar":
        probability = np.clip(value, 1e-7, 1 - 1e-7)
        value = np.log(probability / (1 - probability))
    return value.reshape(-1, 1)


def fit_platt_oof(model_id, score, labels):
    matrix = transform_input(model_id, score)
    y = np.asarray(labels, dtype=np.int64)
    folds = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    oof = np.empty(len(y), dtype=np.float64)
    for train_index, validation_index in folds.split(matrix, y):
        model = LogisticRegression(
            C=1e6, solver="lbfgs", max_iter=1000
        )
        model.fit(matrix[train_index], y[train_index])
        oof[validation_index] = model.predict_proba(
            matrix[validation_index]
        )[:, 1]
    final = LogisticRegression(
        C=1e6, solver="lbfgs", max_iter=1000
    ).fit(matrix, y)
    probability = final.predict_proba(matrix)[:, 1]
    if float(final.coef_[0, 0]) <= 0:
        raise RuntimeError(
            f"Non-positive Platt slope for {model_id}: "
            f"{final.coef_[0, 0]}"
        )
    return final, oof, probability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--existing-binary-calibrator", required=True)
    parser.add_argument("--calibrator-dir", required=True)
    parser.add_argument("--analysis-csv", required=True)
    parser.add_argument("--threshold-json", required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.predictions)
    if set(frame["split"]) != {"calibration"}:
        raise AssertionError(frame["split"].value_counts().to_dict())
    labels = frame["label"].to_numpy(dtype=np.int64)
    if (int(labels.sum()), int((labels == 0).sum())) != (165, 165000):
        raise AssertionError(
            (int(labels.sum()), int((labels == 0).sum()))
        )
    model_ids = [
        column for column in frame.columns if column not in NON_SCORE
    ]
    calibrator_dir = Path(args.calibrator_dir)
    if calibrator_dir.exists():
        raise FileExistsError(calibrator_dir)
    calibrator_dir.mkdir(parents=True)
    analysis = []
    policy = {}
    tie_breaker = frame["sequence_id"].astype(str).to_numpy()
    inherited = joblib.load(args.existing_binary_calibrator)

    for model_id in model_ids:
        score = frame[model_id].to_numpy(dtype=np.float64)
        if model_id == "random":
            model = None
            oof = np.full(len(labels), labels.mean())
            calibrated = oof.copy()
            method = "no_calibration_random_control"
        else:
            model, oof, calibrated = fit_platt_oof(
                model_id, score, labels
            )
            method = "platt_5fold_oof_then_full_fit"
            if model_id == "existing_binary_lamar":
                # Preserve the already frozen binary Phase-2 calibrator for
                # deployment; the Phase-3 refit is retained only as an audit.
                deployment = inherited
                deployment_method = "inherited_phase2_platt"
            else:
                deployment = {
                    "method": "platt",
                    "input": "raw_score",
                    "model": model,
                }
                deployment_method = "phase3_full_calibration_platt"
            joblib_path = calibrator_dir / f"{model_id}.joblib"
            if joblib_path.exists():
                raise FileExistsError(joblib_path)
            joblib.dump(deployment, joblib_path)
        thresholds = {}
        order = stable_order(score, tie_breaker)
        for budget in BUDGETS:
            k = min(budget, len(score))
            cutoff = float(score[order[k - 1]])
            threshold_count = int((score >= cutoff).sum())
            if model is not None:
                calibrated_cutoff = float(
                    model.predict_proba(
                        transform_input(
                            model_id, np.asarray([cutoff])
                        )
                    )[0, 1]
                )
            else:
                calibrated_cutoff = None
            thresholds[str(budget)] = {
                "raw_score_cutoff": cutoff,
                "calibrated_probability_cutoff": calibrated_cutoff,
                "calibration_candidates_at_or_above_cutoff": (
                    threshold_count
                ),
                "requested_budget": budget,
            }
        policy[model_id] = {
            "calibration_method": (
                deployment_method
                if model_id != "random"
                else method
            ),
            "thresholds": thresholds,
        }
        analysis.append(
            {
                "model_id": model_id,
                "method": method,
                "oof_brier": float(
                    brier_score_loss(labels, oof)
                ),
                "oof_ece": expected_calibration_error(labels, oof),
                "full_fit_brier": float(
                    brier_score_loss(labels, calibrated)
                ),
                "full_fit_ece": expected_calibration_error(
                    labels, calibrated
                ),
                "platt_slope": (
                    float(model.coef_[0, 0])
                    if model is not None
                    else math.nan
                ),
                "platt_intercept": (
                    float(model.intercept_[0])
                    if model is not None
                    else math.nan
                ),
            }
        )
    write_frame_new(pd.DataFrame(analysis), args.analysis_csv)
    result = {
        "status": "PASS",
        "calibration_rows": len(frame),
        "calibration_positive": int(labels.sum()),
        "calibration_negative": int((labels == 0).sum()),
        "threshold_definition": (
            "raw score of calibration rank K; deterministic "
            "sequence_id tie-break retained for exact Top-K"
        ),
        "deployment_strategies": [
            "exact_TopK",
            "calibration_quantile_score_threshold",
            "calibration_threshold_then_TopK_cap",
        ],
        "models": policy,
        "test_access": False,
    }
    write_json_new(args.threshold_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
