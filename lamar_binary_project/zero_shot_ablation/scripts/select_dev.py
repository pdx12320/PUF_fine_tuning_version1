#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ablation_common import (
    REPRESENTATION_COLUMNS,
    binary_metrics,
    embedding_matrix,
    embedding_metadata,
    hash_random_probability,
    load_config,
    write_json,
    zero_shot_fit,
    zero_shot_probability,
)


def best_row(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(["dev_average_precision", "dev_pr_auc"], ascending=False).iloc[0]


def logistic_fit(train_x, train_y, dev_x, dev_y, c_value, class_weight):
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(
        C=float(c_value), class_weight=class_weight, solver="liblinear",
        max_iter=3000, random_state=42,
    ).fit(scaler.transform(train_x), train_y)
    probability = model.predict_proba(scaler.transform(dev_x))[:, 1]
    return {"scaler": scaler, "model": model}, probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = Path(cfg["run_dir"])
    if (run / "DEV_SELECTION_COMPLETE").exists():
        raise FileExistsError("DEV_SELECTION_COMPLETE")
    train_path = run / "embeddings/train.parquet"
    dev_path = run / "embeddings/dev.parquet"
    train_meta = embedding_metadata(train_path)
    dev_meta = embedding_metadata(dev_path)
    train_y = train_meta.label.to_numpy(dtype=np.int64)
    dev_y = dev_meta.label.to_numpy(dtype=np.int64)
    if train_meta.label.value_counts().to_dict() != {0: 10280, 1: 1028}:
        raise AssertionError(train_meta.label.value_counts().to_dict())
    if dev_meta.label.value_counts().to_dict() != {0: 1590, 1: 159}:
        raise AssertionError(dev_meta.label.value_counts().to_dict())
    models_dir = run / "models"
    zero_rows, linear_rows = [], []
    zero_dev = dev_meta.copy()
    linear_dev = dev_meta.copy()
    matrices = {}

    for representation in cfg["pooling_methods"]:
        train_x = embedding_matrix(train_path, representation)
        dev_x = embedding_matrix(dev_path, representation)
        matrices[representation] = (train_x, dev_x)
        for method in cfg["zero_shot_methods"]:
            fitted = zero_shot_fit(train_x, train_y, method)
            probability = zero_shot_probability(fitted, dev_x)
            metrics = binary_metrics(dev_y, probability)
            candidate = f"zero_{representation}_{method}"
            joblib.dump(fitted, models_dir / f"{candidate}.joblib")
            zero_dev[candidate] = probability
            zero_rows.append({
                "candidate": candidate, "representation": representation, "method": method,
                "trainable_parameters": 0, **{f"dev_{key}": value for key, value in metrics.items()},
            })
        for c_value in cfg["linear_probe_C"]:
            for class_weight in cfg["linear_probe_class_weight"]:
                fitted, probability = logistic_fit(train_x, train_y, dev_x, dev_y, c_value, class_weight)
                metrics = binary_metrics(dev_y, probability)
                weight_name = "none" if class_weight is None else str(class_weight)
                candidate = f"linear_{representation}_C{c_value:g}_weight_{weight_name}"
                joblib.dump(fitted, models_dir / "linear_probe" / f"{candidate}.joblib")
                linear_dev[candidate] = probability
                linear_rows.append({
                    "candidate": candidate, "representation": representation, "C": float(c_value),
                    "class_weight": weight_name, "trainable_parameters": int(train_x.shape[1] + 1),
                    **{f"dev_{key}": value for key, value in metrics.items()},
                })
    zero_frame = pd.DataFrame(zero_rows)
    linear_frame = pd.DataFrame(linear_rows)
    zero_frame.to_csv(run / "results/zero_shot_dev_grid.csv", index=False)
    linear_frame.to_csv(run / "results/linear_probe_dev_grid.csv", index=False)
    zero_dev.to_parquet(run / "predictions/zero_shot_dev_predictions.parquet", index=False)
    linear_dev.to_parquet(run / "predictions/linear_probe_dev_predictions.parquet", index=False)

    selected_zero = {}
    selected_linear = {}
    for representation in cfg["pooling_methods"]:
        selected_zero[representation] = best_row(
            zero_frame[zero_frame.representation == representation]
        ).to_dict()
        selected_linear[representation] = best_row(
            linear_frame[linear_frame.representation == representation]
        ).to_dict()
    global_zero = best_row(zero_frame).to_dict()
    global_linear = best_row(linear_frame).to_dict()

    random_probability = hash_random_probability(dev_meta.sequence_hash)
    random_metrics = binary_metrics(dev_y, random_probability)
    pd.DataFrame([{
        "candidate": "hash_random", "representation": "none", "method": "predefined_sha256_uniform",
        "trainable_parameters": 0, **{f"dev_{key}": value for key, value in random_metrics.items()},
    }]).to_csv(run / "results/random_dev_metrics.csv", index=False)

    metadata_features = ["median_depth", "gene_coverage", "gc_fraction", "c_count", "entropy"]
    metadata_train = train_meta[metadata_features].to_numpy(dtype=np.float64)
    metadata_dev = dev_meta[metadata_features].to_numpy(dtype=np.float64)
    bias_rows = []
    best_metadata = None
    for c_value in cfg["linear_probe_C"]:
        for class_weight in cfg["linear_probe_class_weight"]:
            fitted, probability = logistic_fit(
                metadata_train, train_y, metadata_dev, dev_y, c_value, class_weight
            )
            metrics = binary_metrics(dev_y, probability)
            row = {
                "model": "metadata_only", "C": float(c_value),
                "class_weight": "none" if class_weight is None else class_weight,
                **{f"dev_{key}": value for key, value in metrics.items()},
            }
            bias_rows.append(row)
            if best_metadata is None or metrics["average_precision"] > best_metadata[0]:
                best_metadata = (metrics["average_precision"], fitted, row)
    joblib.dump(best_metadata[1], models_dir / "metadata_only_logistic.joblib")

    best_representation = str(global_linear["representation"])
    train_embedding, dev_embedding = matrices[best_representation]
    combined_train = np.hstack([train_embedding, metadata_train])
    combined_dev = np.hstack([dev_embedding, metadata_dev])
    best_combined = None
    for c_value in cfg["linear_probe_C"]:
        for class_weight in cfg["linear_probe_class_weight"]:
            fitted, probability = logistic_fit(
                combined_train, train_y, combined_dev, dev_y, c_value, class_weight
            )
            metrics = binary_metrics(dev_y, probability)
            row = {
                "model": "embedding_plus_metadata", "representation": best_representation,
                "C": float(c_value), "class_weight": "none" if class_weight is None else class_weight,
                **{f"dev_{key}": value for key, value in metrics.items()},
            }
            bias_rows.append(row)
            if best_combined is None or metrics["average_precision"] > best_combined[0]:
                best_combined = (metrics["average_precision"], fitted, row)
    joblib.dump(best_combined[1], models_dir / "embedding_plus_metadata_logistic.joblib")
    pd.DataFrame(bias_rows).to_csv(run / "results/shortcut_bias_dev_models.csv", index=False)

    center_train, center_dev = matrices["center"]
    pca = PCA(n_components=20, svd_solver="randomized", random_state=int(cfg["seed"]))
    pca.fit(center_train)
    train_coordinates = pca.transform(center_train)
    dev_coordinates = pca.transform(center_dev)
    joblib.dump(pca, models_dir / "embedding_center_pca.joblib")
    coordinate_frame = pd.concat([
        train_meta[["sequence_id", "split", "label"]].assign(
            PC1=train_coordinates[:, 0], PC2=train_coordinates[:, 1]
        ),
        dev_meta[["sequence_id", "split", "label"]].assign(
            PC1=dev_coordinates[:, 0], PC2=dev_coordinates[:, 1]
        ),
    ], ignore_index=True)
    coordinate_frame.to_parquet(run / "results/embedding_pca_coordinates.parquet", index=False)
    dev_pcs = pd.DataFrame(dev_coordinates, columns=[f"PC{i+1}" for i in range(20)])
    correlation_rows = []
    for pc in dev_pcs.columns:
        for feature in metadata_features:
            correlation_rows.append({
                "component": pc, "feature": feature,
                "pearson": float(dev_pcs[pc].corr(dev_meta[feature], method="pearson")),
                "spearman": float(dev_pcs[pc].corr(dev_meta[feature], method="spearman")),
            })
    pd.DataFrame(correlation_rows).to_csv(
        run / "results/metadata_embedding_correlations.csv", index=False
    )
    selection = {
        "status": "PASS",
        "selection_metric": "dev_average_precision",
        "selected_zero_by_representation": selected_zero,
        "selected_linear_by_representation": selected_linear,
        "global_best_zero_shot": global_zero,
        "global_best_linear_probe": global_linear,
        "random_dev_metrics": random_metrics,
        "shortcut_bias": {
            "metadata_features": metadata_features,
            "best_metadata_only": best_metadata[2],
            "best_embedding_plus_metadata": best_combined[2],
        },
        "PCA": {
            "fit_split": "train", "representation": "center",
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        },
        "mean_masked_mean_equivalence": bool(
            np.array_equal(matrices["mean"][0], matrices["masked_mean"][0])
            and np.array_equal(matrices["mean"][1], matrices["masked_mean"][1])
        ),
    }
    write_json(run / "results/dev_selection.json", selection)
    (run / "DEV_SELECTION_COMPLETE").write_text("PASS\n")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
