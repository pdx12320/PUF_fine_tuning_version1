#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve, roc_curve

from ablation_common import (
    embedding_metadata, load_config, sha256_file, write_json,
)
from plotting import heatmap, line_plot, scatter_plot


DISPLAY_NAMES = {
    "hash_random": "Random baseline",
    "kmer_logistic": "k-mer Logistic",
    "cnn": "CNN",
    "lamar_zero_shot_center": "Lamar zero-shot center",
    "lamar_zero_shot_mean": "Lamar zero-shot mean",
    "lamar_zero_shot_masked_mean": "Lamar zero-shot masked mean",
    "lamar_zero_shot_cls": "Lamar zero-shot CLS",
    "lamar_linear_probe_center": "Lamar linear probe center",
    "lamar_linear_probe_mean": "Lamar linear probe mean",
    "lamar_linear_probe_masked_mean": "Lamar linear probe masked mean",
    "lamar_linear_probe_cls": "Lamar linear probe CLS",
    "frozen_lamar_head": "Frozen Lamar head",
    "partial_lamar_2blocks": "Partial Lamar (2 blocks)",
    "full_lamar": "Full fine-tuning",
    "lora_best": "LoRA Lamar",
}


def concatenate_parquets(paths, output):
    writer = None
    total = 0
    for path in paths:
        source = pq.ParquetFile(path)
        for index in range(source.num_row_groups):
            table = source.read_row_group(index)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            total += len(table)
    if writer is not None:
        writer.close()
    return total


def markdown_table(frame):
    headers = [
        "Model", "Parameters trained", "Dev AP", "Test AP",
        "Test Precision", "Test Recall", "FP/Million",
    ]
    rows = []
    for _, row in frame.iterrows():
        rows.append([
            str(row.display_name), f"{int(row.trainable_parameters):,}",
            f"{row.dev_AP:.6f}", f"{row.test_AP:.6f}",
            f"{row.test_precision:.6f}", f"{row.test_recall:.6f}",
            f"{row.FP_per_million:.3f}",
        ])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", "---:", "---:", "---:", "---:", "---:", "---:"]) + " |",
    ]
    lines.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(lines)


def selected_prediction_files(run, family, representations):
    dev_file = pd.read_parquet(run / f"predictions/{family}_dev_predictions.parquet")
    calibration = pd.read_parquet(run / "predictions/calibration_predictions.parquet")
    test = pd.read_parquet(run / "predictions/all_test_predictions.parquet")
    selection = json.loads((run / "results/dev_selection.json").read_text())
    frames = []
    for representation in representations:
        model_name = f"lamar_{family}_{representation}"
        selected = (
            selection["selected_zero_by_representation"][representation]["candidate"]
            if family == "zero_shot"
            else selection["selected_linear_by_representation"][representation]["candidate"]
        )
        frames.append(pd.DataFrame({
            "sequence_id": dev_file.sequence_id, "split": "dev", "label": dev_file.label,
            "model": model_name, "representation": representation,
            "raw_probability": dev_file[selected], "probability": dev_file[selected],
        }))
        frames.append(pd.DataFrame({
            "sequence_id": calibration.sequence_id, "split": "calibration", "label": calibration.label,
            "model": model_name, "representation": representation,
            "raw_probability": calibration[f"{model_name}_raw"],
            "probability": calibration[f"{model_name}_probability"],
        }))
        frames.append(pd.DataFrame({
            "sequence_id": test.sequence_id, "split": "test", "label": test.label,
            "model": model_name, "representation": representation,
            "raw_probability": test[f"{model_name}_raw"],
            "probability": test[f"{model_name}_probability"],
        }))
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = Path(cfg["run_dir"])
    for marker in ("DEV_SELECTION_COMPLETE", "CALIBRATION_COMPLETE", "TEST_EVALUATION_COMPLETE"):
        if not (run / marker).exists():
            raise RuntimeError(f"Missing {marker}")
    if (run / "SUCCESS").exists():
        raise FileExistsError("SUCCESS")

    embeddings = [
        run / "embeddings/train.parquet", run / "embeddings/dev.parquet",
        run / "embeddings/calibration.parquet", run / "embeddings/test.parquet",
    ]
    embedding_rows = concatenate_parquets(
        embeddings, run / "embeddings/lamar_embeddings.parquet"
    )
    train_manifest = json.loads((run / "embeddings/train.manifest.json").read_text())
    write_json(run / "embeddings/hidden_state_shape.json", {
        "hidden_state_shape": train_manifest["hidden_state_shape_per_batch"],
        "backbone_parameters": train_manifest["backbone_parameters"],
        "pretrained_parameters_including_MLM_head": train_manifest[
            "pretrained_model_parameters_including_MLM_head"
        ],
        "trainable_parameters": 0,
        "checkpoint": train_manifest["checkpoint_path"],
        "checkpoint_sha256": train_manifest["checkpoint_sha256"],
        "CLS_supported": True,
        "mean_masked_mean_equivalent_for_fixed_length_inputs": True,
    })

    representations = cfg["pooling_methods"]
    zero_predictions = selected_prediction_files(run, "zero_shot", representations)
    linear_predictions = selected_prediction_files(run, "linear_probe", representations)
    zero_predictions.to_parquet(run / "predictions/zero_shot_predictions.parquet", index=False)
    linear_predictions.to_parquet(run / "predictions/linear_probe_predictions.parquet", index=False)

    leaderboard = pd.read_csv(run / "results/comparison_leaderboard.csv")
    leaderboard["display_name"] = leaderboard.model.map(DISPLAY_NAMES).fillna(leaderboard.model)
    leaderboard.to_csv(run / "results/comparison_leaderboard.csv", index=False)
    test = pd.read_parquet(run / "predictions/all_test_predictions.parquet")
    y = test.label.to_numpy()
    core = [
        "kmer_logistic", "cnn", "lamar_zero_shot_center",
        "lamar_linear_probe_center", "frozen_lamar_head", "lora_best",
    ]
    pr_series, roc_series = [], []
    for model in core:
        probability = test[f"{model}_probability"].to_numpy()
        precision, recall, _ = precision_recall_curve(y, probability)
        fpr, tpr, _ = roc_curve(y, probability)
        pr_series.append((DISPLAY_NAMES[model], recall, precision))
        roc_series.append((DISPLAY_NAMES[model], fpr, tpr))
    line_plot(pr_series, run / "figures/PR_curves.png", "Locked 1:1000 test — precision-recall", "Recall", "Precision")
    line_plot(roc_series, run / "figures/ROC_curves.png", "Locked 1:1000 test — ROC", "False-positive rate", "True-positive rate", diagonal=True)

    calibration_points = pd.read_csv(run / "results/calibration_curve_points.csv")
    calibration_series = []
    for model in core:
        subset = calibration_points[calibration_points.model == model]
        calibration_series.append((
            DISPLAY_NAMES[model], subset.mean_predicted_probability, subset.observed_frequency
        ))
    line_plot(
        calibration_series, run / "figures/calibration_curve.png",
        "Calibration reliability — 0–1.2% zoom (5-fold OOF)",
        "Predicted probability", "Observed frequency",
        diagonal=True, x_max=0.012, y_max=0.012,
    )
    pca_frame = pd.read_parquet(run / "results/embedding_pca_coordinates.parquet")
    scatter_plot(pca_frame, run / "figures/embedding_PCA.png", "Pretrained Lamar center embedding PCA")
    correlation = pd.read_csv(run / "results/metadata_embedding_correlations.csv")
    heatmap(correlation, run / "figures/metadata_correlation.png", "PCA–metadata Pearson correlation")

    selection = json.loads((run / "results/dev_selection.json").read_text())
    zero_name = f"lamar_zero_shot_{selection['global_best_zero_shot']['representation']}"
    linear_name = f"lamar_linear_probe_{selection['global_best_linear_probe']['representation']}"
    by_name = leaderboard.set_index("model")
    bias = selection["shortcut_bias"]
    metadata_ap = bias["best_metadata_only"]["dev_average_precision"]
    combined_ap = bias["best_embedding_plus_metadata"]["dev_average_precision"]
    zero = by_name.loc[zero_name]
    linear = by_name.loc[linear_name]
    frozen = by_name.loc["frozen_lamar_head"]
    lora = by_name.loc["lora_best"]
    kmer = by_name.loc["kmer_logistic"]
    cnn = by_name.loc["cnn"]
    full = by_name.loc["full_lamar"]
    partial = by_name.loc["partial_lamar_2blocks"]
    ranking = leaderboard.sort_values("test_AP", ascending=False)[
        ["display_name", "trainable_parameters", "dev_AP", "test_AP", "test_precision", "test_recall", "FP_per_million"]
    ]
    table = markdown_table(ranking)
    max_corr = correlation.loc[correlation.pearson.abs().idxmax()]
    report = f"""# Pretrained Lamar zero-shot / linear-probe ablation

All labels are computational. The immutable dataset and existing checkpoints were not modified or retrained. All ablation configurations were frozen by dev, calibration and thresholds were frozen on the 1:1000 calibration split, and the test suite was scored only after `PRETEST_FROZEN.json`.

## Paper-level ablation table

{table}

## Required conclusions

1. **Does pretrained Lamar zero-shot exceed CNN/k-mer?** Best zero-shot is `{zero_name}`: dev AP `{zero.dev_AP:.6f}`, test AP `{zero.test_AP:.6f}`. K-mer test AP is `{kmer.test_AP:.6f}` and CNN test AP is `{cnn.test_AP:.6f}`.
2. **How much editing signal is in the embedding?** The task-unadapted centroid score reaches test AP `{zero.test_AP:.6f}` versus the empirical random baseline `{by_name.loc['hash_random'].test_AP:.6f}`. This is labeled zero-shot representation evaluation even though train labels estimate class centroids.
3. **Center versus mean pooling:** center zero-shot/linear test AP are `{by_name.loc['lamar_zero_shot_center'].test_AP:.6f}` / `{by_name.loc['lamar_linear_probe_center'].test_AP:.6f}`; mean values are `{by_name.loc['lamar_zero_shot_mean'].test_AP:.6f}` / `{by_name.loc['lamar_linear_probe_mean'].test_AP:.6f}`. Mean and masked mean are mathematically equivalent here because every sequence has 101 nucleotides and no padding.
4. **Linear probe gain:** best linear probe improves over best zero-shot by `{linear.test_AP-zero.test_AP:+.6f}` test AP.
5. **Frozen head versus linear probe:** frozen-head test AP `{frozen.test_AP:.6f}` versus linear-probe `{linear.test_AP:.6f}`.
6. **LoRA gain:** LoRA improves over linear probe by `{lora.test_AP-linear.test_AP:+.6f}` test AP.
7. **Representation adaptation versus classifier:** logistic probing tests a simple classifier on fixed embeddings; the remaining LoRA gain is consistent with representation adaptation, but this observational ablation cannot uniquely attribute every gain.
8. **Embedding versus metadata-only:** best embedding-only linear dev AP `{selection['global_best_linear_probe']['dev_average_precision']:.6f}`; metadata-only dev AP `{metadata_ap:.6f}`; combined dev AP `{combined_ap:.6f}`.
9. **Shortcut risk:** the largest absolute PC–metadata Pearson correlation is `{max_corr.pearson:+.3f}` (`{max_corr.component}` with `{max_corr.feature}`). Metadata remains strongly predictive, so coverage/expression-related data-generation bias cannot be excluded even though Lamar receives sequence only.
10. **Complete ordering:** see the frozen test table above. Partial 2-block AP is `{partial.test_AP:.6f}` and full fine-tuning AP is `{full.test_AP:.6f}`; neither model was retrained.

## Final zero-shot and probe configurations

- zero-shot: `{selection['global_best_zero_shot']}`
- linear probe: `{selection['global_best_linear_probe']}`
- pretrained checkpoint: `{cfg['pretrained_checkpoint']}`
- pretrained backbone parameters: `{train_manifest['backbone_parameters']}`
- backbone trainable parameters during embedding extraction: `0`

## Limitations

- “Zero-shot” here means no gradient-based Lamar adaptation; labeled train centroids are still estimated.
- Train negatives are a deterministic 1:10, without-replacement sample from the train-only dynamic pool, matching the selected LoRA ratio.
- External basewise mappability was unavailable in the underlying dataset.
- Mean and masked-mean rows are retained for protocol completeness but are identical for these fixed-length, unpadded inputs.
"""
    (run / "reports/final_zero_shot_report.md").write_text(report)
    manifest = {
        "status": "complete", "embedding_rows": embedding_rows,
        "dataset_dir": cfg["dataset_dir"], "existing_model_dir": cfg["existing_model_dir"],
        "pretrained_checkpoint": cfg["pretrained_checkpoint"],
        "pretrained_checkpoint_sha256": train_manifest["checkpoint_sha256"],
        "trainable_backbone_parameters": 0,
        "train_sampling": train_manifest["source"],
        "dev_selection": selection,
        "test_protocol": cfg["test_protocol"],
        "test_rows": len(test), "test_positive": int(y.sum()), "test_negative": int((y == 0).sum()),
    }
    write_json(run / "results/ablation_manifest.json", manifest)
    checksums = []
    for path in sorted(run.rglob("*")):
        if path.is_file() and path.name not in {"checksums.sha256", "SUCCESS"} and "logs" not in path.parts:
            checksums.append(f"{sha256_file(path)}  {path.relative_to(run)}")
    (run / "checksums.sha256").write_text("\n".join(checksums) + "\n")
    (run / "SUCCESS").write_text("PASS\n")
    parent = run.parent
    (parent / "LATEST_SUCCESSFUL_RUN.txt").write_text(str(run) + "\n")
    print(report)


if __name__ == "__main__":
    main()
