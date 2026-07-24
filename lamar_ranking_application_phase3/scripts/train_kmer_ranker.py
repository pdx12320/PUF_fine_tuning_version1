#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from ranking_common import (
    NegativePool,
    load_yaml,
    positive_indices,
    ranking_key,
    ranking_metrics,
    read_tsv_records,
    seed_everything,
    write_frame_new,
    write_json_new,
)


NUMERIC = ("gc_fraction", "c_count", "entropy")


def numeric_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[float(row[column]) for column in NUMERIC] for row in rows],
        dtype=np.float64,
    )


def feature_matrix(rows, vectorizer, scaler):
    kmers = vectorizer.transform(
        [row["sequence_context"] for row in rows]
    )
    numeric = csr_matrix(scaler.transform(numeric_matrix(rows)))
    return hstack([kmers, numeric], format="csr")


def score_rows(model, vectorizer, scaler, rows):
    return model.decision_function(
        feature_matrix(rows, vectorizer, scaler)
    ).astype(np.float64, copy=False)


def save_model(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    joblib.dump(value, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-epochs", type=int, default=-1)
    args = parser.parse_args()

    master = load_yaml(args.master)
    run_config = json.loads(Path(args.run_config).read_text())
    if run_config["model_type"] != "kmer":
        raise ValueError(run_config["model_type"])
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json_new(output / "config.json", run_config)

    seed = int(run_config["seed"])
    seed_everything(seed)
    dataset_dir = Path(master["dataset_dir"])
    positives_all = read_tsv_records(
        dataset_dir / "train_positives.tsv.gz", "train"
    )
    dev_rows = read_tsv_records(dataset_dir / "dev_1to10.tsv.gz", "dev")
    pool = NegativePool(
        master["negative_pool_sqlite"],
        seed=seed,
        guided_paths=run_config.get("guided_negative_paths", []),
    )
    pairs_per_epoch = int(run_config.get("pairs_per_epoch", 10280))
    bootstrap_ids, bootstrap_manifest = pool.ids_for_epoch(
        pairs_per_epoch,
        run_config["negative_sampling"],
        0,
    )
    bootstrap_negatives = pool.fetch(bootstrap_ids)
    bootstrap_rows = positives_all + bootstrap_negatives
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 6),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    )
    vectorizer.fit(
        [row["sequence_context"] for row in bootstrap_rows]
    )
    scaler = StandardScaler().fit(numeric_matrix(bootstrap_rows))
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(run_config.get("alpha", 1e-5)),
        fit_intercept=False,
        learning_rate="constant",
        eta0=float(run_config["learning_rate"]),
        random_state=seed,
        shuffle=True,
        average=True,
    )

    epochs = int(run_config.get("epochs", 20))
    if args.max_epochs > 0:
        epochs = min(epochs, args.max_epochs)
    history = []
    best_key = None
    best_checkpoint = None
    patience_count = 0
    started = time.time()
    for epoch in range(epochs):
        negative_ids, sampling_manifest = pool.ids_for_epoch(
            pairs_per_epoch,
            run_config["negative_sampling"],
            epoch,
        )
        negatives = pool.fetch(negative_ids)
        indices = positive_indices(
            len(positives_all), pairs_per_epoch, seed, epoch
        )
        positives = [positives_all[int(index)] for index in indices]
        positive_matrix = feature_matrix(
            positives, vectorizer, scaler
        )
        negative_matrix = feature_matrix(
            negatives, vectorizer, scaler
        )
        difference = positive_matrix - negative_matrix
        training_matrix = vstack(
            [difference, -difference], format="csr"
        )
        labels = np.concatenate(
            [
                np.ones(pairs_per_epoch, dtype=np.int64),
                np.zeros(pairs_per_epoch, dtype=np.int64),
            ]
        )
        if epoch == 0:
            model.partial_fit(
                training_matrix, labels, classes=np.array([0, 1])
            )
        else:
            model.partial_fit(training_matrix, labels)
        dev_score = score_rows(
            model, vectorizer, scaler, dev_rows
        )
        metrics = ranking_metrics(
            [row["label"] for row in dev_rows],
            dev_score,
            [row["sequence_id"] for row in dev_rows],
        )
        record = {
            "epoch": epoch + 1,
            "pair_count": pairs_per_epoch,
            "oriented_training_examples": pairs_per_epoch * 2,
            "sampling": sampling_manifest,
            "dev_metrics": metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        current_key = ranking_key(metrics)
        if best_key is None or current_key > best_key:
            best_key = current_key
            patience_count = 0
            best_checkpoint = output / f"checkpoint_epoch{epoch + 1:02d}.joblib"
            save_model(
                best_checkpoint,
                {
                    "vectorizer": vectorizer,
                    "scaler": scaler,
                    "model": model,
                    "numeric_features": list(NUMERIC),
                    "config": run_config,
                },
            )
        else:
            patience_count += 1
        if patience_count >= int(run_config.get("patience", 3)):
            break

    if best_checkpoint is None:
        raise RuntimeError("No checkpoint was created")
    saved = joblib.load(best_checkpoint)
    best_dev_score = score_rows(
        saved["model"],
        saved["vectorizer"],
        saved["scaler"],
        dev_rows,
    )
    best_metrics = ranking_metrics(
        [row["label"] for row in dev_rows],
        best_dev_score,
        [row["sequence_id"] for row in dev_rows],
    )
    predictions = pd.DataFrame(dev_rows)
    predictions["ranking_score"] = best_dev_score
    write_frame_new(
        predictions, output / "dev_fixed_predictions.parquet"
    )
    parameters = int(saved["model"].coef_.size)
    summary = {
        "status": "SUCCESS",
        "model_type": "kmer",
        "config": run_config,
        "best_checkpoint": str(best_checkpoint),
        "best_dev_fixed_metrics": best_metrics,
        "history": history,
        "epochs_completed": len(history),
        "pairs_per_epoch": pairs_per_epoch,
        "actual_pairs_all_epochs": pairs_per_epoch * len(history),
        "vectorizer_fit_rows": len(bootstrap_rows),
        "vectorizer_fit_sampling": bootstrap_manifest,
        "kmer_vocabulary_size": len(vectorizer.vocabulary_),
        "numeric_features": list(NUMERIC),
        "trainable_parameters": parameters,
        "total_parameters": parameters,
        "training_seconds": time.time() - started,
        "peak_gpu_bytes": 0,
    }
    write_json_new(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
