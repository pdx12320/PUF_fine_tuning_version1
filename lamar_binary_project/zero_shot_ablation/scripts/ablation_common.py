#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

SAMPLES = ("CU517_GC_T1", "CU517_GC_T2", "CU517_GC_T3", "CU517_GC_C1", "CU517_GC_C2", "CU517_GC_C3")
META_COLUMNS = [
    "sequence_id", "split", "label", "chrom", "position", "genomic_key", "gene_id",
    "gene_name", "transcript_ids", "sequence_context", "sequence_hash", "leakage_group",
    "negative_type", "true_efficiency", "gc_fraction", "c_count", "entropy",
    "median_depth", "gene_coverage",
]
REPRESENTATION_COLUMNS = {
    "center": "embedding_center",
    "mean": "embedding_mean",
    "masked_mean": "embedding_masked_mean",
    "cls": "embedding_cls",
}


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def shannon_entropy(sequence: str) -> float:
    counts = Counter(sequence)
    return -sum((n / len(sequence)) * math.log2(n / len(sequence)) for n in counts.values())


def normalized_record(row: dict, default_split: str) -> dict:
    sequence = str(row["sequence_context"])
    depths = [float(row.get(f"{sample}_usable_depth", 0) or 0) for sample in SAMPLES]
    median_depth = row.get("median_depth", "")
    if median_depth in (None, "", "nan"):
        median_depth = float(np.median(depths))
    gene_cov = row.get("gene_level_coverage_mean_site_median_depth", "")
    if gene_cov in (None, "", "nan"):
        gene_cov = float(median_depth)
    label = int(row["label"])
    difficulty = str(row.get("negative_difficulty", "positive" if label else "random_strict"))
    seq_hash = str(row.get("sequence_hash") or sequence_hash(sequence))
    split = str(row.get("split") or default_split)
    genomic_key = str(row["genomic_key"])
    return {
        "sequence_id": hashlib.sha256(f"{split}|{genomic_key}|{seq_hash}".encode()).hexdigest(),
        "split": split,
        "label": label,
        "chrom": str(row["chrom"]),
        "position": int(row["position"]),
        "genomic_key": genomic_key,
        "gene_id": str(row.get("gene_id", "NA")),
        "gene_name": str(row.get("gene_name", "NA")),
        "transcript_ids": str(row.get("transcript_ids", "NA")),
        "sequence_context": sequence,
        "sequence_hash": seq_hash,
        "leakage_group": str(row.get("leakage_group", "NA")),
        "negative_type": difficulty,
        "true_efficiency": float(row.get("corrected_editing_efficiency", row.get("true_efficiency", 0)) or 0),
        "gc_fraction": float(row.get("gc_fraction", (sequence.count("G") + sequence.count("C")) / len(sequence))),
        "c_count": int(row.get("c_count", sequence.count("C"))),
        "entropy": float(row.get("sequence_entropy_log2_single_base_101nt", row.get("entropy", shannon_entropy(sequence)))),
        "median_depth": float(median_depth),
        "gene_coverage": float(gene_cov),
    }


def read_fixed_split(path: str | Path, split: str) -> list[dict]:
    with gzip.open(path, "rt", newline="") as handle:
        return [normalized_record(row, split) for row in csv.DictReader(handle, delimiter="\t")]


def read_train_rows(cfg: dict) -> tuple[list[dict], dict]:
    data = Path(cfg["dataset_dir"])
    model_dir = Path(cfg["existing_model_dir"])
    positives = read_fixed_split(data / "train_positives.tsv.gz", "train")
    database = model_dir / "work/train_pool.sqlite"
    with sqlite3.connect(database) as connection:
        total = int(connection.execute("select count(*) from negatives").fetchone()[0])
    count = len(positives) * int(cfg["train_negative_ratio"])
    rng = np.random.default_rng(int(cfg["seed"]) + 1009)
    selected = np.sort(rng.choice(np.arange(1, total + 1, dtype=np.int64), size=count, replace=False))
    columns = "id,chrom,position,genomic_key,gene_id,gene_name,transcript_ids,seq,sequence_hash,leakage_group,gc,entropy,c_count,median_depth,gene_coverage,negative_type"
    raw = []
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        for start in range(0, len(selected), 800):
            chunk = [int(x) for x in selected[start:start + 800]]
            placeholders = ",".join("?" for _ in chunk)
            raw.extend(connection.execute(f"select {columns} from negatives where id in ({placeholders}) order by id", chunk).fetchall())
    negatives = []
    for row in raw:
        value = dict(row)
        value.update({
            "sequence_context": value.pop("seq"), "gc_fraction": value.pop("gc"),
            "label": 0, "split": "train", "true_efficiency": 0.0,
        })
        negatives.append(normalized_record(value, "train"))
    if len(negatives) != count:
        raise AssertionError((len(negatives), count))
    manifest = {
        "positive_count": len(positives), "negative_count": len(negatives),
        "negative_ratio": int(cfg["train_negative_ratio"]), "negative_pool_total": total,
        "sampling": "numpy Generator.choice without replacement from train-only SQLite IDs",
        "sampling_seed": int(cfg["seed"]) + 1009,
        "selected_id_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
    }
    return positives + negatives, manifest


def load_split_rows(cfg: dict, split: str) -> tuple[list[dict], dict]:
    data = Path(cfg["dataset_dir"])
    if split == "train":
        return read_train_rows(cfg)
    filename = {
        "dev": "dev_1to10.tsv.gz",
        "calibration": "calibration_1to1000.tsv.gz",
        "test": "test_1to1000.tsv.gz",
    }[split]
    rows = read_fixed_split(data / filename, split)
    return rows, {"source": str(data / filename), "rows": len(rows)}


def embedding_metadata(path: str | Path) -> pd.DataFrame:
    return pq.read_table(path, columns=META_COLUMNS).to_pandas()


def embedding_matrix(path: str | Path, representation: str) -> np.ndarray:
    column = REPRESENTATION_COLUMNS[representation]
    array = pq.read_table(path, columns=[column]).column(0).combine_chunks()
    values = array.values.to_numpy(zero_copy_only=False)
    return values.reshape(len(array), array.type.list_size).astype(np.float32, copy=False)


def expected_calibration_error(y_true: Sequence[int], probability: Sequence[float], bins: int = 15) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p < right if right < 1 else p <= right)
        if mask.any():
            result += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(result)


def binary_metrics(y_true: Sequence[int], probability: Sequence[float], threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    prediction = p >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    pr_precision, pr_recall, _ = precision_recall_curve(y, p)
    return {
        "pr_auc": float(auc(pr_recall, pr_precision)),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "brier": float(brier_score_loss(y, p)), "ece": expected_calibration_error(y, p),
        "threshold": float(threshold), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fp_per_million": float(fp / max(1, (y == 0).sum()) * 1_000_000),
    }


def fit_calibrator(method: str, raw_probability: np.ndarray, labels: np.ndarray):
    p = np.clip(np.asarray(raw_probability), 1e-7, 1 - 1e-7)
    if method == "none":
        return None
    if method == "platt":
        x = np.log(p / (1 - p)).reshape(-1, 1)
        return LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, labels)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(p, labels)
    raise ValueError(method)


def apply_calibrator(method: str, model, raw_probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(raw_probability), 1e-7, 1 - 1e-7)
    if method == "none":
        return p
    if method == "platt":
        return model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
    return model.predict(p)


def fp_budget_threshold(labels: Sequence[int], probability: Sequence[float], target: int) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    negative = np.sort(p[y == 0])[::-1]
    allowed = int(math.floor(target * len(negative) / 1_000_000))
    if allowed == 0:
        threshold = float(np.nextafter(negative[0], np.inf))
    elif allowed < len(negative):
        threshold = float(np.nextafter(negative[allowed], np.inf))
    else:
        threshold = 0.0
    result = binary_metrics(y, p, threshold)
    result.update({"target_fp_per_million": int(target), "allowed_fp": allowed})
    return result


def zero_shot_fit(matrix: np.ndarray, labels: np.ndarray, method: str) -> dict:
    positive = matrix[labels == 1]
    negative = matrix[labels == 0]
    model = {
        "method": method,
        "positive_centroid": positive.mean(0).astype(np.float32),
        "negative_centroid": negative.mean(0).astype(np.float32),
    }
    if method == "diagonal_mahalanobis":
        model["pooled_variance"] = (matrix.var(0) + 1e-4).astype(np.float32)
    train_score = zero_shot_score(model, matrix)
    model["score_mean"] = float(train_score.mean())
    model["score_sd"] = float(max(train_score.std(), 1e-8))
    return model


def zero_shot_score(model: dict, matrix: np.ndarray) -> np.ndarray:
    positive = model["positive_centroid"]
    negative = model["negative_centroid"]
    method = model["method"]
    if method == "cosine_centroid":
        x_norm = np.linalg.norm(matrix, axis=1).clip(1e-8)
        p = matrix @ positive / (x_norm * max(np.linalg.norm(positive), 1e-8))
        n = matrix @ negative / (x_norm * max(np.linalg.norm(negative), 1e-8))
        return p - n
    if method == "euclidean_centroid":
        return np.sum((matrix - negative) ** 2, axis=1) - np.sum((matrix - positive) ** 2, axis=1)
    variance = model["pooled_variance"]
    return np.sum((matrix - negative) ** 2 / variance, axis=1) - np.sum((matrix - positive) ** 2 / variance, axis=1)


def zero_shot_probability(model: dict, matrix: np.ndarray) -> np.ndarray:
    score = zero_shot_score(model, matrix)
    return expit((score - model["score_mean"]) / model["score_sd"])


def hash_random_probability(sequence_hashes: Iterable[str]) -> np.ndarray:
    scale = float(2**64)
    return np.asarray([int(str(value)[:16], 16) / scale for value in sequence_hashes], dtype=np.float64)
