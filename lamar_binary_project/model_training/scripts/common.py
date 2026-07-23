#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
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
CORE_COLUMNS = [
    "chrom", "position", "genomic_key", "gene_id", "gene_name", "transcript_ids",
    "sequence_context", "label", "split", "leakage_group", "gc_fraction",
    "sequence_entropy_log2_single_base_101nt", "corrected_editing_efficiency",
    "median_depth", "gene_level_coverage_mean_site_median_depth", "negative_difficulty",
]


def load_yaml(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


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


def normalized_record(row: Mapping[str, object]) -> dict:
    sequence = str(row["sequence_context"])
    depths = [float(row.get(f"{sample}_usable_depth", 0) or 0) for sample in SAMPLES]
    median_depth = row.get("median_depth", "")
    if median_depth in (None, "", "nan"):
        median_depth = float(np.median(depths))
    gene_cov = row.get("gene_level_coverage_mean_site_median_depth", "")
    if gene_cov in (None, "", "nan"):
        gene_cov = float(median_depth)
    difficulty = str(row.get("negative_difficulty", "positive" if int(row["label"]) else "random_strict"))
    return {
        "chrom": str(row["chrom"]),
        "position": int(row["position"]),
        "genomic_key": str(row["genomic_key"]),
        "gene_id": str(row.get("gene_id", "NA")),
        "gene_name": str(row.get("gene_name", "NA")),
        "transcript_ids": str(row.get("transcript_ids", "NA")),
        "sequence_context": sequence,
        "sequence_hash": sequence_hash(sequence),
        "label": int(row["label"]),
        "split": str(row.get("split", "train")),
        "leakage_group": str(row.get("leakage_group", "NA")),
        "gc_fraction": float(row.get("gc_fraction", (sequence.count("G") + sequence.count("C")) / len(sequence))),
        "c_count": sequence.count("C"),
        "entropy": float(row.get("sequence_entropy_log2_single_base_101nt", shannon_entropy(sequence))),
        "median_depth": float(median_depth),
        "gene_coverage": float(gene_cov),
        "negative_type": difficulty,
        "true_efficiency": float(row.get("corrected_editing_efficiency", 0) or 0),
    }


def read_tsv_records(path: str | Path) -> list[dict]:
    path = Path(path)
    with gzip.open(path, "rt", newline="") as handle:
        return [normalized_record(row) for row in csv.DictReader(handle, delimiter="\t")]


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


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class NegativePool:
    def __init__(self, sqlite_path: str | Path, seed: int):
        self.path = str(sqlite_path)
        self.seed = int(seed)
        with sqlite3.connect(self.path) as connection:
            self.total = int(connection.execute("select count(*) from negatives").fetchone()[0])
            self.matched_ids = np.fromiter((r[0] for r in connection.execute("select id from negatives where matched=1")), dtype=np.int64)
            self.hard_ids = np.fromiter((r[0] for r in connection.execute("select id from negatives where hard=1")), dtype=np.int64)
            self.easy_ids = np.fromiter((r[0] for r in connection.execute("select id from negatives where hard=0 and matched=0")), dtype=np.int64)
        self.all_ids = np.arange(1, self.total + 1, dtype=np.int64)
        self.dynamic_order = np.random.default_rng(self.seed + 88001).permutation(self.all_ids)

    def _draw(self, source: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        if count > len(source):
            raise ValueError(f"Requested {count} unique negatives from pool of {len(source)}")
        return rng.choice(source, size=count, replace=False)

    def ids_for_epoch(self, count: int, strategy: str, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 1009 * (epoch + 1))
        if strategy == "dynamic_full":
            start = (epoch * count) % len(self.dynamic_order)
            if start + count <= len(self.dynamic_order):
                return self.dynamic_order[start:start + count]
            return np.concatenate([self.dynamic_order[start:], self.dynamic_order[:count - (len(self.dynamic_order) - start)]])
        if strategy == "random":
            return self._draw(self.all_ids, count, rng)
        if strategy == "matched":
            return self._draw(self.matched_ids, count, rng)
        if strategy == "hard":
            return self._draw(self.hard_ids, count, rng)
        if strategy == "mixed":
            counts = (int(round(count * 0.2)), int(round(count * 0.4)))
            easy_n, matched_n = counts
            hard_n = count - easy_n - matched_n
            selected = np.concatenate([
                self._draw(self.easy_ids, easy_n, rng),
                self._draw(self.matched_ids, matched_n, rng),
                self._draw(self.hard_ids, hard_n, rng),
            ])
            selected = np.unique(selected)
            if len(selected) < count:
                remaining = np.setdiff1d(self.all_ids, selected, assume_unique=False)
                selected = np.concatenate([selected, self._draw(remaining, count - len(selected), rng)])
            rng.shuffle(selected)
            return selected
        raise ValueError(strategy)

    def fetch(self, ids: Sequence[int]) -> list[dict]:
        columns = "id,chrom,position,genomic_key,gene_id,gene_name,transcript_ids,seq,sequence_hash,leakage_group,gc,entropy,c_count,median_depth,gene_coverage,negative_type,matched,hard"
        rows = []
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            for start in range(0, len(ids), 800):
                chunk = [int(value) for value in ids[start:start + 800]]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(connection.execute(f"select {columns} from negatives where id in ({placeholders})", chunk).fetchall())
        output = []
        for row in rows:
            value = dict(row)
            value.update({"sequence_context": value.pop("seq"), "gc_fraction": value.pop("gc"), "label": 0, "split": "train", "true_efficiency": 0.0})
            output.append(value)
        return output


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
