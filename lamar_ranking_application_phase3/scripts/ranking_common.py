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
    ndcg_score,
    precision_recall_curve,
    roc_auc_score,
)


BUDGETS = (10, 50, 100, 500, 1000)
SAMPLES = (
    "CU517_GC_T1",
    "CU517_GC_T2",
    "CU517_GC_T3",
    "CU517_GC_C1",
    "CU517_GC_C2",
    "CU517_GC_C3",
)
META_COLUMNS = [
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


def sequence_id(split: str, genomic_key: str, seq_hash: str) -> str:
    text = f"{split}|{genomic_key}|{seq_hash}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shannon_entropy(sequence: str) -> float:
    counts = Counter(sequence)
    return -sum(
        (count / len(sequence)) * math.log2(count / len(sequence))
        for count in counts.values()
    )


def normalized_record(row: Mapping[str, object], default_split: str) -> dict:
    sequence = str(row["sequence_context"])
    if len(sequence) != 101 or sequence[50] != "C":
        raise ValueError(
            f"Invalid sequence for {row.get('genomic_key')}: "
            f"length={len(sequence)} center={sequence[50:51]}"
        )
    split = str(row.get("split") or default_split)
    label = int(row["label"])
    seq_hash = str(row.get("sequence_hash") or sequence_hash(sequence))
    depths = [float(row.get(f"{sample}_usable_depth", 0) or 0) for sample in SAMPLES]
    median_depth = row.get("median_depth", "")
    if median_depth in (None, "", "nan"):
        median_depth = float(np.median(depths))
    gene_cov = row.get("gene_level_coverage_mean_site_median_depth", "")
    if gene_cov in (None, "", "nan"):
        gene_cov = float(median_depth)
    genomic_key = str(row["genomic_key"])
    negative_type = str(
        row.get("negative_type")
        or row.get("negative_difficulty")
        or ("positive" if label else "random_strict")
    )
    return {
        "sequence_id": sequence_id(split, genomic_key, seq_hash),
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
        "negative_type": negative_type,
        "true_efficiency": float(
            row.get("true_efficiency")
            or row.get("corrected_editing_efficiency")
            or 0
        ),
        "gc_fraction": float(
            row.get(
                "gc_fraction",
                (sequence.count("G") + sequence.count("C")) / len(sequence),
            )
        ),
        "c_count": int(row.get("c_count", sequence.count("C"))),
        "entropy": float(
            row.get(
                "entropy",
                row.get(
                    "sequence_entropy_log2_single_base_101nt",
                    shannon_entropy(sequence),
                ),
            )
        ),
        "median_depth": float(median_depth),
        "gene_coverage": float(gene_cov),
    }


def read_tsv_records(path: str | Path, default_split: str) -> list[dict]:
    with gzip.open(path, "rt", newline="") as handle:
        return [
            normalized_record(row, default_split)
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def write_json_new(path: str | Path, value: object) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_frame_new(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".parquet":
        frame.to_parquet(target, index=False)
    elif target.suffix == ".csv":
        frame.to_csv(target, index=False)
    elif target.suffix == ".tsv":
        frame.to_csv(target, sep="\t", index=False)
    else:
        raise ValueError(target)


def stable_order(
    scores: Sequence[float], tie_breaker: Sequence[str]
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    ties = np.asarray(tie_breaker, dtype=str)
    if len(values) != len(ties):
        raise ValueError((len(values), len(ties)))
    if not np.isfinite(values).all():
        raise ValueError("Non-finite ranking score")
    return np.lexsort((ties, -values))


def ranking_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    tie_breaker: Sequence[str],
    budgets: Sequence[int] = BUDGETS,
) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) == 0 or y.sum() == 0 or (y == 0).sum() == 0:
        raise ValueError("Ranking metrics require both classes")
    order = stable_order(s, tie_breaker)
    positive_total = int(y.sum())
    background = float(y.mean())
    precision, recall, _ = precision_recall_curve(y, s)
    result = {
        "rows": int(len(y)),
        "positives": positive_total,
        "negatives": int((y == 0).sum()),
        "background_positive_rate": background,
        "average_precision": float(average_precision_score(y, s)),
        "pr_auc": float(auc(recall, precision)),
        "roc_auc": float(roc_auc_score(y, s)),
        "ndcg": float(ndcg_score(y.reshape(1, -1), s.reshape(1, -1))),
    }
    precision_values = []
    for requested in budgets:
        k = min(int(requested), len(y))
        discovered = int(y[order[:k]].sum())
        p_at_k = discovered / k
        result[f"discovered_at_{requested}"] = discovered
        result[f"precision_at_{requested}"] = float(p_at_k)
        result[f"recall_at_{requested}"] = float(discovered / positive_total)
        result[f"enrichment_at_{requested}"] = float(p_at_k / background)
        precision_values.append(p_at_k)
    result["selection_mean_precision_at_k"] = float(np.mean(precision_values))
    result["selection_rule"] = (
        "maximize unweighted mean Precision@{10,50,100,500,1000}; "
        "tie-break by Precision@100, Precision@500, then AP"
    )
    return result


def ranking_key(metrics: Mapping[str, object]) -> tuple[float, float, float, float]:
    return (
        float(metrics["selection_mean_precision_at_k"]),
        float(metrics["precision_at_100"]),
        float(metrics["precision_at_500"]),
        float(metrics["average_precision"]),
    )


def deterministic_random_scores(ids: Iterable[str], seed: int) -> np.ndarray:
    values = []
    for value in ids:
        digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "big") / 2**64)
    return np.asarray(values, dtype=np.float64)


class NegativePool:
    """Read-only access to the immutable train-only binary negative index."""

    COLUMNS = (
        "id,chrom,position,genomic_key,gene_id,gene_name,transcript_ids,"
        "seq,sequence_hash,leakage_group,gc,entropy,c_count,median_depth,"
        "gene_coverage,negative_type,matched,hard"
    )

    def __init__(
        self,
        sqlite_path: str | Path,
        seed: int,
        guided_paths: Sequence[str | Path] = (),
    ):
        self.path = str(Path(sqlite_path).resolve())
        self.seed = int(seed)
        with self._connect() as connection:
            self.total = int(
                connection.execute("select count(*) from negatives").fetchone()[0]
            )
            self.matched_ids = np.fromiter(
                (
                    row[0]
                    for row in connection.execute(
                        "select id from negatives where matched=1"
                    )
                ),
                dtype=np.int64,
            )
            self.hard_ids = np.fromiter(
                (
                    row[0]
                    for row in connection.execute(
                        "select id from negatives where hard=1"
                    )
                ),
                dtype=np.int64,
            )
            self.easy_ids = np.fromiter(
                (
                    row[0]
                    for row in connection.execute(
                        "select id from negatives where hard=0 and matched=0"
                    )
                ),
                dtype=np.int64,
            )
        self.all_ids = np.arange(1, self.total + 1, dtype=np.int64)
        self.guided_ids = self._load_guided(guided_paths)

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.path}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    @staticmethod
    def _load_guided(paths: Sequence[str | Path]) -> np.ndarray:
        arrays = []
        for path in paths:
            source = Path(path)
            if not source.is_file():
                raise FileNotFoundError(source)
            frame = pd.read_parquet(source, columns=["id"])
            arrays.append(frame["id"].to_numpy(dtype=np.int64))
        if not arrays:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(arrays))

    @staticmethod
    def _draw(
        source: np.ndarray, count: int, rng: np.random.Generator
    ) -> np.ndarray:
        if len(source) == 0:
            raise ValueError("Requested draw from empty pool")
        return rng.choice(source, size=count, replace=count > len(source))

    def ids_for_epoch(
        self, count: int, strategy: str, epoch: int
    ) -> tuple[np.ndarray, dict]:
        rng = np.random.default_rng(self.seed + 1009 * (epoch + 1))
        if strategy == "random":
            selected = self._draw(self.all_ids, count, rng)
            planned = {"random": count}
        elif strategy == "matched":
            selected = self._draw(self.matched_ids, count, rng)
            planned = {"matched": count}
        elif strategy == "hard":
            selected = self._draw(self.hard_ids, count, rng)
            planned = {"hard": count}
        elif strategy == "guided":
            selected = self._draw(self.guided_ids, count, rng)
            planned = {"binary_or_iterative_guided": count}
        elif strategy == "mixed":
            easy_count = int(round(count * 0.20))
            matched_count = int(round(count * 0.40))
            hard_count = count - easy_count - matched_count
            selected = np.concatenate(
                [
                    self._draw(self.easy_ids, easy_count, rng),
                    self._draw(self.matched_ids, matched_count, rng),
                    self._draw(self.hard_ids, hard_count, rng),
                ]
            )
            planned = {
                "easy": easy_count,
                "matched": matched_count,
                "hard": hard_count,
            }
        else:
            raise ValueError(strategy)
        rng.shuffle(selected)
        if len(selected) != count:
            raise AssertionError((len(selected), count))
        return selected.astype(np.int64, copy=False), {
            "strategy": strategy,
            "requested_pairs": int(count),
            "sampled_ids": int(len(selected)),
            "unique_ids": int(len(np.unique(selected))),
            "planned_sources": planned,
            "epoch": int(epoch + 1),
            "sampling_seed": int(self.seed + 1009 * (epoch + 1)),
        }

    def fetch(self, ids: Sequence[int]) -> list[dict]:
        rows = []
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            for start in range(0, len(ids), 800):
                chunk = [int(value) for value in ids[start : start + 800]]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    connection.execute(
                        f"select {self.COLUMNS} from negatives "
                        f"where id in ({placeholders})",
                        chunk,
                    ).fetchall()
                )
        by_id = {int(row["id"]): dict(row) for row in rows}
        output = []
        for identifier in ids:
            row = by_id[int(identifier)].copy()
            row.update(
                {
                    "sequence_context": row.pop("seq"),
                    "gc_fraction": row.pop("gc"),
                    "label": 0,
                    "split": "train",
                    "true_efficiency": 0.0,
                }
            )
            row["sequence_id"] = sequence_id(
                "train", row["genomic_key"], row["sequence_hash"]
            )
            output.append(row)
        if len(output) != len(ids):
            raise AssertionError((len(output), len(ids)))
        return output


def positive_indices(
    positive_count: int, pair_count: int, seed: int, epoch: int
) -> np.ndarray:
    if positive_count <= 0 or pair_count <= 0:
        raise ValueError((positive_count, pair_count))
    base = np.arange(pair_count, dtype=np.int64) % positive_count
    rng = np.random.default_rng(seed + 7919 * (epoch + 1))
    rng.shuffle(base)
    return base


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
