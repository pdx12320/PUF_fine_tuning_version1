#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ranking_common import ranking_metrics, write_frame_new, write_json_new


FEATURES = ["gc_fraction", "c_count", "entropy"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dev-predictions", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    for path in (args.model, args.dev_predictions, args.summary):
        if Path(path).exists():
            raise FileExistsError(path)
    train = pd.read_parquet(
        args.train,
        columns=["label", "sequence_id", *FEATURES],
    )
    dev = pd.read_parquet(args.dev)
    started = time.time()
    scaler = StandardScaler().fit(
        train[FEATURES].to_numpy(dtype=np.float64)
    )
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    ).fit(
        scaler.transform(
            train[FEATURES].to_numpy(dtype=np.float64)
        ),
        train["label"].to_numpy(dtype=np.int64),
    )
    score = model.decision_function(
        scaler.transform(dev[FEATURES].to_numpy(dtype=np.float64))
    )
    target = Path(args.model)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": FEATURES,
            "training_source": args.train,
        },
        target,
    )
    predictions = dev[
        [
            "sequence_id",
            "split",
            "label",
            "genomic_key",
            *FEATURES,
        ]
    ].copy()
    predictions["composition_score"] = score
    write_frame_new(predictions, args.dev_predictions)
    metrics = ranking_metrics(
        dev["label"].to_numpy(),
        score,
        dev["sequence_id"].astype(str),
    )
    summary = {
        "status": "PASS",
        "purpose": "shortcut-bias audit only",
        "features": FEATURES,
        "sequence_only": True,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "trainable_parameters": int(model.coef_.size + model.intercept_.size),
        "dev_metrics": metrics,
        "seconds": time.time() - started,
        "test_access": False,
    }
    write_json_new(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
