#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ranking_common import (
    META_COLUMNS,
    normalized_record,
    read_tsv_records,
    sha256_file,
    write_json_new,
)


def table_from_rows(rows: list[dict]) -> pa.Table:
    frame = pd.DataFrame(
        [{column: row[column] for column in META_COLUMNS} for row in rows]
    )
    return pa.Table.from_pandas(frame, preserve_index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if not (dataset_dir / "SUCCESS").is_file():
        raise RuntimeError("Immutable binary dataset SUCCESS marker missing")
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_positive = int(manifest["counts"]["splits"]["dev"]["positive"])
    expected_negative = int(
        manifest["counts"]["splits"]["dev"]["strict_negative"]
    )

    # The strict-negative chromosome shards intentionally do not repeat split
    # or leakage-group fields. Resolve those immutable assignments from the
    # frozen site-to-group table, then copy the matching negative records.
    dev_negative_groups = {}
    with gzip.open(
        dataset_dir / "site_to_group.tsv.gz", "rt", newline=""
    ) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["split"] == "dev" and int(row["label"]) == 0:
                dev_negative_groups[row["genomic_key"]] = row[
                    "leakage_group"
                ]
    if len(dev_negative_groups) != expected_negative:
        raise AssertionError(
            (len(dev_negative_groups), expected_negative)
        )

    positives = [
        row
        for row in read_tsv_records(
            dataset_dir / "dev_1to10.tsv.gz", "dev"
        )
        if row["label"] == 1
    ]
    by_key = {row["genomic_key"]: row for row in positives}
    if len(by_key) != expected_positive:
        raise AssertionError((len(by_key), expected_positive))

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    batch = list(by_key.values())
    negative_count = 0
    seen_keys = set()
    started = time.time()
    for shard_name in manifest["negative_shards"]:
        shard = Path(shard_name)
        with gzip.open(shard, "rt", newline="") as handle:
            for raw in csv.DictReader(handle, delimiter="\t"):
                genomic_key = raw["genomic_key"]
                leakage_group = dev_negative_groups.get(genomic_key)
                if leakage_group is None:
                    continue
                raw["label"] = "0"
                raw["split"] = "dev"
                raw["leakage_group"] = leakage_group
                batch.append(normalized_record(raw, "dev"))
                seen_keys.add(genomic_key)
                negative_count += 1
                if len(batch) >= 5000:
                    table = table_from_rows(batch)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output,
                            table.schema,
                            compression="zstd",
                            use_dictionary=True,
                        )
                    writer.write_table(table, row_group_size=len(batch))
                    batch = []
    if batch:
        table = table_from_rows(batch)
        if writer is None:
            writer = pq.ParquetWriter(
                output,
                table.schema,
                compression="zstd",
                use_dictionary=True,
            )
        writer.write_table(table, row_group_size=len(batch))
    if writer is not None:
        writer.close()

    parquet = pq.ParquetFile(output)
    actual_rows = parquet.metadata.num_rows
    if negative_count != expected_negative:
        raise AssertionError((negative_count, expected_negative))
    if seen_keys != set(dev_negative_groups):
        raise AssertionError(
            len(set(dev_negative_groups) - seen_keys)
        )
    if actual_rows != expected_positive + expected_negative:
        raise AssertionError(
            (actual_rows, expected_positive + expected_negative)
        )
    result = {
        "status": "PASS",
        "implementation_version": 2,
        "scientific_operation": (
            "derived ranking view only; joined immutable site-to-group "
            "split assignments to immutable strict-negative records without "
            "changing labels, definitions, or splits"
        ),
        "failed_v1_preserved": str(
            output.with_name("dev_ranking_universe.parquet")
        ),
        "test_rows_accessed": 0,
        "positive_count": expected_positive,
        "negative_count": expected_negative,
        "unique_negative_genomic_keys": len(seen_keys),
        "rows": actual_rows,
        "source_dataset": str(dataset_dir),
        "source_manifest_sha256": sha256_file(manifest_path),
        "site_to_group_sha256": sha256_file(
            dataset_dir / "site_to_group.tsv.gz"
        ),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "seconds": time.time() - started,
    }
    write_json_new(output.with_suffix(".manifest.json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
