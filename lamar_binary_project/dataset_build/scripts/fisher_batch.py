#!/usr/bin/env python3
"""Batch two-sided Fisher exact tests using the existing LAMAR SciPy env."""

import csv
import sys

from scipy.stats import fisher_exact


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: fisher_batch.py INPUT.tsv OUTPUT.tsv")
    with open(sys.argv[1]) as source, open(sys.argv[2], "w", newline="") as destination:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.DictWriter(destination, delimiter="\t", fieldnames=["row_index", "pvalue"], lineterminator="\n")
        writer.writeheader()
        for row in reader:
            table = [
                [int(row["treated_alt"]), int(row["treated_ref"])],
                [int(row["control_alt"]), int(row["control_ref"])],
            ]
            result = fisher_exact(table, alternative="two-sided")
            writer.writerow({"row_index": row["row_index"], "pvalue": float(result.pvalue)})


if __name__ == "__main__":
    main()
