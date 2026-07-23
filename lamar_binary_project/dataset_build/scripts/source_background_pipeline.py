#!/usr/bin/env python3
"""Auditable six-replicate background correction for CU5.17 EGFP GC sites."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pysam

try:
    from scipy.stats import fisher_exact as scipy_fisher_exact
except Exception:  # pragma: no cover - covered by the exact fallback
    scipy_fisher_exact = None


SAMPLES = (
    ("CU517_GC_T1", "treated", 1),
    ("CU517_GC_T2", "treated", 2),
    ("CU517_GC_T3", "treated", 3),
    ("CU517_GC_C1", "control", 1),
    ("CU517_GC_C2", "control", 2),
    ("CU517_GC_C3", "control", 3),
)
SAMPLE_NAMES = tuple(x[0] for x in SAMPLES)
TREATED = tuple(x[0] for x in SAMPLES if x[1] == "treated")
CONTROLS = tuple(x[0] for x in SAMPLES if x[1] == "control")
BASES = ("A", "C", "G", "T")
FILTER_FLAGS = 0x4 | 0x100 | 0x200 | 0x400 | 0x800
FILTER_DESCRIPTION = (
    "MAPQ>=30;BQ>=20;exclude_unmapped,secondary,qcfail,duplicate,supplementary;"
    "ignore_overlapping_mates;include_orphans;ACGT_only"
)
STAR_T1_NAME = "CU517_GC_T1.Aligned.sortedByCoord.out.bam"
RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(sequence: str) -> str:
    return sequence.translate(RC)[::-1].upper()


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def mad(values: Sequence[float]) -> float | None:
    if not values:
        return None
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def value_range(values: Sequence[float]) -> float | None:
    return max(values) - min(values) if values else None


def format_value(value, digits: int = 10) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.{digits}g}"
    return str(value)


def open_text(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, newline="")
    return path.open(mode, newline="")


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header in {path}")
        fields = [field.rstrip("\r") for field in reader.fieldnames]
        rows = []
        for raw in reader:
            row = {(key or "").rstrip("\r"): (value or "").rstrip("\r") for key, value in raw.items()}
            rows.append(row)
    return fields, rows


def write_tsv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with open_text(path, "wt") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def exact_fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value; scipy is preferred, this is the fallback."""
    if min(a, b, c, d) < 0:
        raise ValueError("Fisher table counts cannot be negative")
    if scipy_fisher_exact is not None:
        result = scipy_fisher_exact([[a, b], [c, d]], alternative="two-sided")
        return float(getattr(result, "pvalue", result[1]))
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2
    if total == 0:
        return 1.0
    low = max(0, row1 - (total - col1))
    high = min(row1, col1)
    denominator = math.comb(total, row1)

    def probability(x: int) -> float:
        return math.comb(col1, x) * math.comb(total - col1, row1 - x) / denominator

    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(low, high + 1) if probability(x) <= observed + 1e-15))


def benjamini_hochberg(pvalues: Sequence[float | None]) -> list[float | None]:
    adjusted: list[float | None] = [None] * len(pvalues)
    valid = sorted((float(p), index) for index, p in enumerate(pvalues) if p is not None)
    count = len(valid)
    running = 1.0
    for rank_from_end in range(count - 1, -1, -1):
        pvalue, index = valid[rank_from_end]
        rank = rank_from_end + 1
        running = min(running, pvalue * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def parse_mpileup_bases(raw: str, reference: str) -> dict[str, int]:
    counts = {base: 0 for base in BASES}
    counts.update({f"{base}_forward": 0 for base in BASES})
    counts.update({f"{base}_reverse": 0 for base in BASES})
    index = 0
    while index < len(raw):
        char = raw[index]
        if char == "^":
            index += 2
            continue
        if char == "$":
            index += 1
            continue
        if char in "+-":
            match = re.match(r"(\d+)", raw[index + 1 :])
            if not match:
                raise ValueError(f"Malformed indel in mpileup bases: {raw!r}")
            length_text = match.group(1)
            index += 1 + len(length_text) + int(length_text)
            continue
        if char in ".,":
            base = reference.upper()
            direction = "forward" if char == "." else "reverse"
            if base in BASES:
                counts[base] += 1
                counts[f"{base}_{direction}"] += 1
        elif char.upper() in BASES:
            base = char.upper()
            direction = "forward" if char.isupper() else "reverse"
            counts[base] += 1
            counts[f"{base}_{direction}"] += 1
        elif char in "*#<>Nn":
            pass
        else:
            raise ValueError(f"Unexpected mpileup symbol {char!r} in {raw!r}")
        index += 1
    return counts


class RunLogger:
    def __init__(self, run_log: Path, command_log: Path):
        self.run_log = run_log
        self.command_log = command_log

    def log(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.run_log.open("a") as handle:
            handle.write(line + "\n")

    def command(self, command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        display = " ".join(subprocess.list2cmdline([part]) for part in command)
        with self.command_log.open("a") as handle:
            handle.write(f"$ {display}\n")
        result = subprocess.run(command, text=True, capture_output=True)
        with self.command_log.open("a") as handle:
            if result.stdout:
                handle.write(result.stdout.rstrip() + "\n")
            if result.stderr:
                handle.write(result.stderr.rstrip() + "\n")
            handle.write(f"[exit={result.returncode}]\n\n")
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed ({result.returncode}): {display}\n{result.stderr}")
        return result


def find_index(bam_path: Path) -> Path | None:
    candidates = (Path(str(bam_path) + ".bai"), bam_path.with_suffix(".bai"), Path(str(bam_path) + ".csi"))
    return next((path for path in candidates if path.exists()), None)


def ensure_reference_index(reference: Path, samtools: Path, logger: RunLogger) -> Path:
    fai = Path(str(reference) + ".fai")
    if not fai.exists():
        logger.log(f"Reference index missing; creating {fai}")
        logger.command([str(samtools), "faidx", str(reference)])
    if not fai.exists() or fai.stat().st_size == 0:
        raise RuntimeError(f"Reference index was not created: {fai}")
    return fai


def ensure_bam_index(bam_path: Path, samtools: Path, threads: int, logger: RunLogger) -> Path:
    index = find_index(bam_path)
    if index is None:
        logger.log(f"BAM index missing; creating index for {bam_path}")
        logger.command([str(samtools), "index", "-@", str(threads), str(bam_path)])
        index = find_index(bam_path)
    if index is None:
        raise RuntimeError(f"BAM index was not created for {bam_path}")
    return index


def choose_bams(project: Path, samtools: Path, threads: int, logger: RunLogger):
    selected = {}
    t1_candidates = sorted(project.rglob(STAR_T1_NAME))
    valid_t1 = []
    for candidate in t1_candidates:
        result = logger.command([str(samtools), "quickcheck", "-v", str(candidate)], check=False)
        if result.returncode == 0:
            with pysam.AlignmentFile(str(candidate), "rb") as bam:
                if bam.header.to_dict().get("HD", {}).get("SO") == "coordinate":
                    valid_t1.append(candidate)
    if valid_t1:
        t1 = valid_t1[0]
        t1_reason = "original STAR coordinate-sorted T1 found and validated"
    else:
        t1 = project / "bam/markduplicates/CU517_GC_T1.markduplicates.bam"
        t1_reason = (
            "no original STAR T1 BAM found recursively; used valid coordinate-sorted Picard "
            "MarkDuplicates BAM (duplicates marked, not removed)"
        )
    selected["CU517_GC_T1"] = (t1, t1_reason)
    for sample in SAMPLE_NAMES[1:]:
        selected[sample] = (
            project / f"bam/star/{sample}.Aligned.sortedByCoord.out.bam",
            "original STAR coordinate-sorted BAM",
        )
    for sample, (bam_path, _) in selected.items():
        if not bam_path.is_file():
            raise FileNotFoundError(f"Selected BAM for {sample} is missing: {bam_path}")
        logger.command([str(samtools), "quickcheck", "-v", str(bam_path)])
        ensure_bam_index(bam_path, samtools, threads, logger)
        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            sort_order = bam.header.to_dict().get("HD", {}).get("SO")
            if sort_order != "coordinate":
                raise RuntimeError(f"BAM is not coordinate sorted ({sort_order}): {bam_path}")
            bam.check_index()
    return selected, t1_candidates


def candidate_key(row: Mapping[str, str]) -> tuple[str, int, str, str]:
    return row["chrom"], int(row["position"]), row["ref"].upper(), row["alt"].upper()


def audit_candidate_table(path: Path, reference: pysam.FastaFile) -> tuple[dict[str, object], list[dict[str, str]]]:
    fields, rows = read_tsv(path)
    required = {"chrom", "position", "ref", "alt"}
    if not required.issubset(fields):
        raise ValueError(f"Not a candidate table: {path}")
    seen = set()
    duplicates = 0
    one_based_matches = 0
    zero_based_matches = 0
    missing_contigs = set()
    bad_positions = 0
    rule_pass = 0
    rule_fail = 0
    contigs = set()
    for row in rows:
        try:
            key = candidate_key(row)
        except Exception:
            bad_positions += 1
            continue
        chrom, position, ref, alt = key
        contigs.add(chrom)
        if key in seen:
            duplicates += 1
        seen.add(key)
        if chrom not in reference.references:
            missing_contigs.add(chrom)
            continue
        if position < 1 or position > reference.get_reference_length(chrom):
            bad_positions += 1
            continue
        one_based_matches += reference.fetch(chrom, position - 1, position).upper() == ref
        if position < reference.get_reference_length(chrom):
            zero_based_matches += reference.fetch(chrom, position, position + 1).upper() == ref
        strand = row.get("vep_strand", row.get("strand", ""))
        if strand:
            if (strand in {"1", "+", "+1"} and ref == "C" and alt == "T") or (
                strand in {"-1", "-"} and ref == "G" and alt == "A"
            ):
                rule_pass += 1
            else:
                rule_fail += 1
    coordinate = "GRCh38_1_based" if rows and one_based_matches == len(rows) else "inconsistent_or_unknown"
    audit = {
        "path": str(path),
        "row_count": len(rows),
        "column_names": ",".join(fields),
        "chromosome_column": "chrom",
        "position_column": "position",
        "reference_allele_column": "ref",
        "alternate_allele_column": "alt",
        "strand_column": "vep_strand" if "vep_strand" in fields else ("strand" if "strand" in fields else "NA"),
        "coordinate_system": coordinate,
        "chromosome_count": len(contigs),
        "missing_reference_contigs": ",".join(sorted(missing_contigs)) or "none",
        "one_based_ref_matches": one_based_matches,
        "zero_based_ref_matches": zero_based_matches,
        "bad_positions": bad_positions,
        "duplicate_genomic_alleles": duplicates,
        "strand_ref_alt_rule_pass": rule_pass,
        "strand_ref_alt_rule_fail": rule_fail,
    }
    return audit, rows


def discover_candidate_tables(project: Path, reference: pysam.FastaFile):
    audits = []
    rows_by_path = {}
    paths = sorted(set(project.rglob("*.tsv")) | set(project.rglob("*.tsv.gz")))
    for path in paths:
        if "lamar_background_corrected" in path.relative_to(project).parts:
            continue
        try:
            fields, _ = read_tsv_header(path)
            if not {"chrom", "position", "ref", "alt"}.issubset(fields):
                continue
            audit, rows = audit_candidate_table(path, reference)
            audits.append(audit)
            rows_by_path[path.resolve()] = rows
        except (OSError, UnicodeDecodeError, csv.Error, ValueError):
            continue
    return audits, rows_by_path


def read_tsv_header(path: Path) -> tuple[list[str], str]:
    with open_text(path) as handle:
        raw = handle.readline().rstrip("\n")
    fields = [field.rstrip("\r") for field in raw.split("\t")]
    return fields, raw


def validate_selected_candidates(
    name: str,
    path: Path,
    rows: list[dict[str, str]],
    audit: Mapping[str, object],
    reference: pysam.FastaFile,
    bam_contigs: Mapping[str, Mapping[str, int]],
) -> None:
    if audit["coordinate_system"] != "GRCh38_1_based":
        raise RuntimeError(f"ABORT: {name} candidate coordinates are not consistently GRCh38 1-based: {path}")
    if int(audit["duplicate_genomic_alleles"]) != 0:
        raise RuntimeError(f"ABORT: duplicate genomic alleles in {name}: {audit['duplicate_genomic_alleles']}")
    if int(audit["strand_ref_alt_rule_fail"]) != 0:
        raise RuntimeError(f"ABORT: strand/ref/alt inconsistencies in {name}: {audit['strand_ref_alt_rule_fail']}")
    candidate_contigs = {row["chrom"] for row in rows}
    missing_reference = candidate_contigs - set(reference.references)
    if missing_reference:
        raise RuntimeError(f"ABORT: candidate contigs missing from FASTA: {sorted(missing_reference)}")
    for sample, lengths in bam_contigs.items():
        missing = candidate_contigs - set(lengths)
        if missing:
            raise RuntimeError(f"ABORT: candidate contigs missing from {sample} BAM: {sorted(missing)}")
        length_mismatches = [
            chrom
            for chrom in candidate_contigs
            if lengths[chrom] != reference.get_reference_length(chrom)
        ]
        if length_mismatches:
            raise RuntimeError(f"ABORT: contig length mismatch in {sample}: {length_mismatches}")


def cluster_positions(positions: Sequence[int], maximum_gap: int = 200) -> list[tuple[int, int, set[int]]]:
    if not positions:
        return []
    ordered = sorted(set(positions))
    clusters = []
    start = previous = ordered[0]
    members = {ordered[0]}
    for position in ordered[1:]:
        if position - previous <= maximum_gap:
            members.add(position)
        else:
            clusters.append((start, previous + 1, members))
            start = position
            members = {position}
        previous = position
    clusters.append((start, previous + 1, members))
    return clusters


def empty_count() -> dict[str, int]:
    result = {base: 0 for base in BASES}
    result.update({f"{base}_forward": 0 for base in BASES})
    result.update({f"{base}_reverse": 0 for base in BASES})
    return result


def count_sample_sites(
    bam_path: Path,
    positions_by_contig: Mapping[str, Sequence[int]],
    min_mapq: int,
    min_baseq: int,
    logger: RunLogger,
) -> dict[tuple[str, int], dict[str, int]]:
    counts = {
        (chrom, position): empty_count()
        for chrom, positions in positions_by_contig.items()
        for position in positions
    }
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for chrom in sorted(positions_by_contig):
            for start, end, targets in cluster_positions(positions_by_contig[chrom]):
                try:
                    iterator = bam.pileup(
                        chrom,
                        start,
                        end,
                        truncate=True,
                        stepper="all",
                        min_base_quality=min_baseq,
                        min_mapping_quality=min_mapq,
                        flag_filter=FILTER_FLAGS,
                        ignore_overlaps=True,
                        ignore_orphans=False,
                        compute_baq=False,
                        max_depth=1_000_000,
                    )
                    for column in iterator:
                        position = column.reference_pos
                        if position not in targets:
                            continue
                        site_count = counts[(chrom, position)]
                        for pileup_read in column.pileups:
                            alignment = pileup_read.alignment
                            query_position = pileup_read.query_position
                            if pileup_read.is_del or pileup_read.is_refskip or query_position is None:
                                continue
                            if alignment.flag & FILTER_FLAGS or alignment.mapping_quality < min_mapq:
                                continue
                            qualities = alignment.query_qualities
                            if qualities is None or qualities[query_position] < min_baseq:
                                continue
                            sequence = alignment.query_sequence
                            if sequence is None:
                                continue
                            base = sequence[query_position].upper()
                            if base not in BASES:
                                continue
                            direction = "reverse" if alignment.is_reverse else "forward"
                            site_count[base] += 1
                            site_count[f"{base}_{direction}"] += 1
                except Exception as exc:
                    raise RuntimeError(
                        f"ABORT: pileup technical failure for {bam_path} {chrom}:{start + 1}-{end}: {exc}"
                    ) from exc
    logger.log(f"Completed independent pileup for {bam_path.name}: {len(counts)} sites")
    return counts


def load_counts_from_pileup(
    path: Path,
    candidates: Sequence[Mapping[str, str]],
    logger: RunLogger,
) -> dict[str, dict[tuple[str, int], dict[str, int]]]:
    fields, rows = read_tsv(path)
    required = {
        "chrom",
        "position",
        "ref",
        "alt",
        "sample",
        "usable_depth",
        "A_count",
        "C_count",
        "G_count",
        "T_count",
        "forward_alt_count",
        "reverse_alt_count",
    }
    missing = required.difference(fields)
    if missing:
        raise RuntimeError(f"Reusable pileup is missing columns: {sorted(missing)}")
    expected = {candidate_key(row) for row in candidates}
    observed: set[tuple[tuple[str, int, str, str], str]] = set()
    all_counts: dict[str, dict[tuple[str, int], dict[str, int]]] = {
        sample: {} for sample in SAMPLE_NAMES
    }
    for row in rows:
        key = (str(row["chrom"]), int(row["position"]), str(row["ref"]), str(row["alt"]))
        sample = str(row["sample"])
        if key not in expected:
            raise RuntimeError(f"Reusable pileup contains an unexpected allele: {key}")
        if sample not in all_counts:
            raise RuntimeError(f"Reusable pileup contains an unexpected sample: {sample}")
        observation = (key, sample)
        if observation in observed:
            raise RuntimeError(f"Reusable pileup contains a duplicate sample/allele row: {observation}")
        observed.add(observation)
        counts = empty_count()
        for base in BASES:
            counts[base] = int(row[f"{base}_count"])
        alt = key[3]
        counts[f"{alt}_forward"] = int(row["forward_alt_count"])
        counts[f"{alt}_reverse"] = int(row["reverse_alt_count"])
        if sum(counts[base] for base in BASES) != int(row["usable_depth"]):
            raise RuntimeError(f"Reusable pileup depth/base-count mismatch: {observation}")
        all_counts[sample][(key[0], key[1] - 1)] = counts
    expected_observations = len(expected) * len(SAMPLE_NAMES)
    if len(observed) != expected_observations:
        raise RuntimeError(
            f"Reusable pileup row mismatch: expected {expected_observations}, observed {len(observed)}"
        )
    logger.log(f"Loaded and validated reusable pileup counts from {path}: {len(rows)} rows")
    return all_counts


def coverage_status(depth: int, alt_count: int, min_coverage: int) -> str:
    if depth == 0:
        return "inadequate_zero_depth"
    if depth < min_coverage:
        return "inadequate_low_depth"
    if alt_count == 0:
        return "adequate_zero_alt"
    return "adequate_with_alt"


def make_long_rows(
    candidate_rows: Sequence[Mapping[str, str]],
    source_path: Path,
    all_counts: Mapping[str, Mapping[tuple[str, int], Mapping[str, int]]],
    min_coverage: int,
    min_mapq: int,
    min_baseq: int,
) -> list[dict[str, object]]:
    output = []
    metadata = {sample: (group, replicate) for sample, group, replicate in SAMPLES}
    for candidate in candidate_rows:
        chrom, position1, ref, alt = candidate_key(candidate)
        position0 = position1 - 1
        strand = candidate.get("vep_strand", candidate.get("strand", ""))
        for sample in SAMPLE_NAMES:
            group, replicate = metadata[sample]
            counts = all_counts[sample][(chrom, position0)]
            depth = sum(counts[base] for base in BASES)
            ref_count = counts[ref]
            alt_count = counts[alt]
            output.append(
                {
                    "chrom": chrom,
                    "position": position1,
                    "ref": ref,
                    "alt": alt,
                    "transcript_strand": strand,
                    "sample": sample,
                    "group": group,
                    "replicate": replicate,
                    "usable_depth": depth,
                    "ref_count": ref_count,
                    "alt_count": alt_count,
                    "A_count": counts["A"],
                    "C_count": counts["C"],
                    "G_count": counts["G"],
                    "T_count": counts["T"],
                    "edit_rate": alt_count / depth if depth else None,
                    "forward_alt_count": counts[f"{alt}_forward"],
                    "reverse_alt_count": counts[f"{alt}_reverse"],
                    "coverage_status": coverage_status(depth, alt_count, min_coverage),
                    "minimum_mapping_quality": min_mapq,
                    "minimum_base_quality": min_baseq,
                    "filter_flags_decimal": FILTER_FLAGS,
                    "filters_used": FILTER_DESCRIPTION,
                    "candidate_source": str(source_path),
                }
            )
    return output


PILEUP_FIELDS = (
    "chrom",
    "position",
    "ref",
    "alt",
    "transcript_strand",
    "sample",
    "group",
    "replicate",
    "usable_depth",
    "ref_count",
    "alt_count",
    "A_count",
    "C_count",
    "G_count",
    "T_count",
    "edit_rate",
    "forward_alt_count",
    "reverse_alt_count",
    "coverage_status",
    "minimum_mapping_quality",
    "minimum_base_quality",
    "filter_flags_decimal",
    "filters_used",
    "candidate_source",
)


def build_labels(
    candidates: Sequence[Mapping[str, str]],
    long_rows: Sequence[Mapping[str, object]],
    min_coverage: int,
    min_group_replicates: int,
    context_by_key: Mapping[tuple[str, int, str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    by_key_sample = {}
    for row in long_rows:
        key = (str(row["chrom"]), int(row["position"]), str(row["ref"]), str(row["alt"]))
        by_key_sample[(key, str(row["sample"]))] = row
    labels = []
    for candidate in candidates:
        key = candidate_key(candidate)
        sample_rows = {sample: by_key_sample[(key, sample)] for sample in SAMPLE_NAMES}
        rates = {
            sample: (
                float(sample_rows[sample]["edit_rate"])
                if int(sample_rows[sample]["usable_depth"]) >= min_coverage
                and sample_rows[sample]["edit_rate"] is not None
                else None
            )
            for sample in SAMPLE_NAMES
        }
        treated_rates = [rates[sample] for sample in TREATED if rates[sample] is not None]
        control_rates = [rates[sample] for sample in CONTROLS if rates[sample] is not None]
        treated_covered = len(treated_rates)
        control_covered = len(control_rates)
        group_sufficient = treated_covered >= min_group_replicates and control_covered >= min_group_replicates
        treated_median = median(treated_rates)
        control_median = median(control_rates)
        raw_difference = (
            treated_median - control_median
            if group_sufficient and treated_median is not None and control_median is not None
            else None
        )
        corrected = max(raw_difference, 0.0) if raw_difference is not None else None
        pooled = {}
        for group_name, names in (("treated", TREATED), ("control", CONTROLS)):
            covered_names = [sample for sample in names if rates[sample] is not None]
            pooled[f"pooled_{group_name}_ref_count"] = sum(int(sample_rows[sample]["ref_count"]) for sample in covered_names)
            pooled[f"pooled_{group_name}_alt_count"] = sum(int(sample_rows[sample]["alt_count"]) for sample in covered_names)
        fisher_p = None
        if group_sufficient:
            fisher_p = exact_fisher_two_sided(
                pooled["pooled_treated_alt_count"],
                pooled["pooled_treated_ref_count"],
                pooled["pooled_control_alt_count"],
                pooled["pooled_control_ref_count"],
            )
        context = context_by_key[key]
        source_wgs = candidate.get("wgs_variant", "0") not in {"", "0", "NA", "."}
        reasons = []
        if treated_covered < min_group_replicates:
            reasons.append("low_treated_coverage")
        if control_covered < min_group_replicates:
            reasons.append("low_control_coverage")
        if source_wgs:
            reasons.append("source_wgs_variant")
        if context["orientation_qc"] != "pass":
            reasons.append("sequence_orientation_qc_failed")
        training_eligible = not reasons
        if training_eligible and treated_covered == 3 and control_covered == 3 and mad(treated_rates) <= 0.05 and mad(control_rates) <= 0.02:
            confidence = "high"
        elif training_eligible:
            confidence = "moderate"
        else:
            confidence = "low"
        row: dict[str, object] = {
            "chrom": key[0],
            "position": key[1],
            "ref": key[2],
            "alt": key[3],
            "transcript_strand": candidate.get("vep_strand", candidate.get("strand", "")),
        }
        for sample in SAMPLE_NAMES:
            raw_rate = sample_rows[sample]["edit_rate"]
            row[f"{sample}_usable_depth"] = sample_rows[sample]["usable_depth"]
            row[f"{sample}_ref_count"] = sample_rows[sample]["ref_count"]
            row[f"{sample}_alt_count"] = sample_rows[sample]["alt_count"]
            row[f"{sample}_raw_edit_rate"] = raw_rate
            row[f"{sample}_edit_rate"] = rates[sample]
            row[f"{sample}_coverage_status"] = sample_rows[sample]["coverage_status"]
        row.update(
            {
                "treated_median": treated_median,
                "control_median": control_median,
                "treated_mean": mean(treated_rates),
                "control_mean": mean(control_rates),
                "treated_MAD": mad(treated_rates),
                "control_MAD": mad(control_rates),
                "treated_range": value_range(treated_rates),
                "control_range": value_range(control_rates),
                "raw_edit_rate_difference": raw_difference,
                "corrected_editing_efficiency": corrected,
                "sufficiently_covered_treated_replicates": treated_covered,
                "sufficiently_covered_control_replicates": control_covered,
                **pooled,
                "fisher_exact_screening_pvalue": fisher_p,
                "BH_FDR": None,
                "treated_replicate_consistent": treated_covered >= min_group_replicates and mad(treated_rates) <= 0.05,
                "control_replicate_consistent": control_covered >= min_group_replicates and mad(control_rates) <= 0.02,
                "sufficient_coverage_all_six": treated_covered == 3 and control_covered == 3,
                "elevated_control_background": control_median is not None and control_median >= 0.02,
                "training_eligible": training_eligible,
                "label_confidence": confidence,
                "exclusion_reason": ";".join(reasons) if reasons else "none",
                "source_wgs_variant": source_wgs,
                "source_catalogue_overlap": candidate.get("genomic_catalogue_overlap", "NA"),
                "minimum_usable_depth": min_coverage,
                "minimum_covered_replicates_per_group": min_group_replicates,
                "fisher_limitation": "screening_only_reads_are_not_independent_biological_replicates",
            }
        )
        labels.append(row)
    adjusted = benjamini_hochberg([row["fisher_exact_screening_pvalue"] for row in labels])
    for row, fdr in zip(labels, adjusted):
        row["BH_FDR"] = fdr
        row["passes_FDR_0.05"] = fdr is not None and fdr <= 0.05
    return labels


def label_fields() -> list[str]:
    fields = ["chrom", "position", "ref", "alt", "transcript_strand"]
    for sample in SAMPLE_NAMES:
        fields.extend(
            [
                f"{sample}_usable_depth",
                f"{sample}_ref_count",
                f"{sample}_alt_count",
                f"{sample}_raw_edit_rate",
                f"{sample}_edit_rate",
                f"{sample}_coverage_status",
            ]
        )
    fields.extend(
        [
            "treated_median",
            "control_median",
            "treated_mean",
            "control_mean",
            "treated_MAD",
            "control_MAD",
            "treated_range",
            "control_range",
            "raw_edit_rate_difference",
            "corrected_editing_efficiency",
            "sufficiently_covered_treated_replicates",
            "sufficiently_covered_control_replicates",
            "pooled_treated_ref_count",
            "pooled_treated_alt_count",
            "pooled_control_ref_count",
            "pooled_control_alt_count",
            "fisher_exact_screening_pvalue",
            "BH_FDR",
            "passes_FDR_0.05",
            "treated_replicate_consistent",
            "control_replicate_consistent",
            "sufficient_coverage_all_six",
            "elevated_control_background",
            "training_eligible",
            "label_confidence",
            "exclusion_reason",
            "source_wgs_variant",
            "source_catalogue_overlap",
            "minimum_usable_depth",
            "minimum_covered_replicates_per_group",
            "fisher_limitation",
        ]
    )
    return fields


def extract_contexts(
    candidates: Sequence[Mapping[str, str]], reference: pysam.FastaFile, sequence_length: int
) -> dict[tuple[str, int, str, str], dict[str, object]]:
    if sequence_length <= 0 or sequence_length % 2 == 0:
        raise ValueError("Sequence length must be a positive odd number")
    half = sequence_length // 2
    result = {}
    for candidate in candidates:
        key = candidate_key(candidate)
        chrom, position1, ref, alt = key
        position0 = position1 - 1
        start = position0 - half
        end = position0 + half + 1
        strand = candidate.get("vep_strand", candidate.get("strand", ""))
        if start < 0 or end > reference.get_reference_length(chrom):
            sequence = ""
            oriented_ref = "NA"
            oriented_alt = "NA"
            center = "NA"
            qc = "fail_window_out_of_bounds"
        else:
            genomic = reference.fetch(chrom, start, end).upper()
            sequence = genomic if strand in {"1", "+", "+1"} else revcomp(genomic)
            oriented_ref = ref if strand in {"1", "+", "+1"} else revcomp(ref)
            oriented_alt = alt if strand in {"1", "+", "+1"} else revcomp(alt)
            center = sequence[half] if len(sequence) == sequence_length else "NA"
            qc = (
                "pass"
                if len(sequence) == sequence_length
                and center == "C"
                and oriented_ref == "C"
                and oriented_alt == "T"
                else "fail_orientation_or_center"
            )
        result[key] = {
            "sequence_context": sequence,
            "sequence_length": len(sequence),
            "center_index": half,
            "center_base": center,
            "orientation_qc": qc,
            "transcript_oriented_ref": oriented_ref,
            "transcript_oriented_alt": oriented_alt,
        }
    return result


def candidate_summary(labels: Sequence[Mapping[str, object]]) -> dict[str, int]:
    def yes(field: str) -> int:
        return sum(bool(row[field]) for row in labels)

    return {
        "total_sites": len(labels),
        "passed_all_qc": yes("training_eligible"),
        "sufficient_coverage_all_six": yes("sufficient_coverage_all_six"),
        "positive_raw_differences": sum(
            row["raw_edit_rate_difference"] is not None and float(row["raw_edit_rate_difference"]) > 0 for row in labels
        ),
        "became_zero_after_correction": sum(
            row["raw_edit_rate_difference"] is not None
            and float(row["raw_edit_rate_difference"]) <= 0
            and float(row["corrected_editing_efficiency"]) == 0
            for row in labels
        ),
        "failed_low_treated_coverage": sum("low_treated_coverage" in str(row["exclusion_reason"]) for row in labels),
        "failed_low_control_coverage": sum("low_control_coverage" in str(row["exclusion_reason"]) for row in labels),
        "elevated_control_background": yes("elevated_control_background"),
        "passed_FDR_0.05": yes("passes_FDR_0.05"),
        "training_eligible": yes("training_eligible"),
    }


def validate_outputs(labels: Sequence[Mapping[str, object]], expected_rows: int) -> None:
    if len(labels) != expected_rows:
        raise RuntimeError(f"Output row mismatch: expected {expected_rows}, observed {len(labels)}")
    for row in labels:
        raw = row["raw_edit_rate_difference"]
        corrected = row["corrected_editing_efficiency"]
        if corrected is not None and float(corrected) < 0:
            raise RuntimeError(f"Negative corrected efficiency: {row}")
        if raw is None and corrected is not None:
            raise RuntimeError("Missing raw difference must imply missing corrected efficiency")
        if int(row["sufficiently_covered_control_replicates"]) < int(row["minimum_covered_replicates_per_group"]):
            if (
                raw is not None
                or corrected is not None
                or row["training_eligible"]
                or "low_control_coverage" not in str(row["exclusion_reason"])
            ):
                raise RuntimeError("Insufficient control coverage was silently treated as zero/sufficient")
        if int(row["sufficiently_covered_treated_replicates"]) < int(row["minimum_covered_replicates_per_group"]):
            if (
                raw is not None
                or corrected is not None
                or row["training_eligible"]
                or "low_treated_coverage" not in str(row["exclusion_reason"])
            ):
                raise RuntimeError("Insufficient treated coverage was silently treated as zero/sufficient")
        for sample in SAMPLE_NAMES:
            depth = int(row[f"{sample}_usable_depth"])
            alt = int(row[f"{sample}_alt_count"])
            rate = row[f"{sample}_edit_rate"]
            if depth >= int(row["minimum_usable_depth"]):
                if rate is None or not math.isclose(float(rate), alt / depth, rel_tol=1e-10, abs_tol=1e-12):
                    raise RuntimeError(f"Rate/count inconsistency at {row['chrom']}:{row['position']} {sample}")
            elif rate is not None:
                raise RuntimeError("Low-depth replicate must have NA label rate")


def run_direct_validation(
    run_dir: Path,
    candidates: Sequence[Mapping[str, str]],
    all_counts: Mapping[str, Mapping[tuple[str, int], Mapping[str, int]]],
    selected_bams: Mapping[str, tuple[Path, str]],
    reference: Path,
    samtools: Path,
    min_mapq: int,
    min_baseq: int,
    logger: RunLogger,
) -> list[dict[str, object]]:
    randomizer = random.Random(42)
    selected = randomizer.sample(list(candidates), min(20, len(candidates)))
    bed = run_dir / "direct_validation_sites.bed"
    with bed.open("w") as handle:
        for row in sorted(selected, key=lambda item: (item["chrom"], int(item["position"]))):
            position1 = int(row["position"])
            handle.write(f"{row['chrom']}\t{position1 - 1}\t{position1}\n")
    direct_rows = []
    selected_keys = {candidate_key(row) for row in selected}
    for sample in SAMPLE_NAMES:
        bam_path = selected_bams[sample][0]
        result = logger.command(
            [
                str(samtools),
                "mpileup",
                "-B",
                "-q",
                str(min_mapq),
                "-Q",
                str(min_baseq),
                "--ff",
                str(FILTER_FLAGS),
                "-d",
                "1000000",
                "-l",
                str(bed),
                "-f",
                str(reference),
                str(bam_path),
            ]
        )
        direct = {}
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            chrom, position_text, reference_base = parts[:3]
            direct[(chrom, int(position_text))] = parse_mpileup_bases(parts[4], reference_base)
        for key in sorted(selected_keys):
            chrom, position1, ref, alt = key
            generated = all_counts[sample][(chrom, position1 - 1)]
            checked = direct.get((chrom, position1), empty_count())
            fields_match = all(generated[base] == checked[base] for base in BASES)
            fields_match = fields_match and generated[f"{alt}_forward"] == checked[f"{alt}_forward"]
            fields_match = fields_match and generated[f"{alt}_reverse"] == checked[f"{alt}_reverse"]
            direct_rows.append(
                {
                    "chrom": chrom,
                    "position": position1,
                    "ref": ref,
                    "alt": alt,
                    "sample": sample,
                    "generated_A": generated["A"],
                    "direct_A": checked["A"],
                    "generated_C": generated["C"],
                    "direct_C": checked["C"],
                    "generated_G": generated["G"],
                    "direct_G": checked["G"],
                    "generated_T": generated["T"],
                    "direct_T": checked["T"],
                    "generated_forward_alt": generated[f"{alt}_forward"],
                    "direct_forward_alt": checked[f"{alt}_forward"],
                    "generated_reverse_alt": generated[f"{alt}_reverse"],
                    "direct_reverse_alt": checked[f"{alt}_reverse"],
                    "validation_pass": fields_match,
                }
            )
    if not all(row["validation_pass"] for row in direct_rows):
        failures = sum(not row["validation_pass"] for row in direct_rows)
        raise RuntimeError(f"Direct samtools mpileup validation failed for {failures}/{len(direct_rows)} rows")
    return direct_rows


def write_reference_qc(
    path: Path,
    reference: Path,
    fai: Path,
    selected_audits: Mapping[str, Mapping[str, object]],
    bam_contigs: Mapping[str, Mapping[str, int]],
) -> None:
    with path.open("w") as handle:
        handle.write(f"reference\t{reference}\n")
        handle.write(f"fai\t{fai}\n")
        handle.write("fai_status\tpresent_and_readable\n")
        with fai.open() as index_handle:
            index_rows = [line.rstrip("\n").split("\t") for line in index_handle if line.strip()]
        handle.write(f"chromosome_count\t{len(index_rows)}\n")
        for name, audit in selected_audits.items():
            handle.write(f"{name}_coordinate_system\t{audit['coordinate_system']}\n")
            handle.write(f"{name}_one_based_ref_matches\t{audit['one_based_ref_matches']}/{audit['row_count']}\n")
            handle.write(f"{name}_zero_based_ref_matches\t{audit['zero_based_ref_matches']}/{audit['row_count']}\n")
        handle.write("bam_fasta_candidate_contig_compatibility\tpass\n")
        handle.write("\nchromosome\tlength\n")
        for row in index_rows:
            handle.write(f"{row[0]}\t{row[1]}\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(run_dir: Path) -> None:
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    with (run_dir / "checksums.sha256").open("w") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(run_dir)}\n")


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("${IGEM_DATA_ROOT}/CU5.17_EGFP_GC_paper"))
    parser.add_argument("--reference", type=Path, default=Path("${IGEM_DATA_ROOT}/GRCh38.primary_assembly.genome.fa"))
    parser.add_argument("--samtools", type=Path, default=Path("${IGEM_ENV}/bin/samtools"))
    parser.add_argument("--python", type=Path, default=Path("${IGEM_ENV}/bin/python"))
    parser.add_argument("--min-mapq", type=int, default=30)
    parser.add_argument("--min-baseq", type=int, default=20)
    parser.add_argument("--min-coverage", type=int, default=20)
    parser.add_argument("--min-group-replicates", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=101)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--reuse-pileup",
        type=Path,
        help="Reuse a fully validated six-sample pileup table from an interrupted run",
    )
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project = args.project.resolve()
    reference_path = args.reference.resolve()
    output_root = project / "lamar_background_corrected"
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    incomplete = output_root / f".incomplete_run_{timestamp}_{os.getpid()}"
    incomplete.mkdir()
    logger = RunLogger(incomplete / "run.log", incomplete / "commands.log")
    invocation = [str(args.python), str(Path(__file__).resolve()), *sys.argv[1:]]
    with logger.command_log.open("w") as handle:
        handle.write("# Exact pipeline invocation\n$ " + " ".join(subprocess.list2cmdline([x]) for x in invocation) + "\n\n")
    try:
        logger.log("Starting lamar background-corrected analysis")
        if not args.skip_tests:
            test_path = Path(__file__).with_name("test_pipeline.py")
            logger.command(
                [
                    str(args.python),
                    "-m",
                    "unittest",
                    "discover",
                    "-v",
                    "-s",
                    str(test_path.parent),
                    "-p",
                    test_path.name,
                ]
            )
            logger.log("Repository tests passed")
        if not project.is_dir() or not reference_path.is_file() or not args.samtools.is_file():
            raise FileNotFoundError("Project, reference, or samtools path is missing")
        fai = ensure_reference_index(reference_path, args.samtools, logger)
        selected_bams, original_t1_candidates = choose_bams(project, args.samtools, args.threads, logger)

        broad_path = (project / "final/CU5.17_EGFP_GC.site_matrix.tsv.gz").resolve()
        final_path = (
            project / "final_with_293T_catalogue/CU5.17_EGFP_GC.treatment_specific.tsv.gz"
        ).resolve()
        with pysam.FastaFile(str(reference_path)) as reference:
            candidate_audits, rows_by_path = discover_candidate_tables(project, reference)
            audit_by_path = {Path(str(row["path"])).resolve(): row for row in candidate_audits}
            if broad_path not in rows_by_path or final_path not in rows_by_path:
                raise RuntimeError("Required broad or final candidate table was not discovered")
            broad_rows = rows_by_path[broad_path]
            final_rows = rows_by_path[final_path]
            bam_contigs = {}
            bam_headers = {}
            reference_lengths = dict(zip(reference.references, reference.lengths))
            for sample, (bam_path, _) in selected_bams.items():
                with pysam.AlignmentFile(str(bam_path), "rb") as bam:
                    bam_contigs[sample] = dict(zip(bam.references, bam.lengths))
                    bam_headers[sample] = bam.header.to_dict()
            validate_selected_candidates(
                "broad", broad_path, broad_rows, audit_by_path[broad_path], reference, bam_contigs
            )
            validate_selected_candidates(
                "final", final_path, final_rows, audit_by_path[final_path], reference, bam_contigs
            )
            logger.log("Candidate coordinate, allele, duplicate, strand, FASTA, and BAM compatibility checks passed")

            all_candidate_by_key = {candidate_key(row): row for row in broad_rows}
            for row in final_rows:
                all_candidate_by_key.setdefault(candidate_key(row), row)
            all_candidates = list(all_candidate_by_key.values())
            positions_by_contig: dict[str, list[int]] = defaultdict(list)
            for row in all_candidates:
                positions_by_contig[row["chrom"]].append(int(row["position"]) - 1)
            if args.reuse_pileup is not None:
                reuse_path = args.reuse_pileup.resolve()
                if not reuse_path.is_file():
                    raise FileNotFoundError(f"Reusable pileup does not exist: {reuse_path}")
                all_counts = load_counts_from_pileup(reuse_path, all_candidates, logger)
            else:
                all_counts = {}
                for sample in SAMPLE_NAMES:
                    all_counts[sample] = count_sample_sites(
                        selected_bams[sample][0],
                        positions_by_contig,
                        args.min_mapq,
                        args.min_baseq,
                        logger,
                    )
            contexts = extract_contexts(all_candidates, reference, args.sequence_length)

        # Input manifest.
        manifest_fields = (
            "input_type",
            "name",
            "group",
            "replicate",
            "path",
            "size_bytes",
            "mtime_utc",
            "index_path",
            "selected_reason",
        )
        manifest_rows = []
        for sample, group, replicate in SAMPLES:
            bam_path, reason = selected_bams[sample]
            stat = bam_path.stat()
            manifest_rows.append(
                {
                    "input_type": "BAM",
                    "name": sample,
                    "group": group,
                    "replicate": replicate,
                    "path": bam_path,
                    "size_bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "index_path": find_index(bam_path),
                    "selected_reason": reason,
                }
            )
        for input_type, name, path, reason in (
            ("FASTA", "GRCh38", reference_path, "GRCh38 primary assembly reference"),
            ("candidate_table", "broad", broad_path, "priority 1 genuine 9930-site broad matrix"),
            ("candidate_table", "final", final_path, "separate post-catalogue final-candidate analysis"),
        ):
            stat = path.stat()
            manifest_rows.append(
                {
                    "input_type": input_type,
                    "name": name,
                    "group": "NA",
                    "replicate": "NA",
                    "path": path,
                    "size_bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "index_path": fai if input_type == "FASTA" else "NA",
                    "selected_reason": reason,
                }
            )
        if args.reuse_pileup is not None:
            reuse_path = args.reuse_pileup.resolve()
            stat = reuse_path.stat()
            manifest_rows.append(
                {
                    "input_type": "pileup_cache",
                    "name": "reused_six_sample_pileup",
                    "group": "NA",
                    "replicate": "NA",
                    "path": reuse_path,
                    "size_bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "index_path": "NA",
                    "selected_reason": "validated complete 9930-site x 6-sample pileup from interrupted run",
                }
            )
        write_tsv(incomplete / "input_manifest.tsv", manifest_fields, manifest_rows)

        # BAM QC and T1 duplicate investigation.
        bam_qc_rows = []
        t1_metrics = project / "metrics/CU517_GC_T1.markduplicates_metrics.txt"
        t1_metrics_text = t1_metrics.read_text() if t1_metrics.exists() else ""
        percent_match = re.search(r"Unknown Library\t(?:[^\t]*\t){7}([^\t\n]+)", t1_metrics_text)
        t1_duplicate_flags = logger.command(
            [str(args.samtools), "view", "-c", "-f", "1024", str(selected_bams["CU517_GC_T1"][0])]
        ).stdout.strip()
        for sample, group, replicate in SAMPLES:
            bam_path, reason = selected_bams[sample]
            header = bam_headers[sample]
            idxstats = logger.command([str(args.samtools), "idxstats", str(bam_path)]).stdout.splitlines()
            mapped = sum(int(line.split("\t")[2]) for line in idxstats if len(line.split("\t")) >= 4)
            unmapped = sum(int(line.split("\t")[3]) for line in idxstats if len(line.split("\t")) >= 4)
            programs = header.get("PG", [])
            pg_ids = ",".join(str(item.get("ID", "")) for item in programs)
            bam_qc_rows.append(
                {
                    "sample": sample,
                    "group": group,
                    "replicate": replicate,
                    "bam_path": bam_path,
                    "index_path": find_index(bam_path),
                    "quickcheck": "pass",
                    "coordinate_sorted": header.get("HD", {}).get("SO") == "coordinate",
                    "contig_count": len(bam_contigs[sample]),
                    "contig_lengths_match_FASTA": all(
                        chrom in reference_lengths
                        and reference_lengths[chrom] == length
                        for chrom, length in bam_contigs[sample].items()
                    ),
                    "mapped_alignments_from_index": mapped,
                    "unmapped_alignments_from_index": unmapped,
                    "program_record_ids": pg_ids,
                    "duplicate_processing": (
                        "Picard MarkDuplicates; duplicate flags present; REMOVE_DUPLICATES=false"
                        if sample == "CU517_GC_T1"
                        else "STAR BAM; duplicate marking not applied"
                    ),
                    "duplicate_flagged_alignments": t1_duplicate_flags if sample == "CU517_GC_T1" else "not_scanned_unmarked_STAR",
                    "Picard_percent_duplication": percent_match.group(1) if sample == "CU517_GC_T1" and percent_match else "NA",
                    "selected_reason": reason,
                }
            )
        bam_qc_fields = list(bam_qc_rows[0])
        write_tsv(incomplete / "bam_qc.tsv", bam_qc_fields, bam_qc_rows)

        selected_audits = {"broad": audit_by_path[broad_path], "final": audit_by_path[final_path]}
        write_reference_qc(
            incomplete / "reference_qc.txt", reference_path, fai, selected_audits, bam_contigs
        )
        candidate_fields = list(candidate_audits[0]) + ["selected_role"]
        for row in candidate_audits:
            resolved = Path(str(row["path"])).resolve()
            row["selected_role"] = (
                "broad_primary" if resolved == broad_path else "final_separate" if resolved == final_path else "plausible_not_selected"
            )
        write_tsv(incomplete / "candidate_table_qc.tsv", candidate_fields, candidate_audits)

        (incomplete / "selected_candidate_source.md").write_text(
            "# Selected candidate sources\n\n"
            f"- Primary broad analysis: `{broad_path}` ({len(broad_rows):,} rows). This is the genuine broad candidate matrix and has priority over filtered tables.\n"
            f"- Separate final-candidate analysis: `{final_path}` ({len(final_rows):,} rows), after 293T catalogue filtering.\n"
            "- The 3,333 final candidates are not assumed to be a balanced Lamar training set.\n"
            "- Both tables are GRCh38 1-based; every reference allele matches the FASTA at position-1, chromosome names match all selected BAMs, and duplicate genomic alleles were absent.\n"
        )

        summaries = []
        dataset_results = {}
        for dataset_name, candidates, source_path, dataset_dir in (
            ("broad", broad_rows, broad_path, incomplete),
            ("final", final_rows, final_path, incomplete / "final_candidate"),
        ):
            dataset_dir.mkdir(exist_ok=True)
            long_rows = make_long_rows(
                candidates,
                source_path,
                all_counts,
                args.min_coverage,
                args.min_mapq,
                args.min_baseq,
            )
            write_tsv(dataset_dir / "six_sample_pileup_counts.tsv.gz", PILEUP_FIELDS, long_rows)
            contexts_for_set = {candidate_key(row): contexts[candidate_key(row)] for row in candidates}
            labels = build_labels(
                candidates, long_rows, args.min_coverage, args.min_group_replicates, contexts_for_set
            )
            validate_outputs(labels, len(candidates))
            write_tsv(dataset_dir / "background_corrected_labels.tsv.gz", label_fields(), labels)
            excluded = [row for row in labels if not row["training_eligible"]]
            write_tsv(dataset_dir / "excluded_sites.tsv.gz", label_fields(), excluded)
            label_by_key = {
                (str(row["chrom"]), int(row["position"]), str(row["ref"]), str(row["alt"])): row
                for row in labels
            }
            lamar_fields = [
                "chrom",
                "position",
                "ref",
                "alt",
                "transcript_strand",
                "sequence_context",
                "sequence_length",
                "center_index",
                "center_base",
                "orientation_qc",
                "transcript_oriented_ref",
                "transcript_oriented_alt",
                "corrected_editing_efficiency",
                "raw_edit_rate_difference",
                "training_eligible",
                "label_confidence",
                "exclusion_reason",
            ]
            lamar_rows = []
            for candidate in candidates:
                key = candidate_key(candidate)
                label = label_by_key[key]
                lamar_rows.append(
                    {
                        "chrom": key[0],
                        "position": key[1],
                        "ref": key[2],
                        "alt": key[3],
                        "transcript_strand": candidate.get("vep_strand", candidate.get("strand", "")),
                        **contexts[key],
                        "corrected_editing_efficiency": label["corrected_editing_efficiency"],
                        "raw_edit_rate_difference": label["raw_edit_rate_difference"],
                        "training_eligible": label["training_eligible"],
                        "label_confidence": label["label_confidence"],
                        "exclusion_reason": label["exclusion_reason"],
                    }
                )
            write_tsv(dataset_dir / "lamar_ready_metadata.tsv.gz", lamar_fields, lamar_rows)
            summary = candidate_summary(labels)
            summaries.extend(
                {"candidate_set": dataset_name, "metric": metric, "count": count}
                for metric, count in summary.items()
            )
            dataset_results[dataset_name] = (labels, summary)
            logger.log(f"Wrote and validated {dataset_name} outputs: {len(candidates)} sites")

        direct_rows = run_direct_validation(
            incomplete,
            broad_rows,
            all_counts,
            selected_bams,
            reference_path,
            args.samtools,
            args.min_mapq,
            args.min_baseq,
            logger,
        )
        write_tsv(incomplete / "direct_pileup_validation.tsv", list(direct_rows[0]), direct_rows)
        write_tsv(incomplete / "qc_summary.tsv", ("candidate_set", "metric", "count"), summaries)

        versions = [
            f"timestamp_utc\t{timestamp}",
            f"platform\t{platform.platform()}",
            f"python\t{sys.version.replace(chr(10), ' ')}",
            f"pysam\t{pysam.__version__}",
        ]
        try:
            import scipy

            versions.append(f"scipy\t{scipy.__version__}")
        except Exception:
            versions.append("scipy\tnot_available_exact_fallback_used")
        versions.append(
            "samtools\t"
            + logger.command([str(args.samtools), "--version"]).stdout.splitlines()[0]
        )
        (incomplete / "software_versions.txt").write_text("\n".join(versions) + "\n")

        t1_path, t1_reason = selected_bams["CU517_GC_T1"]
        broad_summary = dataset_results["broad"][1]
        final_summary = dataset_results["final"][1]
        repository_audit = f"""# Repository audit

## Existing project

- The project is not a Git working tree and contains no pre-existing automated test suite.
- Existing candidate construction code: `{project / 'helpers/filter_c_to_u_and_compare.py'}`.
- Existing Lamar conversion code: `${IGEM_DATA_ROOT}/auto_sites_to_lamar_csv.py`.
- The latter declares `--window` default 101 and the companion runner declares `WINDOW=101`; this is the evidence for the 101-nt context used here.
- Existing genomic-C export documentation uses coverage 20, supporting the `minimum_usable_depth=20` default.

## Reference and coordinates

- Reference: `{reference_path}` with readable `.fai`.
- Both selected tables are GRCh38 1-based. All rows match the FASTA reference allele at `position-1`; the competing position-as-0-based hypothesis matches only a minority by chance (see `reference_qc.txt`).
- Candidate, BAM, and FASTA chromosome names and lengths passed exact compatibility checks. The pipeline aborts on any mismatch rather than emitting zero coverage.

## T1 consistency investigation

- Recursive search for `{STAR_T1_NAME}` returned {len(original_t1_candidates)} file(s).
- Selected T1: `{t1_path}`.
- Reason: {t1_reason}.
- The Picard metrics command records `REMOVE_DUPLICATES=false` and `REMOVE_SEQUENCING_DUPLICATES=false`; duplicate reads were marked, not removed. Duplicate-flagged alignments were counted in `bam_qc.tsv`.
- This is a real preprocessing inconsistency: T1 is MarkDuplicates output, while T2/T3/C1/C2/C3 are original STAR coordinate-sorted BAMs. All duplicate-flagged alignments are excluded by the same pileup flag filter, but the five STAR BAMs were never duplicate-marked, so identical preprocessing cannot be claimed.

## Counting and labels

- MAPQ >= {args.min_mapq}; base quality >= {args.min_baseq}; usable depth is A+C+G+T after filters.
- Unmapped, secondary, QC-failed, duplicate-marked, and supplementary alignments are excluded. Overlapping mates are collapsed by pileup.
- A replicate is sufficiently covered at usable depth >= {args.min_coverage}. A group is sufficient at >= {args.min_group_replicates}/3 covered replicates; high confidence additionally requires all 3+3 and treated MAD <= 0.05 and control MAD <= 0.02.
- Control background >= 0.02 is flagged as elevated, based on the repository's existing control maximum default.
- Fisher exact p-values are screening statistics only. Reads are not independent biological replicates, so these p-values are not biological validation.
- Twenty deterministic broad-matrix sites were re-counted in all six BAMs with independent `samtools mpileup` calls (120 comparisons), all passing.
"""
        (incomplete / "repository_audit.md").write_text(repository_audit)

        reproduce = (
            f"{args.python} {Path(__file__).resolve()} --project {project} --reference {reference_path} "
            f"--samtools {args.samtools} --python {args.python} --min-mapq {args.min_mapq} "
            f"--min-baseq {args.min_baseq} --min-coverage {args.min_coverage} "
            f"--min-group-replicates {args.min_group_replicates} --sequence-length {args.sequence_length} "
            f"--threads {args.threads}"
            + (f" --reuse-pileup {args.reuse_pileup.resolve()}" if args.reuse_pileup is not None else "")
        )
        analysis_summary = f"""# Analysis summary

Computation completed successfully for a broad {len(broad_rows):,}-site matrix and a separate {len(final_rows):,}-site final-candidate table. This confirms computational processing and QC only; it does **not** experimentally or biologically validate the labels.

## Broad matrix QC

{json.dumps(broad_summary, indent=2, sort_keys=True)}

## Final-candidate QC

{json.dumps(final_summary, indent=2, sort_keys=True)}

## Key limitations

- T1 preprocessing differs from the other five BAMs as documented in `repository_audit.md` and `bam_qc.tsv`.
- Fisher exact tests pool sequencing reads and are screening-only; biological replicate-level inference requires an appropriate replicate-aware model and independent validation.
- The source workflow warned that no WGS VCF was supplied, so the original `wgs_variant` field cannot be treated as a complete genomic-variant exclusion.
- Training eligibility means the documented computational coverage/orientation criteria passed; it is not evidence of experimental validity or of a balanced Lamar training distribution.

## Exact reproduction command

```bash
{reproduce}
```
"""
        (incomplete / "analysis_summary.md").write_text(analysis_summary)
        code_dir = incomplete / "code"
        code_dir.mkdir()
        shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
        test_source = Path(__file__).with_name("test_pipeline.py")
        if test_source.exists():
            shutil.copy2(test_source, code_dir / test_source.name)

        logger.log("All validations passed; finalizing checksums and stable symlink")
        write_checksums(incomplete)
        final_dir = output_root / f"run_{timestamp}"
        if final_dir.exists():
            raise FileExistsError(final_dir)
        incomplete.rename(final_dir)
        temporary_link = output_root / f".latest.{os.getpid()}"
        if temporary_link.exists() or temporary_link.is_symlink():
            temporary_link.unlink()
        temporary_link.symlink_to(final_dir.name)
        os.replace(temporary_link, output_root / "latest")
        print(f"COMPLETED_RUN_DIR={final_dir}", flush=True)
        return 0
    except Exception as exc:
        logger.log(f"FAILED: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
