#!/usr/bin/env python3
"""Build reproducible, RNA-measurable genomic C-to-U negative examples.

The output rows are *computational measured negatives*, not experimentally
proven uneditable sites.  A site is eligible only when all six RNA-seq samples
have sufficient high-quality depth and none has a high-quality C>T (or G>A on
the reverse transcript strand) observation.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import os
import platform
import random
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pysam


SAMPLES = (
    "CU517_GC_T1",
    "CU517_GC_T2",
    "CU517_GC_T3",
    "CU517_GC_C1",
    "CU517_GC_C2",
    "CU517_GC_C3",
)
CANONICAL_CHROMS = tuple([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
BASES = "ACGT"
FILTER_FLAGS = 4 | 256 | 512 | 1024 | 2048
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


class Logger:
    def __init__(self, run_log: Path, command_log: Path):
        self.run_log = run_log
        self.command_log = command_log

    def log(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with self.run_log.open("a") as handle:
            handle.write(line + "\n")

    def command(self, command: Sequence[str]) -> None:
        with self.command_log.open("a") as handle:
            handle.write("$ " + " ".join(subprocess.list2cmdline([part]) for part in command) + "\n")


def open_text(path: Path, mode: str = "rt"):
    return gzip.open(path, mode) if path.suffix == ".gz" else path.open(mode)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(COMPLEMENT)[::-1]


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if start < 0 or end <= start:
            raise ValueError(f"Invalid interval: {(start, end)}")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def subtract_intervals(
    source: Sequence[tuple[int, int]], blockers: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Subtract sorted, merged blockers from sorted, merged source intervals."""
    result: list[tuple[int, int]] = []
    blocker_index = 0
    for start, end in source:
        cursor = start
        while blocker_index < len(blockers) and blockers[blocker_index][1] <= cursor:
            blocker_index += 1
        scan = blocker_index
        while scan < len(blockers) and blockers[scan][0] < end:
            block_start, block_end = blockers[scan]
            if block_start > cursor:
                result.append((cursor, min(block_start, end)))
            cursor = max(cursor, block_end)
            if cursor >= end:
                break
            scan += 1
        if cursor < end:
            result.append((cursor, end))
    return result


def build_unambiguous_gene_intervals(
    gtf_path: Path, chromosomes: Sequence[str], logger: Logger
) -> dict[str, list[tuple[int, int, str]]]:
    allowed = set(chromosomes)
    by_chrom_strand: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    gene_rows = 0
    with open_text(gtf_path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t", 8)
            if len(fields) != 9 or fields[2] != "gene" or fields[0] not in allowed:
                continue
            strand = fields[6]
            if strand not in {"+", "-"}:
                continue
            start0 = int(fields[3]) - 1
            end0 = int(fields[4])
            by_chrom_strand[(fields[0], strand)].append((start0, end0))
            gene_rows += 1
            if line_number % 5_000_000 == 0:
                logger.log(f"Scanned {line_number:,} GTF lines; retained {gene_rows:,} gene rows")

    result: dict[str, list[tuple[int, int, str]]] = {}
    for chrom in chromosomes:
        plus = merge_intervals(by_chrom_strand.get((chrom, "+"), []))
        minus = merge_intervals(by_chrom_strand.get((chrom, "-"), []))
        plus_only = [(start, end, "+") for start, end in subtract_intervals(plus, minus)]
        minus_only = [(start, end, "-") for start, end in subtract_intervals(minus, plus)]
        result[chrom] = sorted(plus_only + minus_only)
    retained_bases = sum(end - start for rows in result.values() for start, end, _ in rows)
    logger.log(
        f"Built unambiguous gene-strand intervals from {gene_rows:,} genes: "
        f"{sum(map(len, result.values())):,} intervals, {retained_bases:,} bases"
    )
    return result


def write_bed(intervals: Mapping[str, Sequence[tuple[int, int, str]]], path: Path) -> None:
    with path.open("w") as handle:
        for chrom in CANONICAL_CHROMS:
            for start, end, _ in intervals.get(chrom, []):
                handle.write(f"{chrom}\t{start}\t{end}\n")


class StrandIndex:
    def __init__(self, intervals: Mapping[str, Sequence[tuple[int, int, str]]]):
        self.rows = {chrom: list(values) for chrom, values in intervals.items()}
        self.starts = {
            chrom: [start for start, _, _ in values] for chrom, values in self.rows.items()
        }

    def get(self, chrom: str, position0: int) -> str | None:
        starts = self.starts.get(chrom)
        if not starts:
            return None
        index = bisect.bisect_right(starts, position0) - 1
        if index < 0:
            return None
        start, end, strand = self.rows[chrom][index]
        return strand if start <= position0 < end else None


def load_exclusion_centers(path: Path) -> dict[str, list[int]]:
    centers: dict[str, list[int]] = defaultdict(list)
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chrom", "position"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Existing label table lacks {sorted(required)}: {path}")
        for row in reader:
            centers[row["chrom"]].append(int(row["position"]))
    return {chrom: sorted(set(values)) for chrom, values in centers.items()}


def is_near(centers: Mapping[str, Sequence[int]], chrom: str, position1: int, distance: int) -> bool:
    values = centers.get(chrom, ())
    index = bisect.bisect_left(values, position1)
    return (index < len(values) and values[index] - position1 <= distance) or (
        index > 0 and position1 - values[index - 1] <= distance
    )


def load_variant_positions(
    paths: Sequence[Path], chromosomes: Sequence[str], logger: Logger
) -> dict[str, set[int]]:
    allowed = set(chromosomes)
    variants: dict[str, set[int]] = defaultdict(set)
    for path in paths:
        count_before = sum(map(len, variants.values()))
        with pysam.VariantFile(str(path)) as variant_file:
            for record in variant_file.fetch():
                if record.contig in allowed:
                    variants[record.contig].add(int(record.pos))
        added = sum(map(len, variants.values())) - count_before
        logger.log(f"Loaded {added:,} new variant positions from {path}")
    return variants


def candidate_context(
    reference: pysam.FastaFile, chrom: str, position1: int, strand: str, flank: int = 50
) -> tuple[str, str, str] | None:
    position0 = position1 - 1
    if position0 - flank < 0 or position0 + flank + 1 > reference.get_reference_length(chrom):
        return None
    genomic = reference.fetch(chrom, position0 - flank, position0 + flank + 1).upper()
    if len(genomic) != 2 * flank + 1 or any(base not in BASES for base in genomic):
        return None
    if strand == "+" and genomic[flank] == "C":
        return genomic, "C", "T"
    if strand == "-" and genomic[flank] == "G":
        oriented = reverse_complement(genomic)
        if oriented[flank] != "C":
            raise AssertionError("Reverse-complement orientation failed")
        return oriented, "G", "A"
    return None


def reservoir_depth_candidates(
    samtools: Path,
    bed: Path,
    bams: Sequence[Path],
    reference_path: Path,
    intervals: Mapping[str, Sequence[tuple[int, int, str]]],
    existing_centers: Mapping[str, Sequence[int]],
    existing_sequences: set[str],
    variant_positions: Mapping[str, set[int]],
    pool_size: int,
    min_mapq: int,
    min_baseq: int,
    min_coverage: int,
    exclusion_distance: int,
    seed: int,
    output_dir: Path,
    logger: Logger,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    command = [
        str(samtools),
        "depth",
        "-q",
        str(min_baseq),
        "-Q",
        str(min_mapq),
        "-s",
        "-b",
        str(bed),
        *map(str, bams),
    ]
    logger.command(command)
    stderr_path = output_dir / "depth.stderr.log"
    rng = random.Random(seed)
    reservoir: list[dict[str, object]] = []
    eligible_seen = 0
    counters: Counter[str] = Counter()
    strand_index = StrandIndex(intervals)
    with pysam.FastaFile(str(reference_path)) as reference, stderr_path.open("w") as stderr_handle:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            bufsize=1024 * 1024,
        )
        assert process.stdout is not None
        for line in process.stdout:
            counters["depth_output_rows"] += 1
            if counters["depth_output_rows"] % 2_000_000 == 0:
                logger.log(
                    f"Scanned {counters['depth_output_rows']:,} depth rows; "
                    f"eligible context rows {eligible_seen:,}"
                )
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2 + len(bams):
                raise RuntimeError(f"Unexpected samtools depth row: {line[:200]!r}")
            chrom, position_text = fields[:2]
            position1 = int(position_text)
            depths = [int(value) for value in fields[2:]]
            if any(depth < min_coverage for depth in depths):
                counters["insufficient_all_six_depth"] += 1
                continue
            strand = strand_index.get(chrom, position1 - 1)
            if strand is None:
                counters["ambiguous_or_missing_strand"] += 1
                continue
            if is_near(existing_centers, chrom, position1, exclusion_distance):
                counters["near_existing_candidate"] += 1
                continue
            if position1 in variant_positions.get(chrom, set()):
                counters["variant_catalogue_overlap"] += 1
                continue
            context_info = candidate_context(reference, chrom, position1, strand)
            if context_info is None:
                counters["not_orientable_c_or_bad_context"] += 1
                continue
            sequence, ref, alt = context_info
            if sequence in existing_sequences:
                counters["identical_existing_sequence"] += 1
                continue
            eligible_seen += 1
            candidate = {
                "chrom": chrom,
                "position": position1,
                "ref": ref,
                "alt": alt,
                "transcript_strand": 1 if strand == "+" else -1,
                "sequence_context": sequence,
                "sequence_length": len(sequence),
                "center_index": len(sequence) // 2,
                "gc_fraction": (sequence.count("G") + sequence.count("C")) / len(sequence),
                **{f"{sample}_depth_prefilter": depth for sample, depth in zip(SAMPLES, depths)},
            }
            if len(reservoir) < pool_size:
                reservoir.append(candidate)
            else:
                replacement = rng.randrange(eligible_seen)
                if replacement < pool_size:
                    reservoir[replacement] = candidate
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"samtools depth failed with exit {return_code}; see {stderr_path}")
    if len(reservoir) < pool_size:
        logger.log(f"Depth scan produced only {len(reservoir):,}/{pool_size:,} requested pool rows")
    else:
        logger.log(f"Depth scan retained a deterministic reservoir of {len(reservoir):,} rows")
    counters["eligible_context_rows_seen"] = eligible_seen
    counters["reservoir_rows"] = len(reservoir)
    return reservoir, dict(counters)


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


def exact_pileup_counts(
    candidates: Sequence[Mapping[str, object]],
    sample_bams: Mapping[str, Path],
    samtools: Path,
    reference: Path,
    min_mapq: int,
    min_baseq: int,
    output_dir: Path,
    logger: Logger,
) -> dict[tuple[str, int], dict[str, dict[str, int]]]:
    bed = output_dir / "candidate_pool_sites.bed"
    ordered = sorted(candidates, key=lambda row: (CANONICAL_CHROMS.index(str(row["chrom"])), int(row["position"])))
    with bed.open("w") as handle:
        for row in ordered:
            position1 = int(row["position"])
            handle.write(f"{row['chrom']}\t{position1 - 1}\t{position1}\n")
    by_key = {(str(row["chrom"]), int(row["position"])): row for row in candidates}
    all_counts: dict[tuple[str, int], dict[str, dict[str, int]]] = {
        key: {} for key in by_key
    }
    stderr_path = output_dir / "mpileup.stderr.log"
    command = [
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
        *[str(sample_bams[sample]) for sample in SAMPLES],
    ]
    logger.command(command)
    with stderr_path.open("w") as stderr_handle:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            bufsize=1024 * 1024,
        )
        assert process.stdout is not None
        observed = 0
        for line in process.stdout:
            parts = line.rstrip("\n").split("\t")
            expected_columns = 3 + 3 * len(SAMPLES)
            if len(parts) < expected_columns:
                raise RuntimeError(f"Unexpected multi-sample mpileup row: {line[:200]!r}")
            chrom, position_text, reference_base = parts[:3]
            key = (chrom, int(position_text))
            if key not in all_counts:
                raise RuntimeError(f"mpileup returned unexpected position: {key}")
            for sample_index, sample in enumerate(SAMPLES):
                bases_column = 4 + 3 * sample_index
                all_counts[key][sample] = parse_mpileup_bases(
                    parts[bases_column], reference_base
                )
            observed += 1
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Multi-sample samtools mpileup failed with exit {return_code}")
    logger.log(
        f"Exact multi-sample mpileup completed: {observed:,} positions x {len(SAMPLES)} samples"
    )
    for sample in SAMPLES:
        for counts in all_counts.values():
            counts.setdefault(sample, {base: 0 for base in BASES})

    pileup_path = output_dir / "candidate_pool_pileup_counts.tsv.gz"
    fields = [
        "chrom", "position", "ref", "alt", "transcript_strand", "sample",
        "usable_depth", "ref_count", "alt_count", "A_count", "C_count", "G_count", "T_count",
    ]
    with gzip.open(pileup_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in ordered:
            key = (str(row["chrom"]), int(row["position"]))
            ref, alt = str(row["ref"]), str(row["alt"])
            for sample in SAMPLES:
                counts = all_counts[key][sample]
                writer.writerow({
                    "chrom": key[0],
                    "position": key[1],
                    "ref": ref,
                    "alt": alt,
                    "transcript_strand": row["transcript_strand"],
                    "sample": sample,
                    "usable_depth": sum(counts.get(base, 0) for base in BASES),
                    "ref_count": counts.get(ref, 0),
                    "alt_count": counts.get(alt, 0),
                    **{f"{base}_count": counts.get(base, 0) for base in BASES},
                })
    return all_counts


def gc_bin(sequence: str) -> int:
    fraction = (sequence.count("G") + sequence.count("C")) / len(sequence)
    return min(9, int(fraction * 10))


def proportional_quotas(counts: Mapping[tuple[int, int], int], total: int) -> dict[tuple[int, int], int]:
    denominator = sum(counts.values())
    if denominator == 0:
        raise ValueError("Cannot derive quotas from an empty positive set")
    raw = {key: total * value / denominator for key, value in counts.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(raw, key=lambda key: (raw[key] - quotas[key], key), reverse=True)
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def select_final_negatives(
    passing: Sequence[dict[str, object]],
    positive_rows: Sequence[Mapping[str, str]],
    target: int,
    min_center_distance: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    positive_strata = Counter(
        (int(row["transcript_strand"]), gc_bin(row["seq"]))
        for row in positive_rows
        if float(row["label"]) > 0
    )
    quotas = proportional_quotas(positive_strata, target)
    available: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in passing:
        available[(int(row["transcript_strand"]), gc_bin(str(row["sequence_context"])))].append(row)
    rng = random.Random(seed)
    for rows in available.values():
        rng.shuffle(rows)

    selected: list[dict[str, object]] = []
    selected_centers: dict[str, list[int]] = defaultdict(list)
    selected_sequences: set[str] = set()

    def try_add(row: dict[str, object]) -> bool:
        chrom, position1 = str(row["chrom"]), int(row["position"])
        sequence = str(row["sequence_context"])
        if sequence in selected_sequences or is_near(selected_centers, chrom, position1, min_center_distance):
            return False
        bisect.insort(selected_centers[chrom], position1)
        selected_sequences.add(sequence)
        selected.append(row)
        return True

    deficits = {}
    used_ids: set[tuple[str, int]] = set()
    for stratum in sorted(quotas):
        wanted = quotas[stratum]
        if wanted <= 0:
            continue
        added = 0
        for row in available.get(stratum, []):
            if try_add(row):
                used_ids.add((str(row["chrom"]), int(row["position"])))
                added += 1
                if added == wanted:
                    break
        if added < wanted:
            deficits[str(stratum)] = wanted - added

    if len(selected) < target:
        leftovers = [
            row for row in passing
            if (str(row["chrom"]), int(row["position"])) not in used_ids
        ]
        rng.shuffle(leftovers)
        for row in leftovers:
            if try_add(row) and len(selected) == target:
                break
    if len(selected) != target:
        raise RuntimeError(
            f"Only {len(selected):,}/{target:,} non-overlapping measured negatives could be selected"
        )
    selected.sort(key=lambda row: (CANONICAL_CHROMS.index(str(row["chrom"])), int(row["position"])))
    observed_strata = Counter(
        (int(row["transcript_strand"]), gc_bin(str(row["sequence_context"]))) for row in selected
    )
    return selected, {
        "target_quotas": {str(key): value for key, value in sorted(quotas.items())},
        "observed_strata": {str(key): value for key, value in sorted(observed_strata.items())},
        "quota_deficits_filled_from_other_strata": deficits,
    }


def input_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--gtf", required=True)
    parser.add_argument("--samtools", required=True)
    parser.add_argument("--existing-labels", required=True)
    parser.add_argument("--v1-manifest", required=True)
    parser.add_argument("--variant-vcf", action="append", default=[])
    parser.add_argument("--bam", action="append", required=True, help="SAMPLE=/path/to.bam")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chromosomes", default=",".join(CANONICAL_CHROMS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=100000)
    parser.add_argument(
        "--verification-size",
        type=int,
        default=None,
        help="Non-overlapping pool rows sent to exact mpileup; default=min(pool, 3*target)",
    )
    parser.add_argument("--target-added-negatives", type=int, default=6300)
    parser.add_argument("--min-mapq", type=int, default=30)
    parser.add_argument("--min-baseq", type=int, default=20)
    parser.add_argument("--min-coverage", type=int, default=20)
    parser.add_argument("--existing-window-exclusion", type=int, default=100)
    parser.add_argument("--negative-center-distance", type=int, default=100)
    args = parser.parse_args()

    reference = Path(args.reference).resolve()
    gtf = Path(args.gtf).resolve()
    samtools = Path(args.samtools).resolve()
    existing_labels = Path(args.existing_labels).resolve()
    v1_manifest = Path(args.v1_manifest).resolve()
    variant_vcfs = [Path(path).resolve() for path in args.variant_vcf]
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = Logger(output_dir / "run.log", output_dir / "commands.log")

    chromosomes = tuple(value for value in args.chromosomes.split(",") if value)
    if not chromosomes or any(chrom not in CANONICAL_CHROMS for chrom in chromosomes):
        raise ValueError("--chromosomes must be a non-empty subset of canonical chromosomes")
    if args.pool_size < args.target_added_negatives:
        raise ValueError("--pool-size must be at least --target-added-negatives")
    sample_bams: dict[str, Path] = {}
    for value in args.bam:
        sample, separator, path_text = value.partition("=")
        if not separator or sample not in SAMPLES or sample in sample_bams:
            raise ValueError(f"Invalid or duplicate --bam value: {value}")
        sample_bams[sample] = Path(path_text).resolve()
    if set(sample_bams) != set(SAMPLES):
        raise ValueError(f"Expected BAMs for {SAMPLES}; observed {sorted(sample_bams)}")
    required_paths = [reference, Path(str(reference) + ".fai"), gtf, samtools, existing_labels, v1_manifest, *variant_vcfs, *sample_bams.values()]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    logger.log("Starting measured-negative construction; no input files will be modified")
    intervals = build_unambiguous_gene_intervals(gtf, chromosomes, logger)
    bed = output_dir / "unambiguous_gene_regions.bed"
    write_bed(intervals, bed)
    existing_centers = load_exclusion_centers(existing_labels)
    with open_text(v1_manifest) as handle:
        v1_rows = list(csv.DictReader(handle, delimiter="\t"))
    existing_sequences = {row["seq"] for row in v1_rows}
    positive_rows = [row for row in v1_rows if float(row["label"]) > 0]
    variants = load_variant_positions(variant_vcfs, chromosomes, logger)

    pool, depth_audit = reservoir_depth_candidates(
        samtools=samtools,
        bed=bed,
        bams=[sample_bams[sample] for sample in SAMPLES],
        reference_path=reference,
        intervals=intervals,
        existing_centers=existing_centers,
        existing_sequences=existing_sequences,
        variant_positions=variants,
        pool_size=args.pool_size,
        min_mapq=args.min_mapq,
        min_baseq=args.min_baseq,
        min_coverage=args.min_coverage,
        exclusion_distance=args.existing_window_exclusion,
        seed=args.seed,
        output_dir=output_dir,
        logger=logger,
    )
    pool_path = output_dir / "candidate_pool.tsv.gz"
    if not pool:
        raise RuntimeError("No depth-qualified candidates were found")
    with gzip.open(pool_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pool[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(pool)

    verification_size = args.verification_size or min(
        args.pool_size, args.target_added_negatives * 3
    )
    if not args.target_added_negatives <= verification_size <= len(pool):
        raise ValueError(
            "verification size must be between target-added-negatives and the observed pool size"
        )
    verification, verification_selection_audit = select_final_negatives(
        pool,
        positive_rows,
        verification_size,
        args.negative_center_distance,
        args.seed + 1,
    )
    verification_path = output_dir / "verification_candidates.tsv.gz"
    with gzip.open(verification_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(verification[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(verification)
    logger.log(
        f"Selected {len(verification):,} non-overlapping candidates for exact six-sample mpileup"
    )

    all_counts = exact_pileup_counts(
        verification,
        sample_bams,
        samtools,
        reference,
        args.min_mapq,
        args.min_baseq,
        output_dir,
        logger,
    )
    passing: list[dict[str, object]] = []
    exact_failures: Counter[str] = Counter()
    for row in verification:
        key = (str(row["chrom"]), int(row["position"]))
        ref, alt = str(row["ref"]), str(row["alt"])
        per_sample = all_counts[key]
        depths = {sample: sum(per_sample[sample].get(base, 0) for base in BASES) for sample in SAMPLES}
        alt_counts = {sample: per_sample[sample].get(alt, 0) for sample in SAMPLES}
        if any(depth < args.min_coverage for depth in depths.values()):
            exact_failures["exact_depth_below_minimum"] += 1
            continue
        if any(count != 0 for count in alt_counts.values()):
            exact_failures["observed_target_alt"] += 1
            continue
        passing.append({
            **row,
            **{f"{sample}_usable_depth": depths[sample] for sample in SAMPLES},
            **{f"{sample}_alt_count": alt_counts[sample] for sample in SAMPLES},
        })
    logger.log(
        f"Exact negative definition passed {len(passing):,}/{len(verification):,} verification sites; "
        f"failures={dict(exact_failures)}"
    )

    selected, selection_audit = select_final_negatives(
        passing,
        positive_rows,
        args.target_added_negatives,
        args.negative_center_distance,
        args.seed,
    )
    negative_path = output_dir / "measured_negatives.tsv.gz"
    output_fields = [
        "chrom", "position", "ref", "alt", "transcript_strand", "sequence_context",
        "sequence_length", "center_index", "center_base", "orientation_qc",
        "transcript_oriented_ref", "transcript_oriented_alt", "corrected_editing_efficiency",
        "raw_edit_rate_difference", "training_eligible", "label_confidence", "exclusion_reason",
        "sample_source", "negative_definition", "gc_fraction", "minimum_usable_depth",
        "minimum_covered_replicates_per_group", "public_293t_variant_catalogue_overlap",
        *[f"{sample}_usable_depth" for sample in SAMPLES],
        *[f"{sample}_alt_count" for sample in SAMPLES],
    ]
    with gzip.open(negative_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, delimiter="\t")
        writer.writeheader()
        for row in selected:
            writer.writerow({
                **{field: row.get(field, "") for field in output_fields},
                "center_base": "C",
                "orientation_qc": "pass",
                "transcript_oriented_ref": "C",
                "transcript_oriented_alt": "T",
                "corrected_editing_efficiency": 0.0,
                "raw_edit_rate_difference": 0.0,
                "training_eligible": 1,
                "label_confidence": "high_computational_negative",
                "exclusion_reason": "none",
                "sample_source": "measured_genomic_negative",
                "negative_definition": "all_six_depth_ge_min_and_all_six_target_alt_count_zero",
                "minimum_usable_depth": args.min_coverage,
                "minimum_covered_replicates_per_group": 3,
                "public_293t_variant_catalogue_overlap": 0,
            })

    if len(selected) != args.target_added_negatives:
        raise AssertionError("Selected negative row count changed during output")
    if any(str(row["sequence_context"])[50] != "C" for row in selected):
        raise AssertionError("A selected sequence is not centered on C")
    if len({str(row["sequence_context"]) for row in selected}) != len(selected):
        raise AssertionError("Selected negative sequences are duplicated")

    summary = {
        "status": "complete",
        "scientific_label": "high-confidence computational measured negative; not proven uneditable",
        "seed": args.seed,
        "candidate_pool_rows": len(pool),
        "verification_candidate_rows": len(verification),
        "exact_definition_pass_rows": len(passing),
        "selected_negative_rows": len(selected),
        "target_added_negatives": args.target_added_negatives,
        "definition": {
            "minimum_mapq": args.min_mapq,
            "minimum_baseq": args.min_baseq,
            "minimum_usable_depth_each_of_six_samples": args.min_coverage,
            "maximum_target_alt_count_each_of_six_samples": 0,
            "existing_candidate_center_exclusion_bp": args.existing_window_exclusion,
            "negative_center_minimum_distance_bp": args.negative_center_distance,
            "transcript_orientation": "unambiguous GENCODE gene strand",
            "variant_exclusion": [str(path) for path in variant_vcfs],
            "gc_and_strand_sampling": "quotas derived from the 7,864 positive v1 rows; deficits filled from remaining strata",
        },
        "depth_scan_audit": depth_audit,
        "exact_failures": dict(exact_failures),
        "selection_audit": selection_audit,
        "verification_selection_audit": verification_selection_audit,
        "inputs": {
            "reference": input_metadata(reference),
            "gtf": input_metadata(gtf),
            "existing_labels": {**input_metadata(existing_labels), "sha256": sha256(existing_labels)},
            "v1_manifest": {**input_metadata(v1_manifest), "sha256": sha256(v1_manifest)},
            "variant_vcfs": [input_metadata(path) for path in variant_vcfs],
            "bams": {sample: input_metadata(path) for sample, path in sample_bams.items()},
        },
        "outputs": {
            "gene_regions_bed": {"path": str(bed), "sha256": sha256(bed)},
            "candidate_pool": {"path": str(pool_path), "sha256": sha256(pool_path)},
            "verification_candidates": {
                "path": str(verification_path),
                "sha256": sha256(verification_path),
            },
            "pileup_counts": {
                "path": str(output_dir / "candidate_pool_pileup_counts.tsv.gz"),
                "sha256": sha256(output_dir / "candidate_pool_pileup_counts.tsv.gz"),
            },
            "measured_negatives": {"path": str(negative_path), "sha256": sha256(negative_path)},
        },
        "versions": {
            "python": platform.python_version(),
            "pysam": pysam.__version__,
            "samtools": subprocess.run([str(samtools), "--version"], text=True, capture_output=True, check=True).stdout.splitlines()[0],
        },
        "arguments": vars(args),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    checksum_paths = [
        bed,
        pool_path,
        verification_path,
        output_dir / "candidate_pool_pileup_counts.tsv.gz",
        negative_path,
        summary_path,
    ]
    with (output_dir / "checksums.sha256").open("w") as handle:
        for path in checksum_paths:
            handle.write(f"{sha256(path)}  {path.name}\n")
    logger.log(f"Completed measured-negative construction: {negative_path}")


if __name__ == "__main__":
    main()
