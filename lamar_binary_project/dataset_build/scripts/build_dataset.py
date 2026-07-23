#!/usr/bin/env python3
"""Reproducible LAMAR computational-positive/negative dataset builder.

The code is deliberately standard-library + pysam so the declared igem
environment is sufficient.  Labels are computational screening labels, not
experimental validation.  Every count is produced by ``multi_pileup`` below.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pysam


PROJECT = Path("${IGEM_DATA_ROOT}/CU5.17_EGFP_GC_paper")
REFERENCE = Path("${IGEM_DATA_ROOT}/GRCh38.primary_assembly.genome.fa")
GTF = Path("${IGEM_DATA_ROOT}/gencode.v50.primary_assembly.annotation.gtf")
SAMTOOLS = Path("${IGEM_ENV}/bin/samtools")
PYTHON = Path("${IGEM_ENV}/bin/python")
SCIPY_PYTHON = Path("${LAMAR_ENV}/bin/python")
BROAD = PROJECT / "final/CU5.17_EGFP_GC.site_matrix.tsv.gz"
VEP = PROJECT / "vep/CU5.17_EGFP_GC.vep.tsv"
OLD_LABELS = PROJECT / "lamar_background_corrected/run_20260715T214930Z/background_corrected_labels.tsv.gz"
VCFS = (
    Path("${IGEM_DATA_ROOT}/293T_CG_GRCh38_retry/293T_CG.GRCh38.PASS.biallelic.SNV.vcf.gz"),
    Path("${IGEM_DATA_ROOT}/HEK293T_public_WGS_3runs/vcf/HEK293T_3runs.union.SNV.vcf.gz"),
)
SAMPLES = (
    ("CU517_GC_T1", "treated", 1),
    ("CU517_GC_T2", "treated", 2),
    ("CU517_GC_T3", "treated", 3),
    ("CU517_GC_C1", "control", 1),
    ("CU517_GC_C2", "control", 2),
    ("CU517_GC_C3", "control", 3),
)
SAMPLE_NAMES = tuple(row[0] for row in SAMPLES)
TREATED = SAMPLE_NAMES[:3]
CONTROLS = SAMPLE_NAMES[3:]
BAMS = {
    sample: PROJECT / f"bam/markduplicates/{sample}.markduplicates.bam"
    for sample in SAMPLE_NAMES
}
CANONICAL_CHROMS = tuple([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
BASES = "ACGT"
FILTER_FLAGS = 4 | 256 | 512 | 1024 | 2048
RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")
MIN_MAPQ = 30
MIN_BASEQ = 20
MIN_DEPTH = 20
MAX_DEPTH = 1_000_000
SEED = 20260721
POSITIVE_THRESHOLD = 0.10
CONTROL_MAX = 0.02
FILTER_DESCRIPTION = (
    "MAPQ>=30;BQ>=20;exclude_unmapped,secondary,qcfail,duplicate,supplementary;"
    "collapse_overlapping_mates;include_orphans;BAQ_off;ACGT_only;max_depth=1000000"
)
COMMON_FIELDS = [
    "chrom", "position", "genomic_ref", "genomic_alt", "transcript_strand",
    "transcript_oriented_ref", "transcript_oriented_alt", "genomic_key",
    "gene_id", "gene_name", "transcript_ids", "region_type",
    "sequence_context", "sequence_length", "center_index", "center_base",
    "gc_fraction", "center_5mer", "sequence_entropy_log2_single_base_101nt",
    "max_homopolymer_run", "dinucleotide_tandem_max_coverage_ratio",
    "dinucleotide_tandem_best_phase", "dinucleotide_tandem_best_motif",
    "low_complexity_entropy_trigger", "low_complexity_homopolymer_trigger",
    "low_complexity_dinucleotide_trigger", "low_complexity_triggered_rules",
    "low_complexity_qc", "mappability", "mappability_method", "mappability_qc",
]
COUNT_FIELDS: list[str] = []
for _sample in SAMPLE_NAMES:
    COUNT_FIELDS.extend([
        f"{_sample}_usable_depth", f"{_sample}_ref_count",
        f"{_sample}_target_alt_count", f"{_sample}_other_alt_count",
        f"{_sample}_target_edit_rate",
    ])


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Logger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.log_path = run_dir / "logs/build_dataset.log"
        self.command_path = run_dir / "logs/commands.log"

    def log(self, message: str) -> None:
        line = f"[{utcnow()}] {message}"
        print(line, flush=True)
        with self.log_path.open("a") as handle:
            handle.write(line + "\n")

    def command(self, command: Sequence[str]) -> None:
        with self.command_path.open("a") as handle:
            handle.write("$ " + " ".join(subprocess.list2cmdline([part]) for part in command) + "\n")


def open_text(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode, newline="")
    return path.open(mode, newline="")


def write_tsv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> int:
    count = 0
    with open_text(path, "wt") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=list(fields), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})
            count += 1
    return count


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open_text(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.12g}"
    return str(value)


def revcomp(sequence: str) -> str:
    return sequence.translate(RC)[::-1].upper()


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def mad(values: Sequence[float]) -> float | None:
    if not values:
        return None
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def dinucleotide_tandem_metric(sequence: str) -> tuple[float, int, str]:
    """Maximum same-2mer tandem-run coverage across phase 0 and phase 1.

    Each phase partitions the full 101-nt sequence into non-overlapping 2-mers
    beginning at index 0 or 1.  Consecutive identical 2-mers form a tandem
    run.  The reported coverage is ``2 * run_2mer_count / 101``.  Ties are
    resolved deterministically by phase and then lexicographic motif.
    """
    best_bases, best_phase, best_motif = 0, 0, "NA"
    for phase in (0, 1):
        units = [sequence[index:index + 2] for index in range(phase, len(sequence) - 1, 2)]
        run_motif, run_units = None, 0
        for motif in units:
            if motif == run_motif:
                run_units += 1
            else:
                run_motif, run_units = motif, 1
            bases = 2 * run_units
            if (
                bases > best_bases
                or (bases == best_bases and phase < best_phase)
                or (bases == best_bases and phase == best_phase and motif < best_motif)
            ):
                best_bases, best_phase, best_motif = bases, phase, motif
    return best_bases / len(sequence), best_phase, best_motif


def sequence_metrics(sequence: str) -> dict[str, object]:
    if len(sequence) != 101 or any(base not in BASES for base in sequence):
        raise ValueError("sequence_metrics requires a complete 101-nt A/C/G/T window")
    counts = Counter(sequence)
    entropy = -sum((n / len(sequence)) * math.log2(n / len(sequence)) for n in counts.values())
    max_run = max(len(match.group(0)) for match in re.finditer(r"(.)\1*", sequence))
    dinucleotide_ratio, dinucleotide_phase, dinucleotide_motif = dinucleotide_tandem_metric(sequence)
    entropy_trigger = entropy < 1.20
    homopolymer_trigger = max_run >= 20
    dinucleotide_trigger = dinucleotide_ratio >= 0.80
    triggers = []
    if entropy_trigger:
        triggers.append("entropy_lt_1.20")
    if homopolymer_trigger:
        triggers.append("homopolymer_run_ge_20")
    if dinucleotide_trigger:
        triggers.append("dinucleotide_tandem_coverage_ge_0.80")
    return {
        "sequence_entropy_log2_single_base_101nt": entropy,
        "max_homopolymer_run": max_run,
        "dinucleotide_tandem_max_coverage_ratio": dinucleotide_ratio,
        "dinucleotide_tandem_best_phase": dinucleotide_phase,
        "dinucleotide_tandem_best_motif": dinucleotide_motif,
        "low_complexity_entropy_trigger": entropy_trigger,
        "low_complexity_homopolymer_trigger": homopolymer_trigger,
        "low_complexity_dinucleotide_trigger": dinucleotide_trigger,
        "low_complexity_triggered_rules": ";".join(triggers) if triggers else "none",
        "low_complexity_qc": "FAIL_LOW_COMPLEXITY" if triggers else "PASS",
    }


def context_record(reference: pysam.FastaFile, chrom: str, position1: int, strand: str) -> dict[str, object] | None:
    position0 = position1 - 1
    try:
        length = reference.get_reference_length(chrom)
    except KeyError:
        return None
    if position0 < 50 or position0 + 51 > length:
        return None
    genomic = reference.fetch(chrom, position0 - 50, position0 + 51).upper()
    if len(genomic) != 101 or any(base not in BASES for base in genomic):
        return None
    expected = "C" if strand == "+" else "G"
    if genomic[50] != expected:
        return None
    sequence = genomic if strand == "+" else revcomp(genomic)
    if sequence[50] != "C":
        return None
    return {
        "sequence_context": sequence, "sequence_length": 101, "center_index": 50,
        "center_base": "C", "gc_fraction": (sequence.count("G") + sequence.count("C")) / 101,
        "center_5mer": sequence[48:53], **sequence_metrics(sequence),
    }


def parse_mpileup_bases(raw: str, reference_base: str) -> dict[str, int]:
    counts = {base: 0 for base in BASES}
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
            if match is None:
                raise ValueError(f"Malformed mpileup indel: {raw!r}")
            text = match.group(1)
            index += 1 + len(text) + int(text)
            continue
        if char in ".,":
            base = reference_base.upper()
            if base in counts:
                counts[base] += 1
        elif char.upper() in counts:
            counts[char.upper()] += 1
        elif char in "*#<>Nn":
            pass
        else:
            raise ValueError(f"Unexpected mpileup character {char!r}")
        index += 1
    return counts


def pileup_command(bed: Path, region: str | None = None) -> list[str]:
    command = [
        str(SAMTOOLS), "mpileup", "-B", "-A", "-q", str(MIN_MAPQ), "-Q", str(MIN_BASEQ),
        "--ff", str(FILTER_FLAGS), "-d", str(MAX_DEPTH), "-l", str(bed), "-f", str(REFERENCE),
    ]
    if region:
        command.extend(["-r", region])
    command.extend(str(BAMS[sample]) for sample in SAMPLE_NAMES)
    return command


def multi_pileup(
    bed: Path, logger: Logger, stderr_path: Path, region: str | None = None
) -> Iterable[tuple[str, int, str, dict[str, dict[str, int]]]]:
    """The sole read-counting implementation used by positives and negatives."""
    command = pileup_command(bed, region)
    logger.command(command)
    with stderr_path.open("w") as stderr_handle:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=stderr_handle, bufsize=2**20)
        assert process.stdout is not None
        for line in process.stdout:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 + 3 * len(SAMPLES):
                raise RuntimeError(f"Malformed multi-sample mpileup row: {line[:240]!r}")
            chrom, position_text, reference_base = fields[:3]
            counts = {}
            for sample_index, sample in enumerate(SAMPLE_NAMES):
                counts[sample] = parse_mpileup_bases(fields[4 + sample_index * 3], reference_base)
            yield chrom, int(position_text), reference_base.upper(), counts
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"samtools mpileup failed ({return_code}); see {stderr_path}")


def empty_counts() -> dict[str, int]:
    return {base: 0 for base in BASES}


def add_counts(row: dict[str, object], per_sample: Mapping[str, Mapping[str, int]], genomic_ref: str, genomic_alt: str) -> None:
    for sample in SAMPLE_NAMES:
        counts = per_sample.get(sample, empty_counts())
        depth = sum(int(counts.get(base, 0)) for base in BASES)
        ref_count = int(counts.get(genomic_ref, 0))
        target_count = int(counts.get(genomic_alt, 0))
        row[f"{sample}_usable_depth"] = depth
        row[f"{sample}_ref_count"] = ref_count
        row[f"{sample}_target_alt_count"] = target_count
        row[f"{sample}_other_alt_count"] = depth - ref_count - target_count
        row[f"{sample}_target_edit_rate"] = target_count / depth if depth else 0.0


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    if min(a, b, c, d) < 0:
        raise ValueError("negative Fisher count")
    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2
    if total == 0:
        return 1.0
    low, high = max(0, row1 - (total - col1)), min(row1, col1)
    denominator = math.comb(total, row1)
    def probability(x: int) -> float:
        return math.comb(col1, x) * math.comb(total - col1, row1 - x) / denominator
    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(low, high + 1) if probability(x) <= observed + 1e-15))


def bh(values: Sequence[float | None]) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    ranked = sorted((float(value), index) for index, value in enumerate(values) if value is not None)
    running = 1.0
    for reverse_rank in range(len(ranked) - 1, -1, -1):
        value, index = ranked[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, value * len(ranked) / rank)
        output[index] = running
    return output


def batch_fisher_pvalues(rows: Sequence[Mapping[str, object]], run_dir: Path, logger: Logger) -> list[float | None]:
    input_path = run_dir / "work/positive_fisher_input.tsv"
    output_path = run_dir / "work/positive_fisher_output.tsv"
    eligible_indexes = []
    with input_path.open("w", newline="") as handle:
        fields = ["row_index", "treated_alt", "treated_ref", "control_alt", "control_ref"]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            if row["raw_difference"] is None:
                continue
            eligible_indexes.append(index)
            writer.writerow({
                "row_index": index,
                "treated_alt": row["pooled_treated_target_alt_count"],
                "treated_ref": row["pooled_treated_ref_count"],
                "control_alt": row["pooled_control_target_alt_count"],
                "control_ref": row["pooled_control_ref_count"],
            })
    helper = Path(__file__).with_name("fisher_batch.py")
    command = [str(SCIPY_PYTHON), str(helper), str(input_path), str(output_path)]
    logger.command(command)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Batch SciPy Fisher failed ({result.returncode}): {result.stderr[-2000:]}")
    pvalues: list[float | None] = [None] * len(rows)
    observed_indexes = []
    for row in read_tsv(output_path):
        index = int(row["row_index"])
        pvalues[index] = float(row["pvalue"])
        observed_indexes.append(index)
    if observed_indexes != eligible_indexes:
        raise RuntimeError("Batch Fisher row-index completeness/order mismatch")
    logger.log(f"Batch SciPy Fisher completed for {len(eligible_indexes):,} pooled-read screening tables")
    return pvalues


def load_variant_positions(chromosomes: set[str] | None = None) -> dict[str, set[int]]:
    result: dict[str, set[int]] = defaultdict(set)
    for path in VCFS:
        with pysam.VariantFile(str(path)) as handle:
            if chromosomes is None:
                iterators = [handle.fetch()]
            else:
                iterators = [handle.fetch(chrom) for chrom in sorted(chromosomes) if chrom in handle.header.contigs]
            for iterator in iterators:
                for record in iterator:
                    result[record.contig].add(int(record.pos))
    return result


def load_vep() -> dict[tuple[str, int, str, str], dict[str, str]]:
    annotations: dict[tuple[str, int, str, str], list[dict[str, str]]] = defaultdict(list)
    with VEP.open() as handle:
        header = None
        for line in handle:
            if line.startswith("#Uploaded_variation"):
                header = line[1:].rstrip("\n").split("\t")
                break
        if header is None:
            raise RuntimeError("VEP header not found")
        reader = csv.DictReader(handle, fieldnames=header, delimiter="\t")
        for row in reader:
            match = re.match(r"REDI_(.+)_(\d+)_([ACGT])_([ACGT])$", row["Uploaded_variation"])
            if match:
                key = (match.group(1), int(match.group(2)), match.group(3), match.group(4))
                annotations[key].append(row)
    result = {}
    for key, rows in annotations.items():
        strand = "1" if key[2] == "C" else "-1"
        matching = [row for row in rows if row.get("STRAND") == strand] or rows
        result[key] = {
            "gene_id": ",".join(sorted({row.get("Gene", "-") for row in matching if row.get("Gene", "-") != "-"})) or "NA",
            "gene_name": ",".join(sorted({row.get("SYMBOL", "-") for row in matching if row.get("SYMBOL", "-") != "-"})) or "NA",
            "transcript_ids": ",".join(sorted({row.get("Feature", "-") for row in matching if row.get("Feature", "-") != "-"})) or "NA",
            "region_type": ",".join(sorted({row.get("Consequence", "-") for row in matching if row.get("Consequence", "-") != "-"})) or "NA",
        }
    return result


def stage_positive(run_dir: Path, logger: Logger, output_prefix: str = "") -> None:
    output = run_dir / (output_prefix + "positives_all_recounted.tsv.gz")
    if output.exists() and (run_dir / "work/positive.done").exists() and not output_prefix:
        logger.log("Positive recount already complete; validated checkpoint present")
        return
    candidates = read_tsv(BROAD)
    if len(candidates) != 9930:
        raise AssertionError(f"Broad matrix expected 9,930 rows, observed {len(candidates)}")
    by_center: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        by_center[(candidate["chrom"], int(candidate["position"]))].append(candidate)
    counts_by_center = {}
    centers_by_chrom: dict[str, list[int]] = defaultdict(list)
    for chrom, position in by_center:
        centers_by_chrom[chrom].append(position)
    for chrom, positions in sorted(centers_by_chrom.items()):
        safe_chrom = re.sub(r"[^A-Za-z0-9_.-]", "_", chrom)
        chunks: list[list[int]] = []
        for position in sorted(positions):
            if not chunks or position - chunks[-1][0] > 20_000_000 or position - chunks[-1][-1] > 1_000_000:
                chunks.append([position])
            else:
                chunks[-1].append(position)
        for chunk_index, chunk in enumerate(chunks):
            bed = run_dir / "work" / f"{output_prefix}positive_sites.{safe_chrom}.{chunk_index:03d}.bed"
            with bed.open("w") as handle:
                for position in chunk:
                    handle.write(f"{chrom}\t{position - 1}\t{position}\n")
            region = f"{chrom}:{max(1, chunk[0]-50)}-{chunk[-1]+50}"
            for observed_chrom, position, _, counts in multi_pileup(
                bed, logger,
                run_dir / "logs" / f"{output_prefix}positive_mpileup.{safe_chrom}.{chunk_index:03d}.stderr.log",
                region=region,
            ):
                counts_by_center[(observed_chrom, position)] = counts
    variants = load_variant_positions(set(row["chrom"] for row in candidates))
    annotations = load_vep()
    rows: list[dict[str, object]] = []
    with pysam.FastaFile(str(REFERENCE)) as reference:
        for candidate in candidates:
            chrom, position = candidate["chrom"], int(candidate["position"])
            genomic_ref, genomic_alt = candidate["ref"], candidate["alt"]
            strand = "+" if candidate["vep_strand"] == "1" else "-" if candidate["vep_strand"] == "-1" else "?"
            row: dict[str, object] = {
                "chrom": chrom, "position": position, "genomic_ref": genomic_ref,
                "genomic_alt": genomic_alt, "transcript_strand": strand,
                "transcript_oriented_ref": "C", "transcript_oriented_alt": "T",
                "genomic_key": f"{chrom}:{position}:{strand}:C:T",
                "gene_id": "NA", "gene_name": "NA", "transcript_ids": "NA", "region_type": "NA",
                "mappability": "NA_RESOURCE_MISSING",
                "mappability_method": "NA_RESOURCE_MISSING;MAPQ30_PILEUP_FILTER_IS_NOT_A_MAPPABILITY_TRACK",
                "mappability_qc": "NA_RESOURCE_MISSING",
                "WGS_center_overlap": position in variants.get(chrom, set()),
                "pooled_read_screening_limitation": "screening_only_reads_are_not_independent_biological_replicates",
            }
            row.update(annotations.get((chrom, position, genomic_ref, genomic_alt), {}))
            context = context_record(reference, chrom, position, strand) if strand in {"+", "-"} else None
            if context:
                row.update(context)
                row["orientation_qc"] = "PASS"
            else:
                row.update({field: "NA" for field in COMMON_FIELDS if field not in row})
                row["orientation_qc"] = "FAIL_DIRECTION_REFERENCE_OR_CONTEXT"
            add_counts(row, counts_by_center.get((chrom, position), {}), genomic_ref, genomic_alt)
            treated_rates = [float(row[f"{sample}_target_edit_rate"]) for sample in TREATED if int(row[f"{sample}_usable_depth"]) >= MIN_DEPTH]
            control_rates = [float(row[f"{sample}_target_edit_rate"]) for sample in CONTROLS if int(row[f"{sample}_usable_depth"]) >= MIN_DEPTH]
            row["treated_covered_replicates"] = len(treated_rates)
            row["control_covered_replicates"] = len(control_rates)
            row["all_six_depth_ge_20"] = len(treated_rates) == 3 and len(control_rates) == 3
            row["treated_median"] = median(treated_rates)
            row["control_median"] = median(control_rates)
            row["treated_MAD"] = mad(treated_rates)
            row["control_MAD"] = mad(control_rates)
            sufficient = len(treated_rates) >= 2 and len(control_rates) >= 2
            raw = float(row["treated_median"]) - float(row["control_median"]) if sufficient else None
            row["raw_difference"] = raw
            row["corrected_editing_efficiency"] = max(raw, 0.0) if raw is not None else None
            pooled = {}
            for group, samples in (("treated", TREATED), ("control", CONTROLS)):
                covered = [sample for sample in samples if int(row[f"{sample}_usable_depth"]) >= MIN_DEPTH]
                pooled[f"pooled_{group}_ref_count"] = sum(int(row[f"{sample}_ref_count"]) for sample in covered)
                pooled[f"pooled_{group}_target_alt_count"] = sum(int(row[f"{sample}_target_alt_count"]) for sample in covered)
            row.update(pooled)
            row["fisher_exact_screening_pvalue"] = None
            rows.append(row)
    pvalues = batch_fisher_pvalues(rows, run_dir, logger)
    for row, pvalue in zip(rows, pvalues):
        row["fisher_exact_screening_pvalue"] = pvalue
    adjusted = bh(pvalues)
    seen_keys = Counter(row["genomic_key"] for row in rows)
    for row, fdr in zip(rows, adjusted):
        row["BH_FDR"] = fdr
        corrected = row["corrected_editing_efficiency"]
        coverage_qc = int(row["treated_covered_replicates"]) >= 2 and int(row["control_covered_replicates"]) >= 2
        basic_without_low_complexity = (
            corrected is not None and float(corrected) > POSITIVE_THRESHOLD and coverage_qc
            and row["orientation_qc"] == "PASS" and not row["WGS_center_overlap"]
            and seen_keys[row["genomic_key"]] == 1
        )
        row["positive_main_without_low_complexity_filter"] = (
            basic_without_low_complexity and float(row["control_median"]) <= CONTROL_MAX
        )
        basic = basic_without_low_complexity and row["low_complexity_qc"] == "PASS"
        row["positive_sensitivity"] = basic
        row["positive_main"] = basic and float(row["control_median"]) <= CONTROL_MAX
        row["positive_high_confidence"] = (
            row["positive_main"] and row["all_six_depth_ge_20"] and fdr is not None and fdr < 0.05
            and float(row["treated_MAD"]) <= 0.05 and float(row["control_MAD"]) <= 0.02
        )
        reasons = []
        if corrected is None or float(corrected) <= POSITIVE_THRESHOLD: reasons.append("corrected_efficiency_not_gt_0.10")
        if int(row["treated_covered_replicates"]) < 2: reasons.append("treated_coverage_lt_2of3")
        if int(row["control_covered_replicates"]) < 2: reasons.append("control_coverage_lt_2of3")
        if row["control_median"] is None or float(row["control_median"]) > CONTROL_MAX: reasons.append("control_median_gt_0.02")
        if row["orientation_qc"] != "PASS": reasons.append("orientation_or_context_qc")
        if row["WGS_center_overlap"]: reasons.append("WGS_center_overlap")
        if row["low_complexity_qc"] != "PASS": reasons.append("obvious_low_complexity")
        if seen_keys[row["genomic_key"]] != 1: reasons.append("duplicate_genomic_key")
        row["positive_main_exclusion_reason"] = ";".join(reasons) if reasons else "none"

    metric_fields = [
        "treated_covered_replicates", "control_covered_replicates", "all_six_depth_ge_20",
        "treated_median", "control_median", "treated_MAD", "control_MAD", "raw_difference",
        "corrected_editing_efficiency", "pooled_treated_ref_count", "pooled_treated_target_alt_count",
        "pooled_control_ref_count", "pooled_control_target_alt_count", "fisher_exact_screening_pvalue",
        "BH_FDR", "WGS_center_overlap", "orientation_qc", "positive_main", "positive_high_confidence",
        "positive_sensitivity", "positive_main_without_low_complexity_filter",
        "positive_main_exclusion_reason", "pooled_read_screening_limitation",
    ]
    fields = COMMON_FIELDS + COUNT_FIELDS + metric_fields
    if output_prefix:
        write_tsv(output, fields, rows)
        logger.log(f"Smoke positive recount wrote {len(rows):,} rows")
        return
    write_tsv(output, fields, rows)
    write_tsv(run_dir / "positives_main.tsv.gz", fields, (row for row in rows if row["positive_main"]))
    write_tsv(run_dir / "positives_high_confidence.tsv.gz", fields, (row for row in rows if row["positive_high_confidence"]))
    write_tsv(run_dir / "work/positives_sensitivity.tsv.gz", fields, (row for row in rows if row["positive_sensitivity"]))
    write_tsv(
        run_dir / "positives_main_without_low_complexity_filter.tsv.gz", fields,
        (row for row in rows if row["positive_main_without_low_complexity_filter"]),
    )
    write_tsv(
        run_dir / "positives_low_complexity_excluded.tsv.gz", fields,
        (
            row for row in rows
            if row["positive_main_without_low_complexity_filter"] and not row["positive_main"]
        ),
    )
    funnel = []
    predicates = [
        ("broad_candidate_rows", lambda row: True),
        ("corrected_efficiency_gt_0.10", lambda row: row["corrected_editing_efficiency"] is not None and float(row["corrected_editing_efficiency"]) > 0.10),
        ("treated_and_control_each_depth_ge20_in_at_least_2of3", lambda row: int(row["treated_covered_replicates"]) >= 2 and int(row["control_covered_replicates"]) >= 2),
        ("orientation_sequence_center_qc", lambda row: row["orientation_qc"] == "PASS"),
        ("WGS_center_absent", lambda row: not row["WGS_center_overlap"]),
        ("obvious_low_complexity_absent", lambda row: row["low_complexity_qc"] == "PASS"),
        ("unique_genomic_key", lambda row: seen_keys[row["genomic_key"]] == 1),
        ("control_median_le_0.02_positive_main", lambda row: bool(row["positive_main"])),
    ]
    retained = rows
    for order, (stage, predicate) in enumerate(predicates, 1):
        retained = [row for row in retained if predicate(row)]
        funnel.append({"order": order, "filter_stage": stage, "retained_rows": len(retained)})
    write_tsv(run_dir / "positive_filter_funnel.tsv", ["order", "filter_stage", "retained_rows"], funnel)
    old_by_key = {(row["chrom"], int(row["position"]), row["ref"], row["alt"]): row for row in read_tsv(OLD_LABELS)}
    audit_rows = []
    for row in rows:
        key = (str(row["chrom"]), int(row["position"]), str(row["genomic_ref"]), str(row["genomic_alt"]))
        old = old_by_key.get(key, {})
        old_eff_text = old.get("corrected_editing_efficiency", "NA")
        old_eff = float(old_eff_text) if old_eff_text not in {"", "NA"} else None
        old_positive = old_eff is not None and old_eff > POSITIVE_THRESHOLD and old.get("training_eligible") == "1"
        audit_rows.append({
            "chrom": key[0], "position": key[1], "genomic_ref": key[2], "genomic_alt": key[3],
            "old_corrected_editing_efficiency": old_eff, "new_corrected_editing_efficiency": row["corrected_editing_efficiency"],
            "old_comparable_positive_gt_0.10_and_training_eligible": old_positive,
            "new_positive_main": row["positive_main"], "agreement": old_positive == bool(row["positive_main"]),
            "old_input_role": "audit_only_not_used_for_new_label",
        })
    write_tsv(run_dir / "positive_old_vs_new_audit.tsv.gz", list(audit_rows[0]), audit_rows)
    (run_dir / "work/positive.done").write_text(utcnow() + "\n")
    logger.log(
        f"Positive recount complete: main={sum(bool(r['positive_main']) for r in rows):,}; "
        f"high_confidence={sum(bool(r['positive_high_confidence']) for r in rows):,}; "
        f"sensitivity={sum(bool(r['positive_sensitivity']) for r in rows):,}"
    )


def gtf_attributes(text: str) -> dict[str, str]:
    result = {}
    for match in re.finditer(r'(\S+) "([^"]*)";', text):
        result[match.group(1)] = match.group(2)
    return result


def prepare_gtf_shards(run_dir: Path, logger: Logger) -> list[str]:
    manifest = run_dir / "work/gtf_shards/manifest.tsv"
    if manifest.exists() and (run_dir / "work/gtf_shards.done").exists():
        return [row["chrom"] for row in read_tsv(manifest)]
    allowed = set(CANONICAL_CHROMS + ("chrM",))
    handles: dict[str, object] = {}
    counts: Counter[str] = Counter()
    try:
        with GTF.open() as source:
            for line_number, line in enumerate(source, 1):
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t", 8)
                if len(fields) != 9 or fields[2] != "exon" or fields[0] not in allowed or fields[6] not in {"+", "-"}:
                    continue
                chrom = fields[0]
                attrs = gtf_attributes(fields[8])
                if chrom not in handles:
                    handles[chrom] = gzip.open(run_dir / f"work/gtf_shards/{chrom}.exons.tsv.gz", "wt")
                    handles[chrom].write("start0\tend0\tstrand\tgene_id\tgene_name\ttranscript_id\tregion_type\n")
                handles[chrom].write(
                    "\t".join([
                        str(int(fields[3]) - 1), fields[4], fields[6], attrs.get("gene_id", "NA"),
                        attrs.get("gene_name", "NA"), attrs.get("transcript_id", "NA"),
                        attrs.get("transcript_type", attrs.get("gene_type", "exon")),
                    ]) + "\n"
                )
                counts[chrom] += 1
                if line_number % 5_000_000 == 0:
                    logger.log(f"GTF scan: {line_number:,} lines; retained {sum(counts.values()):,} exon rows")
    finally:
        for handle in handles.values():
            handle.close()
    rows = [
        {"chrom": chrom, "exon_rows": counts[chrom], "path": run_dir / f"work/gtf_shards/{chrom}.exons.tsv.gz"}
        for chrom in CANONICAL_CHROMS + ("chrM",) if counts[chrom]
    ]
    write_tsv(manifest, ["chrom", "exon_rows", "path"], rows)
    (run_dir / "work/gtf_shards.done").write_text(utcnow() + "\n")
    logger.log(f"Prepared {len(rows)} chromosome GTF exon shards containing {sum(counts.values()):,} rows")
    return [str(row["chrom"]) for row in rows]


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


class ExonIndex:
    def __init__(self, rows: Sequence[dict[str, str]], strand: str):
        self.rows = sorted(
            (
                int(row["start0"]), int(row["end0"]), row["gene_id"], row["gene_name"],
                row["transcript_id"], row["region_type"],
            )
            for row in rows if row["strand"] == strand
        )
        self.starts = [row[0] for row in self.rows]
        self.prefix_max_end: list[int] = []
        maximum = -1
        for row in self.rows:
            maximum = max(maximum, row[1])
            self.prefix_max_end.append(maximum)

    def annotations(self, position0: int) -> dict[str, str] | None:
        index = bisect.bisect_right(self.starts, position0) - 1
        hits = []
        while index >= 0 and self.prefix_max_end[index] > position0:
            row = self.rows[index]
            if row[0] <= position0 < row[1]:
                hits.append(row)
            index -= 1
        if not hits:
            return None
        return {
            "gene_id": ",".join(sorted({row[2] for row in hits})),
            "gene_name": ",".join(sorted({row[3] for row in hits})),
            "transcript_ids": ",".join(sorted({row[4] for row in hits})),
            "region_type": "exon:" + ",".join(sorted({row[5] for row in hits})),
        }


def candidate_pileup_rows(
    bed: Path, logger: Logger, stderr_path: Path, chrom: str
) -> Iterable[tuple[str, int, str, dict[str, dict[str, int]]]]:
    """Same pileup implementation with a lossless reported-depth prefilter.

    mpileup reported depth is an upper bound on A+C+G+T usable depth, so a
    site whose reported depths do not reach 20 in 2/3 samples in each group
    cannot satisfy strict, relaxed, or near-zero definitions.
    """
    command = pileup_command(bed, chrom)
    logger.command(command)
    with stderr_path.open("w") as stderr_handle:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=stderr_handle, bufsize=2**20)
        assert process.stdout is not None
        for line in process.stdout:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 + 3 * len(SAMPLES):
                raise RuntimeError(f"Malformed multi-sample mpileup row: {line[:240]!r}")
            reference_base = fields[2].upper()
            if reference_base not in {"C", "G"}:
                continue
            reported = [int(fields[3 + sample_index * 3]) for sample_index in range(len(SAMPLES))]
            if sum(depth >= MIN_DEPTH for depth in reported[:3]) < 2 or sum(depth >= MIN_DEPTH for depth in reported[3:]) < 2:
                continue
            counts = {
                sample: parse_mpileup_bases(fields[4 + sample_index * 3], reference_base)
                for sample_index, sample in enumerate(SAMPLE_NAMES)
            }
            yield fields[0], int(fields[1]), reference_base, counts
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"samtools mpileup failed ({return_code}); see {stderr_path}")


NEGATIVE_EXTRA_FIELDS = [
    "minimum_depth", "median_depth", "mean_depth", "treated_covered_replicates",
    "control_covered_replicates", "all_six_depth_ge_20", "all_six_target_alt_zero",
    "treated_median", "control_median", "raw_difference", "broad_candidate_center_overlap",
    "WGS_center_overlap", "motif_similar", "negative_difficulty", "negative_definition",
    "gene_expression_coverage_summary", "external_mappability_resource_status",
    "would_be_strict_without_low_complexity_filter",
    "would_be_relaxed_without_low_complexity_filter",
    "would_be_near_zero_without_low_complexity_filter",
]


def negative_row(
    chrom: str,
    position: int,
    reference_base: str,
    counts: Mapping[str, Mapping[str, int]],
    annotation: Mapping[str, str],
    context: Mapping[str, object],
    positive_5mers: set[str],
) -> dict[str, object]:
    strand = "+" if reference_base == "C" else "-"
    genomic_alt = "T" if strand == "+" else "A"
    row: dict[str, object] = {
        "chrom": chrom, "position": position, "genomic_ref": reference_base,
        "genomic_alt": genomic_alt, "transcript_strand": strand,
        "transcript_oriented_ref": "C", "transcript_oriented_alt": "T",
        "genomic_key": f"{chrom}:{position}:{strand}:C:T", **annotation, **context,
        "mappability": "NA_RESOURCE_MISSING",
        "mappability_method": "NA_RESOURCE_MISSING;MAPQ30_PILEUP_FILTER_IS_NOT_A_MAPPABILITY_TRACK",
        "mappability_qc": "NA_RESOURCE_MISSING",
        "external_mappability_resource_status": "NA_RESOURCE_MISSING",
    }
    add_counts(row, counts, reference_base, genomic_alt)
    depths = [int(row[f"{sample}_usable_depth"]) for sample in SAMPLE_NAMES]
    treated_rates = [float(row[f"{sample}_target_edit_rate"]) for sample in TREATED if int(row[f"{sample}_usable_depth"]) >= MIN_DEPTH]
    control_rates = [float(row[f"{sample}_target_edit_rate"]) for sample in CONTROLS if int(row[f"{sample}_usable_depth"]) >= MIN_DEPTH]
    row["minimum_depth"] = min(depths)
    row["median_depth"] = statistics.median(depths)
    row["mean_depth"] = statistics.fmean(depths)
    row["treated_covered_replicates"] = len(treated_rates)
    row["control_covered_replicates"] = len(control_rates)
    row["all_six_depth_ge_20"] = all(depth >= MIN_DEPTH for depth in depths)
    alt_counts = [int(row[f"{sample}_target_alt_count"]) for sample in SAMPLE_NAMES]
    row["all_six_target_alt_zero"] = all(value == 0 for value in alt_counts)
    row["treated_median"] = median(treated_rates)
    row["control_median"] = median(control_rates)
    row["raw_difference"] = (
        float(row["treated_median"]) - float(row["control_median"])
        if len(treated_rates) >= 2 and len(control_rates) >= 2 else None
    )
    row["motif_similar"] = row["center_5mer"] in positive_5mers
    row["negative_difficulty"] = "motif_similar_strict" if row["motif_similar"] else "random_strict"
    row["gene_expression_coverage_summary"] = (
        f"six_sample_site_depth_mean={fmt(row['mean_depth'])};median={fmt(row['median_depth'])}"
    )
    return row


def stage_negative_chrom(run_dir: Path, logger: Logger, chrom: str, smoke: bool = False) -> None:
    suffix = ".smoke" if smoke else ""
    done = run_dir / f"work/count_shards/{chrom}{suffix}.done"
    if done.exists():
        logger.log(f"Negative shard {chrom}{suffix} already complete")
        return
    exon_path = run_dir / f"work/gtf_shards/{chrom}.exons.tsv.gz"
    exon_rows = read_tsv(exon_path)
    plus_index, minus_index = ExonIndex(exon_rows, "+"), ExonIndex(exon_rows, "-")
    union = merge_intervals((int(row["start0"]), int(row["end0"])) for row in exon_rows)
    if smoke:
        # Deterministic bounded one-megabase exon-union subset.
        bounded = []
        remaining = 1_000_000
        for start, end in union:
            if remaining <= 0:
                break
            retained_end = min(end, start + remaining)
            bounded.append((start, retained_end))
            remaining -= retained_end - start
        union = bounded
    bed = run_dir / f"work/candidate_shards/{chrom}{suffix}.exon_union.bed"
    with bed.open("w") as handle:
        for start, end in union:
            handle.write(f"{chrom}\t{start}\t{end}\n")
    broad_centers: dict[str, set[int]] = defaultdict(set)
    for row in read_tsv(BROAD):
        broad_centers[row["chrom"]].add(int(row["position"]))
    variants = load_variant_positions({chrom})
    positives = read_tsv(run_dir / "positives_main.tsv.gz")
    positive_5mers = {row["center_5mer"] for row in positives}
    strict_rows: list[dict[str, object]] = []
    relaxed_rows: list[dict[str, object]] = []
    near_rows: list[dict[str, object]] = []
    low_complexity_excluded_rows: list[dict[str, object]] = []
    counters: Counter[str] = Counter()
    with pysam.FastaFile(str(REFERENCE)) as reference:
        for _, position, reference_base, counts in candidate_pileup_rows(
            bed, logger, run_dir / f"logs/{chrom}{suffix}.mpileup.stderr.log", chrom
        ):
            counters["reported_depth_prefilter_pass"] += 1
            strand = "+" if reference_base == "C" else "-"
            annotation = (plus_index if strand == "+" else minus_index).annotations(position - 1)
            if annotation is None:
                counters["not_transcript_oriented_exonic_C"] += 1
                continue
            counters["transcript_oriented_exonic_C"] += 1
            context = context_record(reference, chrom, position, strand)
            if context is None:
                counters["bad_101nt_context_or_center"] += 1
                continue
            counters["context_qc_pass"] += 1
            if position in variants.get(chrom, set()):
                counters["WGS_center_overlap"] += 1
                continue
            counters["WGS_center_absent"] += 1
            if position in broad_centers.get(chrom, set()):
                counters["broad_candidate_ambiguous_center"] += 1
                continue
            counters["broad_candidate_center_absent"] += 1
            row = negative_row(chrom, position, reference_base, counts, annotation, context, positive_5mers)
            row["broad_candidate_center_overlap"] = False
            row["WGS_center_overlap"] = False
            strict = bool(row["all_six_depth_ge_20"]) and bool(row["all_six_target_alt_zero"])
            covered_zero = all(
                int(row[f"{sample}_target_alt_count"]) == 0
                for sample in SAMPLE_NAMES if int(row[f"{sample}_usable_depth"]) >= MIN_DEPTH
            )
            relaxed = (
                int(row["treated_covered_replicates"]) >= 2
                and int(row["control_covered_replicates"]) >= 2 and covered_zero
            )
            all_rates_low = all(float(row[f"{sample}_target_edit_rate"]) <= 0.005 for sample in SAMPLE_NAMES)
            near = bool(row["all_six_depth_ge_20"]) and all_rates_low and row["raw_difference"] is not None and float(row["raw_difference"]) <= 0
            row["would_be_strict_without_low_complexity_filter"] = strict
            row["would_be_relaxed_without_low_complexity_filter"] = relaxed
            row["would_be_near_zero_without_low_complexity_filter"] = near
            if context["low_complexity_qc"] != "PASS":
                counters["low_complexity_any_eligible_negative_definition"] += int(strict or relaxed or near)
                counters["low_complexity_strict_excluded"] += int(strict)
                counters["low_complexity_relaxed_excluded"] += int(relaxed and not strict)
                counters["low_complexity_near_zero_excluded"] += int(near and not strict)
                if strict or relaxed or near:
                    row["negative_definition"] = "excluded_by_low_complexity_OR_rule"
                    row["negative_difficulty"] = "low_complexity_excluded_audit_only"
                    low_complexity_excluded_rows.append(row)
                continue
            counters["low_complexity_qc_pass"] += 1
            if strict:
                row["negative_definition"] = "strict_all_six_depth_ge20_and_all_six_target_alt_zero"
                strict_rows.append(row)
                counters["negative_strict"] += 1
            elif relaxed:
                row["negative_definition"] = "relaxed_each_group_2of3_depth_ge20_and_covered_target_alt_zero"
                row["negative_difficulty"] = "relaxed_not_for_primary_label"
                relaxed_rows.append(row)
                counters["negative_relaxed_only"] += 1
            if near and not strict:
                near_row = dict(row)
                near_row["negative_definition"] = "near_zero_all_six_depth_ge20_rates_le0.005_and_treated_minus_control_le0"
                near_row["negative_difficulty"] = "near_zero_sensitivity_only"
                near_rows.append(near_row)
                counters["negative_near_zero_only"] += 1
    fields = COMMON_FIELDS + COUNT_FIELDS + NEGATIVE_EXTRA_FIELDS
    target_dirs = {
        "strict": run_dir / "negative_universe_strict",
        "relaxed": run_dir / "negative_universe_relaxed",
        "near": run_dir / "negative_near_zero",
        "low_complexity_excluded": run_dir / "negative_low_complexity_excluded",
    }
    for directory in target_dirs.values():
        directory.mkdir(exist_ok=True)
    write_tsv(target_dirs["strict"] / f"{chrom}{suffix}.tsv.gz", fields, strict_rows)
    write_tsv(target_dirs["relaxed"] / f"{chrom}{suffix}.tsv.gz", fields, relaxed_rows)
    write_tsv(target_dirs["near"] / f"{chrom}{suffix}.tsv.gz", fields, near_rows)
    write_tsv(
        target_dirs["low_complexity_excluded"] / f"{chrom}{suffix}.tsv.gz",
        fields, low_complexity_excluded_rows,
    )
    write_tsv(
        run_dir / f"work/count_shards/{chrom}{suffix}.funnel.tsv",
        ["chrom", "metric", "count"],
        ({"chrom": chrom, "metric": key, "count": value} for key, value in sorted(counters.items())),
    )
    done.write_text(json.dumps({"completed_utc": utcnow(), "counts": counters}, sort_keys=True) + "\n")
    logger.log(
        f"Negative shard {chrom}{suffix}: strict={len(strict_rows):,}; "
        f"relaxed-only={len(relaxed_rows):,}; near-zero-only={len(near_rows):,}; "
        f"low-complexity-excluded={len(low_complexity_excluded_rows):,}"
    )


def stage_negative(run_dir: Path, logger: Logger, chromosomes: Sequence[str] | None = None, smoke: bool = False) -> None:
    available = prepare_gtf_shards(run_dir, logger)
    selected = list(chromosomes) if chromosomes else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Chromosomes absent from GTF shards: {unknown}")
    for chrom in selected:
        stage_negative_chrom(run_dir, logger, chrom, smoke=smoke)
    if smoke:
        return
    funnel_rows = []
    for chrom in selected:
        funnel_rows.extend(read_tsv(run_dir / f"work/count_shards/{chrom}.funnel.tsv"))
    totals: Counter[str] = Counter()
    for row in funnel_rows:
        totals[row["metric"]] += int(row["count"])
    write_tsv(
        run_dir / "negative_filter_funnel.tsv", ["metric", "count"],
        ({"metric": metric, "count": count} for metric, count in sorted(totals.items())),
    )
    qc_rows = [
        {"metric": "strict_rows", "value": totals["negative_strict"], "status": "INFO"},
        {"metric": "relaxed_only_rows", "value": totals["negative_relaxed_only"], "status": "INFO"},
        {"metric": "near_zero_only_rows", "value": totals["negative_near_zero_only"], "status": "INFO"},
        {"metric": "external_basewise_mappability_resource", "value": "NA_RESOURCE_MISSING; MAPQ30 pileup filtering is not treated as mappability", "status": "LIMITATION"},
        {"metric": "low_complexity_method", "value": "101nt single-base Shannon log2 entropy<1.20 OR any-base homopolymer>=20 OR deterministic phase0/phase1 tandem-2mer coverage>=0.80", "status": "INFO"},
    ]
    write_tsv(run_dir / "negative_qc_summary.tsv", ["metric", "value", "status"], qc_rows)
    (run_dir / "work/negative.done").write_text(utcnow() + "\n")
    logger.log(f"All negative shards complete: strict={totals['negative_strict']:,}")


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = bytearray(size)

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def iter_tsv(path: Path) -> Iterable[dict[str, str]]:
    with open_text(path) as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def strict_shards(run_dir: Path) -> list[Path]:
    return sorted(path for path in (run_dir / "negative_universe_strict").glob("*.tsv.gz") if ".smoke." not in path.name)


def site_metadata(run_dir: Path, logger: Logger) -> tuple[list[dict[str, object]], list[str]]:
    positive_rows = read_tsv(run_dir / "positives_main.tsv.gz")
    metadata: list[dict[str, object]] = []
    for row in positive_rows:
        metadata.append({
            "genomic_key": row["genomic_key"], "chrom": row["chrom"], "position": int(row["position"]),
            "strand": row["transcript_strand"], "gene_ids": row["gene_id"],
            "sequence": row["sequence_context"], "label": 1,
            "efficiency": float(row["corrected_editing_efficiency"]),
            "gc": float(row["gc_fraction"]), "region": row["region_type"],
        })
    shards = strict_shards(run_dir)
    for shard in shards:
        for row in iter_tsv(shard):
            metadata.append({
                "genomic_key": row["genomic_key"], "chrom": row["chrom"], "position": int(row["position"]),
                "strand": row["transcript_strand"], "gene_ids": row["gene_id"],
                "sequence": row["sequence_context"], "label": 0, "efficiency": 0.0,
                "gc": float(row["gc_fraction"]), "region": row["region_type"],
            })
    logger.log(f"Loaded minimal leakage metadata: positives={len(positive_rows):,}; strict negatives={len(metadata)-len(positive_rows):,}")
    return metadata, [str(path) for path in shards]


def build_leakage_groups(metadata: Sequence[Mapping[str, object]], logger: Logger) -> tuple[list[str], list[int]]:
    uf = UnionFind(len(metadata))
    first_gene: dict[str, int] = {}
    first_sequence: dict[str, int] = {}
    first_center: dict[tuple[str, int], int] = {}
    by_chrom_strand: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        genes = [gene for gene in str(row["gene_ids"]).split(",") if gene not in {"", "NA", "-"}]
        for gene in genes:
            if gene in first_gene:
                uf.union(index, first_gene[gene])
            else:
                first_gene[gene] = index
        sequence = str(row["sequence"])
        if sequence in first_sequence:
            uf.union(index, first_sequence[sequence])
        else:
            first_sequence[sequence] = index
        center = (str(row["chrom"]), int(row["position"]))
        if center in first_center:
            uf.union(index, first_center[center])
        else:
            first_center[center] = index
        by_chrom_strand[(str(row["chrom"]), str(row["strand"]))].append(index)
    for indexes in by_chrom_strand.values():
        indexes.sort(key=lambda index: int(metadata[index]["position"]))
        for left, right in zip(indexes, indexes[1:]):
            if int(metadata[right]["position"]) - int(metadata[left]["position"]) <= 100:
                uf.union(left, right)
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(metadata)):
        members[uf.find(index)].append(index)
    group_id_by_root = {}
    for root, indexes in members.items():
        digest = hashlib.sha256()
        for key in sorted(str(metadata[index]["genomic_key"]) for index in indexes):
            digest.update(key.encode())
            digest.update(b"\0")
        group_id_by_root[root] = "LG_" + digest.hexdigest()[:16]
    group_ids = [group_id_by_root[uf.find(index)] for index in range(len(metadata))]
    logger.log(
        f"Leakage grouping complete: {len(members):,} groups; gene/window/exact-sequence/center unions applied"
    )
    return group_ids, [len(members[uf.find(index)]) for index in range(len(metadata))]


def split_groups(
    metadata: Sequence[Mapping[str, object]], group_ids: Sequence[str]
) -> dict[str, str]:
    summaries: dict[str, Counter[str]] = defaultdict(Counter)
    for row, group_id in zip(metadata, group_ids):
        summaries[group_id]["total"] += 1
        summaries[group_id]["positive"] += int(row["label"])
        if int(row["label"]) == 1:
            efficiency_bin = min(5, int(float(row["efficiency"]) / 0.1))
            summaries[group_id][f"positive_efficiency_bin:{efficiency_bin}"] += 1
            summaries[group_id][f"positive_chrom:{row['chrom']}"] += 1
            summaries[group_id][f"positive_region:{region_class(str(row['region']))}"] += 1
            summaries[group_id][f"positive_gc_bin:{min(9, int(float(row['gc']) * 10))}"] += 1
    names = ("train", "dev", "calibration", "test")
    proportions = {"train": 0.70, "dev": 0.10, "calibration": 0.10, "test": 0.10}
    total_positive = sum(value["positive"] for value in summaries.values())
    total_rows = sum(value["total"] for value in summaries.values())
    balance_features = sorted({
        feature
        for summary in summaries.values()
        for feature in summary
        if feature.startswith("positive_") and feature != "positive"
    })
    feature_totals = {
        feature: sum(summary[feature] for summary in summaries.values())
        for feature in balance_features
    }
    assigned = {name: Counter() for name in names}
    output: dict[str, str] = {}
    rng = random.Random(SEED)
    tie = {group_id: rng.random() for group_id in summaries}
    ordered = sorted(
        summaries,
        key=lambda group_id: (
            summaries[group_id]["positive"] == 0,
            -summaries[group_id]["positive"], -summaries[group_id]["total"], tie[group_id], group_id,
        ),
    )
    for group_id in ordered:
        summary = summaries[group_id]
        best_name, best_score = None, None
        for candidate_name in names:
            # Compare the total objective over all four splits.  Scoring only the
            # candidate split makes an existing positive-count imbalance act as
            # a split-specific constant for negative-only groups, which can send
            # nearly every such group to one split even when row targets are far
            # from 70/10/10/10.
            score = 0.0
            for evaluated_name in names:
                addition = summary if evaluated_name == candidate_name else Counter()
                pos_target = max(1.0, total_positive * proportions[evaluated_name])
                row_target = max(1.0, total_rows * proportions[evaluated_name])
                pos_after = assigned[evaluated_name]["positive"] + addition["positive"]
                row_after = assigned[evaluated_name]["total"] + addition["total"]
                pos_error = (pos_after - pos_target) / pos_target
                row_error = (row_after - row_target) / row_target
                score += 8.0 * pos_error * pos_error + row_error * row_error
                for feature in balance_features:
                    target = feature_totals[feature] * proportions[evaluated_name]
                    if target <= 0:
                        continue
                    after = assigned[evaluated_name][feature] + addition[feature]
                    error = (after - target) / target
                    score += 0.15 * error * error
            if best_score is None or score < best_score or (score == best_score and candidate_name < str(best_name)):
                best_name, best_score = candidate_name, score
        assert best_name is not None
        output[group_id] = best_name
        assigned[best_name].update(summary)
    return output


def with_label(row: Mapping[str, str], label: int, split: str, group_id: str) -> dict[str, object]:
    return {**row, "label": label, "split": split, "leakage_group": group_id}


def depth_bin(row: Mapping[str, str]) -> int:
    depths = [int(row[f"{sample}_usable_depth"]) for sample in SAMPLE_NAMES]
    value = statistics.median(depths)
    return min(12, int(math.log2(max(1.0, value))))


def gc_bin(row: Mapping[str, str]) -> int:
    return min(9, int(float(row["gc_fraction"]) * 10))


def chrom_class(chrom: str) -> str:
    return "autosome" if chrom in {f"chr{i}" for i in range(1, 23)} else "sex_or_mito"


def region_class(region: str) -> str:
    text = region.lower()
    if "coding" in text or "missense" in text or "synonymous" in text:
        return "coding_like"
    if "utr" in text:
        return "UTR"
    if "intron" in text:
        return "intron_like"
    return "other_exonic"


def gene_ids(row: Mapping[str, object]) -> list[str]:
    value = row.get("gene_id", row.get("gene_ids", ""))
    return sorted({gene for gene in str(value).split(",") if gene not in {"", "NA", "-"}})


def row_site_median_depth(row: Mapping[str, object]) -> float:
    return float(statistics.median(int(row[f"{sample}_usable_depth"]) for sample in SAMPLE_NAMES))


def build_gene_coverage_summary(
    run_dir: Path, shard_paths: Sequence[str], logger: Logger
) -> dict[str, dict[str, float | int]]:
    aggregates: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"site_count": 0, "positive_site_count": 0, "strict_negative_site_count": 0, "site_median_depth_sum": 0.0}
    )

    def consume(row: Mapping[str, object], label: int) -> None:
        depth = row_site_median_depth(row)
        for gene in gene_ids(row):
            aggregate = aggregates[gene]
            aggregate["site_count"] = int(aggregate["site_count"]) + 1
            count_field = "positive_site_count" if label == 1 else "strict_negative_site_count"
            aggregate[count_field] = int(aggregate[count_field]) + 1
            aggregate["site_median_depth_sum"] = float(aggregate["site_median_depth_sum"]) + depth

    for row in iter_tsv(run_dir / "positives_main.tsv.gz"):
        consume(row, 1)
    for shard_text in shard_paths:
        for row in iter_tsv(Path(shard_text)):
            consume(row, 0)

    output_rows = []
    result: dict[str, dict[str, float | int]] = {}
    for gene in sorted(aggregates):
        aggregate = aggregates[gene]
        site_count = int(aggregate["site_count"])
        mean_depth = float(aggregate["site_median_depth_sum"]) / site_count
        coverage_bin = min(15, int(math.log2(max(1.0, mean_depth))))
        result[gene] = {**aggregate, "mean_site_median_depth": mean_depth, "coverage_bin_log2": coverage_bin}
        output_rows.append({
            "gene_id": gene,
            "modeling_universe_site_count": site_count,
            "positive_site_count": int(aggregate["positive_site_count"]),
            "strict_negative_site_count": int(aggregate["strict_negative_site_count"]),
            "mean_of_six_sample_site_median_depth": mean_depth,
            "coverage_bin_log2": coverage_bin,
            "definition": "mean of per-site median usable depth across six BAMs over retained positive_main and strict-negative modeling-universe sites annotated to this gene; not TPM",
        })
    if not output_rows:
        raise RuntimeError("No gene-level coverage summaries could be constructed")
    write_tsv(run_dir / "gene_coverage_summary.tsv.gz", list(output_rows[0]), output_rows)
    logger.log(f"Gene-level coverage proxy written for {len(output_rows):,} genes")
    return result


def add_gene_coverage_fields(
    row: dict[str, object], gene_coverage: Mapping[str, Mapping[str, float | int]]
) -> dict[str, object]:
    values = [float(gene_coverage[gene]["mean_site_median_depth"]) for gene in gene_ids(row) if gene in gene_coverage]
    if values:
        value = float(statistics.median(values))
        coverage_bin: int | str = min(15, int(math.log2(max(1.0, value))))
        status = "MODEL_UNIVERSE_GENE_COVERAGE_PROXY"
    else:
        value = float("nan")
        coverage_bin = "NA_NO_GENE_ID"
        status = "NA_NO_GENE_ID"
    row["gene_level_coverage_mean_site_median_depth"] = value
    row["gene_level_coverage_bin_log2"] = coverage_bin
    row["gene_level_coverage_status"] = status
    row["gene_expression_coverage_summary"] = (
        f"gene_level_mean_site_median_depth={value};gene_level_bin={coverage_bin};"
        f"site_six_sample_depth_median={row_site_median_depth(row)};not_TPM"
    )
    return row


def stable_sample(rows: Sequence[dict[str, object]], count: int, seed_text: str) -> list[dict[str, object]]:
    if count > len(rows):
        count = len(rows)
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{SEED}|{seed_text}|{row['genomic_key']}".encode()).hexdigest(),
    )
    return ranked[:count]


def write_combined(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty fixed dataset: {path}")
    fields = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fields:
                fields.append(field)
    write_tsv(path, fields, rows)


def build_fixed_sets(
    run_dir: Path,
    positives: Mapping[str, list[dict[str, object]]],
    negatives: Mapping[str, list[dict[str, object]]],
    logger: Logger,
) -> dict[str, dict[str, int | float | bool]]:
    summary: dict[str, dict[str, int | float | bool]] = {}
    train_pos = positives["train"]
    write_combined(run_dir / "train_positives.tsv.gz", train_pos)

    dev_pos = positives["dev"]
    dev_neg = stable_sample(negatives["dev"], len(dev_pos) * 10, "dev_1to10")
    write_combined(run_dir / "dev_1to10.tsv.gz", dev_pos + dev_neg)
    train_5mers = {str(row["center_5mer"]) for row in train_pos}
    dev_hard_candidates = [row for row in negatives["dev"] if str(row["center_5mer"]) in train_5mers]
    dev_hard = stable_sample(dev_hard_candidates, max(len(dev_pos) * 10, min(1000, len(dev_hard_candidates))), "dev_hard")
    if not dev_hard:
        raise RuntimeError("No dev hard negatives sharing a train-positive center 5-mer")
    write_combined(run_dir / "dev_hard_negatives.tsv.gz", dev_hard)

    for split, filename in (("calibration", "calibration_1to1000.tsv.gz"), ("test", "test_1to1000.tsv.gz")):
        all_pos = positives[split]
        available_neg = negatives[split]
        used_pos_count = min(len(all_pos), len(available_neg) // 1000)
        selected_pos = stable_sample(all_pos, used_pos_count, f"{split}_positive")
        selected_neg = stable_sample(available_neg, used_pos_count * 1000, f"{split}_negative")
        if not selected_pos:
            raise RuntimeError(f"{split} cannot realize even one strict 1:1000 fixed set")
        write_combined(run_dir / filename, selected_pos + selected_neg)
        summary[split] = {
            "positive_all": len(all_pos), "positive_used": len(selected_pos), "negative_used": len(selected_neg),
            "achieved_1to1000": len(selected_neg) == 1000 * len(selected_pos),
            "prevalence": len(selected_pos) / (len(selected_pos) + len(selected_neg)),
        }
    write_combined(run_dir / "test_positive_all.tsv.gz", positives["test"])
    summary["train"] = {"positive": len(train_pos), "negative_pool": len(negatives["train"])}
    summary["dev"] = {"positive": len(dev_pos), "negative_1to10": len(dev_neg), "hard_negative": len(dev_hard)}
    logger.log(f"Fixed dev/calibration/test sets written: {json.dumps(summary, sort_keys=True)}")
    return summary


def verify_leakage(metadata: Sequence[Mapping[str, object]], group_ids: Sequence[str], group_split: Mapping[str, str]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    site_splits = [group_split[group] for group in group_ids]
    for name, key_function in (
        ("gene_id_not_cross_split", lambda row: [gene for gene in str(row["gene_ids"]).split(",") if gene not in {"", "NA", "-"}]),
        ("exact_sequence_not_cross_split", lambda row: [str(row["sequence"])]),
        ("genomic_key_not_cross_split", lambda row: [str(row["genomic_key"])]),
        ("genomic_center_not_cross_split", lambda row: [f"{row['chrom']}:{row['position']}"]),
    ):
        observed: dict[str, str] = {}
        violations = 0
        for row, split in zip(metadata, site_splits):
            for key in key_function(row):
                if key in observed and observed[key] != split:
                    violations += 1
                observed[key] = split
        checks.append({"assertion": name, "status": "PASS" if violations == 0 else "FAIL", "violations": violations})
    group_observed: dict[str, str] = {}
    group_violations = 0
    for group, split in zip(group_ids, site_splits):
        if group in group_observed and group_observed[group] != split:
            group_violations += 1
        group_observed[group] = split
    checks.append({"assertion": "leakage_group_not_cross_split", "status": "PASS" if group_violations == 0 else "FAIL", "violations": group_violations})
    window_violations = 0
    by_key: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for row, split in zip(metadata, site_splits):
        by_key[(str(row["chrom"]), str(row["strand"]))].append((int(row["position"]), split))
    for rows in by_key.values():
        rows.sort()
        for left, right in zip(rows, rows[1:]):
            if right[0] - left[0] <= 100 and right[1] != left[1]:
                window_violations += 1
    checks.append({"assertion": "overlapping_101nt_windows_not_cross_split", "status": "PASS" if window_violations == 0 else "FAIL", "violations": window_violations})
    return checks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(run_dir: Path) -> None:
    """Write checksums only after the final logger message has been flushed."""
    checksums = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"checksums.sha256", "SUCCESS"}:
            checksums.append(f"{sha256_file(path)}  {path.relative_to(run_dir)}")
    (run_dir / "checksums.sha256").write_text("\n".join(checksums) + "\n")


def stage_finalize(run_dir: Path, logger: Logger) -> None:
    metadata, shard_paths = site_metadata(run_dir, logger)
    group_ids, group_sizes = build_leakage_groups(metadata, logger)
    group_split = split_groups(metadata, group_ids)
    gene_coverage = build_gene_coverage_summary(run_dir, shard_paths, logger)
    assignments = []
    for row, group, group_size in zip(metadata, group_ids, group_sizes):
        assignments.append({
            "genomic_key": row["genomic_key"], "chrom": row["chrom"], "position": row["position"],
            "transcript_strand": row["strand"], "gene_id": row["gene_ids"], "label": row["label"],
            "leakage_group": group, "leakage_group_size": group_size, "split": group_split[group],
        })
    write_tsv(run_dir / "site_to_group.tsv.gz", list(assignments[0]), assignments)
    write_tsv(run_dir / "split_assignments.tsv.gz", list(assignments[0]), assignments)
    split_by_key = {str(row["genomic_key"]): str(row["split"]) for row in assignments}
    group_by_key = {str(row["genomic_key"]): str(row["leakage_group"]) for row in assignments}

    positives: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in iter_tsv(run_dir / "positives_main.tsv.gz"):
        split = split_by_key[row["genomic_key"]]
        enriched_positive: dict[str, object] = dict(row)
        add_gene_coverage_fields(enriched_positive, gene_coverage)
        positives[split].append(with_label(enriched_positive, 1, split, group_by_key[row["genomic_key"]]))
    negatives: dict[str, list[dict[str, object]]] = defaultdict(list)
    pool_manifest = []
    (run_dir / "train_negative_pool").mkdir(parents=True, exist_ok=True)
    train_5mers = {row["center_5mer"] for row in positives["train"]}
    matched_strata = {
        (
            depth_bin(row), row["gene_level_coverage_bin_log2"], gc_bin(row), row["center_5mer"], region_class(row["region_type"]),
            chrom_class(row["chrom"]), row["mappability_qc"],
        )
        for row in positives["train"]
    }
    for shard_text in shard_paths:
        shard = Path(shard_text)
        train_rows = []
        matched_rows = []
        hard_rows = []
        for row in iter_tsv(shard):
            split = split_by_key[row["genomic_key"]]
            enriched_negative: dict[str, object] = dict(row)
            add_gene_coverage_fields(enriched_negative, gene_coverage)
            enriched = with_label(enriched_negative, 0, split, group_by_key[row["genomic_key"]])
            negatives[split].append(enriched)
            if split == "train":
                enriched["sampling_pool_random"] = 1
                stratum = (
                    depth_bin(enriched), enriched["gene_level_coverage_bin_log2"], gc_bin(enriched), enriched["center_5mer"], region_class(str(enriched["region_type"])),
                    chrom_class(row["chrom"]), row["mappability_qc"],
                )
                enriched["sampling_pool_matched"] = int(stratum in matched_strata)
                enriched["sampling_pool_hard"] = int(row["center_5mer"] in train_5mers)
                enriched["matching_stratum"] = "|".join(map(str, stratum))
                train_rows.append(enriched)
                if enriched["sampling_pool_matched"]:
                    matched_rows.append(enriched)
                if enriched["sampling_pool_hard"]:
                    hard_rows.append(enriched)
        if train_rows:
            output = run_dir / "train_negative_pool" / shard.name
            write_combined(output, train_rows)
            pool_manifest.append({
                "pool": "random_strict_all", "path": output, "rows": len(train_rows), "sha256": sha256_file(output),
                "definition": "all train-split strict computational negatives; dynamic sampling without replacement per draw",
            })
        if matched_rows:
            output = run_dir / "train_negative_pool" / shard.name.replace(".tsv.gz", ".matched.tsv.gz")
            write_combined(output, matched_rows)
            pool_manifest.append({
                "pool": "matched_strict", "path": output, "rows": len(matched_rows), "sha256": sha256_file(output),
                "definition": "exact bin match on site depth,gene-level coverage proxy,GC,5-mer,region,mappability status,chromosome class to train positives",
            })
        if hard_rows:
            output = run_dir / "train_negative_pool" / shard.name.replace(".tsv.gz", ".hard.tsv.gz")
            write_combined(output, hard_rows)
            pool_manifest.append({
                "pool": "hard_strict", "path": output, "rows": len(hard_rows), "sha256": sha256_file(output),
                "definition": "train strict negatives sharing center 5-mer with train positives; test never mined",
            })
    write_tsv(run_dir / "train_negative_pool/manifest.tsv", list(pool_manifest[0]), pool_manifest)

    fixed_summary = build_fixed_sets(run_dir, positives, negatives, logger)
    assertions = verify_leakage(metadata, group_ids, group_split)
    strict_failures = 0
    strict_rows = 0
    for split_rows in negatives.values():
        for row in split_rows:
            strict_rows += 1
            if not all(int(row[f"{sample}_usable_depth"]) >= MIN_DEPTH and int(row[f"{sample}_target_alt_count"]) == 0 for sample in SAMPLE_NAMES):
                strict_failures += 1
    assertions.extend([
        {"assertion": "strict_all_six_depth_ge20_and_alt_zero", "status": "PASS" if strict_failures == 0 else "FAIL", "violations": strict_failures},
        {"assertion": "all_sequence_context_length_101_center_C", "status": "PASS" if all(len(str(row["sequence"])) == 101 and str(row["sequence"])[50] == "C" for row in metadata) else "FAIL", "violations": 0 if all(len(str(row["sequence"])) == 101 and str(row["sequence"])[50] == "C" for row in metadata) else 1},
        {"assertion": "calibration_true_1to1000", "status": "PASS" if fixed_summary["calibration"]["achieved_1to1000"] else "FAIL", "violations": 0 if fixed_summary["calibration"]["achieved_1to1000"] else 1},
        {"assertion": "test_true_1to1000", "status": "PASS" if fixed_summary["test"]["achieved_1to1000"] else "FAIL", "violations": 0 if fixed_summary["test"]["achieved_1to1000"] else 1},
        {"assertion": "test_not_used_for_hard_negative_mining", "status": "PASS", "violations": 0},
    ])
    write_tsv(run_dir / "qc_assertions.tsv", ["assertion", "status", "violations"], assertions)
    failures = [row for row in assertions if row["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"QC assertions failed: {failures}")

    old_audit = read_tsv(run_dir / "positive_old_vs_new_audit.tsv.gz")
    old_agree = sum(row["agreement"] == "1" for row in old_audit)
    main_count = len(read_tsv(run_dir / "positives_main.tsv.gz"))
    high_count = len(read_tsv(run_dir / "positives_high_confidence.tsv.gz"))
    positive_without_lc_rows = read_tsv(run_dir / "positives_main_without_low_complexity_filter.tsv.gz")
    positive_lc_excluded_rows = read_tsv(run_dir / "positives_low_complexity_excluded.tsv.gz")
    positive_without_lc_count = len(positive_without_lc_rows)
    positive_lc_excluded = len(positive_lc_excluded_rows)
    positive_lc_rate = positive_lc_excluded / positive_without_lc_count if positive_without_lc_count else 0.0
    negative_lc_excluded_rows = []
    for path in sorted((run_dir / "negative_low_complexity_excluded").glob("*.tsv.gz")):
        if ".smoke." not in path.name:
            negative_lc_excluded_rows.extend(
                row for row in iter_tsv(path)
                if row["would_be_strict_without_low_complexity_filter"] == "1"
            )
    negative_lc_excluded = len(negative_lc_excluded_rows)
    negative_without_lc_count = strict_rows + negative_lc_excluded
    negative_lc_rate = negative_lc_excluded / negative_without_lc_count if negative_without_lc_count else 0.0
    positive_over_10pct = positive_lc_rate > 0.10
    positive_over_2x_negative = positive_lc_rate > 2 * negative_lc_rate
    low_complexity_review_required = positive_over_10pct or positive_over_2x_negative

    low_complexity_audit_rows = []
    for population, before, excluded, excluded_rows in (
        ("positive_main_without_low_complexity_filter", positive_without_lc_count, positive_lc_excluded, positive_lc_excluded_rows),
        ("negative_strict_without_low_complexity_filter", negative_without_lc_count, negative_lc_excluded, negative_lc_excluded_rows),
    ):
        low_complexity_audit_rows.append({
            "population": population, "eligible_before_low_complexity_filter": before,
            "excluded_by_OR_rule": excluded, "retained_after_filter": before - excluded,
            "exclusion_proportion": excluded / before if before else 0.0,
            "entropy_trigger_count": sum(row["low_complexity_entropy_trigger"] == "1" for row in excluded_rows),
            "homopolymer_trigger_count": sum(row["low_complexity_homopolymer_trigger"] == "1" for row in excluded_rows),
            "dinucleotide_trigger_count": sum(row["low_complexity_dinucleotide_trigger"] == "1" for row in excluded_rows),
            "OR_logic": "entropy_lt_1.20_OR_homopolymer_ge20_OR_phase_checked_dinucleotide_coverage_ge0.80",
        })
    write_tsv(run_dir / "low_complexity_audit.tsv", list(low_complexity_audit_rows[0]), low_complexity_audit_rows)
    sensitivity_manifest_rows = [
        {
            "component": "positive_main_without_low_complexity_filter", "path": run_dir / "positives_main_without_low_complexity_filter.tsv.gz",
            "rows": positive_without_lc_count,
            "reconstruction": "use directly; all other positive_main rules applied, low-complexity rule not applied",
        },
        {
            "component": "strict_negative_retained", "path": run_dir / "negative_universe_strict",
            "rows": strict_rows,
            "reconstruction": "union with strict rows in negative_low_complexity_excluded",
        },
        {
            "component": "strict_negative_low_complexity_excluded", "path": run_dir / "negative_low_complexity_excluded",
            "rows": negative_lc_excluded,
            "reconstruction": "select would_be_strict_without_low_complexity_filter=1 then union with retained strict shards",
        },
    ]
    write_tsv(
        run_dir / "low_complexity_sensitivity_manifest.tsv",
        ["component", "path", "rows", "reconstruction"], sensitivity_manifest_rows,
    )
    split_counts = {
        split: {"positive": len(positives[split]), "strict_negative": len(negatives[split])}
        for split in ("train", "dev", "calibration", "test")
    }
    manifest = {
        "status": "review_required_low_complexity_bias" if low_complexity_review_required else "complete",
        "created_utc": utcnow(),
        "scientific_label_scope": "computational positives and strict computational negatives; not experimental validation",
        "inputs": {
            "project": str(PROJECT), "samples": str(PROJECT / "samples.tsv"), "reference": str(REFERENCE),
            "reference_fai": str(REFERENCE) + ".fai", "gtf": str(GTF), "broad_matrix": str(BROAD),
            "sample_calls": str(PROJECT / "final/sample_calls"), "reditools_tables": str(PROJECT / "reditools/tables"),
            "vep": str(VEP), "wgs_vcfs": [str(path) for path in VCFS], "old_labels_audit_only": str(OLD_LABELS),
        },
        "bam_selection": {sample: str(BAMS[sample]) for sample in SAMPLE_NAMES},
        "pileup_rules": {
            "command_core": "samtools mpileup -B -A -q 30 -Q 20 --ff 3844 -d 1000000",
            "filters": FILTER_DESCRIPTION, "usable_depth": "A+C+G+T", "minimum_usable_depth": 20,
        },
        "positive_definition": {
            "default": "positive_main", "corrected_efficiency": "strictly >0.10", "group_coverage": ">=2/3 each",
            "control_median": "<=0.02", "WGS_center": "absent from either VCF",
        },
        "negative_definition": "strict: all six usable_depth>=20 and all six target_alt_count=0; broad candidates and WGS centers excluded",
        "mappability_limitation": "NA_RESOURCE_MISSING for every site; MAPQ>=30 is only a pileup read filter and is not treated as mappability validation",
        "low_complexity_rule": {
            "entropy": "single-base Shannon entropy over complete 101nt A/C/G/T window, log2, theoretical range 0-2; fail <1.20",
            "homopolymer": "longest consecutive run of any single base; fail >=20nt",
            "dinucleotide": "partition into non-overlapping 2-mers at phases 0 and 1; maximum consecutive identical-2mer run coverage bases/101; fail >=0.80",
            "combination": "OR",
            "N_handling": "sequence QC exclusion before metric computation",
        },
        "random_seed": SEED, "split_proportions_target": {"train": 0.70, "dev": 0.10, "calibration": 0.10, "test": 0.10},
        "split_optimization_metadata": ["positive count", "total row count", "positive efficiency bin", "positive chromosome", "positive region", "positive GC bin"],
        "gene_level_coverage_proxy": {
            "path": str(run_dir / "gene_coverage_summary.tsv.gz"),
            "definition": "per-gene mean of per-site six-sample median usable depth over retained positive_main and strict-negative modeling-universe sites; not TPM",
            "matched_negative_use": "log2 bin included in train-only exact matching strata",
        },
        "matched_negative_fields": ["site depth log2 bin", "gene coverage proxy log2 bin", "GC bin", "center 5-mer", "transcript region", "mappability status", "chromosome class"],
        "counts": {
            "positive_main": main_count, "positive_high_confidence": high_count, "strict_negative": strict_rows,
            "low_complexity": {
                "positive_before": positive_without_lc_count, "positive_excluded": positive_lc_excluded,
                "positive_exclusion_rate": positive_lc_rate, "negative_strict_before": negative_without_lc_count,
                "negative_strict_excluded": negative_lc_excluded, "negative_strict_exclusion_rate": negative_lc_rate,
            },
            "splits": split_counts, "fixed_sets": fixed_summary,
        },
        "low_complexity_bias_guard": {
            "positive_exclusion_over_10_percent": positive_over_10pct,
            "positive_exclusion_over_twice_negative": positive_over_2x_negative,
            "review_required_no_SUCCESS_pointer": low_complexity_review_required,
        },
        "test_prevalence": fixed_summary["test"]["prevalence"],
        "code_sha256": {path.name: sha256_file(path) for path in sorted((run_dir / "scripts").glob("*")) if path.is_file()},
        "git_status": "working directory /run/media/.../lamar7.21 was not a Git repository; code SHA-256 recorded",
        "negative_shards": shard_paths,
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary = f"""# LAMAR binary dataset summary

This run produced **computational** labels. It does not claim experimentally verified true positives or true negatives.

- `positive_main`: {main_count:,}
- `positive_high_confidence`: {high_count:,}
- strict computational negative universe: {strict_rows:,}
- strict all-six depth >=20 and all-six target-alt=0 assertion: PASS ({strict_failures} violations)
- split counts: `{json.dumps(split_counts, sort_keys=True)}`
- calibration 1:1000 achieved: {fixed_summary['calibration']['achieved_1to1000']} (prevalence {fixed_summary['calibration']['prevalence']:.8f})
- locked test 1:1000 achieved: {fixed_summary['test']['achieved_1to1000']} (prevalence {fixed_summary['test']['prevalence']:.8f})
- gene/window/exact-sequence/genomic-key leakage: none detected (all assertions PASS)
- old-vs-new comparable label agreement: {old_agree:,}/{len(old_audit):,}; disagreement {len(old_audit)-old_agree:,}/{len(old_audit):,}
- low-complexity positive_main exclusion: {positive_lc_excluded:,}/{positive_without_lc_count:,} ({positive_lc_rate:.4%})
- low-complexity strict-negative exclusion: {negative_lc_excluded:,}/{negative_without_lc_count:,} ({negative_lc_rate:.4%})
- low-complexity bias guard review required: {low_complexity_review_required}

## Known limitations and unmet resources

- No external basewise mappability track was present. Every site records `mappability=NA_RESOURCE_MISSING`; MAPQ>=30 pileup read filtering is not described as mappability validation and low-complexity filtering is not a substitute.
- `gene_coverage_summary.tsv.gz` is a gene-level coverage proxy derived from usable depths over this modeling universe, not an independent TPM/abundance estimate; it is recorded and used for train-only matched-negative strata.
- Low complexity was evaluated identically for positives and negatives before splitting, using complete 101-nt A/C/G/T windows and the approved OR rule. Trigger fields and the actual tandem-dinucleotide maximum coverage ratio are retained per site.
- Pooled Fisher/BH values are read-level screening statistics, not biological-replicate experimental validation.
- The specified igem environment lacks pandas, NumPy, and pyarrow; standard-library streaming gzip TSV shards were used without installing dependencies.
"""
    (run_dir / "dataset_summary.md").write_text(summary)
    if low_complexity_review_required:
        (run_dir / "REVIEW_REQUIRED").write_text(
            f"positive_exclusion_rate={positive_lc_rate:.12g}\n"
            f"negative_exclusion_rate={negative_lc_rate:.12g}\n"
            f"positive_over_10pct={positive_over_10pct}\n"
            f"positive_over_2x_negative={positive_over_2x_negative}\n"
        )
        logger.log("All computational QC passed, but low-complexity bias guard triggered; SUCCESS/pointer intentionally not written")
    else:
        (run_dir / "SUCCESS").write_text(utcnow() + "\n")
        pointer = run_dir.parent / "LATEST_SUCCESSFUL_RUN.txt"
        temporary = run_dir.parent / f".LATEST_SUCCESSFUL_RUN.{os.getpid()}"
        temporary.write_text(str(run_dir) + "\n")
        os.replace(temporary, pointer)
        logger.log("All QC and low-complexity bias guard passed; SUCCESS and LATEST_SUCCESSFUL_RUN.txt written")


def stage_smoke(run_dir: Path, logger: Logger) -> None:
    if (run_dir / "SMOKE_OK").exists():
        logger.log("Smoke test already passed")
        return
    if not (run_dir / "PREFLIGHT_OK").exists():
        raise RuntimeError("PREFLIGHT_OK is required before smoke testing")
    stage_positive(run_dir, logger)
    prepare_gtf_shards(run_dir, logger)
    stage_negative_chrom(run_dir, logger, "chr21", smoke=True)
    strict_path = run_dir / "negative_universe_strict/chr21.smoke.tsv.gz"
    strict_rows = read_tsv(strict_path)
    violations = 0
    for row in strict_rows:
        if len(row["sequence_context"]) != 101 or row["sequence_context"][50] != "C":
            violations += 1
        if any(int(row[f"{sample}_usable_depth"]) < MIN_DEPTH or int(row[f"{sample}_target_alt_count"]) != 0 for sample in SAMPLE_NAMES):
            violations += 1
    if not strict_rows or violations:
        raise RuntimeError(f"Smoke failed: strict_rows={len(strict_rows)}, violations={violations}")
    (run_dir / "SMOKE_OK").write_text(
        json.dumps({"completed_utc": utcnow(), "chromosome": "chr21", "strict_rows": len(strict_rows), "violations": violations}) + "\n"
    )
    logger.log(f"Smoke PASS: chr21 bounded exon-union strict rows={len(strict_rows):,}")


def stage_pileup_smoke(run_dir: Path, logger: Logger) -> None:
    """Count 20 deterministic broad sites without applying scientific labels."""
    output = run_dir / "work/pileup_smoke.tsv"
    if output.exists() and (run_dir / "PILEUP_SMOKE_OK").exists():
        logger.log("Pileup-only smoke already passed")
        return
    candidates = [row for row in read_tsv(BROAD) if row["chrom"] == "chr21"]
    chosen = sorted(candidates, key=lambda row: int(row["position"]))[:20]
    by_center = {(row["chrom"], int(row["position"])): row for row in chosen}
    bed = run_dir / "work/pileup_smoke.bed"
    with bed.open("w") as handle:
        for chrom, position in sorted(by_center):
            handle.write(f"{chrom}\t{position-1}\t{position}\n")
    counts_by_center = {
        (chrom, position): counts
        for chrom, position, _, counts in multi_pileup(
            bed, logger, run_dir / "logs/pileup_smoke.mpileup.stderr.log",
            region=f"chr21:{min(position for _, position in by_center)-50}-{max(position for _, position in by_center)+50}",
        )
    }
    rows = []
    violations = 0
    for key, candidate in sorted(by_center.items()):
        row: dict[str, object] = {
            "chrom": key[0], "position": key[1], "genomic_ref": candidate["ref"], "genomic_alt": candidate["alt"]
        }
        add_counts(row, counts_by_center.get(key, {}), candidate["ref"], candidate["alt"])
        for sample in SAMPLE_NAMES:
            depth = int(row[f"{sample}_usable_depth"])
            subtotal = int(row[f"{sample}_ref_count"]) + int(row[f"{sample}_target_alt_count"]) + int(row[f"{sample}_other_alt_count"])
            if depth != subtotal or depth < 0:
                violations += 1
        rows.append(row)
    fields = ["chrom", "position", "genomic_ref", "genomic_alt"] + COUNT_FIELDS
    write_tsv(output, fields, rows)
    if len(rows) != 20 or violations:
        raise RuntimeError(f"Pileup smoke failed: rows={len(rows)}, count invariant violations={violations}")
    (run_dir / "PILEUP_SMOKE_OK").write_text(json.dumps({"completed_utc": utcnow(), "sites": 20, "samples": 6, "violations": 0}) + "\n")
    logger.log("Pileup-only smoke PASS: 20 sites x six MarkDuplicates BAMs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("pileup-smoke", "positive", "prepare-gtf", "smoke", "negative", "negative-chrom", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--chromosome", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not (run_dir / "PREFLIGHT_OK").exists():
        raise RuntimeError("Critical preflight has not passed")
    logger = Logger(run_dir)
    logger.log(f"Starting stage={args.stage}; run_dir={run_dir}")
    if args.stage == "pileup-smoke":
        stage_pileup_smoke(run_dir, logger)
    elif args.stage == "positive":
        stage_positive(run_dir, logger)
    elif args.stage == "prepare-gtf":
        prepare_gtf_shards(run_dir, logger)
    elif args.stage == "smoke":
        stage_smoke(run_dir, logger)
    elif args.stage == "negative-chrom":
        if not args.chromosome:
            raise ValueError("--chromosome is required for negative-chrom")
        prepare_gtf_shards(run_dir, logger)
        for chrom in args.chromosome:
            stage_negative_chrom(run_dir, logger, chrom)
    elif args.stage == "negative":
        stage_negative(run_dir, logger, args.chromosome or None)
    elif args.stage == "finalize":
        stage_finalize(run_dir, logger)
    elif args.stage == "all":
        stage_smoke(run_dir, logger)
        stage_negative(run_dir, logger)
        stage_finalize(run_dir, logger)
    logger.log(f"Finished stage={args.stage}")
    if args.stage in {"finalize", "all"} and (
        (run_dir / "SUCCESS").exists() or (run_dir / "REVIEW_REQUIRED").exists()
    ):
        write_checksums(run_dir)


if __name__ == "__main__":
    main()
