#!/usr/bin/env python3
"""Fail-fast input audit for the 2026-07-21 LAMAR binary dataset build."""

from __future__ import annotations

import csv
import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pysam


PROJECT = Path("${IGEM_DATA_ROOT}/CU5.17_EGFP_GC_paper")
REFERENCE = Path("${IGEM_DATA_ROOT}/GRCh38.primary_assembly.genome.fa")
GTF = Path("${IGEM_DATA_ROOT}/gencode.v50.primary_assembly.annotation.gtf")
SAMTOOLS = Path("${IGEM_ENV}/bin/samtools")
BAM_DIR = PROJECT / "bam/markduplicates"
SAMPLE_ROWS = (
    ("T1", "CU517_GC_T1", "treated", "1", "SRR27885768"),
    ("T2", "CU517_GC_T2", "treated", "2", "SRR27885766"),
    ("T3", "CU517_GC_T3", "treated", "3", "SRR27885765"),
    ("C1", "CU517_GC_C1", "control", "1", "SRR27885767"),
    ("C2", "CU517_GC_C2", "control", "2", "SRR27885764"),
    ("C3", "CU517_GC_C3", "control", "3", "SRR27885763"),
)
OTHER_INPUTS = (
    PROJECT / "samples.tsv",
    REFERENCE,
    Path(str(REFERENCE) + ".fai"),
    GTF,
    PROJECT / "final/CU5.17_EGFP_GC.site_matrix.tsv.gz",
    PROJECT / "final/sample_calls",
    PROJECT / "reditools/tables",
    PROJECT / "vep/CU5.17_EGFP_GC.vep.tsv",
    Path("${IGEM_DATA_ROOT}/293T_CG_GRCh38_retry/293T_CG.GRCh38.PASS.biallelic.SNV.vcf.gz"),
    Path("${IGEM_DATA_ROOT}/293T_CG_GRCh38_retry/293T_CG.GRCh38.PASS.biallelic.SNV.vcf.gz.tbi"),
    Path("${IGEM_DATA_ROOT}/HEK293T_public_WGS_3runs/vcf/HEK293T_3runs.union.SNV.vcf.gz"),
    Path("${IGEM_DATA_ROOT}/HEK293T_public_WGS_3runs/vcf/HEK293T_3runs.union.SNV.vcf.gz.tbi"),
    PROJECT / "lamar_background_corrected/run_20260715T214930Z/background_corrected_labels.tsv.gz",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_samples() -> list[dict[str, str]]:
    with (PROJECT / "samples.tsv").open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def bai_for(bam: Path) -> Path:
    candidates = (Path(str(bam) + ".bai"), bam.with_suffix(".bai"))
    hits = [path for path in candidates if path.is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one readable BAI for {bam}; found {hits}")
    with hits[0].open("rb") as handle:
        if not handle.read(4):
            raise RuntimeError(f"Empty or unreadable BAI: {hits[0]}")
    return hits[0]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: preflight.py RUN_DIR")
    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    failures: list[str] = []
    for path in (*OTHER_INPUTS, SAMTOOLS):
        if not path.exists():
            failures.append(f"missing_required_input:{path}")
    if failures:
        raise RuntimeError(";".join(failures))

    observed_samples = read_samples()
    manifest_expected = {
        (row["sample"], row["group"], row["replicate"], row["srr"])
        for row in observed_samples
    }
    expected = {(long_name, group, replicate, sra) for _, long_name, group, replicate, sra in SAMPLE_ROWS}
    if manifest_expected != expected or len(observed_samples) != 6:
        raise RuntimeError(f"samples.tsv does not exactly match expected six rows: {manifest_expected}")

    fai = {}
    with Path(str(REFERENCE) + ".fai").open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            fai[fields[0]] = int(fields[1])

    all_bams = sorted(BAM_DIR.glob("*.bam"))
    quick_rows: list[dict[str, object]] = []
    compat_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    selected: dict[str, tuple[Path, Path]] = {}
    for short, long_name, group, replicate, sra in SAMPLE_ROWS:
        hits = [path for path in all_bams if long_name in path.name]
        if len(hits) != 1:
            raise RuntimeError(f"{short}/{long_name} matched {len(hits)} BAMs: {hits}")
        bam = hits[0]
        bai = bai_for(bam)
        quick = subprocess.run([str(SAMTOOLS), "quickcheck", "-v", str(bam)], text=True, capture_output=True)
        quick_rows.append({
            "sample": short, "sample_name": long_name, "bam": bam,
            "return_code": quick.returncode, "stdout": quick.stdout.strip(),
            "stderr": quick.stderr.strip(), "status": "PASS" if quick.returncode == 0 else "FAIL",
        })
        if quick.returncode != 0:
            failures.append(f"samtools_quickcheck:{long_name}")
        header_text = subprocess.run(
            [str(SAMTOOLS), "view", "-H", str(bam)], check=True, text=True, capture_output=True
        ).stdout
        with pysam.AlignmentFile(str(bam), "rb") as handle:
            bam_sq = dict(zip(handle.references, handle.lengths))
            try:
                handle.check_index()
                index_status = "PASS"
            except Exception as exc:
                index_status = f"FAIL:{exc}"
                failures.append(f"bam_index_check:{long_name}")
        missing_from_ref = sorted(set(bam_sq) - set(fai))
        length_mismatch = sorted(name for name, length in bam_sq.items() if fai.get(name) != length)
        missing_from_bam = sorted(set(fai) - set(bam_sq))
        exact = not missing_from_ref and not length_mismatch and not missing_from_bam
        sample_in_pg = long_name in header_text
        has_rg = any(line.startswith("@RG\t") for line in header_text.splitlines())
        if not exact:
            failures.append(f"reference_compatibility:{long_name}")
        if not sample_in_pg:
            failures.append(f"header_provenance:{long_name}")
        compat_rows.append({
            "sample": short, "sample_name": long_name, "bam_sq_count": len(bam_sq),
            "fai_contig_count": len(fai), "missing_from_reference_count": len(missing_from_ref),
            "length_mismatch_count": len(length_mismatch), "missing_from_bam_count": len(missing_from_bam),
            "exact_contig_and_length_match": int(exact), "sample_name_in_PG_provenance": int(sample_in_pg),
            "RG_header_present": int(has_rg), "bai_readable_by_pysam": index_status,
            "status": "PASS" if exact and sample_in_pg and index_status == "PASS" else "FAIL",
            "note": "No @RG/SM is expected in these BAMs; @PG STAR/MarkDuplicates command provenance was used",
        })
        bam_stat, bai_stat = bam.stat(), bai.stat()
        header_sha = sha256_bytes(header_text.encode())
        for kind, path, stat, digest, scope in (
            ("bam", bam, bam_stat, header_sha, "samtools_view_header_only"),
            ("bai", bai, bai_stat, sha256_file(bai), "entire_file"),
        ):
            manifest_rows.append({
                "input_kind": kind, "sample": short, "sample_name": long_name, "group": group,
                "replicate": replicate, "sra_accession": sra, "path": path,
                "size_bytes": stat.st_size, "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "sha256": digest, "sha256_scope": scope,
            })
        selected[long_name] = (bam, bai)

    for path in OTHER_INPUTS:
        stat = path.stat()
        manifest_rows.append({
            "input_kind": "directory" if path.is_dir() else "input", "sample": "", "sample_name": "",
            "group": "", "replicate": "", "sra_accession": "", "path": path,
            "size_bytes": stat.st_size, "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "sha256": "NA", "sha256_scope": "recorded_path_size_mtime_only",
        })

    write_tsv(run_dir / "input_manifest.tsv", list(manifest_rows[0]), manifest_rows)
    write_tsv(run_dir / "bam_quickcheck.tsv", list(quick_rows[0]), quick_rows)
    write_tsv(run_dir / "reference_compatibility.tsv", list(compat_rows[0]), compat_rows)
    def package_version(name: str) -> str:
        try:
            from importlib.metadata import version
            return version(name)
        except Exception:
            return "NOT_INSTALLED"

    versions = [
        f"preflight_utc={datetime.now(timezone.utc).isoformat()}",
        f"platform={platform.platform()}", f"python={platform.python_version()}",
        f"python_executable={sys.executable}", f"pysam={pysam.__version__}",
        f"pandas={package_version('pandas')}", f"numpy={package_version('numpy')}",
        f"samtools_path={SAMTOOLS}",
        subprocess.run([str(SAMTOOLS), "--version"], text=True, capture_output=True, check=True).stdout.rstrip(),
    ]
    (run_dir / "software_versions.txt").write_text("\n".join(versions) + "\n")
    if failures:
        (run_dir / "PREFLIGHT_FAILED").write_text("\n".join(failures) + "\n")
        raise RuntimeError("Critical preflight failures: " + ";".join(failures))
    (run_dir / "PREFLIGHT_OK").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    print("PREFLIGHT PASS: six unique MarkDuplicates BAMs, indexes, provenance, and reference dictionaries")


if __name__ == "__main__":
    main()
