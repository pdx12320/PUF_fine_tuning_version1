# Continuation validation checklist

## Read-only preflight

- [ ] Confirm the intended run identifier; do not rely only on `LATEST`.
- [ ] Confirm `SUCCESS` content.
- [ ] Verify the run checksum manifest.
- [ ] Confirm dataset and model input checksums match recorded values.
- [ ] Check train/dev/calibration/test counts.
- [ ] Check sequence length 101 and center base `C`.
- [ ] Confirm zero cross-split gene, group, sequence, locus, and window leakage.
- [ ] Confirm no test-derived parameter will be changed.
- [ ] Record software and checkpoint versions.
- [ ] Inspect free disk, memory, GPU, and active jobs.

## Before a new computation

- [ ] Create a new UTC timestamped output directory.
- [ ] Write config and input manifest before processing.
- [ ] Start with a smoke fixture.
- [ ] Set explicit CPU/GPU/memory limits.
- [ ] Use `nohup` or the institutional scheduler with persistent logs.
- [ ] Obtain explicit approval for full-data/high-resource execution.

## Completion

- [ ] Check process exit code and logs.
- [ ] Check expected files, schemas, and exact row counts.
- [ ] Run scientific QC assertions.
- [ ] Generate SHA-256 checksums.
- [ ] Write `SUCCESS` only after every required check passes.
- [ ] Update `LATEST_SUCCESSFUL_RUN.txt` only after `SUCCESS`.
- [ ] Report scientific limitations separately from computational completion.
