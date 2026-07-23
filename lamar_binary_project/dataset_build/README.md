# Dataset construction archive

This folder is the public, path-sanitized snapshot of dataset run
`run_20260721T231520Z`.

## Verified counts

- `positive_main`: 1,513
- `positive_high_confidence`: 1,457
- strict computational negatives: 2,821,734
- all-six depth-at-least-20 and all-six target-alt-zero violations: 0
- old/new label agreement among comparable broad candidates: 9,731/9,930
- low-complexity exclusions: 0/1,513 positives and 2,087/2,823,821 potential
  strict negatives

The dataset reached exact 1:1000 calibration and locked-test ratios without
replacement. All leakage assertions passed.

## Contents

- `scripts/`: executed source and tests;
- `manifests/`: input, BAM, reference, software, and train-pool manifests;
- `results/`: filter funnels, QC, positive recount/audit outputs, compact dev
  data, and held-out positive list;
- `provenance/`: original server checksums and success markers.

Large negative shards and fixed 1:1000 files are indexed in
`provenance/checksums.server.sha256` and intentionally not committed.
