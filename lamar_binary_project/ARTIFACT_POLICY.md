# Artifact inclusion policy

## What is committed

- complete source scripts and experiment configurations;
- dataset manifests, QC assertions, filter funnels, positive recount tables,
  low-complexity audits, and compact fixed dev files;
- all aggregate baseline, hyperparameter, calibration, threshold, subgroup,
  locked-test, and ablation result tables;
- false-positive and false-negative error-analysis tables;
- all generated figures and the Wiki copies;
- compact baseline models, all three final LoRA adapters, calibration objects,
  zero-shot centroids, PCA, and linear probes;
- original server SHA-256 manifests and run-completion markers.

## What is not committed

| Artifact class | Approximate scale | Reason |
| --- | ---: | --- |
| Full strict/relaxed/near-zero negative universes and train pool | hundreds of MB | Derived full-data tables; preserved by server checksum manifest |
| Fixed 1:1000 calibration/test TSV files and leakage maps | about 100 MB total | Large row-level data; immutable server artifacts |
| Train-pool SQLite databases | 2.6 GB across attempts | Runtime index, not a publication result |
| Full/partial backbone checkpoints | 343 MB and 113 MB largest files | Reconstructable from base checkpoint plus configuration; unsuitable for ordinary Git |
| Full LAMAR embedding matrices | 4.45 GB | Multi-gigabyte derived arrays |
| Calibration/test and all-representation prediction matrices | hundreds of MB | Row-level derived predictions; aggregate metrics and error cases are committed |
| Raw BAM/BAI, reference FASTA/GTF, VCF, and upstream calls | external inputs | Immutable source data, never copied into this repository |

Nothing was omitted silently: each successful run's
`checksums.server.sha256` lists original artifacts and hashes. Manifest JSON
files record row counts, schemas, checkpoint hashes, and protocol markers.

## Public-copy transformation

The original run snapshots contained machine-specific absolute paths. Public
text files replace them with environment-variable placeholders. Consequently,
the server checksum manifests are provenance indexes rather than checksums of
the transformed text files. Use `PUBLIC_CHECKSUMS.sha256` for the GitHub copy.
