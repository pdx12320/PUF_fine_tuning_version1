# Validation record

## Window preflight

All 27 assertions passed, including:

- exact window set: 21, 41, 61, and 101 bp;
- train positives: 1,028;
- immutable negative pool: 1,975,244 rows;
- complete dev: 282,325 rows;
- complete-dev labels: 159 positive and 282,166 negative;
- all source sequences: 101 nt;
- all biological centers: C at index 50;
- unique dev sequence and genomic identifiers;
- disjoint train/dev leakage groups;
- correct centered definition for every train and dev crop;
- 354 optimizer steps per epoch and 7,080 total steps.

## Formal-run invariants

The four formal summaries were checked for:

- `status == SUCCESS`;
- exactly 7,080 optimizer steps;
- exactly 282,325 complete-dev predictions;
- identical 20-epoch negative-selection hash sequence.

The four prediction tables were then read independently. Their sequence-ID
order and labels were identical, all probabilities were finite, and AP,
ROC-AUC, P@10, and P@100 were recomputed from the stored predictions. The
independent metrics matched the saved summaries.

The 101 bp final trainable checkpoint SHA-256 matched the original
`s1_frozen_center` checkpoint:

```text
c945f7783ae1a3fc4ac5f654c8792b0c800284ae3fd07f11cc0ab27b64436b42
```

## Rank-4 spectrum coverage

Checkpoint validation found:

- 48 LoRA A tensors and 48 paired B tensors;
- every A tensor had shape 4 x 768;
- every B tensor had shape 768 x 4;
- complete coverage of 12 layers and Q/K/V/O modules;
- all computed singular values finite and ordered;
- float32 numerical rank 4 for all 48 updates.

For an independent numerical check, four representative 768 x 768 update
matrices were explicitly materialized and decomposed with a direct full SVD:

- layer 0 K;
- layer 4 K;
- layer 11 Q;
- layer 11 O.

Their singular values agreed with the thin-QR/core-SVD method to a maximum
relative error below `2.7e-12`.

## Excluded artifacts

This public folder does not include:

- raw or processed source datasets;
- the 2 GB negative SQLite pool;
- full 282,325-row prediction tables;
- model checkpoints;
- environment caches or Python bytecode;
- server-specific absolute paths;
- calibration or test artifacts.
