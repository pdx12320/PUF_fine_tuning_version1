# Experimental design

## Research question

Given a transcript-oriented 101-nt sequence centered on cytosine, estimate
whether the locus is a reliable computational C-editing candidate worth
experimental follow-up. The target is binary discovery, not editing-efficiency
regression.

## Input experiment

Six RNA-seq samples were analyzed:

- treated: T1, T2, T3
- control: C1, C2, C3

Every count used the same MarkDuplicates BAM pileup semantics:

- mapping quality at least 30;
- base quality at least 20;
- no unmapped, secondary, QC-failed, duplicate-flagged, supplementary reads;
- paired-end overlap counted once;
- usable depth is filtered A+C+G+T depth;
- minimum qualifying usable depth is 20.

For positive recounting, per-replicate target-edit rates were summarized by
treated and control medians and median absolute deviations. The computational
effect was

`max(treated_median - control_median, 0)`.

## Computational-positive definition

The main positive class required:

1. corrected editing efficiency strictly greater than 0.10;
2. at least two of three treated and two of three control replicates with
   usable depth at least 20;
3. control median no greater than 0.02;
4. valid 101-nt transcript-oriented sequence with center `C`;
5. no central-site overlap with either WGS VCF;
6. deterministic genomic-key deduplication.

The high-confidence audit subset additionally required all six depths at least
20, BH-FDR below 0.05, treated MAD at most 0.05, and control MAD at most 0.02.

## Strict computational-negative definition

Exonic transcript-oriented C sites were enumerated from primary-assembly
annotation. A strict negative required all of the following:

- all six BAMs have usable depth at least 20;
- all six target-alt counts equal zero;
- the site is absent from the broad candidate matrix and all positive pools;
- the center is absent from both WGS VCFs;
- direction, reference base, complete 101-nt sequence, and sequence QC pass;
- low-complexity filters do not trigger.

Low complexity used an OR rule applied identically before splitting:

- base-2 single-nucleotide Shannon entropy below 1.20;
- any homopolymer run at least 20 nt;
- maximum phased dinucleotide-repeat coverage at least 0.80.

External basewise mappability was unavailable and is recorded as
`NA_RESOURCE_MISSING`; MAPQ filtering and low-complexity filtering are not
described as equivalent to mappability validation.

## Leakage-safe splitting

Candidate sites were grouped before sampling. Sites share a leakage group if
they share a gene, genomic center, exact sequence, or overlapping
transcript-oriented genomic window. Deterministic group assignment produced:

| Split | Positives | Fixed negatives used for evaluation |
| --- | ---: | ---: |
| train | 1,028 | dynamic train-only strict pool |
| dev | 159 | 1,590 |
| calibration | 165 | 165,000 |
| locked test | 161 | 161,000 |

The full strict-negative universe contains 2,821,734 sites. No gene,
leakage-group, exact sequence, genomic key, or held-out row crossed splits.

## Model comparison

Sequence-only baselines:

- 1- to 4-mer logistic regression with GC, C count, and entropy;
- one-hot 101-nt CNN.

LAMAR comparisons:

- frozen backbone with center, mean, and attention heads;
- partial unfreezing of the final 1, 2, or 4 transformer blocks;
- LoRA target-module and rank/dropout ablations;
- full fine-tuning;
- zero-gradient centroid scoring of pretrained embeddings;
- frozen-embedding logistic linear probes.

Training negatives were redrawn from the train split only. Sampling ratios,
negative difficulty classes, losses, learning rates, LoRA configurations, and
regularization were selected using dev AP. The selected configuration was
repeated for seeds 42, 43, and 44.

## Calibration and thresholding

Raw sigmoid scores are not interpreted as 1:1000 population probabilities.
No-calibration, Platt, and isotonic methods were compared on the fixed
calibration split. The final operating threshold maximized recall subject to
no more than 100 false positives per million calibration negatives.

The final model used Platt scaling and threshold `0.24298369973628076`.
Calibration achieved 96.97 FP/M. The unchanged threshold produced 136.65 FP/M
on locked test; the threshold must not be altered using test outcomes.

## Locked-test results

| Metric | Value |
| --- | ---: |
| Average precision | 0.171958 |
| PR-AUC | 0.169754 |
| Precision | 0.476190 |
| Recall | 0.124224 |
| F1 | 0.197044 |
| MCC | 0.242859 |
| Brier score | 0.00090393 |
| ECE | 0.00017023 |
| FP per million negatives | 136.646 |
| ROC-AUC, supplementary | 0.957783 |

At the frozen threshold there were 20 true positives, 22 false positives,
141 false negatives, and 160,978 true negatives.

## Representation ablation

Center-token pretrained representations were substantially stronger than
mean or CLS pooling. Test AP values were:

- zero-gradient center centroid: 0.036180;
- center linear probe: 0.066262;
- frozen LAMAR head: 0.064390;
- LoRA LAMAR: 0.171958;
- CNN: 0.013912;
- k-mer logistic: 0.006598.

The linear probe and frozen head were similar, while LoRA added 0.105695
absolute test AP over the linear probe. This is consistent with useful
representation adaptation, but the observational ablation does not uniquely
attribute every gain.

## Shortcut analysis

On dev, the fixed-embedding linear probe AP was 0.698583, a metadata-only
logistic model reached 0.759266, and their combination reached 0.813666.
The largest absolute embedding-PC correlation was PC3 with GC fraction
(`r=0.637`). The model receives sequence only, but GC/C-count and
coverage-related data-generation bias cannot be excluded.
