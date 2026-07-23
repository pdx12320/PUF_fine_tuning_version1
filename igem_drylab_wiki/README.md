# Sequence-informed prioritization of C-to-U RNA-editing candidates

## Overview

C-to-U RNA editing can diversify transcript function, but experimental
validation across every transcriptomic cytosine is impractical. We developed a
dry-lab system that ranks 101-nucleotide (nt) sequence contexts for targeted
follow-up. The model addresses a screening question: **which central
cytosines are sufficiently supported to justify experimental validation?**

The study combines unified RNA-seq recounting, expression-supported negative
selection, leakage-controlled splitting, sequence baselines, pretrained LAMAR
representations, parameter-efficient adaptation and prevalence-aware
calibration. It does not predict editing efficiency. Its outputs are
computational priorities, not experimentally verified editing events.

![End-to-end dry-lab workflow](assets/pipeline_overview.svg)

## Why the dataset required a new design

A classifier can appear successful when its labels encode technical shortcuts.
Two risks were especially important here. First, an uncovered cytosine cannot
be treated as unedited because no read tested that hypothesis. Second, random
row splitting can place related genes, overlapping windows or duplicate
sequences in both training and evaluation data.

We therefore designed the dataset around three requirements:

1. positives and negatives must use identical pileup semantics;
2. strict negatives must have direct expression and coverage evidence;
3. related sequence contexts must remain within one data split.

This design shifts the task from separating called sites from arbitrary
genomic positions to separating two deeply measured computational classes.

## Recounting six RNA-seq samples

Three treated and three control MarkDuplicates BAMs were processed with one
pileup implementation. Every sample used the same read-level filters:

- mapping quality of at least 30;
- base quality of at least 20;
- exclusion of unmapped, secondary, QC-failed, duplicate-flagged and
  supplementary reads;
- one count for overlapping paired-end observations;
- usable depth defined as filtered A+C+G+T depth;
- a minimum qualifying usable depth of 20.

The model input was normalized to transcript orientation. A positive-strand
C-to-T event and a negative-strand G-to-A event therefore map to the same
C-to-U interpretation. Every retained context contains 101 nt and has C at
the central zero-based index 50.

For each candidate, replicate editing rates were summarized with treated and
control medians. Replicate variability was measured with the median absolute
deviation. Corrected editing efficiency was calculated as:

`max(treated_median - control_median, 0)`.

The accompanying Fisher and Benjamini-Hochberg values are pooled-read
screening statistics. They do not constitute biological-replicate
experimental validation.

## Defining computational positives

All positive candidates began with a broad 9,930-site matrix, but their counts
were recomputed from the six selected BAMs. A main computational positive
required:

1. corrected editing efficiency strictly greater than 0.10;
2. at least two depth-qualified treated and two depth-qualified control
   replicates;
3. a control median no greater than 0.02;
4. a valid transcript-oriented 101-nt sequence centered on C;
5. no central-site overlap with either whole-genome sequencing VCF;
6. deterministic genomic-key deduplication.

These criteria yielded 1,513 main computational positives. A
high-confidence audit subset also required usable depth of at least 20 in all
six samples and BH-FDR below 0.05. Treated MAD could not exceed 0.05, while
control MAD could not exceed 0.02. This subset contained 1,457 sites.

## Defining expression-supported strict negatives

Strict computational negatives were not selected merely because they were
absent from the candidate table. We enumerated transcript-oriented exonic
cytosines and required direct RNA-seq evidence at every retained site.

Each strict negative satisfied all of the following:

- usable depth of at least 20 in all six samples;
- target-alternative count exactly zero in all six samples;
- no overlap with a main, sensitivity or broad ambiguous candidate center;
- no central-site occurrence in either whole-genome sequencing VCF;
- valid strand, reference base and complete 101-nt context;
- no low-complexity trigger.

The resulting universe contained 2,821,734 strict computational negatives.
The all-six coverage and zero-target-alternative assertion had zero violations.
Finite read depth cannot prove universal editing absence, so these sites remain
computational negatives.

## Applying identical sequence-complexity rules

The same deterministic rule was applied to both classes before splitting. A
site was marked as low complexity when any condition was met:

- base-2 single-nucleotide Shannon entropy below 1.20;
- a homopolymer run of at least 20 nt;
- maximum two-phase dinucleotide-repeat coverage of at least 0.80.

No main computational positive was removed. The rule excluded 2,087 of
2,823,821 potential strict negatives, corresponding to 0.0739%.

External basewise mappability was unavailable and is recorded as
`NA_RESOURCE_MISSING`. Mapping-quality filtering and sequence-complexity
filtering are not equivalent to external mappability validation.

## Preventing information leakage

The complete positive and negative universes were grouped before sampling.
Sites shared a leakage group when they had any of the following relationships:

- the same gene;
- the same genomic center;
- the same 101-nt sequence;
- overlapping strand-aware 101-nt genomic windows.

Deterministic assignment then produced four immutable splits:

| Split | Positives | Negatives used in evaluation | Purpose |
| --- | ---: | ---: | --- |
| Train | 1,028 | Dynamic train-only pool | Parameter learning |
| Development | 159 | 1,590 | Model and hyperparameter selection |
| Calibration | 165 | 165,000 | Calibration and threshold selection |
| Locked test | 161 | 161,000 | One final evaluation |

Quality-control assertions found no gene, leakage group, exact sequence or
genomic key crossing splits. No locked-test row appeared in train,
development or calibration data.

The locked test remained excluded from model selection, hard-negative mining,
calibration fitting and threshold choice. It was opened only after these
decisions had been frozen.

## Establishing baselines before adapting LAMAR

We first trained two sequence-only baselines. Logistic regression used 1-mer
to 4-mer counts, GC fraction, C count and sequence entropy. A convolutional
neural network used one-hot encoded 101-nt sequences.

These models tested whether a foundation model added information beyond short
motifs and simple composition. Their locked-test AP values were 0.006598 and
0.013912, respectively.

## Comparing LAMAR training strategies

We evaluated four adaptation strategies:

1. a frozen LAMAR backbone with a trainable classification head;
2. partial fine-tuning of the final one, two or four transformer blocks;
3. LoRA adaptation of attention projections;
4. full fine-tuning as a resource-intensive comparison.

Training negatives were sampled only from the train split and refreshed
between epochs. Development-set experiments compared ratios from 1:1 to 1:20,
random, matched and hard negatives, multiple losses, learning rates,
regularization settings and LoRA configurations.

The selected model used center-token pooling and LoRA on q/k/v/o attention
projections. It used rank 4, alpha 8, dropout 0.05 and binary cross-entropy.
Dynamic random strict negatives were sampled at 1:10. The backbone and head
learning rates were `1e-5` and `1e-4`.

The selected configuration used batch size 16, two-step gradient accumulation,
warmup ratio 0.03, no weight decay and early stopping. Repeated training with
seeds 42, 43 and 44 gave development AP of `0.808719 ± 0.013597`.

## Testing information already present in pretrained LAMAR

The original pretrained checkpoint was loaded without adapters, fine-tuned
weights or a classification head. All 85,851,793 backbone parameters remained
frozen during embedding extraction.

We evaluated center, mean, masked-mean and CLS representations. Labeled train
examples estimated class centroids for zero-gradient scoring. Logistic
regression then tested how much information a 769-parameter linear probe could
extract from fixed embeddings.

The term “zero-shot” denotes zero gradient-based LAMAR adaptation in this
study. It is not label-free because train labels estimate the centroids.

## Model comparison on the locked test

All model choices, calibration functions and thresholds were frozen before the
locked test was opened.

| Model | Trained parameters | Development AP | Test AP | Precision | Recall | FP/M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA LAMAR | 297,217 | 0.825438 | 0.171958 | 0.476190 | 0.124224 | 136.646 |
| Partial LAMAR, two blocks | 14,179,585 | 0.775930 | 0.103259 | 0.391304 | 0.055901 | 86.957 |
| Full LAMAR fine-tuning | 85,854,098 | 0.799066 | 0.096254 | 0.400000 | 0.062112 | 93.168 |
| LAMAR center linear probe | 769 | 0.698583 | 0.066262 | 0.333333 | 0.024845 | 49.689 |
| Frozen LAMAR head | 2,305 | 0.686136 | 0.064390 | 0.285714 | 0.049689 | 124.224 |
| LAMAR center centroid | 0 | 0.591302 | 0.036180 | 0.066667 | 0.006211 | 86.957 |
| Convolutional neural network | 59,969 | 0.365052 | 0.013912 | 0.111111 | 0.012422 | 99.379 |
| K-mer logistic regression | 344 | 0.330512 | 0.006598 | 0 | 0 | 0 |

![Locked-test precision-recall curves](assets/PR_curves.png)

The unadapted center representation exceeded both simple baselines. The linear
probe raised test AP from 0.036180 to 0.066262. Its performance was similar to
the frozen neural head, which reached 0.064390.

LoRA improved test AP by another 0.105695 over the linear probe. This pattern
is consistent with task-specific representation adaptation beyond a simple
classifier. The ablation remains observational and does not uniquely assign
every performance gain.

Center position carried substantially more information than global pooling.
Mean-pooling test AP was 0.004307 for centroid scoring and 0.005391 for the
linear probe.

## Calibrating scores for a 1:1000 screening setting

The training ratio does not represent screening prevalence. Raw sigmoid scores
were therefore not interpreted as real-world probabilities. Uncalibrated,
Platt and isotonic mappings were compared only on the fixed calibration set.

Platt scaling was selected. The final threshold,
`0.24298369973628076`, maximized recall under a target of no more than 100
false positives per million calibration negatives. Calibration achieved
96.97 FP/M.

The unchanged model, calibrator and threshold were then applied once to the
locked test:

| Metric | Value |
| --- | ---: |
| Average precision | 0.171958 |
| PR-AUC | 0.169754 |
| Precision | 0.476190 |
| Recall | 0.124224 |
| F1 | 0.197044 |
| Matthews correlation coefficient | 0.242859 |
| Brier score | 0.00090393 |
| Expected calibration error | 0.00017023 |
| False positives per million negatives | 136.646 |
| ROC-AUC, supplementary | 0.957783 |

At the frozen threshold, the model identified 20 computational positives and
22 strict computational negatives. It missed 141 computational positives.
The calibration target was exceeded on test, but the threshold was not changed
after this observation.

![Locked-test ROC curves](assets/ROC_curves.png)

![Probability calibration](assets/calibration_curve.png)

## Interpreting pretrained representations

The embedding analysis asks whether pretrained LAMAR encoded information
related to the editing label before task adaptation. The center centroid
achieved test AP 0.036180, compared with an empirical random baseline of
0.001143. A linear classifier extracted additional information, while LoRA
adaptation produced the largest gain.

These comparisons support two bounded conclusions. First, pretrained
center-token representations contain sequence information associated with the
computational editing label. Second, task adaptation improves this
representation. Neither result establishes a molecular mechanism.

![Pretrained center-embedding PCA](assets/embedding_PCA.png)

## Checking shortcut learning

The model receives sequence only, but sequence can correlate with coverage,
expression and candidate-generation rules. We therefore compared fixed
embeddings with metadata-only and combined models on the development set.

| Development model | AP |
| --- | ---: |
| LAMAR center embedding | 0.698583 |
| Metadata only | 0.759266 |
| Embedding plus metadata | 0.813666 |

The largest absolute embedding-PC correlation was between PC3 and GC fraction
(`r=0.637`). Metadata remained strongly predictive, so coverage, expression
and sequence-composition bias cannot be excluded.

Among the 100 highest-scoring strict negatives, motif-similar sequences
accounted for 81 zero-shot cases and 99 linear-probe cases. These results
identify motif-similar negatives as a major failure mode rather than proof of
unobserved editing.

![Embedding and metadata correlations](assets/metadata_correlation.png)

## What the model can and cannot support

The model can rank candidate sequences under a declared false-positive budget.
It can support selection of a smaller, auditable validation panel.

The model cannot:

- prove editing in a new biological context;
- establish that a strict negative is universally unedited;
- replace biological replicates or orthogonal validation;
- estimate editing efficiency;
- establish causal recognition by a specific editor;
- transfer calibrated probabilities unchanged to another assay or cell type.

The labels derive from one six-sample system. External basewise mappability was
not available. Sequence composition and dataset-generation bias remain
measurable. These limitations define the boundary of every result reported
here.

## Reproducibility and open documentation

The public archive contains dataset and training source code, experiment
configurations, QC assertions, filter funnels, aggregate prediction tables,
error analyses, figures and compact model artifacts. It also preserves server
and public-copy checksums.

Three immutable successful runs define the study:

- dataset construction: `run_20260721T231520Z`;
- model comparison: `run_20260722T203752Z`;
- pretrained-representation ablation: `run_20260723T045155Z`.

Multi-gigabyte embeddings, full negative pools and complete prediction matrices
remain outside ordinary Git history. Their hashes are retained in the
successful-run manifests. Environment-specific paths use placeholders in the
public archive.

The complete process, code and provenance are documented in the repository
[home page](../README.md) and the
[reproducibility guide](../lamar_binary_project/REPRODUCIBILITY.md).

## Conclusion

Leakage-controlled data construction and realistic calibration were essential
for evaluating C-to-U RNA-editing prioritization. Pretrained LAMAR
representations captured more label-associated sequence information than
k-mer and convolutional baselines. A small linear probe extracted part of this
signal, while LoRA adaptation produced the strongest locked-test performance.

The selected model reached 47.6% precision and 12.4% recall at the frozen
operating point. Its predictions remain hypotheses for experimental follow-up.
Prospective testing on independently generated data is the necessary next
step.
