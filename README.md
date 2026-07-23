# LAMAR identifies sequence contexts for C-to-U RNA-editing follow-up

Experimental validation cannot cover every cytosine observed across the
transcriptome. This project develops a sequence-level screening system that
ranks 101-nucleotide (nt) contexts for targeted C-to-U RNA-editing follow-up.
The work combines a leakage-controlled computational dataset, simple sequence
baselines, pretrained LAMAR representations, parameter-efficient adaptation,
probability calibration and a single locked-test evaluation.

The model answers one practical question: **which sequence contexts should be
prioritized for experimental validation?** It does not estimate editing
efficiency, establish an editing mechanism or replace prospective experiments.
Every positive and negative label in this repository is computational.

![Study workflow](igem_drylab_wiki/assets/pipeline_overview.svg)

## Study at a glance

| Component | Frozen outcome |
| --- | ---: |
| Main computational positives | 1,513 |
| High-confidence computational positives | 1,457 |
| Strict computational-negative universe | 2,821,734 |
| Input length | 101 nt |
| Train positives | 1,028 |
| Development set | 159 positives + 1,590 negatives |
| Calibration set | 165 positives + 165,000 negatives |
| Locked test | 161 positives + 161,000 negatives |
| Selected model | Center-pooled LAMAR with q/k/v/o LoRA, rank 4 |
| Locked-test average precision | 0.171958 |
| Locked-test precision | 0.476190 |
| Locked-test recall | 0.124224 |
| Locked-test false positives per million negatives | 136.646 |

At the frozen operating threshold, the selected model returned 42 candidates
from the locked test. Twenty were computational positives and 22 were strict
computational negatives. Accuracy is deliberately not emphasized because a
1:1000 screening task is dominated by the negative class.

## From RNA-seq reads to defensible labels

### Unified six-sample recounting

Three treated and three control RNA-seq samples were recounted from
MarkDuplicates BAMs. Positive and negative sites used the same pileup
implementation and read filters:

- mapping quality of at least 30;
- base quality of at least 20;
- exclusion of unmapped, secondary, QC-failed, duplicate-flagged and
  supplementary reads;
- one count for overlapping paired-end observations;
- usable depth defined as filtered A+C+G+T depth;
- a minimum qualifying usable depth of 20.

The sequence context was oriented to the transcript. Positive-strand C-to-T
events and negative-strand G-to-A events therefore share one representation.
Every retained sequence contains 101 nt, and its central base at index 50 is C.

### Computational positives

Candidate positives began with a broad 9,930-site matrix, but every site was
recounted from the six selected BAMs. The corrected editing efficiency was
defined as:

`max(treated_median - control_median, 0)`.

The main computational-positive class required:

1. corrected editing efficiency strictly greater than 0.10;
2. at least two depth-qualified replicates in each experimental group;
3. a control median no greater than 0.02;
4. a valid transcript-oriented 101-nt context centered on C;
5. no central-site overlap with either whole-genome sequencing VCF;
6. deterministic genomic-key deduplication.

This procedure yielded 1,513 main computational positives. A
high-confidence audit subset imposed all-six coverage, false-discovery-rate
and replicate-variability criteria. It retained 1,457 sites.

### Expression-supported strict negatives

An unreported event is not automatically a negative. Strict computational
negatives were enumerated from exonic transcript-oriented C sites and required
direct RNA-seq coverage evidence. Every strict negative had usable depth of at
least 20 and zero target-alternative reads in all six samples.

The site also had to remain outside every positive pool and the broad candidate
matrix. Central whole-genome variants, incomplete contexts, ambiguous
orientation and failed sequence quality control were excluded. These rules
produced 2,821,734 strict computational negatives, with zero violations of the
all-six coverage and zero-alternative-read assertion.

### Low-complexity control

The same deterministic low-complexity rule was applied to both classes before
data splitting. A sequence was excluded when any condition was met:

- base-2 single-nucleotide Shannon entropy below 1.20;
- a homopolymer run of at least 20 nt;
- phased dinucleotide-repeat coverage of at least 0.80.

No main computational positive was removed. The rule removed 2,087 of
2,823,821 potential strict negatives, corresponding to 0.0739%.

External basewise mappability was unavailable. The dataset records this as
`NA_RESOURCE_MISSING`. Mapping-quality and low-complexity filters are not
presented as substitutes for external mappability validation.

## Splitting before sampling prevents leakage

The complete positive and negative universes were grouped before sampling.
Sites entered the same leakage group when they shared a gene, genomic center,
exact sequence or overlapping strand-aware 101-nt genomic window.

Deterministic group assignment produced four immutable splits:

| Split | Positives | Evaluation negatives | Role |
| --- | ---: | ---: | --- |
| Train | 1,028 | Dynamic train-only pool | Parameter learning |
| Development | 159 | 1,590 | Architecture and hyperparameter selection |
| Calibration | 165 | 165,000 | Calibration and threshold selection |
| Locked test | 161 | 161,000 | One final evaluation |

Post-split assertions detected no crossing gene, leakage group, exact sequence
or genomic key. No locked-test row appeared in train, development or
calibration data. The locked test was not used for negative mining, model
selection, probability calibration or threshold adjustment.

## Model development

### Establishing sequence-only baselines

Two compact baselines tested whether LAMAR learned more than short motifs or
base composition:

- logistic regression using 1-mer to 4-mer counts, GC fraction, C count and
  sequence entropy;
- a convolutional neural network operating on one-hot encoded 101-nt inputs.

The locked-test average precision (AP) was 0.006598 for the k-mer model and
0.013912 for the convolutional network.

### Comparing LAMAR adaptation strategies

The model study compared four training modes:

1. a frozen LAMAR backbone with a trainable classification head;
2. partial fine-tuning of the final one, two or four transformer blocks;
3. low-rank adaptation (LoRA) of attention projections;
4. full fine-tuning as a resource-intensive comparison.

Training negatives were drawn only from the train split and refreshed between
epochs. Staged development-set experiments compared sampling ratios from 1:1
to 1:20, negative difficulty, loss functions, learning rates, regularization,
warmup, batch size and LoRA configuration.

The selected configuration used:

- center-token pooling;
- LoRA on q/k/v/o attention projections;
- rank 4, alpha 8 and dropout 0.05;
- binary cross-entropy;
- dynamic random strict negatives at a 1:10 ratio;
- backbone and head learning rates of `1e-5` and `1e-4`;
- batch size 16 with two-step gradient accumulation;
- warmup ratio 0.03, no weight decay and early stopping.

The final configuration was repeated with seeds 42, 43 and 44. Its
development AP was `0.808719 ± 0.013597`.

## What pretraining contributed

The original pretrained checkpoint was evaluated without loading an adapter,
fine-tuned checkpoint or classification head. All 85,851,793 backbone
parameters remained frozen during embedding extraction.

Center-token representations were evaluated in two ways. Centroid scoring used
no gradient-based LAMAR adaptation, although labeled train examples estimated
the class centroids. Logistic regression then tested how much information a
769-parameter linear probe could extract from fixed embeddings.

| Model | Trained parameters | Development AP | Locked-test AP |
| --- | ---: | ---: | ---: |
| K-mer logistic regression | 344 | 0.330512 | 0.006598 |
| Convolutional neural network | 59,969 | 0.365052 | 0.013912 |
| LAMAR center centroid | 0 | 0.591302 | 0.036180 |
| LAMAR center linear probe | 769 | 0.698583 | 0.066262 |
| Frozen LAMAR head | 2,305 | 0.686136 | 0.064390 |
| Full LAMAR fine-tuning | 85,854,098 | 0.799066 | 0.096254 |
| Partial LAMAR, two blocks | 14,179,585 | 0.775930 | 0.103259 |
| LoRA LAMAR | 297,217 | 0.825438 | 0.171958 |

The unadapted center representation exceeded both simple baselines. The linear
probe and frozen neural head performed similarly. LoRA added 0.105695
absolute test AP over the linear probe, consistent with useful task-specific
representation adaptation.

Center location was crucial. Mean-pooling test AP was 0.004307 for centroid
scoring and 0.005391 for the linear probe. These results support a
position-specific sequence signal rather than a uniformly distributed
sequence summary.

## Calibration and locked-test evaluation

Training ratios do not represent the expected screening prevalence. Raw
sigmoid outputs were therefore not treated as population probabilities.
Uncalibrated, Platt and isotonic mappings were compared only on the fixed
1:1000 calibration set.

Platt scaling was selected. The final threshold,
`0.24298369973628076`, maximized recall under a target of at most 100 false
positives per million calibration negatives. The calibration set achieved
96.97 FP/M.

The unchanged model, calibrator and threshold were then evaluated once on the
locked test:

| Metric | Locked-test value |
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

The calibration target was exceeded on the locked test. We did not tighten the
threshold after observing this result, because such a change would convert the
test into a development set.

![Locked-test precision-recall curves](igem_drylab_wiki/assets/PR_curves.png)

## Shortcut analysis and limitations

Sequence-only input prevents direct use of coverage metadata, but it does not
eliminate correlations introduced during dataset construction. A metadata-only
logistic model reached development AP 0.759266. The fixed LAMAR embedding
reached 0.698583, and the combined representation reached 0.813666.

The strongest embedding-principal-component association was between PC3 and GC
fraction (`r=0.637`). Coverage, expression and sequence-composition shortcuts
therefore remain plausible. Prospective data from an independently generated
experiment are needed to measure transportability.

Additional limitations include one six-sample biological system, missing
external mappability validation and finite-depth negative labels. A strict
computational negative is strongly supported by coverage and zero observed
target reads, but it is not proof of universal editing absence.

## Repository map

| Path | Contents |
| --- | --- |
| [`lamar_binary_project/dataset_build/`](lamar_binary_project/dataset_build/) | Dataset scripts, manifests, QC summaries, filter funnels and selected compact outputs |
| [`lamar_binary_project/model_training/`](lamar_binary_project/model_training/) | Training code, experiment configurations, leaderboard, selected models and error analyses |
| [`lamar_binary_project/zero_shot_ablation/`](lamar_binary_project/zero_shot_ablation/) | Embedding extraction, probes, calibration, ablation results and figures |
| [`igem_drylab_wiki/README.md`](igem_drylab_wiki/README.md) | Single-page English iGEM dry-lab Wiki |
| [`codex_handoff/`](codex_handoff/) | Frozen study rules, file guide and next-agent handoff |
| [`tools/`](tools/) | Public-archive and copied-binary validation |

## Reproducibility

Three immutable successful runs define the study:

- dataset construction: `run_20260721T231520Z`;
- model comparison: `run_20260722T203752Z`;
- pretrained-representation ablation: `run_20260723T045155Z`.

Start with
[`lamar_binary_project/REPRODUCIBILITY.md`](lamar_binary_project/REPRODUCIBILITY.md).
The public snapshot uses placeholders such as `${LAMAR_WORK_ROOT}` and
`${IGEM_DATA_ROOT}` instead of user-specific paths. Configuration files,
software records, run markers and original server checksums preserve the
successful-run provenance.

Validate the public archive with:

```bash
python3 tools/validate_public_archive.py
python3 tools/verify_copied_binary_checksums.py
shasum -a 256 -c PUBLIC_CHECKSUMS.sha256
```

The complete server output is approximately 8.5 GB. Ordinary Git history
contains source code, compact models, paper-level tables, error analyses,
figures and documentation. Multi-gigabyte embeddings, full negative pools,
SQLite stores and complete prediction matrices remain outside Git.
[`ARTIFACT_POLICY.md`](lamar_binary_project/ARTIFACT_POLICY.md) records every
omission class, while the server manifests retain their SHA-256 checksums.

## Responsible interpretation

Use this system to prioritize candidates under a declared false-positive
budget. Report outputs as computational positives or strict computational
negatives. Pooled-read screening statistics are not biological-replicate
validation, and calibrated probabilities are conditional on this data
pipeline.

The recommended next step is prospective evaluation in a separately generated
experiment. The model, calibration function and operating threshold should be
frozen before those labels are inspected.
