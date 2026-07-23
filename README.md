# AI-guided RNA editing design with LAMAR

> **Prioritizing candidate RNA sites for programmable C-to-U editing using pretrained RNA language models.**

| The biological challenge | Our computational approach |
| --- | --- |
| Experimental validation cannot test every cytosine in the transcriptome. | Our model ranks candidate editing contexts before wet-lab validation. |

![RNA sequence to PUF-APOBEC validation workflow](igem_drylab_wiki/assets/ai_guided_rna_design.svg)

This repository evaluates the sequence-prioritization layer of a broader
programmable RNA-editing workflow. It asks which 101-nucleotide (nt) contexts
should move forward to PUF-APOBEC testing.

The model does not replace experiments. It reduces the search space and helps
allocate experimental effort to a smaller, ranked candidate set.

## Project motivation

> **Takeaway:** Programmable editors need both molecular targeting and an
> informed choice of where to edit.

APOBEC cytidine deaminase activity can support C-to-U conversion in engineered
RNA-editing systems. PUF repeat proteins provide a programmable route to RNA
recognition. Together, these components motivate targeted PUF-APOBEC editor
design.

Choosing an effective central cytosine remains difficult. Local sequence
context can influence whether a candidate resembles sites with observed
editing evidence. Testing thousands of possible sites individually would be
slow and resource intensive.

LAMAR provides a pretrained representation of RNA sequence. We adapt that
representation to rank candidate contexts before experimental testing.

| Biological component | Design role |
| --- | --- |
| APOBEC | Catalytic C-to-U editing activity |
| PUF | Programmable RNA recognition |
| LAMAR | Sequence-based candidate prioritization |
| Wet lab | Experimental confirmation and editor characterization |

## From prediction to experimental design

> **Takeaway:** The model converts a transcriptome-scale search problem into an
> experimentally manageable shortlist.

**Current decision path**

`Thousands of cytosines` → `candidate screening` → `LAMAR ranking` →
`prioritized candidates` → `experimental testing`

For each candidate cytosine, the model receives a transcript-oriented 101-nt
sequence. It returns a ranking score for membership in the computational
editing-positive class.

The score can help decide which candidates enter PUF-APOBEC design and
validation. PUF-binding compatibility, off-target assessment and experimental
performance remain separate design criteria.

## Model performance

> **Takeaway:** LoRA adaptation extracted a stronger editing-associated signal
> than simple sequence baselines or an unchanged LAMAR backbone.

![Locked-test average precision comparison](igem_drylab_wiki/assets/model_performance.svg)

LoRA LAMAR increased locked-test average precision (AP) to **0.171958**.
The corresponding values were 0.006598 for k-mer logistic regression,
0.013912 for the convolutional neural network and 0.064390 for frozen LAMAR.
Full fine-tuning reached 0.096254.

![Locked-test precision-recall curves](igem_drylab_wiki/assets/PR_curves.png)

In a 1:1000 screening setting, the model prioritizes candidates under a
false-positive budget. At the frozen operating point, it achieved:

| Metric | Locked-test value |
| --- | ---: |
| Average precision | 0.171958 |
| Precision | 0.476190 |
| Recall | 0.124224 |
| False positives per million negatives | 136.646 |

The model returned 42 candidates from the locked test. Twenty were
computational positives and 22 were strict computational negatives. These are
computational labels, not experimentally verified outcomes.

## How LAMAR ranks a candidate

> **Takeaway:** The known candidate position is central to the prediction.

![LAMAR model architecture](igem_drylab_wiki/assets/lamar_model_architecture.svg)

The prediction path has four main steps:

1. orient a 101-nt RNA sequence around a central C;
2. encode the sequence with the pretrained LAMAR transformer;
3. use the hidden representation at center index 50;
4. adapt attention with LoRA and produce a candidate probability.

Center pooling was substantially stronger than global mean pooling. This
result supports a position-specific signal around the candidate cytosine.

## Future direction: programmable RNA editing design

> **Takeaway:** Sequence ranking is one component of a future integrated editor
> design workflow.

![Proposed future RNA-editor design pipeline](igem_drylab_wiki/assets/future_design_pipeline.svg)

The present study evaluates LAMAR sequence ranking. A future integration could
combine that score with:

- transcriptome-wide cytosine scanning;
- PUF-binding compatibility;
- off-target filtering;
- editor construction and wet-lab testing.

This integrated pipeline has not yet been completed. The diagram defines a
future engineering direction rather than a validated end-to-end system.

---

## Technical foundation

> **Takeaway:** The complete benchmark remains available below the
> project-level narrative.

The sections below preserve the complete benchmark design for readers who want
to inspect the evidence, reproduce the workflow or assess its limitations.

## Study at a glance

> **Takeaway:** The evaluation uses fixed, leakage-controlled splits and a
> realistic 1:1000 final screening ratio.

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

Accuracy is not emphasized because the 1:1000 task is dominated by the
negative class.

## From RNA-seq reads to computational labels

> **Takeaway:** Positive and negative sites were measured with the same
> six-sample pileup rules.

### Unified recounting

Three treated and three control RNA-seq samples were recounted from
MarkDuplicates BAMs. Every site used:

- mapping quality of at least 30;
- base quality of at least 20;
- exclusion of unmapped, secondary, QC-failed, duplicate-flagged and
  supplementary reads;
- one count for overlapping paired-end observations;
- usable depth defined as filtered A+C+G+T depth;
- a minimum qualifying usable depth of 20.

The sequence was normalized to transcript orientation. Positive-strand C-to-T
and negative-strand G-to-A events therefore share one C-to-U representation.
Every retained sequence contains 101 nt, with C at zero-based index 50.

### Computational positives

Candidates began with a broad 9,930-site matrix, but all counts were recomputed
from the six selected BAMs. Corrected editing efficiency was:

`max(treated_median - control_median, 0)`.

The main computational-positive class required:

1. corrected editing efficiency strictly greater than 0.10;
2. at least two depth-qualified replicates in each group;
3. a control median no greater than 0.02;
4. a valid transcript-oriented 101-nt context centered on C;
5. no central-site overlap with either whole-genome sequencing VCF;
6. deterministic genomic-key deduplication.

This procedure yielded 1,513 main computational positives. The
high-confidence audit subset added all-six coverage, false-discovery-rate and
replicate-variability criteria. It retained 1,457 sites.

### Expression-supported strict negatives

An unreported site was not automatically treated as negative. Strict
computational negatives required direct RNA-seq coverage evidence.

Every retained strict negative had usable depth of at least 20 and zero target
alternative reads in all six samples. It also remained outside every positive
pool and the broad candidate matrix.

Central whole-genome variants, incomplete contexts, ambiguous orientation and
failed sequence quality control were excluded. The final universe contained
2,821,734 strict computational negatives. The all-six coverage and zero-alt
assertion had zero violations.

### Low-complexity control

The same deterministic rule was applied to both classes before splitting. A
sequence was excluded when any condition was met:

- base-2 single-nucleotide Shannon entropy below 1.20;
- a homopolymer run of at least 20 nt;
- phased dinucleotide-repeat coverage of at least 0.80.

No main computational positive was removed. The rule removed 2,087 of
2,823,821 potential strict negatives, corresponding to 0.0739%.

External basewise mappability was unavailable. The dataset records
`NA_RESOURCE_MISSING`. Mapping-quality and low-complexity filters are not
presented as external mappability validation.

## Leakage-controlled splitting

> **Takeaway:** Related genes, overlapping windows, loci and identical
> sequences never crossed data splits.

The full positive and negative universes were grouped before sampling. Sites
entered the same leakage group when they shared:

- a gene;
- a genomic center;
- an exact 101-nt sequence;
- an overlapping strand-aware genomic window.

Deterministic assignment produced four immutable splits:

| Split | Positives | Evaluation negatives | Role |
| --- | ---: | ---: | --- |
| Train | 1,028 | Dynamic train-only pool | Parameter learning |
| Development | 159 | 1,590 | Architecture and hyperparameter selection |
| Calibration | 165 | 165,000 | Calibration and threshold selection |
| Locked test | 161 | 161,000 | One final evaluation |

Post-split assertions found no crossing gene, leakage group, exact sequence or
genomic key. No locked-test row appeared in train, development or calibration.

The locked test was not used for negative mining, model selection, calibration
or threshold adjustment.

## Model development

> **Takeaway:** Baselines were established first, and every model decision used
> development data rather than the locked test.

### Sequence baselines

Two compact baselines tested whether LAMAR learned more than short motifs or
base composition:

- logistic regression using 1-mer to 4-mer counts, GC fraction, C count and
  sequence entropy;
- a convolutional neural network operating on one-hot encoded sequences.

### LAMAR adaptation strategies

The model study compared:

1. a frozen LAMAR backbone with a trainable classification head;
2. partial fine-tuning of the final one, two or four transformer blocks;
3. LoRA adaptation of attention projections;
4. full fine-tuning as a resource-intensive comparison.

Training negatives came only from the train split and were refreshed between
epochs. Staged development experiments compared sampling ratios, negative
difficulty, losses, learning rates, regularization, warmup and LoRA settings.

The selected configuration used:

- center-token pooling;
- LoRA on q/k/v/o attention projections;
- rank 4, alpha 8 and dropout 0.05;
- binary cross-entropy;
- dynamic random strict negatives at 1:10;
- backbone and head learning rates of `1e-5` and `1e-4`;
- batch size 16 with two-step gradient accumulation;
- warmup ratio 0.03, no weight decay and early stopping.

The configuration was repeated with seeds 42, 43 and 44. Development AP was
`0.808719 ± 0.013597`.

## What pretraining contributed

> **Takeaway:** Pretrained center representations contained editing-associated
> sequence information before task-specific gradient updates.

The original checkpoint was evaluated without an adapter, fine-tuned weights
or classification head. All 85,851,793 backbone parameters remained frozen
during embedding extraction.

Labeled train examples estimated class centroids for zero-gradient scoring. A
769-parameter logistic probe then tested information in fixed embeddings.

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

The center centroid exceeded both simple baselines. The linear probe and frozen
head performed similarly. LoRA added 0.105695 absolute test AP over the linear
probe, consistent with useful task-specific representation adaptation.

Mean-pooling test AP was 0.004307 for centroid scoring and 0.005391 for the
linear probe. The known center position was therefore essential.

## Calibration and frozen evaluation

> **Takeaway:** Threshold selection occurred at realistic prevalence before the
> locked test was opened.

Training ratios do not represent screening prevalence. Raw sigmoid scores were
therefore not treated as population probabilities. Uncalibrated, Platt and
isotonic mappings were compared on the fixed 1:1000 calibration set.

Platt scaling was selected. The threshold `0.24298369973628076` maximized
recall under a target of at most 100 false positives per million calibration
negatives. Calibration achieved 96.97 FP/M.

The unchanged model, calibrator and threshold were evaluated once:

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

The calibration target was exceeded on the locked test. We did not adjust the
threshold after observing this result.

## Shortcut analysis and scientific boundaries

> **Takeaway:** LAMAR learned sequence information, but dataset-generation
> shortcuts cannot be excluded.

A metadata-only logistic model reached development AP 0.759266. The fixed
LAMAR embedding reached 0.698583, and their combination reached 0.813666.

The strongest embedding-principal-component association was between PC3 and GC
fraction (`r=0.637`). Coverage, expression and composition shortcuts remain
plausible, even though the model receives sequence only.

Additional limitations include:

- labels from one six-sample biological system;
- unavailable external basewise mappability;
- finite-depth computational-negative labels;
- no completed PUF-binding or off-target integration;
- no prospective PUF-APOBEC validation reported in this repository.

A strict computational negative is supported by coverage and zero observed
target reads. It is not proof of universal editing absence.

## Repository map

> **Takeaway:** The repository separates data construction, model development,
> ablation analysis, Wiki content and provenance.

| Path | Contents |
| --- | --- |
| [`lamar_binary_project/dataset_build/`](lamar_binary_project/dataset_build/) | Dataset code, manifests, QC and filter funnels |
| [`lamar_binary_project/model_training/`](lamar_binary_project/model_training/) | Training code, configurations, selected models and error analyses |
| [`lamar_binary_project/zero_shot_ablation/`](lamar_binary_project/zero_shot_ablation/) | Representation extraction, probes, calibration and ablation results |
| [`igem_drylab_wiki/README.md`](igem_drylab_wiki/README.md) | Single-page English iGEM dry-lab Wiki |
| [`codex_handoff/`](codex_handoff/) | Frozen study rules and next-agent handoff |
| [`tools/`](tools/) | Public-archive validation |

## Reproducibility

> **Takeaway:** Frozen run identifiers, configurations and checksums preserve
> the successful computational record.

Three immutable successful runs define the study:

- dataset construction: `run_20260721T231520Z`;
- model comparison: `run_20260722T203752Z`;
- pretrained-representation ablation: `run_20260723T045155Z`.

Start with
[`lamar_binary_project/REPRODUCIBILITY.md`](lamar_binary_project/REPRODUCIBILITY.md).
The public snapshot uses environment placeholders instead of user-specific
paths. Configurations, software records, run markers and server checksums
preserve provenance.

Validate the public archive with:

```bash
python3 tools/validate_public_archive.py
python3 tools/verify_copied_binary_checksums.py
shasum -a 256 -c PUBLIC_CHECKSUMS.sha256
```

The complete server output is approximately 8.5 GB. Ordinary Git history
contains code, compact models, result tables, error analyses, figures and
documentation. Multi-gigabyte derived artifacts remain outside Git.
[`ARTIFACT_POLICY.md`](lamar_binary_project/ARTIFACT_POLICY.md) records these
omissions and their checksum coverage.

## Responsible use

> **Takeaway:** Model scores prioritize experiments but cannot establish
> biological editing on their own.

Use the model to prioritize candidates under a declared false-positive budget.
Report outputs as computational positives or strict computational negatives.
Pooled-read screening statistics are not biological-replicate validation.

Prospective evaluation should freeze the model, calibration function and
operating threshold before new labels are inspected.
