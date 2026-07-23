# Sequence-informed prioritization of C-to-U RNA-editing candidates

We developed a leakage-controlled dry-lab pipeline that ranks 101-nt
transcript-oriented sequence contexts for experimental follow-up. Six
MarkDuplicates RNA-seq BAMs were recounted with identical MAPQ/base-quality
and read-flag rules. Main computational positives required corrected editing
efficiency above 0.10, replicate coverage, low control editing, sequence QC,
and WGS exclusion. Strict computational negatives required usable depth at
least 20 and zero target-alt reads in all six samples.

The resulting universe contained 1,513 computational positives and 2,821,734
strict computational negatives. Sites sharing genes, overlapping windows,
exact sequences, or genomic centers were grouped before splitting. The fixed
calibration and locked-test sets each used approximately 1:1000 prevalence.

We compared k-mer logistic regression, CNN, frozen LAMAR, partial unfreezing,
LoRA, and full fine-tuning. A staged dev-only search selected center-token
q/k/v/o LoRA with rank 4 and dynamic 1:10 random strict negatives. Platt
scaling and threshold selection used only the 1:1000 calibration set.

On the locked test, the final model achieved AP 0.171958, precision 0.476190,
recall 0.124224, and 136.646 false positives per million negatives. An
unadapted pretrained center representation achieved AP 0.036180, above CNN
0.013912 and k-mer logistic 0.006598. A 769-parameter linear probe reached
0.066262, similar to a frozen neural head at 0.064390; LoRA's additional
0.105695 AP supports representation adaptation beyond a simple classifier.

Metadata remained predictive and embedding PCs correlated with GC and C
content, so sequence-composition and dataset-generation shortcuts cannot be
excluded. These outputs are computational prioritizations, not experimentally
verified edits. Prospective experimental validation on independently generated
data is required.

![Workflow](assets/pipeline_overview.svg)

![Locked-test precision-recall curves](assets/PR_curves.png)
