# Responsibility, interpretation, and limitations

## What the model can support

The output is a prioritization score. It can help choose a smaller set of
sequences for targeted validation under a declared false-positive budget.

## What the model cannot establish

- It does not prove that a site is edited in a new biological context.
- It does not replace biological replicates, orthogonal sequencing, or
  targeted experimental validation.
- It does not estimate editing efficiency.
- It does not establish causal recognition by a particular editor.
- A calibrated probability is conditional on this data-generation pipeline
  and should not be transferred unchanged to a new cell type or assay.

## Known technical limitations

- No external basewise mappability track was available.
- Labels are derived from one six-sample experimental system.
- The negative class is stringent but remains a computational negative class;
  finite sequencing depth cannot prove universal absence of editing.
- Coverage and sequence-composition biases are predictive.
- Full and partial fine-tuning were evaluated once in the locked suite, but
  their resource cost was high relative to LoRA.
- The 100-FP/M calibration target was exceeded on locked test because of
  sampling variability and distribution shift.

## Safe reporting language

Use:

- “computational positive”;
- “strict computational negative”;
- “pooled-read screening statistic”;
- “candidate for experimental follow-up.”

Avoid:

- “experimentally validated positive”;
- “true unedited site”;
- “biological-replicate significance” when referring to pooled Fisher/BH;
- “mappability passed” when the resource was unavailable.

## Recommended next validation

The next scientific step is prospective evaluation on a new, separately
generated experiment. Select the operating threshold before seeing new
labels, preserve gene/window separation from the current data, and report both
candidate yield and false positives per million negatives.
