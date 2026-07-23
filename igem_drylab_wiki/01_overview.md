# Dry lab: prioritizing C-to-U RNA-editing candidates with LAMAR

## Why candidate prioritization matters

Experimental validation of every possible transcriptomic cytosine is
impractical. A useful dry-lab system should reduce the candidate space while
remaining honest about uncertainty: it should rank sequences for follow-up,
not claim that a prediction is an experimentally verified edit.

We formulated this as sequence-level binary discovery:

- **input:** a 101-nt transcript-oriented sequence centered on `C`;
- **output:** a score for whether the sequence belongs to a reliable
  computational C-editing candidate class;
- **positive definition:** corrected editing efficiency greater than 0.10
  after unified six-sample recounting;
- **negative definition:** a covered exonic C with zero target-alt reads in all
  six samples and stringent exclusion/QC rules.

![End-to-end dry-lab workflow](assets/pipeline_overview.svg)

## Design principles

The workflow was built around five safeguards:

1. recount positives and negatives with identical pileup rules;
2. require expression evidence for negatives rather than treating unobserved
   sites as negative;
3. separate genes, overlapping windows, loci, and duplicate sequences across
   splits;
4. calibrate at realistic 1:1000 prevalence;
5. open the locked test once, after all choices and thresholds were frozen.

## Final scale

- 1,513 main computational positives;
- 1,457 high-confidence computational positives;
- 2,821,734 strict computational negatives;
- 1,028 train positives;
- 159:1,590 dev positives:negatives;
- 165:165,000 calibration positives:negatives;
- 161:161,000 locked-test positives:negatives.

The model is therefore evaluated in the low-prevalence regime relevant to
candidate screening, where false-positive control matters more than accuracy.
