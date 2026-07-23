# Building a defensible computational dataset

## Unified read counting

Three treated and three control RNA-seq MarkDuplicates BAMs were analyzed with
the same read filters:

- MAPQ at least 30 and base quality at least 20;
- unmapped, secondary, QC-failed, duplicate-flagged, and supplementary reads
  excluded;
- paired-end overlaps counted once;
- usable depth defined as filtered A+C+G+T depth;
- minimum qualifying usable depth of 20.

For positive candidates, treated and control replicate rates were summarized
with medians and median absolute deviations. Corrected efficiency was the
positive part of treated median minus control median.

## Main computational positives

Sites passed when corrected efficiency was strictly greater than 0.10,
treated and control each had at least two depth-qualified replicates, control
median was at most 0.02, the transcript-oriented 101-nt sequence was valid,
and neither WGS VCF contained the central site.

The final count was 1,513. An additional high-confidence flag required all six
depths at least 20, BH-FDR below 0.05, treated MAD at most 0.05, and control
MAD at most 0.02; 1,457 sites passed.

## Expression-supported strict negatives

Negatives were not defined as “not in the candidate table.” Every strict
negative had:

- usable depth at least 20 in all six samples;
- target-alt count exactly zero in all six samples;
- no overlap with any positive or broad ambiguous candidate center;
- no central WGS variant;
- valid transcript orientation and 101-nt context;
- no low-complexity trigger.

This produced 2,821,734 strict computational negatives. The all-six depth and
zero-alt assertion had zero violations.

## Sequence-complexity rule

The same OR rule was applied to positives and negatives before splitting:

- base-2 nucleotide Shannon entropy below 1.20;
- longest homopolymer run at least 20 nt;
- maximum two-phase dinucleotide-repeat coverage at least 0.80.

No main positive was removed; 2,087 of 2,823,821 potential strict negatives
were removed (0.0739%).

## Leakage control

Rows were grouped before sampling. A group joined sites sharing a gene,
genomic center, exact sequence, or overlapping strand-aware 101-nt genomic
window. Post-split assertions found:

- no gene crossing splits;
- no leakage group crossing splits;
- no exact sequence crossing splits;
- no genomic key crossing splits;
- no test row appearing in train, dev, or calibration.

This matters because random row splitting would allow nearly identical
sequence contexts from one gene to appear in both training and evaluation.

## Limitations at the data layer

External basewise mappability was unavailable. The dataset records
`NA_RESOURCE_MISSING`; MAPQ filtering and low-complexity filtering are not
claimed to replace mappability validation. Pooled Fisher/BH values are
read-level screening statistics, not biological-replicate experimental
validation.
