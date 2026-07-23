# LAMAR binary dataset summary

This run produced **computational** labels. It does not claim experimentally verified true positives or true negatives.

- `positive_main`: 1,513
- `positive_high_confidence`: 1,457
- strict computational negative universe: 2,821,734
- strict all-six depth >=20 and all-six target-alt=0 assertion: PASS (0 violations)
- split counts: `{"calibration": {"positive": 165, "strict_negative": 282160}, "dev": {"positive": 159, "strict_negative": 282166}, "test": {"positive": 161, "strict_negative": 282164}, "train": {"positive": 1028, "strict_negative": 1975244}}`
- calibration 1:1000 achieved: True (prevalence 0.00099900)
- locked test 1:1000 achieved: True (prevalence 0.00099900)
- gene/window/exact-sequence/genomic-key leakage: none detected (all assertions PASS)
- old-vs-new comparable label agreement: 9,731/9,930; disagreement 199/9,930
- low-complexity positive_main exclusion: 0/1,513 (0.0000%)
- low-complexity strict-negative exclusion: 2,087/2,823,821 (0.0739%)
- low-complexity bias guard review required: False

## Known limitations and unmet resources

- No external basewise mappability track was present. Every site records `mappability=NA_RESOURCE_MISSING`; MAPQ>=30 pileup read filtering is not described as mappability validation and low-complexity filtering is not a substitute.
- `gene_coverage_summary.tsv.gz` is a gene-level coverage proxy derived from usable depths over this modeling universe, not an independent TPM/abundance estimate; it is recorded and used for train-only matched-negative strata.
- Low complexity was evaluated identically for positives and negatives before splitting, using complete 101-nt A/C/G/T windows and the approved OR rule. Trigger fields and the actual tandem-dinucleotide maximum coverage ratio are retained per site.
- Pooled Fisher/BH values are read-level screening statistics, not biological-replicate experimental validation.
- The specified igem environment lacks pandas, NumPy, and pyarrow; standard-library streaming gzip TSV shards were used without installing dependencies.
