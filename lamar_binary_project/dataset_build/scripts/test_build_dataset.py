#!/usr/bin/env python3
import unittest

import build_dataset as pipeline


class PipelineUnitTests(unittest.TestCase):
    def test_parse_mpileup(self):
        counts = pipeline.parse_mpileup_bases(".,Aa^F.$+2tt-1c*<>Nn", "C")
        self.assertEqual(counts, {"A": 2, "C": 3, "G": 0, "T": 0})

    def test_bh(self):
        self.assertEqual(pipeline.bh([0.01, 0.04, None, 0.03]), [0.03, 0.04, None, 0.04])

    def test_merge_intervals(self):
        self.assertEqual(pipeline.merge_intervals([(5, 10), (1, 3), (3, 5), (20, 21)]), [(1, 10), (20, 21)])

    def test_union_find(self):
        union = pipeline.UnionFind(4)
        union.union(0, 1)
        union.union(2, 3)
        self.assertEqual(union.find(0), union.find(1))
        self.assertNotEqual(union.find(0), union.find(2))

    def test_sequence_metrics(self):
        metrics = pipeline.sequence_metrics("C" * 101)
        self.assertEqual(metrics["sequence_entropy_log2_single_base_101nt"], 0.0)
        self.assertEqual(metrics["max_homopolymer_run"], 101)
        self.assertEqual(metrics["low_complexity_qc"], "FAIL_LOW_COMPLEXITY")

    def test_dinucleotide_both_phases(self):
        ratio0, phase0, motif0 = pipeline.dinucleotide_tandem_metric("AC" * 50 + "A")
        self.assertAlmostEqual(ratio0, 100 / 101)
        self.assertEqual((phase0, motif0), (0, "AC"))
        ratio1, phase1, motif1 = pipeline.dinucleotide_tandem_metric("G" + "AC" * 50)
        self.assertAlmostEqual(ratio1, 100 / 101)
        self.assertEqual((phase1, motif1), (1, "AC"))

    def test_entropy_log2_range_and_n_rejection(self):
        balanced = pipeline.sequence_metrics("ACGT" * 25 + "A")
        self.assertGreater(balanced["sequence_entropy_log2_single_base_101nt"], 1.99)
        self.assertLessEqual(balanced["sequence_entropy_log2_single_base_101nt"], 2.0)
        with self.assertRaises(ValueError):
            pipeline.sequence_metrics("N" + "A" * 100)

    def test_gene_level_coverage_fields(self):
        row = {"gene_id": "ENSG2,ENSG1,ENSG2"}
        row.update({f"{sample}_usable_depth": depth for sample, depth in zip(pipeline.SAMPLE_NAMES, (20, 24, 28, 32, 36, 40))})
        coverage = {
            "ENSG1": {"mean_site_median_depth": 8.0},
            "ENSG2": {"mean_site_median_depth": 32.0},
        }
        self.assertEqual(pipeline.gene_ids(row), ["ENSG1", "ENSG2"])
        pipeline.add_gene_coverage_fields(row, coverage)
        self.assertEqual(row["gene_level_coverage_mean_site_median_depth"], 20.0)
        self.assertEqual(row["gene_level_coverage_bin_log2"], 4)
        self.assertIn("not_TPM", row["gene_expression_coverage_summary"])

    def test_split_groups_balances_negative_only_groups_globally(self):
        metadata = []
        group_ids = []
        for index in range(1000):
            metadata.append({
                "label": int(index < 100),
                "efficiency": 0.2 if index < 100 else 0.0,
                "chrom": f"chr{1 + index % 2}",
                "region": "exonic",
                "gc": 0.5,
            })
            group_ids.append(f"group_{index:04d}")
        split = pipeline.split_groups(metadata, group_ids)
        counts = {name: 0 for name in ("train", "dev", "calibration", "test")}
        positives = dict(counts)
        for row, group in zip(metadata, group_ids):
            counts[split[group]] += 1
            positives[split[group]] += row["label"]
        self.assertLessEqual(abs(counts["train"] - 700), 2)
        self.assertLessEqual(abs(counts["dev"] - 100), 2)
        self.assertLessEqual(abs(counts["calibration"] - 100), 2)
        self.assertLessEqual(abs(counts["test"] - 100), 2)
        self.assertEqual(positives, {"train": 70, "dev": 10, "calibration": 10, "test": 10})


if __name__ == "__main__":
    unittest.main()
