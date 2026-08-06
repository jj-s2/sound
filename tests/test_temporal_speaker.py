import unittest

from xh202615.contracts import EvidenceWindow
from xh202615.temporal_speaker import aggregate_windows


class TemporalAggregationTest(unittest.TestCase):
    def test_topk_coverage_consistency_and_quality(self):
        windows = (
            EvidenceWindow(0.0, 1.0, 0.8, 0.6),
            EvidenceWindow(1.0, 3.0, 0.4, 0.9),
            EvidenceWindow(3.0, 4.0, 0.7, None),
        )
        result = aggregate_windows(windows, top_k=2, target_threshold=0.6)
        self.assertAlmostEqual(result.topk_similarity, 0.75)
        self.assertAlmostEqual(result.temporal_coverage, 0.5)
        self.assertAlmostEqual(result.consistency, 0.65)
        self.assertAlmostEqual(result.quality, 0.8)

    def test_empty_windows_return_missing_aggregates(self):
        result = aggregate_windows((), top_k=2, target_threshold=0.6)
        self.assertIsNone(result.topk_similarity)
        self.assertIsNone(result.temporal_coverage)
        self.assertIsNone(result.consistency)
        self.assertIsNone(result.quality)

    def test_top_k_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            aggregate_windows((), top_k=0, target_threshold=0.6)

    def test_aggregation_orders_windows_and_quality_uses_only_known_windows(self):
        windows = (
            EvidenceWindow(2.0, 4.0, 0.4, 0.2),
            EvidenceWindow(0.0, 1.0, 0.9, None),
            EvidenceWindow(1.0, 2.0, 0.8, 0.8),
        )
        result = aggregate_windows(windows, top_k=2, target_threshold=0.8)
        self.assertAlmostEqual(result.topk_similarity, 0.85)
        self.assertAlmostEqual(result.temporal_coverage, 0.5)
        self.assertAlmostEqual(result.consistency, 0.75)
        self.assertAlmostEqual(result.quality, 0.4)

    def test_invalid_target_threshold_must_be_finite(self):
        with self.assertRaisesRegex(ValueError, "target_threshold"):
            aggregate_windows((), top_k=1, target_threshold=float("nan"))


if __name__ == "__main__":
    unittest.main()
