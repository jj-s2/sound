import unittest

from scripts.evaluate_temporal_head import gated_text, select_candidate_text


class TemporalEvaluationTest(unittest.TestCase):
    def test_rejected_sample_emits_empty_text(self):
        self.assertEqual(gated_text(0.4, "原始识别", threshold=0.5), "")

    def test_accepted_sample_preserves_frozen_asr_text(self):
        self.assertEqual(gated_text(0.6, "原始识别", threshold=0.5), "原始识别")

    def test_rescue_selects_raw_only_for_confident_target_with_text(self):
        self.assertEqual(
            select_candidate_text(0.8, "打开空调", "", threshold=0.7),
            ("打开空调", "raw"),
        )

    def test_rescue_preserves_fusion_when_raw_is_empty_or_target_is_uncertain(self):
        self.assertEqual(
            select_candidate_text(0.8, "", "拒识", threshold=0.7),
            ("拒识", "fusion_empty_raw_fallback"),
        )
        self.assertEqual(
            select_candidate_text(0.2, "打开空调", "拒识", threshold=0.7),
            ("拒识", "fusion"),
        )


if __name__ == "__main__":
    unittest.main()
