import argparse
import unittest

from scripts.merge_asr_fallback import should_use_fallback


def args(**overrides):
    values = {
        "min_primary_length": 8,
        "max_primary_domain_score": 0,
        "use_robustness_trigger": False,
        "short_text_length": 6,
        "enable_short_non_domain": False,
        "incomplete_text_length": 12,
        "max_incomplete_domain_score": 2,
        "long_text_length": 14,
        "long_text_max_domain_score": 1,
        "very_long_text_length": 18,
        "min_length_reduction_ratio": 0.0,
        "require_fallback_nonempty": True,
        "prefer_higher_domain_score": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class MergeAsrFallbackTest(unittest.TestCase):
    def test_legacy_trigger_replaces_non_domain_text(self):
        use, reason = should_use_fallback("高等专业学校第三附属", "", args(require_fallback_nonempty=False))
        self.assertTrue(use)
        self.assertTrue(reason.startswith("primary_non_domain"))

    def test_robustness_trigger_replaces_incomplete_command_text(self):
        use, reason = should_use_fallback(
            "播放前一段时间因为正在治",
            "",
            args(use_robustness_trigger=True, require_fallback_nonempty=False),
        )
        self.assertTrue(use)
        self.assertTrue(reason.startswith("incomplete_command_text"))

    def test_robustness_trigger_keeps_complete_command(self):
        use, reason = should_use_fallback(
            "打开客厅空调",
            "",
            args(use_robustness_trigger=True, require_fallback_nonempty=False),
        )
        self.assertFalse(use)
        self.assertEqual(reason, "keep_primary")


if __name__ == "__main__":
    unittest.main()
