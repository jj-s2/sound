import unittest

from xh202615.robustness import should_enhance_for_robustness


class RobustnessTest(unittest.TestCase):
    def test_selects_long_non_domain_text(self):
        decision = should_enhance_for_robustness(
            "\u9ad8\u7b49\u4e13\u4e1a\u5b66\u6821\u7b2c\u4e09\u9644\u5c5e",
            min_text_length=8,
            max_domain_score=0,
        )
        self.assertTrue(decision.enhance)

    def test_keeps_domain_command(self):
        decision = should_enhance_for_robustness(
            "\u6253\u5f00\u5ba2\u5385\u7a7a\u8c03",
            min_text_length=8,
            max_domain_score=0,
        )
        self.assertFalse(decision.enhance)

    def test_selects_incomplete_domain_text(self):
        decision = should_enhance_for_robustness(
            "\u6211\u8981\u505a\u996d\u4e86\u7136\u540e\u5b83\u81ea\u52a8\u964d\u4e0b\u55ef",
            min_text_length=8,
            max_domain_score=0,
            incomplete_text_length=10,
            max_incomplete_domain_score=2,
        )
        self.assertTrue(decision.enhance)

    def test_keeps_complete_command_even_when_long(self):
        decision = should_enhance_for_robustness(
            "\u4f60\u597d\u79d1\u6155\u6253\u5f00\u5ba2\u5385\u7a7a\u8c03\u6e29\u5ea6\u8c03\u5230\u4e8c\u5341\u516d\u5ea6",
            min_text_length=8,
            max_domain_score=0,
            incomplete_text_length=10,
            max_incomplete_domain_score=2,
        )
        self.assertFalse(decision.enhance)

    def test_short_non_domain_is_optional(self):
        conservative = should_enhance_for_robustness("\u9ad8\u7b49\u5b66\u6821", short_text_length=4)
        aggressive = should_enhance_for_robustness(
            "\u9ad8\u7b49\u5b66\u6821",
            short_text_length=4,
            enable_short_non_domain=True,
        )
        self.assertFalse(conservative.enhance)
        self.assertTrue(aggressive.enhance)

    def test_keeps_empty_text(self):
        decision = should_enhance_for_robustness("", min_text_length=8, max_domain_score=0)
        self.assertFalse(decision.enhance)


if __name__ == "__main__":
    unittest.main()
