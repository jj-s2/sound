import unittest

from xh202615.metrics import cer_stats, is_rejection
from xh202615.text import clean_asr_text, normalize_text


class MetricsTest(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text(" 空调，打开！ "), "空调打开")

    def test_clean_asr_text_tags(self):
        self.assertEqual(clean_asr_text("<|zh|><|NEUTRAL|>打开空调"), "打开空调")
        self.assertEqual(normalize_text("<|zh|><|NEUTRAL|>打开空调。"), "打开空调")

    def test_cer_exact(self):
        self.assertEqual(cer_stats("打开空调", "打开空调").cer, 0.0)

    def test_cer_delete_all(self):
        stats = cer_stats("打开空调", "")
        self.assertEqual(stats.deletions, 4)
        self.assertEqual(stats.cer, 1.0)

    def test_rejection(self):
        self.assertTrue(is_rejection(""))
        self.assertTrue(is_rejection("   "))
        self.assertFalse(is_rejection("打开空调"))


if __name__ == "__main__":
    unittest.main()
