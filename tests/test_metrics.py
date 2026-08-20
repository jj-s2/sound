import unittest

from xh202615.metrics import cer_stats, is_rejection
from xh202615.text import clean_asr_text, normalize_text


class MetricsTest(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text(" 空调，打开！ "), "空调打开")

    def test_clean_asr_text_tags(self):
        self.assertEqual(clean_asr_text("<|zh|><|NEUTRAL|>打开空调"), "打开空调")
        self.assertEqual(normalize_text("<|zh|><|NEUTRAL|>打开空调。"), "打开空调")

    def test_clean_asr_text_removes_spaces_between_cjk_characters(self):
        self.assertEqual(clean_asr_text("洗 碗 机 暂 停 工 作"), "洗碗机暂停工作")

    def test_clean_asr_text_collapses_large_repeated_phrases(self):
        self.assertEqual(clean_asr_text("空调空调空调打开"), "空调打开")
        self.assertEqual(clean_asr_text("暂停暂停暂停"), "暂停")

    def test_clean_asr_text_keeps_normal_repetition_and_english_spacing(self):
        self.assertEqual(clean_asr_text("哈哈 打开 Wi Fi"), "哈哈打开 Wi Fi")

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
