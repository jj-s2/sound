import unittest

from xh202615.command_corrector import correct_command_text


class CommandCorrectorTest(unittest.TestCase):
    def test_fixed_rules(self):
        cfg = {"enabled": True, "min_similarity": 0.88}
        self.assertEqual(correct_command_text("丰富六十", cfg).text, "风速六十")
        self.assertEqual(correct_command_text("开启之不温", cfg).text, "开启智控温")

    def test_phrase_similarity_for_short_text(self):
        cfg = {"enabled": True, "min_similarity": 0.88, "max_phrase_changes": 1}
        self.assertEqual(correct_command_text("五风干", cfg).text, "无风感")

    def test_long_non_domain_text_is_not_corrected(self):
        cfg = {"enabled": True, "min_similarity": 0.88}
        text = "视频普京打冰球庆祝六十三岁生日"
        self.assertEqual(correct_command_text(text, cfg).text, text)

    def test_keeps_valid_choushi(self):
        cfg = {"enabled": True, "min_similarity": 0.88}
        self.assertEqual(correct_command_text("调到抽湿模式", cfg).text, "调到抽湿模式")


if __name__ == "__main__":
    unittest.main()
