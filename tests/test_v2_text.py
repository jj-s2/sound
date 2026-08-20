import unittest

from xh202615.backends import SpeakerScores
from xh202615.postprocess import postprocess_asr_text
from xh202615.text_router import analyze_text, route_by_text


class V2TextTest(unittest.TestCase):
    def test_postprocess_command_homophones(self):
        cfg = {"enabled": True, "domain_context_required": True}
        self.assertEqual(postprocess_asr_text("关闭时机", cfg).text, "关闭洗衣机")
        self.assertEqual(postprocess_asr_text("颜接下降", cfg).text, "烟机下降")

    def test_postprocess_leaves_non_context_text(self):
        cfg = {"enabled": True, "domain_context_required": True}
        self.assertEqual(postprocess_asr_text("这个时机不对", cfg).text, "这个时机不对")

    def test_text_evidence(self):
        evidence = analyze_text("打开客厅空调")
        self.assertGreater(evidence.domain_score, 0)
        self.assertGreater(evidence.device_hits, 0)

    def test_text_router_rejects_non_domain_with_weak_speaker(self):
        cfg = {
            "text_router": {
                "enabled": True,
                "reject": {
                    "speaker_similarity_max": 0.60,
                    "target_probability_max": 0.98,
                    "min_speaker_votes": 2,
                    "min_text_length": 10,
                    "max_domain_score": 0,
                },
            }
        }
        scores = SpeakerScores(target_probability=0.90, global_similarity=0.58, topk_similarity=0.58)
        result = route_by_text("高等专业学校第三附属", scores, cfg)
        self.assertTrue(result.reject)

    def test_text_router_keeps_domain_text(self):
        cfg = {
            "text_router": {
                "enabled": True,
                "reject": {
                    "speaker_similarity_max": 0.60,
                    "target_probability_max": 0.98,
                    "min_speaker_votes": 2,
                    "min_text_length": 10,
                    "max_domain_score": 0,
                },
            }
        }
        scores = SpeakerScores(target_probability=0.90, global_similarity=0.58, topk_similarity=0.58)
        result = route_by_text("打开客厅空调温度二十五度", scores, cfg)
        self.assertFalse(result.reject)


if __name__ == "__main__":
    unittest.main()
