import unittest

from xh202615.backends import SpeakerScores
from xh202615.text_router import analyze_text, route_by_text


class TextRouterIntentTest(unittest.TestCase):
    def test_protects_broad_assistant_intents(self):
        examples = [
            "\u64ad\u653e\u5468\u6770\u4f26\u7684\u6b4c",
            "ladygaga\u6700\u65b0\u7684\u4e13\u8f91\u53eb\u4ec0\u4e48",
            "\u6211\u8981\u51fa\u95e8\u4e86",
            "\u5e2e\u6211\u8bbe\u4e00\u4e2a\u95f9\u949f",
        ]
        for text in examples:
            with self.subTest(text=text):
                evidence = analyze_text(text)
                self.assertGreater(evidence.domain_score, 0)
                self.assertGreater(evidence.assistant_intent_score, 0)

    def test_router_keeps_broad_assistant_intent_with_weak_speaker(self):
        cfg = {
            "text_router": {
                "enabled": True,
                "reject": {
                    "speaker_similarity_max": 0.60,
                    "target_probability_max": 0.98,
                    "min_speaker_votes": 2,
                    "min_text_length": 6,
                    "max_domain_score": 0,
                },
            }
        }
        scores = SpeakerScores(target_probability=0.90, global_similarity=0.58, topk_similarity=0.58)
        result = route_by_text("\u64ad\u653e\u4e00\u9996\u513f\u6b4c", scores, cfg)
        self.assertFalse(result.reject)


if __name__ == "__main__":
    unittest.main()
