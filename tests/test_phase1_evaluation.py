import unittest
from pathlib import Path

from xh202615.data import Sample
from xh202615.evaluation import evaluate_rows


def sample(sample_id: str, split: str, label: str | None) -> Sample:
    return Sample(
        id=sample_id,
        split=split,
        wakeup_audio=Path(f"{sample_id}-wake.wav"),
        wakeup_text="小助手",
        command_audio=Path(f"{sample_id}-cmd.wav"),
        label=label,
    )


class Phase1EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.samples = [
            sample("1", "pos", "打开空调"),
            sample("2", "pos", "关闭灯光"),
            sample("3", "neg", None),
            sample("4", "neg", None),
        ]

    def test_exact_metrics_and_per_sample_shape(self):
        report = evaluate_rows(
            self.samples,
            [
                {"id": "1", "recognition_text": "打开空"},
                {"id": "2", "recognition_text": ""},
                {"id": "3", "recognition_text": ""},
                {"id": "4", "recognition_text": "你好"},
            ],
        )

        self.assertEqual(
            report.metrics,
            {
                "avg_cer": 5 / 8,
                "avg_rr": 1 / 2,
                "pos_count": 2,
                "neg_count": 2,
                "missing_predictions": 0,
                "substitutions": 0,
                "insertions": 0,
                "deletions": 5,
                "ref_chars": 8,
                "false_reject_rate": 1 / 2,
                "false_accept_rate": 1 / 2,
            },
        )
        self.assertEqual([row["id"] for row in report.per_sample], ["1", "2", "3", "4"])
        self.assertEqual(report.per_sample[0]["errors"], 1)
        self.assertTrue(report.per_sample[2]["rr_correct"])

    def test_missing_empty_counts_as_empty_prediction(self):
        report = evaluate_rows(self.samples, [{"id": "1", "recognition_text": "打开空调"}])
        self.assertEqual(report.metrics["missing_predictions"], 3)
        self.assertEqual(report.metrics["pos_count"], 2)
        self.assertEqual(report.metrics["neg_count"], 2)
        self.assertEqual(report.metrics["false_reject_rate"], 1 / 2)
        self.assertEqual(report.metrics["avg_rr"], 1.0)

    def test_missing_skip_excludes_missing_samples_but_reports_count(self):
        report = evaluate_rows(
            self.samples,
            [{"id": "1", "recognition_text": "打开空调"}],
            missing_policy="skip",
        )
        self.assertEqual(report.metrics["missing_predictions"], 3)
        self.assertEqual(report.metrics["pos_count"], 1)
        self.assertEqual(report.metrics["neg_count"], 0)
        self.assertEqual(report.metrics["ref_chars"], 4)
        self.assertEqual(len(report.per_sample), 1)

    def test_duplicate_prediction_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate prediction id '1'"):
            evaluate_rows(
                self.samples,
                [
                    {"id": "1", "recognition_text": "打开空调"},
                    {"id": "1", "recognition_text": "关闭空调"},
                ],
            )

    def test_bucket_evaluation_uses_metadata_without_leaking_labels(self):
        report = evaluate_rows(
            self.samples,
            [
                {"id": "1", "recognition_text": "打开空调"},
                {"id": "2", "recognition_text": "关闭灯光"},
                {"id": "3", "recognition_text": ""},
                {"id": "4", "recognition_text": "噪声"},
            ],
            metadata_by_id={
                "1": {"snr_db": -5},
                "2": {"snr_db": 5},
                "3": {"snr_db": -5},
                "4": {"snr_db": 5},
            },
            bucket_fields=("snr_db",),
        )
        self.assertIn("snr_db=-5", report.buckets)
        self.assertEqual(report.buckets["snr_db=-5"]["pos_count"], 1)
        self.assertEqual(report.buckets["snr_db=-5"]["neg_count"], 1)
        self.assertNotIn("label", report.to_dict())

    def test_unknown_missing_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing_policy"):
            evaluate_rows(self.samples, [], missing_policy="invent")


if __name__ == "__main__":
    unittest.main()
