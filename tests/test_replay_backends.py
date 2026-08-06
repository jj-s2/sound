import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from xh202615.backends import AsrResult
from xh202615.contracts import (
    BackendMetadata,
    EvidenceWindow,
    TemporalSpeakerEvidence,
)
from xh202615.data import Sample
from xh202615.replay_backends import (
    GlobalScoreReplayBackend,
    TemporalEvidenceReplayBackend,
    TranscriptReplayBackend,
)


class ReplayBackendsTest(unittest.TestCase):
    def setUp(self):
        self.sample_1 = Sample(
            id="1",
            split="pos",
            wakeup_audio=Path("wake-1.wav"),
            wakeup_text="小助手",
            command_audio=Path("command-1.wav"),
            label="ignored",
        )
        self.sample_2 = Sample(
            id="2",
            split="neg",
            wakeup_audio=Path("wake-2.wav"),
            wakeup_text="小助手",
            command_audio=Path("command-2.wav"),
            label="ignored",
        )
        self.unknown = Sample(
            id="unknown",
            split="neg",
            wakeup_audio=Path("wake-unknown.wav"),
            wakeup_text="小助手",
            command_audio=Path("command-unknown.wav"),
            label="ignored",
        )

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_transcript_replay_reads_jsonl_and_marks_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asr.jsonl"
            self._write_jsonl(
                path,
                [
                    {"id": "1", "recognition_text": "打开空调"},
                    {"id": "2", "text": "关闭空调"},
                ],
            )
            backend = TranscriptReplayBackend(path)
            self.assertTrue(backend.metadata.replay)
            self.assertEqual(backend.transcribe(self.sample_1).text, "打开空调")
            self.assertIsInstance(backend.transcribe(self.sample_1), AsrResult)
            missing = backend.transcribe(self.unknown)
            self.assertEqual(missing.text, "")
            self.assertIsNone(missing.confidence)
            self.assertEqual(missing.error, "missing_prediction")
            self.assertEqual(missing.metadata, backend.metadata)

    def test_transcript_replay_reads_csv_aliases_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "asr.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "text"])
                writer.writeheader()
                writer.writerow({"sample_id": "1", "text": "打开空调"})
            backend = TranscriptReplayBackend(csv_path)
            self.assertEqual(backend.transcribe(self.sample_1).text, "打开空调")

            duplicate_path = Path(directory) / "duplicate.jsonl"
            self._write_jsonl(
                duplicate_path,
                [{"id": "1", "text": "a"}, {"sample_id": "1", "text": "b"}],
            )
            with self.assertRaisesRegex(ValueError, "duplicate id '1'"):
                TranscriptReplayBackend(duplicate_path)

    def test_global_score_replay_creates_explicit_degenerate_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "id",
                        "global_similarity",
                        "target_probability",
                        "overlap_probability",
                        "duration",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "1",
                        "global_similarity": "0.67",
                        "target_probability": "0.8",
                        "overlap_probability": "0.2",
                        "duration": "2.5",
                    }
                )
            backend = GlobalScoreReplayBackend(path)
            evidence = backend.score(self.sample_1)
            self.assertTrue(backend.metadata.replay)
            self.assertEqual(evidence.windows[0].similarity, 0.67)
            self.assertEqual(evidence.windows[0].start_sec, 0.0)
            self.assertEqual(evidence.windows[0].end_sec, 2.5)
            self.assertEqual(evidence.global_similarity, 0.67)
            self.assertEqual(evidence.topk_similarity, 0.67)
            self.assertIsNone(evidence.temporal_coverage)
            self.assertIsNone(evidence.consistency)
            self.assertEqual(evidence.target_probability, 0.8)
            self.assertEqual(evidence.overlap_probability, 0.2)
            self.assertIn("degenerate", backend.replay_window_note)
            self.assertEqual(backend.score(self.unknown).error, "missing_evidence")
            self.assertEqual(backend.score(self.unknown).windows, ())

    def test_global_score_replay_defaults_duration_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "global_similarity"])
                writer.writeheader()
                writer.writerow({"id": "1", "global_similarity": "0.67"})
            evidence = GlobalScoreReplayBackend(path).score(self.sample_1)
            self.assertEqual(evidence.windows[0].end_sec, 1.0)

            duplicate_path = Path(directory) / "duplicate.csv"
            with duplicate_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "global_similarity"])
                writer.writeheader()
                writer.writerow({"id": "1", "global_similarity": "0.6"})
                writer.writerow({"id": "1", "global_similarity": "0.7"})
            with self.assertRaisesRegex(ValueError, "duplicate id '1'"):
                GlobalScoreReplayBackend(duplicate_path)

    def test_temporal_evidence_replay_uses_strict_contract_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "temporal.jsonl"
            evidence = TemporalSpeakerEvidence(
                id="1",
                backend=BackendMetadata("fixture", "model", replay=False),
                enrollment_source="wake.wav",
                command_source="command.wav",
                windows=(EvidenceWindow(0.0, 1.0, 0.75, 0.8),),
                target_probability=0.9,
            )
            self._write_jsonl(path, [evidence.to_dict()])
            backend = TemporalEvidenceReplayBackend(path)
            loaded = backend.score(self.sample_1)
            expected = replace(evidence, backend=replace(evidence.backend, replay=True))
            self.assertEqual(loaded, expected)
            self.assertTrue(loaded.backend.replay)
            self.assertEqual(backend.score(self.unknown).error, "missing_evidence")

            malformed_path = Path(directory) / "malformed.jsonl"
            malformed = evidence.to_dict()
            malformed["windows"][0]["end_sec"] = "not-a-number"
            self._write_jsonl(malformed_path, [malformed])
            with self.assertRaisesRegex(ValueError, r"TemporalSpeakerEvidence.*sample '1'.*windows\[0\]"):
                TemporalEvidenceReplayBackend(malformed_path)

    def test_temporal_evidence_replay_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            evidence = TemporalSpeakerEvidence(
                id="1",
                backend=BackendMetadata("fixture", "model"),
                enrollment_source="wake.wav",
                command_source="command.wav",
            ).to_dict()
            self._write_jsonl(path, [evidence, dict(evidence)])
            with self.assertRaisesRegex(ValueError, "duplicate id '1'"):
                TemporalEvidenceReplayBackend(path)


if __name__ == "__main__":
    unittest.main()
