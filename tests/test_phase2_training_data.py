"""Tests for the Phase-2 public/synthetic training manifest interface."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from xh202615.training_data import (
    TrainingManifestRow,
    assert_valid_training_manifest,
    read_training_manifest,
    validate_training_manifest,
)


class Phase2TrainingDataTests(unittest.TestCase):
    def make_row(self, **changes):
        values = {
            "row_id": "synthetic-train-001",
            "split": "train",
            "source": "synthetic_tone_fixture",
            "enrollment_audio": Path("audio/enrollment/spk-train-01.wav"),
            "target_audio": Path("audio/target/spk-train-01.wav"),
            "mixture_audio": Path("audio/mixture/synthetic-train-001.wav"),
            "target_speaker_id": "spk-train-01",
            "interferer_speaker_id": "spk-train-02",
            "target_present": True,
            "overlap_ratio": 0.4,
            "snr_db": None,
            "sir_db": 3.0,
            "text": "illustrative synthetic command",
            "seed": 202615,
        }
        values.update(changes)
        return TrainingManifestRow(**values)

    def codes(self, rows, **kwargs):
        return {issue.code for issue in validate_training_manifest(rows, **kwargs)}

    def test_round_trip_is_json_compatible_and_resolves_relative_paths(self):
        row = self.make_row()
        payload = row.to_dict()
        self.assertEqual(json.loads(json.dumps(payload, sort_keys=True)), payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            restored = TrainingManifestRow.from_dict(payload, base_dir=base_dir)
            self.assertEqual(restored.row_id, row.row_id)
            self.assertEqual(
                restored.enrollment_audio,
                base_dir.resolve(strict=False) / row.enrollment_audio,
            )
            self.assertEqual(restored.to_dict()["seed"], 202615)

    def test_read_manifest_reports_malformed_json_and_row_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.jsonl"
            path.write_text("{not-json}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"line 1.*malformed JSON"):
                read_training_manifest(path)

            path.write_text(json.dumps({"row_id": "broken"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"line 1.*broken.*missing"):
                read_training_manifest(path)

    def test_duplicate_row_ids_are_rejected(self):
        first = self.make_row()
        second = self.make_row(target_speaker_id="spk-train-03")
        self.assertIn("duplicate_row_id", self.codes((first, second)))

    def test_speaker_cannot_cross_splits_in_either_role(self):
        train = self.make_row()
        val = self.make_row(
            row_id="synthetic-val-001",
            split="val",
            target_speaker_id="spk-val-01",
            interferer_speaker_id="spk-train-01",
            enrollment_audio=Path("audio/enrollment/spk-val-01.wav"),
            target_audio=Path("audio/target/spk-val-01.wav"),
        )
        self.assertIn("speaker_split_leakage", self.codes((train, val)))

    def test_forbidden_root_rejects_resolved_audio_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            forbidden = root / "evaluation_only"
            manifest = root / "configs" / "manifest.jsonl"
            row = self.make_row(
                enrollment_audio=Path("../evaluation_only/enrollment.wav"),
                target_audio=Path("../public/target.wav"),
                mixture_audio=Path("../public/mixture.wav"),
            )
            issues = validate_training_manifest(
                (row,), manifest_path=manifest, forbidden_roots=(forbidden,)
            )
            self.assertIn("forbidden_path", {issue.code for issue in issues})
            self.assertTrue(
                any("enrollment_audio" in issue.message for issue in issues)
            )

    def test_invalid_ranges_and_non_finite_ratios_are_rejected(self):
        for ratio in (-0.01, 1.01, math.inf, math.nan):
            with self.subTest(ratio=ratio):
                self.assertIn(
                    "invalid_overlap_ratio",
                    self.codes((self.make_row(overlap_ratio=ratio),)),
                )
        self.assertIn(
            "invalid_snr_db", self.codes((self.make_row(snr_db=math.inf),))
        )
        self.assertIn(
            "invalid_sir_db", self.codes((self.make_row(sir_db=math.nan),))
        )

    def test_negative_target_requires_absent_text_but_keeps_enrolled_speaker(self):
        valid_negative = self.make_row(
            row_id="synthetic-negative-001",
            target_present=False,
            text=None,
            target_speaker_id="enrolled-speaker-negative",
        )
        self.assertEqual(validate_training_manifest((valid_negative,)), ())

        invalid_negative = self.make_row(target_present=False, text="must not leak")
        self.assertIn("target_absent_text", self.codes((invalid_negative,)))

    def test_overlap_or_noise_requires_mixture_audio(self):
        row = self.make_row(mixture_audio=None)
        self.assertIn("mixture_audio_required", self.codes((row,)))

    def test_valid_synthetic_row_and_assertion(self):
        row = self.make_row()
        self.assertEqual(validate_training_manifest((row,)), ())
        self.assertEqual(assert_valid_training_manifest((row,)), (row,))

    def test_same_target_and_interferer_are_rejected(self):
        row = self.make_row(interferer_speaker_id="spk-train-01")
        self.assertIn("same_speaker_id", self.codes((row,)))


if __name__ == "__main__":
    unittest.main()
