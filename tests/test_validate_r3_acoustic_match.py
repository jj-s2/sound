"""Tests for the R3 aggregate-match validator (Task 4)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.prepare_r3_counterfactual import build_r3_manifest
from scripts.validate_r3_acoustic_match import (
    _assert_report_redacted,
    main,
    validate_match,
)
from xh202615.acoustic_profile import profile_audio_paths, write_aggregate_profile


class ValidateR3AcousticMatchTests(unittest.TestCase):
    SR = 16_000

    def _make_corpus(self, root: Path, speaker_count: int = 10) -> Path:
        corpus = root / "aishell"
        transcript = corpus / "transcript" / "aishell_transcript_v0.8.txt"
        transcript.parent.mkdir(parents=True)
        lines = []
        for si in range(1, speaker_count + 1):
            sid = f"S{si:04d}"
            sdir = corpus / "wav" / "train" / sid
            sdir.mkdir(parents=True)
            for ui, dur in enumerate((0.4, 0.6, 0.8, 1.0), start=1):
                uid = f"{sid}{ui:04d}"
                n = int(self.SR * dur)
                t = np.arange(n) / self.SR
                sf.write(str(sdir / f"{uid}.wav"),
                         0.1 * np.sin(2 * np.pi * (180 + si * 30 + ui) * t), self.SR)
                lines.append(f"{uid}\t文本 {uid}")
        transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return corpus

    def _make_assets(self, root: Path, count: int = 6) -> tuple[Path, Path]:
        rir_root = root / "rir_assets"
        noise_root = root / "noise_assets"
        rir_root.mkdir(parents=True)
        noise_root.mkdir(parents=True)
        impulse = np.zeros(160, dtype=np.float32)
        impulse[3] = 0.8
        for i in range(count):
            sf.write(str(rir_root / f"rir_{i:02d}.wav"), impulse, self.SR)
            n = 800 + i * 10
            t = np.arange(n) / self.SR
            sf.write(str(noise_root / f"noise_{i:02d}.wav"),
                     0.02 * np.sin(2 * np.pi * (80 + i * 10) * t), self.SR)
        return rir_root, noise_root

    def _make_profile(self, root: Path, duration: float = 0.8) -> Path:
        wavs = []
        for i in range(4):
            w = root / f"p_{i}.wav"
            n = int(self.SR * duration)
            t = np.arange(n) / self.SR
            sf.write(str(w), (0.1 * np.sin(2 * np.pi * (300 + i * 40) * t)).astype(np.float32), self.SR)
            wavs.append(w)
        path = root / "profile.json"
        write_aggregate_profile(path, profile_audio_paths(wavs))
        return path

    def _generate(self, root: Path) -> tuple[Path, Path, Path, Path]:
        corpus = self._make_corpus(root)
        rir_root, noise_root = self._make_assets(root)
        profile = self._make_profile(root, duration=0.8)
        output = root / "out"
        build_r3_manifest(
            corpus, output, dataset_a_root=root / "da",
            pairs=3, seed=7, rir_root=rir_root, noise_root=noise_root,
            acoustic_profile=profile,
        )
        return output, profile, corpus, root / "da"

    # ------------------------------------------------------------- happy path

    def test_validate_aggregate_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, profile, _, _ = self._generate(root)
            report_path = root / "report.json"
            report = validate_match(profile, output / "manifest.jsonl", report_path)
            self.assertTrue(report_path.is_file())
            self.assertIn("profile_hash", report)
            self.assertEqual(len(report["profile_hash"]), 64)
            self.assertIn("generated_hash", report)
            self.assertGreater(report["matched_pair_count"], 0)
            self.assertEqual(report["unreadable_audio_count"], 0)
            self.assertIn("comparison", report)
            dur = report["comparison"]["duration"]
            self.assertIn("0.5", dur)
            self.assertIn("norm_diff", dur["0.5"])
            self.assertTrue(np.isfinite(dur["0.5"]["norm_diff"]))
            self.assertIn("checks", report)
            self.assertTrue(report["overall_passed"])

    def test_profile_hash_propagation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, profile, _, _ = self._generate(root)
            report = validate_match(profile, output / "manifest.jsonl")
            profile_payload = json.loads(profile.read_text(encoding="utf-8"))
            self.assertEqual(report["profile_hash"], profile_payload["hash"])

    # ------------------------------------------------------ missing audio

    def test_missing_audio_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, profile, _, _ = self._generate(root)
            # Delete one mixture WAV to force an unreadable path.
            mixtures = sorted((output / "mixture").rglob("*.wav"))
            self.assertGreater(len(mixtures), 0)
            mixtures[0].unlink()
            report = validate_match(profile, output / "manifest.jsonl")
            self.assertGreaterEqual(report["unreadable_audio_count"], 1)
            self.assertFalse(report["overall_passed"])
            check_names = {c["name"] for c in report["checks"]}
            self.assertIn("all_audio_readable", check_names)

    # ------------------------------------------------ counterfactual issues

    def test_broken_pair_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, profile, _, _ = self._generate(root)
            manifest = output / "manifest.jsonl"
            lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
            # Drop the last row -> one pair now has a single member.
            manifest.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            report = validate_match(profile, manifest)
            self.assertGreater(len(report["manifest_issues"]), 0)
            self.assertFalse(report["overall_passed"])

    # ----------------------------------------------------------- redaction

    def test_report_has_no_paths_or_prohibited_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, profile, corpus, _ = self._generate(root)
            report_path = root / "report.json"
            validate_match(profile, output / "manifest.jsonl", report_path)
            text = report_path.read_text(encoding="utf-8")
            self.assertNotIn(str(corpus), text)
            self.assertNotIn(str(output), text)
            payload = json.loads(text)
            for forbidden in ("paths", "files", "ids", "transcript", "label", "text", "prediction"):
                self.assertNotIn(forbidden, payload)

    def test_redaction_guard_rejects_dataset_a_root(self):
        report = {"profile_hash": "x", "comparison": {}, "path_leak": "F:/datasetA/secret.wav"}
        with self.assertRaisesRegex(ValueError, "Dataset-A|redact"):
            _assert_report_redacted(report, Path("F:/datasetA"))

    def test_redaction_guard_rejects_prohibited_key(self):
        report = {"profile_hash": "x", "comparison": {}, "labels": ["a", "b"]}
        with self.assertRaisesRegex(ValueError, "prohibited"):
            _assert_report_redacted(report, None)

    def test_dataset_a_audio_root_rejected_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, profile, _, _ = self._generate(root)
            da = root / "datasetA"
            da.mkdir()
            # Inject a manifest line whose mixture path resolves under Dataset-A.
            manifest = output / "manifest.jsonl"
            lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
            bad = json.loads(lines[0])
            bad["row_id"] = "r3-train-0001-pos"
            bad["pair_id"] = "r3-train-0001"
            bad["mixture_audio"] = str(da / "leak.wav")
            manifest.write_text(json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Dataset-A"):
                validate_match(profile, manifest, dataset_a_root=da)

    # --------------------------------------------------------------- CLI

    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, profile, _, _ = self._generate(root)
            report_path = root / "cli_report.json"
            main(
                [
                    "--profile", str(profile),
                    "--manifest", str(output / "manifest.jsonl"),
                    "--output", str(report_path),
                ]
            )
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("profile_hash", report)


if __name__ == "__main__":
    unittest.main()
