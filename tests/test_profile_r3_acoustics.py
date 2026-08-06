"""Tests for the acoustic profiler CLI and public comparator (Task 2)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts import profile_r3_acoustics as cli
from xh202615.acoustic_profile import profile_audio_paths, write_aggregate_profile


class ProfileR3AcousticsCLITests(unittest.TestCase):
    SR = 16_000

    def _make_wav_root(self, root: Path, count: int = 4) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            n = int(self.SR * 0.3)
            t = np.arange(n) / self.SR
            tone = (0.1 * np.sin(2.0 * np.pi * (220.0 + i * 40.0) * t)).astype(np.float32)
            sf.write(str(root / f"utt_{i:03d}.wav"), tone, self.SR)
        return root

    # ------------------------------------------------------------- profiling

    def test_profiles_tiny_wav_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_root = self._make_wav_root(root / "speech", count=4)
            out = root / "profile.json"
            cli.main(
                [
                    "--audio-root", str(wav_root),
                    "--output", str(out),
                    "--kind", "public_speech",
                    "--max-files", "10",
                    "--seed", "1",
                ]
            )
            self.assertTrue(out.is_file())
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["profiling"]["kind"], "public_speech")
            self.assertEqual(payload["profiling"]["discovered_count"], 4)
            self.assertEqual(payload["profiling"]["profiled_count"], 4)
            self.assertEqual(payload["file_count"], 4)
            self.assertEqual(len(payload["hash"]), 64)
            self.assertIn("duration", payload["metrics"])

    def test_output_contains_no_input_paths_or_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_root = self._make_wav_root(root / "corpus", count=3)
            out = root / "profile.json"
            cli.main(
                [
                    "--audio-root", str(wav_root),
                    "--output", str(out),
                    "--kind", "public_speech",
                ]
            )
            text = out.read_text(encoding="utf-8")
            self.assertNotIn(str(wav_root), text)
            self.assertNotIn("utt_000", text)
            for forbidden in ("paths", "files", "ids", "transcript", "label", "text", "prediction"):
                self.assertNotIn(forbidden, payload_keys := json.loads(text))
                self.assertNotIn(forbidden, json.loads(text).get("profiling", {}))

    def test_rejects_jsonl_root_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "labels.jsonl"
            fake.write_text('{"id":"x","label":"y"}\n', encoding="utf-8")
            out = root / "profile.json"
            with self.assertRaises(ValueError):
                cli.main(
                    [
                        "--audio-root", str(fake),
                        "--output", str(out),
                        "--kind", "dataseta_audio",
                    ]
                )
            self.assertFalse(out.exists())

    def test_rejects_non_directory_file_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "manifest.json"
            fake.write_text("{}", encoding="utf-8")
            out = root / "profile.json"
            with self.assertRaises(ValueError):
                cli.main(
                    [
                        "--audio-root", str(fake),
                        "--output", str(out),
                        "--kind", "public_noise",
                    ]
                )
            self.assertFalse(out.exists())

    def test_max_files_bounds_profiled_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_root = self._make_wav_root(root / "many", count=20)
            out = root / "profile.json"
            cli.main(
                [
                    "--audio-root", str(wav_root),
                    "--output", str(out),
                    "--kind", "public_rir",
                    "--max-files", "5",
                    "--seed", "42",
                ]
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["profiling"]["discovered_count"], 20)
            self.assertEqual(payload["profiling"]["profiled_count"], 5)
            self.assertEqual(payload["file_count"], 5)

    def test_max_files_selection_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_root = self._make_wav_root(root / "many", count=20)
            out_a = root / "a.json"
            out_b = root / "b.json"
            for out in (out_a, out_b):
                cli.main(
                    [
                        "--audio-root", str(wav_root),
                        "--output", str(out),
                        "--kind", "public_speech",
                        "--max-files", "5",
                        "--seed", "42",
                    ]
                )
            a = json.loads(out_a.read_text(encoding="utf-8"))
            b = json.loads(out_b.read_text(encoding="utf-8"))
            self.assertEqual(a["hash"], b["hash"])

    def test_never_opens_non_wav_files_in_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_root = self._make_wav_root(root / "mixed", count=3)
            # Garbage non-WAV files that would crash if opened as audio.
            (wav_root / "labels.jsonl").write_text("not audio at all\n", encoding="utf-8")
            (wav_root / "meta.json").write_text("{not valid json", encoding="utf-8")
            out = root / "profile.json"
            cli.main(
                [
                    "--audio-root", str(wav_root),
                    "--output", str(out),
                    "--kind", "dataseta_audio",
                ]
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
            # Only the 3 WAVs are profiled; non-WAV files are ignored.
            self.assertEqual(payload["profiling"]["discovered_count"], 3)
            self.assertEqual(payload["file_count"], 3)

    # --------------------------------------------------------------- compare

    def _write_profile(self, path: Path, freqs, durations) -> None:
        wavs = []
        tmp = path.parent
        for i, (f, d) in enumerate(zip(freqs, durations)):
            wav = tmp / f"s_{i}.wav"
            n = int(self.SR * d)
            t = np.arange(n) / self.SR
            sf.write(str(wav), (0.1 * np.sin(2 * np.pi * f * t)).astype(np.float32), self.SR)
            wavs.append(wav)
        write_aggregate_profile(path, profile_audio_paths(wavs))

    def test_compare_mode_emits_normalized_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "ref.json"
            cand = root / "cand.json"
            self._write_profile(ref, [220.0, 330.0], [0.4, 0.5])
            self._write_profile(cand, [660.0, 880.0], [0.6, 0.7])
            report_path = root / "report.json"
            cli.main(
                [
                    "--reference-profile", str(ref),
                    "--candidate-profile", str(cand),
                    "--output", str(report_path),
                ]
            )
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["reference_hash"], json.loads(ref.read_text())["hash"])
            self.assertEqual(report["candidate_hash"], json.loads(cand.read_text())["hash"])
            dur = report["metrics"]["duration"]
            self.assertIn("0.5", dur)
            self.assertIn("norm_diff", dur["0.5"])
            self.assertIn("abs_diff", dur["0.5"])
            # No paths leak into the comparison report.
            text = report_path.read_text(encoding="utf-8")
            self.assertNotIn(str(root), text)
            for forbidden in ("paths", "files", "ids", "transcript", "label", "text", "prediction"):
                self.assertNotIn(forbidden, report)

    def test_compare_mode_requires_both_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "ref.json"
            self._write_profile(ref, [220.0], [0.4])
            with self.assertRaises(ValueError):
                cli.main(
                    [
                        "--reference-profile", str(ref),
                        "--output", str(root / "report.json"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
