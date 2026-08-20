"""Tests for the aggregate acoustic profiling core (Task 1).

The profiler computes deterministic, aggregate-only acoustic statistics from
audio files. Serialized profiles must contain only counts, feature quantiles,
histograms, configuration, and a SHA-256 hash -- never input paths, filenames,
IDs, transcripts, labels, or per-file vectors.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from xh202615.acoustic_profile import (
    AudioStats,
    merge_profiles,
    profile_audio_paths,
    read_aggregate_profile,
    write_aggregate_profile,
)


class AcousticProfileTests(unittest.TestCase):
    SR = 16_000

    def _write_tone(
        self,
        path: Path,
        freq: float = 440.0,
        duration: float = 1.0,
        amp: float = 0.1,
        sr: int = 16_000,
    ) -> None:
        n = int(sr * duration)
        t = np.arange(n) / sr
        tone = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        sf.write(str(path), tone, sr)

    def _write_tone_with_silence(
        self,
        path: Path,
        freq: float = 440.0,
        amp: float = 0.1,
        sr: int = 16_000,
        silence_front: float = 0.5,
        tone_dur: float = 1.0,
    ) -> None:
        n_sil = int(sr * silence_front)
        n_tone = int(sr * tone_dur)
        t = np.arange(n_tone) / sr
        sig = np.concatenate(
            [
                np.zeros(n_sil, dtype=np.float32),
                (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32),
            ]
        )
        sf.write(str(path), sig, sr)

    def _write_clipped(self, path: Path, freq: float = 300.0, sr: int = 16_000) -> None:
        n = int(sr * 1.0)
        t = np.arange(n) / sr
        # Amplitude above full scale, then hard clipped -> samples pinned at +/-1.
        raw = 1.5 * np.sin(2.0 * np.pi * freq * t)
        clipped = np.clip(raw, -1.0, 1.0).astype(np.float32)
        sf.write(str(path), clipped, sr)

    # ------------------------------------------------------------------ basics

    def test_profiles_basic_features_of_pure_tone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "tone.wav"
            self._write_tone(wav, freq=440.0, duration=1.0, amp=0.1)
            stats = profile_audio_paths([wav])

            self.assertIsInstance(stats, AudioStats)
            self.assertEqual(stats.file_count, 1)
            self.assertAlmostEqual(stats.total_duration_seconds, 1.0, places=1)

            dur = stats.metrics["duration"]["quantiles"]
            self.assertAlmostEqual(dur["0.5"], 1.0, places=1)
            # RMS of a unit-amplitude sine is amp / sqrt(2).
            rms = stats.metrics["rms"]["quantiles"]
            self.assertAlmostEqual(rms["0.5"], 0.1 / np.sqrt(2.0), places=2)
            peak = stats.metrics["peak"]["quantiles"]
            self.assertAlmostEqual(peak["0.5"], 0.1, places=2)
            # A continuous tone is active throughout.
            active = stats.metrics["active_speech_ratio"]["quantiles"]
            self.assertGreater(active["0.5"], 0.9)
            # Spectral centroid of a pure 440 Hz tone is near 440 Hz.
            centroid = stats.metrics["spectral_centroid"]["quantiles"]
            self.assertAlmostEqual(centroid["0.5"], 440.0, delta=80.0)
            # A pure tone is not flat.
            flat = stats.metrics["spectral_flatness"]["quantiles"]
            self.assertLess(flat["0.5"], 0.2)
            rolloff = stats.metrics["spectral_rolloff"]["quantiles"]
            self.assertGreater(rolloff["0.5"], 0.0)
            self.assertLessEqual(rolloff["0.5"], self.SR / 2.0 + 1.0)

    def test_silence_runs_and_active_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "gap.wav"
            self._write_tone_with_silence(wav, silence_front=0.5, tone_dur=1.0)
            stats = profile_audio_paths([wav])
            active = stats.metrics["active_speech_ratio"]["quantiles"]
            # 1.0s tone out of 1.5s -> ~0.67 active.
            self.assertGreater(active["0.5"], 0.3)
            self.assertLess(active["0.5"], 0.9)
            runs = stats.metrics["silence_run"]["quantiles"]
            self.assertGreater(runs["0.5"], 0.0)

    def test_clipping_rate_is_nonzero_for_clipped_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "clip.wav"
            self._write_clipped(wav)
            stats = profile_audio_paths([wav])
            clip = stats.metrics["clipping_rate"]["quantiles"]
            self.assertGreater(clip["0.5"], 0.0)
            peak = stats.metrics["peak"]["quantiles"]
            self.assertAlmostEqual(peak["0.5"], 1.0, places=2)

    def test_source_sample_rate_and_channel_counts_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav8k = root / "eight.wav"
            self._write_tone(wav8k, freq=300.0, duration=0.5, sr=8_000)
            stats = profile_audio_paths([wav8k])
            self.assertEqual(stats.sample_rate_counts.get("8000"), 1)
            self.assertEqual(stats.channel_counts.get("1"), 1)

    def test_stationary_noise_estimate_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "gap.wav"
            self._write_tone_with_silence(wav, silence_front=0.5, tone_dur=1.0)
            stats = profile_audio_paths([wav])
            self.assertTrue(np.isfinite(stats.stationary_noise_estimate))
            # Quiet frames exist -> estimate is small but non-negative.
            self.assertGreaterEqual(stats.stationary_noise_estimate, 0.0)

    # -------------------------------------------------------- serialization

    def test_config_hash_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "tone.wav"
            self._write_tone(wav)
            stats = profile_audio_paths([wav])
            self.assertEqual(len(stats.hash), 64)
            int(stats.hash, 16)  # valid hex
            self.assertIn("version", stats.config)
            self.assertEqual(stats.config["quantiles"], [0.05, 0.25, 0.5, 0.75, 0.95])

    def test_json_round_trip_preserves_values_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wavs = []
            for i, freq in enumerate((440.0, 880.0, 220.0)):
                w = root / f"t{i}.wav"
                self._write_tone(w, freq=freq, duration=0.5 + 0.1 * i)
                wavs.append(w)
            stats = profile_audio_paths(wavs)
            out = root / "profile.json"
            write_aggregate_profile(out, stats)
            reread = read_aggregate_profile(out)
            self.assertEqual(reread.file_count, stats.file_count)
            self.assertEqual(reread.total_duration_seconds, stats.total_duration_seconds)
            self.assertEqual(reread.hash, stats.hash)
            self.assertEqual(
                reread.metrics["duration"]["quantiles"],
                stats.metrics["duration"]["quantiles"],
            )
            self.assertEqual(
                reread.metrics["spectral_centroid"]["histogram"],
                stats.metrics["spectral_centroid"]["histogram"],
            )

    def test_serialized_json_has_no_input_paths_or_prohibited_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav = root / "secret_tone.wav"
            self._write_tone(wav)
            stats = profile_audio_paths([wav])
            out = root / "profile.json"
            write_aggregate_profile(out, stats)
            text = out.read_text(encoding="utf-8")

            # No input path or filename leaks into the aggregate profile.
            self.assertNotIn(str(wav), text)
            self.assertNotIn(str(root), text)
            self.assertNotIn("secret_tone", text)
            payload = json.loads(text)
            for forbidden in (
                "paths", "files", "file_paths", "ids", "transcript",
                "label", "labels", "text", "prediction", "predictions",
            ):
                self.assertNotIn(forbidden, payload, f"prohibited key {forbidden!r}")
            self.assertNotIn("paths", payload.get("config", {}))

    def test_deterministic_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "tone.wav"
            self._write_tone(wav, freq=660.0, duration=0.7)
            first = profile_audio_paths([wav])
            second = profile_audio_paths([wav])
            self.assertEqual(first.hash, second.hash)
            self.assertEqual(
                first.metrics["spectral_centroid"]["quantiles"],
                second.metrics["spectral_centroid"]["quantiles"],
            )

    # ------------------------------------------------------------------- merge

    def test_merge_profiles_matches_direct_profiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group_a = []
            group_b = []
            for i, freq in enumerate((440.0, 880.0, 220.0, 660.0)):
                wa = root / f"a{i}.wav"
                wb = root / f"b{i}.wav"
                self._write_tone(wa, freq=freq, duration=0.6 + 0.05 * i)
                self._write_tone(wb, freq=freq * 1.5, duration=0.4 + 0.05 * i)
                group_a.append(wa)
                group_b.append(wb)
            profile_a = profile_audio_paths(group_a)
            profile_b = profile_audio_paths(group_b)
            merged = merge_profiles([profile_a, profile_b])
            direct = profile_audio_paths(group_a + group_b)

            self.assertEqual(merged.file_count, direct.file_count)
            self.assertAlmostEqual(
                merged.total_duration_seconds, direct.total_duration_seconds, places=6
            )
            # Exact in-memory merge reproduces the direct profile hash.
            self.assertEqual(merged.hash, direct.hash)

    def test_merge_of_read_back_profiles_sums_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wavs_a = []
            wavs_b = []
            for i in range(3):
                wa = root / f"a{i}.wav"
                wb = root / f"b{i}.wav"
                self._write_tone(wa, freq=440.0 + i * 100, duration=0.5)
                self._write_tone(wb, freq=880.0 + i * 100, duration=0.7)
                wavs_a.append(wa)
                wavs_b.append(wb)
            pa = profile_audio_paths(wavs_a)
            pb = profile_audio_paths(wavs_b)
            out_a = root / "a.json"
            out_b = root / "b.json"
            write_aggregate_profile(out_a, pa)
            write_aggregate_profile(out_b, pb)
            merged = merge_profiles([read_aggregate_profile(out_a), read_aggregate_profile(out_b)])
            self.assertEqual(merged.file_count, len(wavs_a) + len(wavs_b))
            self.assertAlmostEqual(
                merged.total_duration_seconds,
                pa.total_duration_seconds + pb.total_duration_seconds,
                places=6,
            )

    def test_empty_path_list_returns_valid_empty_profile(self):
        stats = profile_audio_paths([])
        self.assertEqual(stats.file_count, 0)
        self.assertEqual(stats.total_duration_seconds, 0.0)
        self.assertEqual(len(stats.hash), 64)


if __name__ == "__main__":
    unittest.main()
