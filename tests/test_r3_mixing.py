"""Tests for the R3 matched-counterfactual renderer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from xh202615.r3_data import validate_r3_manifest
from xh202615.r3_mixing import (
    RendererConfig,
    compute_nuisance_fingerprint,
    render_counterfactual_pair,
)


class R3MixingTests(unittest.TestCase):
    SR = 16_000

    def tone(self, freq, duration=0.05, amp=0.2):
        n = int(self.SR * duration)
        t = np.arange(n) / self.SR
        return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    def render(self, tmp, *, config=None, seed=1234, pair_id="pair-001"):
        cfg = config or RendererConfig()
        rng = np.random.default_rng(seed)
        out = Path(tmp)
        return render_counterfactual_pair(
            pair_id=pair_id,
            split="train",
            config=cfg,
            rng=rng,
            enrollment=self.tone(220),
            target=self.tone(440),
            interferers=(self.tone(330), self.tone(550)),
            noise=self.tone(110, amp=0.05),
            target_source_id="src-target",
            interferer_source_ids=("src-int-1", "src-int-2"),
            noise_source_id="noise-1",
            target_rir=None,
            target_rir_id=None,
            interferer_rirs=(None, None),
            interferer_rir_ids=(),
            renderer_family="r3-train",
            enrollment_path=out / "enroll.wav",
            mixture_paths=(out / "mix_pos.wav", out / "mix_neg.wav"),
            clean_target_paths=(out / "clean_pos.wav", out / "clean_neg.wav"),
        ), cfg

    def _equal_modulo_paths(self, r1, r2):
        d1, d2 = r1.to_dict(), r2.to_dict()
        for key in ("enrollment_audio", "mixture_audio", "clean_target_audio"):
            d1.pop(key)
            d2.pop(key)
        self.assertEqual(d1, d2)

    def test_deterministic_rendering(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            (pos1, neg1), _ = self.render(a, seed=42)
            (pos2, neg2), _ = self.render(b, seed=42)
            self._equal_modulo_paths(pos1, pos2)
            self._equal_modulo_paths(neg1, neg2)
            a1, _ = sf.read(str(pos1.mixture_audio), dtype="float64")
            a2, _ = sf.read(str(pos2.mixture_audio), dtype="float64")
            np.testing.assert_array_equal(a1, a2)
            c1, _ = sf.read(str(pos1.clean_target_audio), dtype="float64")
            c2, _ = sf.read(str(pos2.clean_target_audio), dtype="float64")
            np.testing.assert_array_equal(c1, c2)

    def test_sibling_nuisance_equality(self):
        with tempfile.TemporaryDirectory() as tmp:
            (pos, neg), _ = self.render(tmp)
            for field in (
                "pair_id", "split", "enrollment_audio", "target_source_id",
                "interferer_source_ids", "noise_source_id", "target_rir_id",
                "interferer_rir_ids", "renderer_family", "snr_db", "sir_db",
                "overlap_ratio", "codec", "clip_threshold", "nuisance_fingerprint",
            ):
                self.assertEqual(getattr(pos, field), getattr(neg, field), field)
            self.assertTrue(pos.target_present)
            self.assertFalse(neg.target_present)
            self.assertNotEqual(pos.row_id, neg.row_id)
            self.assertNotEqual(pos.mixture_audio, neg.mixture_audio)
            self.assertNotEqual(pos.clean_target_audio, neg.clean_target_audio)

    def test_one_positive_one_negative_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            (pos, neg), _ = self.render(tmp)
            self.assertEqual(pos.pair_id, neg.pair_id)
            self.assertEqual({pos.target_present, neg.target_present}, {True, False})

    def test_differing_target_component_only(self):
        cfg = RendererConfig(codec="lowpass8k", clip_threshold=1.0, target_peak=0.98)
        with tempfile.TemporaryDirectory() as tmp:
            (pos, neg), _ = self.render(tmp, config=cfg)
            positive, _ = sf.read(str(pos.mixture_audio), dtype="float64")
            negative, _ = sf.read(str(neg.mixture_audio), dtype="float64")
            clean, _ = sf.read(str(pos.clean_target_audio), dtype="float64")
            # The positive is exactly the negative plus the rendered target
            # component (additive form avoids float subtraction rounding).
            np.testing.assert_array_equal(positive, negative + clean)
            neg_clean, _ = sf.read(str(neg.clean_target_audio), dtype="float64")
            np.testing.assert_array_equal(neg_clean, np.zeros_like(neg_clean))

    def test_nuisance_fingerprint_is_computed_not_accepted(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            (pos, neg), cfg = self.render(a, seed=7)
            self.assertEqual(len(pos.nuisance_fingerprint), 64)
            int(pos.nuisance_fingerprint, 16)  # valid hex
            self.assertEqual(pos.nuisance_fingerprint, neg.nuisance_fingerprint)
            nuisance = {
                "target_source_id": "src-target",
                "interferer_source_ids": ["src-int-1", "src-int-2"],
                "noise_source_id": "noise-1",
                "target_rir_id": None,
                "interferer_rir_ids": [],
                "renderer_family": "r3-train",
                "snr_db": cfg.snr_db,
                "sir_db": cfg.sir_db,
                "overlap_ratio": cfg.overlap_ratio,
                "codec": cfg.codec,
                "clip_threshold": cfg.clip_threshold,
                "channel_response": list(cfg.channel_response),
                "sample_rate": cfg.sample_rate,
            }
            self.assertEqual(pos.nuisance_fingerprint, compute_nuisance_fingerprint(nuisance))
            (pos2, _), _ = self.render(b, seed=7, config=RendererConfig(snr_db=99.0))
            self.assertNotEqual(pos.nuisance_fingerprint, pos2.nuisance_fingerprint)

    def test_pair_passes_task1_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            (pos, neg), _ = self.render(tmp)
            self.assertEqual(validate_r3_manifest((pos, neg)), ())


if __name__ == "__main__":
    unittest.main()
