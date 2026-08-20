"""Tests for the R7 enrollment-conditioned speaker-verification reject score.

Pure-Python tests (no WeSpeaker, no audio I/O) cover the cosine helpers, the
score-variant assembly, the public-val Overall-optimal variant selection, and
the Youden-J surrogate calibration. Fail-closed behaviour is asserted for
non-finite scores, shape mismatches, missing scores, and single-class splits.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from xh202615.data import Sample
from xh202615.speaker_score import (
    SCORE_VARIANTS,
    cosine_similarity,
    cosine_tensor,
    score_from_variants,
    select_score_variant,
    variant_scores,
)
from xh202615.tse_presence import overall_at_threshold, overall_from_metrics
from scripts.train_tse import calibrate_speaker_youden


def _sample(sid: str, label: str | None, split: str = "val") -> Sample:
    return Sample(
        id=sid, split=split, wakeup_audio=".", wakeup_text="",
        command_audio=".", label=label,
    )


def _scene():
    """Enhanced cosine discriminates perfectly; mixture/max do not."""
    samples = [
        _sample("p0", "你好"), _sample("p1", "世界"),
        _sample("n0", None), _sample("n1", None),
    ]
    asr = {"p0": "你好", "p1": "世界", "n0": "干扰", "n1": "噪音"}
    enhanced = {"p0": 0.60, "p1": 0.55, "n0": 0.10, "n1": 0.15}
    mixture = {"p0": 0.20, "p1": 0.30, "n0": 0.80, "n1": 0.70}
    return samples, asr, enhanced, mixture


class CosineSimilarityTests(unittest.TestCase):
    def test_identical_is_one(self):
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=12)

    def test_gain_invariant(self):
        """The cosine must be invariant to a positive gain (score-shift proof)."""
        v = np.array([0.3, -0.7, 0.1, 0.9], dtype=np.float64)
        self.assertAlmostEqual(cosine_similarity(v, 2.5 * v), 1.0, places=12)
        self.assertAlmostEqual(cosine_similarity(v, -1.0 * v), -1.0, places=12)

    def test_orthogonal_is_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 1.0], dtype=np.float64)
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=12)

    def test_zero_vector_returns_zero_not_nan(self):
        v = np.array([1.0, 2.0], dtype=np.float64)
        z = np.zeros(2, dtype=np.float64)
        self.assertEqual(cosine_similarity(v, z), 0.0)
        self.assertEqual(cosine_similarity(z, z), 0.0)
        self.assertTrue(math.isfinite(cosine_similarity(z, v)))

    def test_torch_tensor_input(self):
        v = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(cosine_similarity(v, v.numpy()), 1.0, places=12)

    def test_shape_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            cosine_similarity(np.zeros(3), np.zeros(4))

    def test_non_finite_raises(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            cosine_similarity(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


class CosineTensorTests(unittest.TestCase):
    def test_batched_matches_scalar(self):
        a = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
        b = torch.tensor([[2.0, 4.0, 6.0], [1.0, 1.0, 0.0]])
        cos = cosine_tensor(a, b)
        self.assertEqual(cos.shape, (2,))
        self.assertAlmostEqual(float(cos[0]), 1.0, places=6)  # a[0] || b[0]
        # a[1]·b[1] = 1; norms 1 and sqrt(2) -> 1/sqrt(2)
        self.assertAlmostEqual(float(cos[1]), 1.0 / math.sqrt(2.0), places=6)

    def test_shape_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            cosine_tensor(torch.zeros(2, 3), torch.zeros(2, 4))


class VariantScoresTests(unittest.TestCase):
    def test_max_is_max_of_raws(self):
        vs = variant_scores(0.6, 0.2)
        self.assertEqual(vs["enhanced_cosine"], 0.6)
        self.assertEqual(vs["mixture_cosine"], 0.2)
        self.assertEqual(vs["max_cosine"], 0.6)
        self.assertEqual(set(vs), set(SCORE_VARIANTS))

    def test_non_finite_raises(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            variant_scores(float("nan"), 0.2)
        with self.assertRaisesRegex(ValueError, "finite"):
            variant_scores(0.1, float("inf"))


class ScoreFromVariantsTests(unittest.TestCase):
    def test_picks_each_variant(self):
        row = {"enhanced_cosine": 0.6, "mixture_cosine": 0.2, "max_cosine": 0.6}
        self.assertEqual(score_from_variants(row, "enhanced_cosine"), 0.6)
        self.assertEqual(score_from_variants(row, "mixture_cosine"), 0.2)
        self.assertEqual(score_from_variants(row, "max_cosine"), 0.6)

    def test_max_derived_from_raws_not_stored(self):
        # max_cosine stored value disagrees with raws -> derived from raws.
        row = {"enhanced_cosine": 0.6, "mixture_cosine": 0.2, "max_cosine": 0.99}
        self.assertEqual(score_from_variants(row, "max_cosine"), 0.6)

    def test_unknown_variant_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown speaker score variant"):
            score_from_variants({}, "bogus")

    def test_missing_field_raises(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            score_from_variants({"mixture_cosine": 0.2}, "enhanced_cosine")


class SelectScoreVariantTests(unittest.TestCase):
    def test_picks_discriminative_variant_with_perfect_overall(self):
        samples, asr, enhanced, mixture = _scene()
        scores_by_variant = {"enhanced_cosine": enhanced, "mixture_cosine": mixture}
        result = select_score_variant(
            samples, asr, scores_by_variant,
            overall_at_threshold=overall_at_threshold,
            overall_from_metrics=overall_from_metrics,
        )
        self.assertEqual(result["score_type"], "enhanced_cosine")
        self.assertEqual(result["threshold_source"], "public_val_max_overall")
        self.assertAlmostEqual(result["metrics"]["overall"], 1.0, places=6)
        # Threshold sits between the absent (<=0.15) and present (>=0.55) scores.
        self.assertGreater(result["threshold"], 0.15)
        self.assertLessEqual(result["threshold"], 0.55)
        # Per-variant diagnostics present for every variant considered.
        self.assertEqual(set(result["per_variant"]), {"enhanced_cosine", "mixture_cosine"})
        self.assertGreater(result["per_variant"]["enhanced_cosine"]["overall"],
                           result["per_variant"]["mixture_cosine"]["overall"])

    def test_includes_max_variant_when_both_raws_present(self):
        samples, asr, enhanced, mixture = _scene()
        scores_by_variant = {"enhanced_cosine": enhanced, "mixture_cosine": mixture}
        result = select_score_variant(
            samples, asr, scores_by_variant,
            overall_at_threshold=overall_at_threshold,
            overall_from_metrics=overall_from_metrics,
        )
        # max_cosine is derived inside select_score_variant? No: it must be
        # passed in. Re-run with all three to confirm max is considered.
        scores_by_variant["max_cosine"] = {
            sid: max(enhanced[sid], mixture[sid]) for sid in enhanced
        }
        result = select_score_variant(
            samples, asr, scores_by_variant,
            overall_at_threshold=overall_at_threshold,
            overall_from_metrics=overall_from_metrics,
        )
        self.assertEqual(set(result["per_variant"]), set(SCORE_VARIANTS))

    def test_missing_score_fails_closed(self):
        samples, asr, enhanced, _ = _scene()
        # Drop one sample's score -> fail closed.
        broken = {"p0": 0.6, "p1": 0.55, "n0": 0.1}
        with self.assertRaisesRegex(ValueError, "missing scores"):
            select_score_variant(
                samples, asr, {"enhanced_cosine": broken},
                overall_at_threshold=overall_at_threshold,
                overall_from_metrics=overall_from_metrics,
            )

    def test_single_class_fails_closed(self):
        samples = [_sample("p0", "你好"), _sample("p1", "世界")]
        asr = {"p0": "你好", "p1": "世界"}
        scores = {"p0": 0.9, "p1": 0.8}
        with self.assertRaisesRegex(ValueError, "both classes"):
            select_score_variant(
                samples, asr, {"enhanced_cosine": scores},
                overall_at_threshold=overall_at_threshold,
                overall_from_metrics=overall_from_metrics,
            )

    def test_no_variants_raises(self):
        samples, asr, _, _ = _scene()
        with self.assertRaisesRegex(ValueError, "no score variants"):
            select_score_variant(
                samples, asr, {},
                overall_at_threshold=overall_at_threshold,
                overall_from_metrics=overall_from_metrics,
            )


class CalibrateSpeakerYoudenTests(unittest.TestCase):
    def test_selects_best_auc_variant(self):
        # enhanced perfectly separates; mixture does not.
        enhanced = np.array([0.6, 0.55, 0.1, 0.15])
        mixture = np.array([0.2, 0.3, 0.2, 0.3])  # no separation -> AUC 0.5
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        result = calibrate_speaker_youden(enhanced, mixture, labels)
        self.assertEqual(result["score_type"], "enhanced_cosine")
        self.assertAlmostEqual(result["auc"], 1.0, places=6)
        self.assertEqual(result["threshold_source"], "public_val_youden_j")
        self.assertEqual(set(result["per_variant"]), set(SCORE_VARIANTS))

    def test_single_class_fails_closed(self):
        enhanced = np.array([0.6, 0.55])
        mixture = np.array([0.2, 0.3])
        labels = np.array([1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "cannot calibrate"):
            calibrate_speaker_youden(enhanced, mixture, labels)

    def test_shape_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "share a shape"):
            calibrate_speaker_youden(
                np.array([0.6, 0.1]), np.array([0.2, 0.3, 0.4]), np.array([1.0, 0.0])
            )


if __name__ == "__main__":
    unittest.main()
