"""Tests for the lightweight enrollment-conditioned target-speaker extractor.

TDD tests for Task 3 of the R3 domain-matched TSE pilot.  All tests run on CPU.
No Dataset-A dependency.
"""

from __future__ import annotations

import unittest

import torch

from xh202615.target_extractor import (
    FiLMCRNExtractor,
    enhance_waveform,
    istft_waveform,
    multi_resolution_stft_loss,
    negative_si_sdr_loss,
    stft_waveform,
)


class StftIstftTests(unittest.TestCase):
    """Boundary-correct STFT/ISTFT round-trip and length matching."""

    def test_roundtrip_recovers_signal(self):
        torch.manual_seed(0)
        waveform = torch.randn(2, 16000)
        spec = stft_waveform(waveform, n_fft=512, hop_length=128)
        self.assertEqual(spec.ndim, 3)
        self.assertEqual(spec.shape[0], 2)
        recovered = istft_waveform(spec, n_fft=512, hop_length=128, length=16000)
        self.assertEqual(recovered.shape, waveform.shape)
        self.assertTrue(torch.allclose(recovered, waveform, atol=1e-4))

    def test_output_length_matches_input_for_various_lengths(self):
        for n in (8000, 16000, 16001, 32000, 48000):
            wave = torch.randn(1, n)
            spec = stft_waveform(wave, n_fft=512, hop_length=128)
            recovered = istft_waveform(spec, n_fft=512, hop_length=128, length=n)
            self.assertEqual(recovered.shape[-1], n, f"length {n} mismatch")

    def test_stft_returns_complex(self):
        wave = torch.randn(1, 16000)
        spec = stft_waveform(wave, n_fft=512, hop_length=128)
        self.assertTrue(spec.is_complex(), "spectrogram should be complex-valued")


class FiLMCRNExtractorTests(unittest.TestCase):
    """Model construction, parameter budget, and conditioning."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = FiLMCRNExtractor(embedding_dim=256, channels=(16, 32, 64))
        self.model.eval()

    def test_parameter_count_between_2m_and_5m(self):
        count = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(count, 2_000_000, f"param count {count} not > 2M")
        self.assertLess(count, 5_000_000, f"param count {count} not < 5M")

    def test_forward_returns_mask_matching_spectrogram_shape(self):
        waveform = torch.randn(2, 16000)
        emb = torch.randn(2, 256)
        spec = stft_waveform(
            waveform,
            n_fft=self.model.n_fft,
            hop_length=self.model.hop_length,
            win_length=self.model.win_length,
        )
        mask = self.model(spec, emb)
        self.assertEqual(mask.shape, spec.shape)
        self.assertTrue(mask.is_complex())

    def test_conditioning_sensitivity_different_embeddings(self):
        """Different enrollment embeddings must produce different outputs."""
        waveform = torch.randn(2, 16000)
        emb_a = torch.randn(2, 256)
        emb_b = torch.randn(2, 256)
        with torch.no_grad():
            out_a = enhance_waveform(self.model, waveform, emb_a)
            out_b = enhance_waveform(self.model, waveform, emb_b)
        self.assertEqual(out_a.shape, waveform.shape)
        self.assertFalse(
            torch.allclose(out_a, out_b, atol=1e-5),
            "different embeddings produced identical outputs",
        )


class EnhanceWaveformTests(unittest.TestCase):
    """End-to-end enhancement: length, finiteness, zero-input safety."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = FiLMCRNExtractor(embedding_dim=256, channels=(16, 32, 64))
        self.model.eval()

    def test_output_length_matches_input_and_finite(self):
        waveform = torch.randn(2, 16000)
        emb = torch.randn(2, 256)
        out = enhance_waveform(self.model, waveform, emb)
        self.assertEqual(out.shape, waveform.shape)
        self.assertTrue(torch.isfinite(out).all(), "output contains non-finite values")

    def test_output_length_matches_for_various_lengths(self):
        emb = torch.randn(1, 256)
        for n in (8000, 16000, 16001, 32000):
            wave = torch.randn(1, n)
            out = enhance_waveform(self.model, wave, emb)
            self.assertEqual(out.shape[-1], n, f"length {n} mismatch")
            self.assertTrue(torch.isfinite(out).all())

    def test_zero_silent_input_safety(self):
        """All-zero waveform must not produce NaN/Inf."""
        zero_wave = torch.zeros(2, 16000)
        emb = torch.randn(2, 256)
        out = enhance_waveform(self.model, zero_wave, emb)
        self.assertEqual(out.shape, zero_wave.shape)
        self.assertTrue(torch.isfinite(out).all(), "zero input produced non-finite output")

    def test_batch_of_one(self):
        wave = torch.randn(1, 16000)
        emb = torch.randn(1, 256)
        out = enhance_waveform(self.model, wave, emb)
        self.assertEqual(out.shape, wave.shape)
        self.assertTrue(torch.isfinite(out).all())


class LossTests(unittest.TestCase):
    """Numerically stable losses with CPU backward pass."""

    def test_multi_resolution_stft_loss_scalar_and_finite(self):
        enhanced = torch.randn(2, 16000)
        target = torch.randn(2, 16000)
        loss = multi_resolution_stft_loss(enhanced, target)
        self.assertEqual(loss.ndim, 0, "loss must be scalar")
        self.assertTrue(torch.isfinite(loss))

    def test_negative_si_sdr_loss_scalar_and_finite(self):
        enhanced = torch.randn(2, 16000)
        target = torch.randn(2, 16000)
        loss = negative_si_sdr_loss(enhanced, target)
        self.assertEqual(loss.ndim, 0, "loss must be scalar")
        self.assertTrue(torch.isfinite(loss))

    def test_si_sdr_loss_zero_when_identical(self):
        """SI-SDR should be high (loss very negative) when enhanced == target."""
        target = torch.randn(2, 16000) * 0.1
        loss = negative_si_sdr_loss(target, target)
        self.assertTrue(loss < -50, f"identical signals should give loss < -50, got {loss}")

    def test_cpu_backward_pass_runs(self):
        """Full forward + loss + backward must work on CPU."""
        model = FiLMCRNExtractor(embedding_dim=256, channels=(16, 32, 64))
        model.train()
        waveform = torch.randn(2, 16000)
        target = torch.randn(2, 16000)
        emb = torch.randn(2, 256)
        enhanced = enhance_waveform(model, waveform, emb)
        loss = multi_resolution_stft_loss(enhanced, target) + negative_si_sdr_loss(
            enhanced, target
        )
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        self.assertTrue(has_grad, "no parameter received a non-zero gradient")

    def test_loss_backward_with_stft_loss_only(self):
        """multi_resolution_stft_loss must be independently differentiable."""
        enhanced = torch.randn(2, 16000, requires_grad=True)
        target = torch.randn(2, 16000)
        loss = multi_resolution_stft_loss(enhanced, target)
        loss.backward()
        self.assertIsNotNone(enhanced.grad)
        self.assertTrue(torch.isfinite(enhanced.grad).all())

    def test_loss_backward_with_si_sdr_only(self):
        """negative_si_sdr_loss must be independently differentiable."""
        enhanced = torch.randn(2, 16000, requires_grad=True)
        target = torch.randn(2, 16000)
        loss = negative_si_sdr_loss(enhanced, target)
        loss.backward()
        self.assertIsNotNone(enhanced.grad)
        self.assertTrue(torch.isfinite(enhanced.grad).all())

    def test_zero_target_si_sdr_safety(self):
        """Zero target must not produce NaN/Inf."""
        enhanced = torch.randn(2, 16000)
        target = torch.zeros(2, 16000)
        loss = negative_si_sdr_loss(enhanced, target)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
