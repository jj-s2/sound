"""Lightweight enrollment-conditioned target-speaker extractor.

A compact encoder-recurrent-bottleneck-decoder complex-ratio-mask (CRM) network
with FiLM (Feature-wise Linear Modulation) conditioning from the enrollment
embedding at the bottleneck.  Dependencies are limited to torch + numpy (no
SpeechBrain, torchaudio model packages, or other external model packages).

Public interfaces
------------------
- :class:`FiLMCRNExtractor` – the model.
- :func:`stft_waveform` / :func:`istft_waveform` – boundary-correct STFT/ISTFT.
- :func:`enhance_waveform` – end-to-end enhancement (STFT -> mask -> ISTFT).
- :func:`multi_resolution_stft_loss` / :func:`negative_si_sdr_loss` – losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# STFT / ISTFT helpers
# ---------------------------------------------------------------------------

def _resolve_window(
    win_length: int | None,
    n_fft: int,
    window: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a Hann window of *win_length* samples (or *n_fft* if None)."""
    if window is not None:
        return window
    win = win_length if win_length is not None else n_fft
    return torch.hann_window(win, device=device, dtype=dtype)


def stft_waveform(
    waveform: torch.Tensor,
    *,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int | None = None,
    window: torch.Tensor | None = None,
    center: bool = True,
) -> torch.Tensor:
    """Compute the STFT of a waveform with boundary-correct padding.

    Parameters
    ----------
    waveform
        Real-valued tensor of shape ``[batch, samples]`` (or ``[samples]``).
    n_fft
        FFT size.
    hop_length
        Hop size in samples.
    win_length
        Window length (defaults to *n_fft*).
    window
        Optional window tensor of length *win_length*.
    center
        If ``True``, the signal is padded so that frame *t* is centered at
        sample *t* * *hop_length*, ensuring ISTFT can recover the original
        length exactly.

    Returns
    -------
    Complex spectrogram of shape ``[batch, n_fft // 2 + 1, frames]``.
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    win = win_length if win_length is not None else n_fft
    window = _resolve_window(win_length, n_fft, window, waveform.device, waveform.dtype)
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win,
        window=window,
        center=center,
        return_complex=True,
    )


def istft_waveform(
    spectrogram: torch.Tensor,
    *,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int | None = None,
    window: torch.Tensor | None = None,
    center: bool = True,
    length: int | None = None,
) -> torch.Tensor:
    """Compute the ISTFT of a complex spectrogram with exact length matching.

    Parameters
    ----------
    spectrogram
        Complex tensor of shape ``[batch, n_fft // 2 + 1, frames]``.
    n_fft
        FFT size.
    hop_length
        Hop size in samples.
    win_length
        Window length (defaults to *n_fft*).
    window
        Optional window tensor of length *win_length*.
    center
        If ``True``, undo center padding.
    length
        Target output length in samples.  When provided, the output is
        truncated or zero-padded to exactly this length so that
        ``istft_waveform(stft_waveform(x, ...), length=N).shape[-1] == N``.

    Returns
    -------
    Waveform of shape ``[batch, samples]``.
    """
    win = win_length if win_length is not None else n_fft
    real_dtype = spectrogram.real.dtype if spectrogram.is_complex() else spectrogram.dtype
    window = _resolve_window(win_length, n_fft, window, spectrogram.device, real_dtype)
    return torch.istft(
        spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win,
        window=window,
        center=center,
        length=length,
        return_complex=False,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _group_norm(channels: int, max_groups: int = 4) -> nn.GroupNorm:
    """GroupNorm with at most *max_groups* groups (batch-statistic-free)."""
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class FiLMCRNExtractor(nn.Module):
    """Compact CRN with FiLM-conditioned GRU bottleneck.

    The encoder uses ``len(channels)`` stride-2 (frequency-only) Conv2d stages
    to progressively reduce the frequency axis.  A multi-layer GRU processes
    the temporal sequence at the bottleneck; FiLM modulation
    (``gamma * x + beta``) injects the enrollment embedding.  The decoder
    mirrors the encoder with ConvTranspose2d stages and U-Net skip connections.

    A complex ratio mask is predicted (``tanh``-bounded real and imaginary
    parts) and applied to the input spectrogram.

    Parameters
    ----------
    embedding_dim
        Dimensionality of the enrollment embedding (default 256).
    channels
        Channel progression for the encoder stages (default ``(16, 32, 64)``).
    n_fft, hop_length, win_length
        STFT configuration stored on the model for use by
        :func:`enhance_waveform`.
    gru_hidden
        Hidden size of the bottleneck GRU.
    gru_layers
        Number of stacked GRU layers.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        channels: tuple[int, ...] = (16, 32, 64),
        *,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int | None = None,
        gru_hidden: int = 256,
        gru_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.channels = tuple(channels)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.gru_hidden = gru_hidden

        # --- Compute bottleneck frequency dimension -----------------------
        kernel_f, pad_f, stride_f = 5, 2, 2
        freq = n_fft // 2 + 1
        for _ in self.channels:
            freq = (freq + 2 * pad_f - kernel_f) // stride_f + 1
        self._bottleneck_freq = freq
        bottleneck_dim = self.channels[-1] * freq

        # --- Encoder -------------------------------------------------------
        in_ch = 2  # real + imaginary
        self.encoder = nn.ModuleList()
        for out_ch in self.channels:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
                    _group_norm(out_ch),
                    nn.PReLU(out_ch),
                )
            )
            in_ch = out_ch

        # --- Bottleneck: GRU + FiLM + projection ---------------------------
        self.gru = nn.GRU(bottleneck_dim, gru_hidden, num_layers=gru_layers, batch_first=True)
        self.film_gamma = nn.Linear(embedding_dim, gru_hidden)
        self.film_beta = nn.Linear(embedding_dim, gru_hidden)
        self.proj = nn.Linear(gru_hidden, bottleneck_dim)

        # --- Decoder (mirror of encoder with skip connections) ------------
        # Pairs: (ConvTranspose2d in, out) for each stage
        rev = list(reversed(self.channels))  # e.g. [64, 32, 16]
        dec_pairs = list(zip(rev, rev[1:] + [2]))  # [(64,32), (32,16), (16,2)]

        self.decoder_t = nn.ModuleList()  # ConvTranspose2d stages
        self.decoder_c = nn.ModuleList()  # post-concat Conv2d stages (None for last)

        for i, (dec_in, dec_out) in enumerate(dec_pairs):
            self.decoder_t.append(
                nn.ConvTranspose2d(dec_in, dec_out, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))
            )
            if i < len(dec_pairs) - 1:
                # After upsampling, concat with the encoder skip of the same
                # channel count, then reduce with a 3x3 conv.
                self.decoder_c.append(
                    nn.Sequential(
                        nn.Conv2d(dec_out * 2, dec_out, kernel_size=3, padding=1),
                        _group_norm(dec_out),
                        nn.PReLU(dec_out),
                    )
                )
            else:
                self.decoder_c.append(None)  # type: ignore[arg-type]

    def forward(
        self, complex_spec: torch.Tensor, enrollment_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Predict a complex ratio mask.

        Parameters
        ----------
        complex_spec
            Complex spectrogram ``[batch, freq, frames]``.
        enrollment_embedding
            Real tensor ``[batch, embedding_dim]``.

        Returns
        -------
        Complex mask ``[batch, freq, frames]`` (``tanh``-bounded).
        """
        # Stack real/imag as input channels: [batch, 2, freq, frames]
        x = torch.stack([complex_spec.real, complex_spec.imag], dim=1)

        # Encoder with skip collection
        skips: list[torch.Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)

        # Reshape for GRU: [batch, C, F, T] -> [batch, T, C*F]
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)

        # GRU
        gru_out, _ = self.gru(x)  # [batch, T, gru_hidden]

        # FiLM conditioning from enrollment embedding
        gamma = self.film_gamma(enrollment_embedding)  # [batch, gru_hidden]
        beta = self.film_beta(enrollment_embedding)  # [batch, gru_hidden]
        x = gamma.unsqueeze(1) * gru_out + beta.unsqueeze(1)  # [batch, T, gru_hidden]

        # Project back to bottleneck dimension and reshape
        x = self.proj(x)  # [batch, T, C*F]
        x = x.reshape(b, t, c, f).permute(0, 2, 3, 1)  # [batch, C, F, T]

        # Decoder with U-Net skip connections
        for i, (dec_t, dec_c) in enumerate(zip(self.decoder_t, self.decoder_c)):
            x = dec_t(x)  # ConvTranspose2d: upsample frequency
            if dec_c is not None:
                skip = skips[-(i + 2)]  # corresponding encoder skip
                x = torch.cat([x, skip], dim=1)
                x = dec_c(x)

        # x: [batch, 2, freq, frames] -> bounded complex mask
        mask = torch.tanh(x)
        return torch.complex(mask[:, 0], mask[:, 1])


# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------

def enhance_waveform(
    model: FiLMCRNExtractor,
    waveform: torch.Tensor,
    enrollment_embedding: torch.Tensor,
) -> torch.Tensor:
    """Enhance a mixture waveform conditioned on an enrollment embedding.

    Parameters
    ----------
    model
        A :class:`FiLMCRNExtractor` (its STFT config is used).
    waveform
        Mixture waveform ``[batch, samples]``.
    enrollment_embedding
        Speaker embedding ``[batch, embedding_dim]``.

    Returns
    -------
    Enhanced waveform ``[batch, samples]`` matching the input length.
    """
    if waveform.ndim != 2:
        raise ValueError(f"waveform must be 2-D [batch, samples], got shape {waveform.shape}")
    if enrollment_embedding.ndim != 2:
        raise ValueError(
            f"enrollment_embedding must be 2-D [batch, dim], got shape {enrollment_embedding.shape}"
        )
    if waveform.shape[0] != enrollment_embedding.shape[0]:
        raise ValueError(
            f"batch mismatch: waveform {waveform.shape[0]} vs embedding {enrollment_embedding.shape[0]}"
        )

    length = waveform.shape[-1]
    spec = stft_waveform(
        waveform,
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        win_length=model.win_length,
    )
    mask = model(spec, enrollment_embedding)
    enhanced_spec = mask * spec
    return istft_waveform(
        enhanced_spec,
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        win_length=model.win_length,
        length=length,
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def multi_resolution_stft_loss(
    enhanced: torch.Tensor,
    target: torch.Tensor,
    *,
    fft_configs: tuple[tuple[int, int, int], ...] | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Multi-resolution spectral convergence + log-magnitude STFT loss.

    Parameters
    ----------
    enhanced, target
        Waveforms ``[batch, samples]``.
    fft_configs
        Tuple of ``(n_fft, hop_length, win_length)`` triples.  Defaults to
        three resolutions covering short, medium, and long windows.
    eps
        Small constant for numerical stability.

    Returns
    -------
    Scalar loss tensor (mean over resolutions).
    """
    if fft_configs is None:
        fft_configs = ((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048))

    if enhanced.ndim == 1:
        enhanced = enhanced.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)

    total = torch.zeros((), device=enhanced.device, dtype=enhanced.dtype)
    for n_fft, hop, win in fft_configs:
        est_spec = stft_waveform(enhanced, n_fft=n_fft, hop_length=hop, win_length=win)
        ref_spec = stft_waveform(target, n_fft=n_fft, hop_length=hop, win_length=win)

        est_mag = est_spec.abs()
        ref_mag = ref_spec.abs()

        # Spectral convergence (Frobenius-normalised)
        sc = torch.norm(est_mag - ref_mag, p="fro") / (torch.norm(ref_mag, p="fro") + eps)

        # Log spectral magnitude L1
        log_mag = F.l1_loss(torch.log(est_mag + eps), torch.log(ref_mag + eps))

        total = total + sc + log_mag

    return total / len(fft_configs)


def negative_si_sdr_loss(
    enhanced: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative SI-SDR loss (to be minimised).

    SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)

    where ``s_target`` is the projection of *target* onto *enhanced* and
    ``e_noise = enhanced - s_target``.

    Parameters
    ----------
    enhanced, target
        Waveforms ``[batch, samples]``.
    eps
        Small constant for numerical stability (avoids division by zero and
        log10(0)).

    Returns
    -------
    Scalar loss tensor (mean over batch).  Zero-target samples contribute 0.
    """
    if enhanced.ndim == 1:
        enhanced = enhanced.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)

    # Zero-mean (standard SI-SDR pre-processing)
    enhanced = enhanced - enhanced.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Per-sample projection
    dot_et = (enhanced * target).sum(dim=-1)  # [batch]
    dot_tt = (target * target).sum(dim=-1)  # [batch]

    scale = dot_et / (dot_tt + eps)  # [batch]
    s_target = scale.unsqueeze(-1) * target  # [batch, samples]
    e_noise = enhanced - s_target  # [batch, samples]

    dot_ss = (s_target * s_target).sum(dim=-1)  # [batch]
    dot_nn = (e_noise * e_noise).sum(dim=-1)  # [batch]

    si_sdr = 10.0 * torch.log10((dot_ss + eps) / (dot_nn + eps))  # [batch]

    # Zero-target safety: where ||target||^2 is negligible, set SI-SDR to 0
    valid = dot_tt > eps
    si_sdr = torch.where(valid, si_sdr, torch.zeros_like(si_sdr))

    return -si_sdr.mean()
