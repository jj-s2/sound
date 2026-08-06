"""Enrollment-conditioned speaker-verification reject score (R7).

The R6 reject signal was an absolute presence probability (``sigmoid`` of a
trained logit). Its *distribution* shifted between the public synthetic domain
and the blind Dataset-A domain, so a public-calibrated threshold rejected
~96 % of real positives. This module replaces that gating signal with a
**speaker-verification cosine** between the frozen WeSpeaker enrollment
embedding and the WeSpeaker embedding of the enhanced (and mixture) signal:

    enhanced_cosine = cos(WeSpeaker(enrollment), WeSpeaker(TSE_enhanced))
    mixture_cosine  = cos(WeSpeaker(enrollment), WeSpeaker(mixture))
    max_cosine      = max(enhanced_cosine, mixture_cosine)

The cosine is a *relative*, enrollment-conditioned comparison. WeSpeaker's
embedding space (CN-Celeb ResNet34, 256-D) is pretrained to be
speaker-discriminative across recording conditions, so the cosine - and hence a
fixed threshold on it - is far more stable across the public->Dataset-A domain
gap than an absolute head trained from scratch on synthetic data.

Differentiability / compute
---------------------------
WeSpeaker's ``compute_features``/``model`` are wrapped in ``torch.no_grad()``,
so the cosine is a **non-differentiable, post-hoc signal**: it does not
backprop into the TSE mask. This is deliberate - the reconstruction objective
(SI-SDR + multi-resolution STFT to the clean target) already drives
``enhanced ≈ clean target`` for present rows, so no differentiable speaker loss
is needed. Compute cost is one frozen-WeSpeaker forward per signal per sample
(enhanced + mixture); see the R7 design doc.

Data boundary
-------------
Threshold calibration uses ONLY public samples (the public manifest's val
split). No Dataset-A field is read here. The functions are score-agnostic
helpers; the evaluator chooses which score field to gate on.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F

EMBEDDING_DIM = 256
SAMPLE_RATE = 16_000

# The three monotone score variants considered for gating. ``max_cosine`` falls
# back to the mixture when the TSE enhanced output is unreliable, so the reject
# decision does not depend solely on extractor quality.
SCORE_VARIANTS: tuple[str, ...] = ("enhanced_cosine", "mixture_cosine", "max_cosine")


class SpeakerEncoder(Protocol):
    """Embeds 16 kHz mono waveforms into a fixed speaker-embedding space.

    Implementations MUST be frozen (no gradient through the encoder) and MUST
    return embeddings in the same space as the cached enrollment vectors. The
    default :class:`WeSpeakerEncoder` satisfies both; tests inject a mock.
    """

    def embed_tensor(self, waveform: torch.Tensor) -> torch.Tensor:
        """Embed a batch ``[B, samples]`` -> raw embeddings ``[B, dim]``."""
        ...

    def embed_audio(self, audio: np.ndarray) -> np.ndarray:
        """Embed a 1-D float32 waveform ``[samples]`` -> ``[dim]`` numpy."""
        ...

    def embed_path(self, path: str | Path) -> np.ndarray:
        """Read + embed an audio file -> ``[dim]`` numpy (enrollment path)."""
        ...


# ---------------------------------------------------------------------------
# Pure score helpers (no WeSpeaker, no audio I/O - fully unit-testable)
# ---------------------------------------------------------------------------

def _as_numpy_vector(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("embedding must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("embedding must be finite")
    return arr


def cosine_similarity(a: object, b: object, *, eps: float = 1e-8) -> float:
    """Cosine similarity in [-1, 1] between two 1-D embeddings.

    Both vectors are L2-normalised first, so callers may pass raw or pre-
    normalised WeSpeaker embeddings; the result is identical. Raises on empty
    or non-finite input (fail-closed).
    """
    va = _as_numpy_vector(a)
    vb = _as_numpy_vector(b)
    if va.shape != vb.shape:
        raise ValueError(
            f"embedding shape mismatch: {va.shape} vs {vb.shape}"
        )
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < eps or nb < eps:
        # A zero vector is unordered relative to every vector; return 0 (a
        # neutral, non-accepting score) rather than a NaN that would poison
        # calibration.
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def cosine_tensor(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Batched cosine ``[B]`` between ``[B, dim]`` embeddings (L2-normalised)."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected [B, dim] tensors, got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape != b.shape:
        raise ValueError(f"batch/shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    an = F.normalize(a.float(), dim=-1, eps=eps)
    bn = F.normalize(b.float(), dim=-1, eps=eps)
    return (an * bn).sum(dim=-1)


def variant_scores(enhanced_cosine: float, mixture_cosine: float) -> dict[str, float]:
    """Assemble the three gating variants from the two raw cosines.

    Fails closed on non-finite inputs so a NaN/inf score can never silently
    become a ``max`` that gates incorrectly.
    """
    for name, value in (("enhanced_cosine", enhanced_cosine),
                        ("mixture_cosine", mixture_cosine)):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
    enhanced = float(enhanced_cosine)
    mixture = float(mixture_cosine)
    return {
        "enhanced_cosine": enhanced,
        "mixture_cosine": mixture,
        "max_cosine": max(enhanced, mixture),
    }


def score_from_variants(row_scores: Mapping[str, float], variant: str) -> float:
    """Pick the gating score for a row given the chosen variant.

    Validates the variant and the underlying fields so a malformed audio map
    cannot silently fall back to a different score.
    """
    if variant not in SCORE_VARIANTS:
        raise ValueError(f"unknown speaker score variant {variant!r}; expected one of {SCORE_VARIANTS}")
    if variant == "max_cosine":
        # Derive max from the two raw cosines rather than trusting a stored
        # max field (which could disagree with the stored raws).
        enhanced = row_scores.get("enhanced_cosine")
        mixture = row_scores.get("mixture_cosine")
        if enhanced is None or mixture is None:
            raise ValueError("max_cosine requires enhanced_cosine and mixture_cosine")
        return max(float(enhanced), float(mixture))
    value = row_scores.get(variant)
    if value is None:
        raise ValueError(f"score variant {variant!r} is missing from the audio map")
    return float(value)


# ---------------------------------------------------------------------------
# Threshold calibration with variant selection (public-val Overall only)
# ---------------------------------------------------------------------------

def select_score_variant(
    samples: Sequence,
    asr_by_id: Mapping[str, str],
    scores_by_variant: Mapping[str, Mapping[str, float]],
    *,
    overall_at_threshold: Callable,
    overall_from_metrics: Callable,
    max_candidates: int = 2000,
) -> dict:
    """Pick the cosine variant + threshold maximising public-val Overall.

    For each variant in :data:`SCORE_VARIANTS` that is present in
    ``scores_by_variant``, sweep the threshold (unique scores + boundaries,
    quantile-subsampled for very large sets, exactly mirroring
    :func:`xh202615.tse_presence.calibrate_threshold_overall`) and score the
    official ``Overall = ((1-CER)+RR)/2``. Select the variant+threshold with
    the highest Overall, tie-broken toward higher RR (protect rejection) then
    lower CER. Returns the variant, threshold, metrics, and per-variant
    diagnostics. Fails loudly if no variant has both classes.
    """
    # Local import to avoid a circular dependency at module load time
    # (tse_presence imports from training_data/data; this module is imported by
    # the trainer and evaluator which already import tse_presence).
    from .tse_presence import (
        REJECT_TEXT,
    )  # noqa: F401  - keeps the public-val evaluator semantics coupled

    materialized = list(samples)
    present_ids = [str(s.id) for s in materialized if s.label is not None]
    absent_ids = [str(s.id) for s in materialized if s.label is None]
    if not present_ids or not absent_ids:
        raise ValueError(
            f"variant selection needs both classes; got {len(present_ids)} pos / "
            f"{len(absent_ids)} neg"
        )

    per_variant: list[dict] = []
    for variant in SCORE_VARIANTS:
        score_map = scores_by_variant.get(variant)
        if score_map is None:
            continue
        # Fail closed if any sample lacks a score for this variant.
        missing = [sid for sid in (present_ids + absent_ids) if sid not in score_map]
        if missing:
            raise ValueError(
                f"variant {variant!r} is missing scores for {len(missing)} sample(s): "
                f"{missing[:5]}"
            )
        scores = sorted({float(score_map[sid]) for sid in (present_ids + absent_ids)})
        if len(scores) > max_candidates:
            idx = np.linspace(0, len(scores) - 1, max_candidates).astype(int)
            scores = sorted({scores[i] for i in idx})
        candidates = [min(scores) - 1.0] + scores + [max(scores) + 1.0]
        best: dict | None = None
        for thr in candidates:
            metrics = overall_at_threshold(materialized, asr_by_id, score_map, thr)
            key = (metrics["overall"], metrics["avg_rr"], -metrics["avg_cer"])
            if best is None or key > best["key"]:
                best = {"key": key, "threshold": float(thr), "metrics": metrics}
        per_variant.append({
            "variant": variant,
            "threshold": best["threshold"],
            "metrics": best["metrics"],
            "n_candidates": len(candidates),
        })

    if not per_variant:
        raise ValueError("no score variants available for calibration")

    # Select the best variant by Overall, then RR, then -CER.
    best_variant = max(
        per_variant,
        key=lambda pv: (
            pv["metrics"]["overall"],
            pv["metrics"]["avg_rr"],
            -pv["metrics"]["avg_cer"],
        ),
    )
    return {
        "score_type": best_variant["variant"],
        "threshold": float(best_variant["threshold"]),
        "threshold_source": "public_val_max_overall",
        "metrics": best_variant["metrics"],
        "n_pos": len(present_ids),
        "n_neg": len(absent_ids),
        "per_variant": {
            pv["variant"]: {
                "threshold": pv["threshold"],
                "overall": pv["metrics"]["overall"],
                "avg_cer": pv["metrics"]["avg_cer"],
                "avg_rr": pv["metrics"]["avg_rr"],
                "false_reject_rate": pv["metrics"]["false_reject_rate"],
                "false_accept_rate": pv["metrics"]["false_accept_rate"],
            }
            for pv in per_variant
        },
    }


# ---------------------------------------------------------------------------
# WeSpeaker-backed encoder (frozen, VAD off, no_grad)
# ---------------------------------------------------------------------------

def _load_mono_16k(path: str | Path) -> np.ndarray:
    """Read a 16 kHz mono float32 waveform (averages multi-channel)."""
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}: {path}")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"audio is empty or non-finite: {path}")
    return np.ascontiguousarray(mono)


class WeSpeakerEncoder:
    """Frozen WeSpeaker speaker encoder used for enrollment AND enhanced/mixture.

    A single instance serves both the enrollment-embedding cache (``embed_path``)
    and the reject-score embedding (``embed_tensor``), guaranteeing the two live
    in the same speaker space. Weights are frozen, VAD is off (so silence /
    suppressed enhanced output yields a stable, comparable embedding rather than
    ``None``), and every forward is under ``torch.no_grad()``. The fbank frontend
    config matches WeSpeaker's ``Speaker.compute_features`` exactly
    (80 mel bins, 25 ms / 10 ms, hamming window, per-utterance CMN).
    """

    def __init__(self, model_name: str, device: torch.device) -> None:
        try:
            import wespeaker  # noqa: import-outside-toplevel (env-dependent)
            from torchaudio.compliance import kaldi  # noqa: import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("WeSpeaker + torchaudio are required for the speaker score") from exc
        self._kaldi = kaldi
        speaker = wespeaker.load_model(model_name)
        if callable(getattr(speaker, "set_device", None)):
            speaker.set_device(str(device))
        speaker.set_vad(False)  # stable embeddings on silence / suppressed output
        self._speaker = speaker
        self._model = speaker.model
        self.device = torch.device(device)
        # Freeze weights; the encoder is a fixed feature extractor.
        for param in self._model.parameters():
            param.requires_grad_(False)
        self._model.eval()
        self.model_name = model_name

    @property
    def frontend_type(self) -> str:
        return getattr(self._model, "frontend_type", "fbank")

    def _fbank_single(self, waveform: torch.Tensor) -> torch.Tensor:
        """fbank + per-utterance CMN for one waveform ``[1, samples]`` -> ``[1, T, D]``.

        Mirrors WeSpeaker's ``Speaker.compute_features`` exactly: ``kaldi.fbank``
        on a single utterance returns ``[T, D]``; CMN subtracts the mean over
        time (dim 0); ``unsqueeze(0)`` adds the batch dim WeSpeaker's ResNet
        expects. Batched callers loop over this to stay byte-for-byte consistent
        with the enrollment-embedding path.
        """
        waveform = waveform.to(self.device).to(torch.float32)
        feat = self._kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            sample_frequency=SAMPLE_RATE,
            window_type="hamming",
        )  # [T, D] for a single utterance
        if feat.ndim == 3:
            feat = feat.squeeze(0)  # tolerate a leading singleton dim
        feat = feat - feat.mean(dim=0)  # per-utterance CMN over time
        return feat.unsqueeze(0)  # [1, T, D]

    @torch.no_grad()
    def embed_tensor(self, waveform: torch.Tensor) -> torch.Tensor:
        """Embed ``[B, samples]`` -> raw embeddings ``[B, dim]`` (no_grad).

        Processes one utterance at a time so the fbank/CMN path is identical to
        :meth:`embed_path` (the enrollment path); the two embeddings therefore
        share the exact same space.
        """
        if waveform.ndim != 2:
            raise ValueError(f"waveform must be 2-D [B, samples], got {tuple(waveform.shape)}")
        embs: list[torch.Tensor] = []
        for i in range(waveform.shape[0]):
            feat = self._fbank_single(waveform[i : i + 1])
            outputs = self._model(feat)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
            if outputs.ndim != 2 or outputs.shape[-1] != EMBEDDING_DIM:
                raise ValueError(
                    f"unexpected WeSpeaker embedding shape: {tuple(outputs.shape)}; "
                    f"expected [B, {EMBEDDING_DIM}]"
                )
            embs.append(outputs)  # [1, dim]
        return torch.cat(embs, dim=0)  # [B, dim]

    def embed_audio(self, audio: np.ndarray) -> np.ndarray:
        """Embed a 1-D float32 waveform ``[samples]`` -> ``[dim]`` numpy."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("audio must be a non-empty finite float32 waveform")
        wav = torch.from_numpy(audio).reshape(1, -1)
        emb = self.embed_tensor(wav).reshape(-1)
        return emb.detach().cpu().numpy()

    def embed_path(self, path: str | Path) -> np.ndarray:
        """Read + embed an audio file -> ``[dim]`` numpy (enrollment path)."""
        return self.embed_audio(_load_mono_16k(path))


def build_wespeaker_encoder(model_name: str, device: torch.device) -> WeSpeakerEncoder:
    """Construct the default frozen WeSpeaker encoder (env-dependent)."""
    return WeSpeakerEncoder(model_name, device)
