"""Deterministic matched-counterfactual renderer for the R3 TSE pilot.

Positive and negative siblings share every nuisance choice (enrollment, target
source metadata, interferers, placements, RIRs, noise, gains, SNR/SIR/overlap,
channel response, codec simulation, clipping, and one common pair-level peak
scale). The negative waveform removes only the target component; the siblings
are never independently normalized. Dependencies are limited to NumPy +
soundfile.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from xh202615.r3_data import R3MixtureRow


@dataclass(frozen=True)
class RendererConfig:
    """Acoustic scene parameters shared by both counterfactual siblings."""

    sample_rate: int = 16_000
    snr_db: float | None = 5.0
    sir_db: float | None = 0.0
    overlap_ratio: float = 0.5
    codec: str = "pcm16"
    clip_threshold: float = 1.0
    target_peak: float = 0.98
    channel_response: tuple[float, ...] = (1.0,)


def _rms(audio: np.ndarray) -> float:
    a = np.asarray(audio, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(a))))


def _apply_rir(audio: np.ndarray, rir: np.ndarray | None) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    if rir is None:
        return a
    rir = np.asarray(rir, dtype=np.float64)
    if _rms(a) == 0.0:
        return a
    if _rms(rir) == 0.0:
        raise ValueError("RIR audio is silent")
    normalized = rir / max(float(np.max(np.abs(rir))), 1e-8)
    full = len(a) + len(normalized) - 1
    fft_len = 1 << (full - 1).bit_length()
    convolved = np.fft.irfft(
        np.fft.rfft(a, fft_len) * np.fft.rfft(normalized, fft_len), fft_len
    )[:full]
    direct = int(np.argmax(np.abs(normalized)))
    segment = convolved[direct : direct + len(a)]
    if len(segment) < len(a):
        segment = np.pad(segment, (0, len(a) - len(segment)))
    target_rms = _rms(a)
    seg_rms = _rms(segment)
    if seg_rms == 0.0:
        raise ValueError("RIR convolution produced silence")
    return segment * (target_rms / seg_rms)


def _crop_or_repeat(audio: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    a = np.asarray(audio, dtype=np.float64)
    if a.size == 0:
        return np.zeros(length, dtype=np.float64)
    if len(a) >= length:
        start = int(rng.integers(0, len(a) - length + 1))
        return a[start : start + length]
    repeats = (length + len(a) - 1) // len(a)
    return np.tile(a, repeats)[:length]


def _channel_response(audio: np.ndarray, response: tuple[float, ...]) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    if response is None or len(response) <= 1:
        return a
    return np.convolve(a, np.asarray(response, dtype=np.float64), mode="same")


def _codec_simulate(audio: np.ndarray, codec: str) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    if codec == "pcm16":
        return np.round(a * 32768.0) / 32768.0
    if codec == "mulaw8":
        mu = 255.0
        sign = np.sign(a)
        encoded = sign * np.log1p(mu * np.abs(a)) / np.log1p(mu)
        quantized = np.round(encoded * 127.0) / 127.0
        return np.sign(quantized) * (np.power(1.0 + mu, np.abs(quantized)) - 1.0) / mu
    if codec == "lowpass8k":
        coeffs = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        return np.convolve(a, coeffs, mode="same")
    raise ValueError(f"unknown codec: {codec!r}")


def _clip(audio: np.ndarray, threshold: float) -> np.ndarray:
    return np.clip(np.asarray(audio, dtype=np.float64), -float(threshold), float(threshold))


def _place_interferer(rendered: np.ndarray, length: int, overlap_ratio: float, rng: np.random.Generator) -> np.ndarray:
    overlap_samples = max(1, int(round(length * float(overlap_ratio))))
    overlap_samples = min(length, overlap_samples)
    crop = _crop_or_repeat(rendered, overlap_samples, rng)
    max_start = length - overlap_samples
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    placed = np.zeros(length, dtype=np.float64)
    placed[start : start + overlap_samples] = crop
    return placed


def render_pair_audio(
    *,
    target: np.ndarray,
    interferers: tuple[np.ndarray, ...],
    noise: np.ndarray | None,
    target_rir: np.ndarray | None,
    interferer_rirs: tuple[np.ndarray | None, ...],
    config: RendererConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render (positive_mixture, negative_mixture, clean_target) arrays.

    The positive and negative share interferers, placements, RIRs, noise, gains,
    channel response, codec, clipping, and one common pair-level peak scale. The
    negative omits only the target component.
    """
    length = len(target)
    if length == 0:
        raise ValueError("target audio must be non-empty")

    target_component = _apply_rir(target, target_rir)
    target_rms = _rms(target_component)
    if target_rms == 0.0:
        raise ValueError("target audio is silent after rendering")

    sir_ratio = (10.0 ** (config.sir_db / 20.0)) if config.sir_db is not None else None
    interferer_components: list[np.ndarray] = []
    for interferer, rir in zip(interferers, interferer_rirs):
        rendered = _apply_rir(interferer, rir)
        placed = _place_interferer(rendered, length, config.overlap_ratio, rng)
        if sir_ratio is not None:
            interferer_rms = _rms(placed)
            if interferer_rms > 0.0:
                placed = placed * (target_rms / (sir_ratio * interferer_rms))
        interferer_components.append(placed)
    interferer_sum = (
        sum(interferer_components)
        if interferer_components
        else np.zeros(length, dtype=np.float64)
    )

    if noise is not None and config.snr_db is not None:
        noise_component = _crop_or_repeat(noise, length, rng)
        noise_rms = _rms(noise_component)
        signal_rms = _rms(target_component + interferer_sum)
        if noise_rms > 0.0 and signal_rms > 0.0:
            noise_component = noise_component * (
                signal_rms / (10.0 ** (config.snr_db / 20.0) * noise_rms)
            )
        else:
            noise_component = np.zeros(length, dtype=np.float64)
    else:
        noise_component = np.zeros(length, dtype=np.float64)

    base = interferer_sum + noise_component
    target_channel = _channel_response(target_component, config.channel_response)
    base_channel = _channel_response(base, config.channel_response)

    # Codec is applied per stream before the shared peak scale so the positive is
    # exactly the negative plus the rendered target reference. Both siblings
    # share the same codec, peak scale, and clip threshold and are never
    # independently normalized.
    clean_raw = _codec_simulate(target_channel, config.codec)
    negative_raw = _codec_simulate(base_channel, config.codec)
    positive_raw = negative_raw + clean_raw

    peak = max(
        float(np.max(np.abs(positive_raw))),
        float(np.max(np.abs(negative_raw))),
        float(np.max(np.abs(clean_raw))),
    ) if length else 0.0
    scale = (config.target_peak / peak) if peak > 0.0 else 1.0
    clean_target = _clip(clean_raw * scale, config.clip_threshold)
    negative = _clip(negative_raw * scale, config.clip_threshold)
    positive = _clip(negative + clean_target, config.clip_threshold)

    return positive, negative, clean_target


def _build_nuisance_dict(
    *,
    target_source_id,
    interferer_source_ids,
    noise_source_id,
    target_rir_id,
    interferer_rir_ids,
    renderer_family,
    config: RendererConfig,
) -> dict:
    return {
        "target_source_id": target_source_id,
        "interferer_source_ids": list(interferer_source_ids),
        "noise_source_id": noise_source_id,
        "target_rir_id": target_rir_id,
        "interferer_rir_ids": list(interferer_rir_ids),
        "renderer_family": renderer_family,
        "snr_db": config.snr_db,
        "sir_db": config.sir_db,
        "overlap_ratio": config.overlap_ratio,
        "codec": config.codec,
        "clip_threshold": config.clip_threshold,
        "channel_response": list(config.channel_response),
        "sample_rate": config.sample_rate,
    }


def compute_nuisance_fingerprint(nuisance: dict) -> str:
    """SHA-256 digest over canonical (sorted) JSON of the nuisance values."""
    payload = json.dumps(nuisance, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float64), sample_rate, subtype="DOUBLE")


def load_mono_16k(path: str | Path, sample_rate: int = 16_000) -> np.ndarray:
    """Read audio as mono float32 resampled to ``sample_rate`` Hz."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr <= 0 or audio.shape[0] == 0:
        raise ValueError(f"audio is empty or has an invalid sample rate: {path}")
    mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"audio contains non-finite samples: {path}")
    if sr != sample_rate:
        out_len = max(1, int(round(len(mono) * sample_rate / sr)))
        positions = np.arange(out_len, dtype=np.float64) * (sr / sample_rate)
        mono = np.interp(
            positions,
            np.arange(len(mono), dtype=np.float64),
            mono.astype(np.float64),
            left=float(mono[0]),
            right=float(mono[-1]),
        ).astype(np.float32)
    return mono


def render_counterfactual_pair(
    *,
    pair_id: str,
    split: str,
    config: RendererConfig,
    rng: np.random.Generator,
    enrollment: np.ndarray,
    target: np.ndarray,
    interferers: tuple[np.ndarray, ...],
    noise: np.ndarray | None,
    target_source_id: str,
    interferer_source_ids: tuple[str, ...],
    noise_source_id: str | None,
    target_rir: np.ndarray | None,
    target_rir_id: str | None,
    interferer_rirs: tuple[np.ndarray | None, ...],
    interferer_rir_ids: tuple[str, ...],
    renderer_family: str,
    enrollment_path: Path,
    mixture_paths: tuple[Path, Path],
    clean_target_paths: tuple[Path, Path],
) -> tuple[R3MixtureRow, R3MixtureRow]:
    """Render WAVs for a counterfactual pair and return the two manifest rows."""
    interferers = tuple(interferers)
    interferer_rirs = tuple(interferer_rirs)
    interferer_source_ids = tuple(interferer_source_ids)
    interferer_rir_ids = tuple(interferer_rir_ids)
    if not interferers:
        raise ValueError("at least one interferer is required")
    if len(interferer_rirs) != len(interferers):
        raise ValueError("interferer_rirs must align with interferers")
    if len(interferer_source_ids) != len(interferers):
        raise ValueError("interferer_source_ids must align with interferers")
    if len(interferer_rir_ids) not in (0, len(interferer_source_ids)):
        raise ValueError("interferer_rir_ids length must be 0 or match interferer_source_ids")

    positive_mixture, negative_mixture, clean_target = render_pair_audio(
        target=target,
        interferers=interferers,
        noise=noise,
        target_rir=target_rir,
        interferer_rirs=interferer_rirs,
        config=config,
        rng=rng,
    )
    _write_wav(enrollment_path, enrollment, config.sample_rate)
    _write_wav(mixture_paths[0], positive_mixture, config.sample_rate)
    _write_wav(mixture_paths[1], negative_mixture, config.sample_rate)
    _write_wav(clean_target_paths[0], clean_target, config.sample_rate)
    _write_wav(clean_target_paths[1], np.zeros_like(clean_target), config.sample_rate)

    nuisance = _build_nuisance_dict(
        target_source_id=target_source_id,
        interferer_source_ids=interferer_source_ids,
        noise_source_id=noise_source_id,
        target_rir_id=target_rir_id,
        interferer_rir_ids=interferer_rir_ids,
        renderer_family=renderer_family,
        config=config,
    )
    fingerprint = compute_nuisance_fingerprint(nuisance)

    common = dict(
        pair_id=pair_id,
        split=split,
        enrollment_audio=enrollment_path,
        target_source_id=target_source_id,
        interferer_source_ids=interferer_source_ids,
        noise_source_id=noise_source_id,
        target_rir_id=target_rir_id,
        interferer_rir_ids=interferer_rir_ids,
        renderer_family=renderer_family,
        snr_db=config.snr_db,
        sir_db=config.sir_db,
        overlap_ratio=config.overlap_ratio,
        codec=config.codec,
        clip_threshold=config.clip_threshold,
        nuisance_fingerprint=fingerprint,
    )
    positive = R3MixtureRow(
        row_id=f"{pair_id}-pos",
        target_present=True,
        mixture_audio=mixture_paths[0],
        clean_target_audio=clean_target_paths[0],
        **common,
    )
    negative = R3MixtureRow(
        row_id=f"{pair_id}-neg",
        target_present=False,
        mixture_audio=mixture_paths[1],
        clean_target_audio=clean_target_paths[1],
        **common,
    )
    return positive, negative
