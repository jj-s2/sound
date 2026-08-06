"""Utilities for SNR-based robustness evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _require_soundfile():
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "soundfile is required for audio file I/O. Install soundfile or use the in-memory SNR helpers."
        ) from exc
    return sf


def _require_resample_poly():
    try:
        from scipy.signal import resample_poly
    except ModuleNotFoundError as exc:
        raise RuntimeError("scipy is required for sample-rate conversion.") from exc
    return resample_poly


def read_mono_audio(path: str | Path) -> tuple[np.ndarray, int]:
    sf = _require_soundfile()
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def write_mono_audio(path: str | Path, audio: np.ndarray, sr: int) -> None:
    sf = _require_soundfile()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    sf.write(str(path), audio, sr)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def ensure_sample_rate(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(audio, dtype=np.float32)
    resample_poly = _require_resample_poly()
    gcd = math.gcd(src_sr, dst_sr)
    up = dst_sr // gcd
    down = src_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def fit_noise_to_length(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if noise.size == 0:
        return rng.standard_normal(length).astype(np.float32)
    if noise.size >= length:
        start = int(rng.integers(0, noise.size - length + 1))
        return noise[start : start + length].astype(np.float32)
    repeat = int(math.ceil(length / noise.size))
    tiled = np.tile(noise, repeat)
    return tiled[:length].astype(np.float32)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Mix clean speech and noise at a target SNR in dB."""

    clean = np.asarray(clean, dtype=np.float32)
    noise = fit_noise_to_length(np.asarray(noise, dtype=np.float32), clean.size, rng)
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if clean.size == 0:
        return clean
    if clean_rms <= 1e-8:
        return clean
    if noise_rms <= 1e-8:
        noise = rng.standard_normal(clean.size).astype(np.float32)
        noise_rms = rms(noise)

    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / max(noise_rms, 1e-8))
    mixed = clean + scaled_noise
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    return mixed.astype(np.float32)


def estimate_snr_db(clean: np.ndarray, mixed: np.ndarray) -> float:
    """Estimate SNR when the clean signal is known."""

    clean = np.asarray(clean, dtype=np.float32)
    mixed = np.asarray(mixed, dtype=np.float32)
    length = min(clean.size, mixed.size)
    if length == 0:
        return 0.0
    clean = clean[:length]
    noise = mixed[:length] - clean
    return 20.0 * math.log10(max(rms(clean), 1e-8) / max(rms(noise), 1e-8))
