"""Aggregate acoustic profiling core (Task 1).

Computes deterministic, aggregate-only acoustic statistics from audio files
using only NumPy and soundfile. There is no ASR, speaker-recognition, or label
input. Serialized profiles contain only counts, feature quantiles, histograms,
configuration, and a SHA-256 hash -- never input paths, filenames, IDs,
transcripts, labels, or per-file vectors.

The in-memory :class:`AudioStats` additionally retains an internal ``_raw`` bag
of per-file scalar measurements so that :func:`merge_profiles` can reproduce a
direct profile of the union exactly. ``_raw`` is never serialized; profiles read
back from JSON merge via their histograms instead.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

PROFILE_VERSION = "acoustic-profile-v1"
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
CLIP_THRESHOLD = 0.99
ROLLOFF_PERCENTILE = 0.85
_HIST_BINS = 256

# Fixed, domain-wide histogram ranges so any two profiles' bin counts sum
# exactly (merge by addition). Values outside a range are clipped into the
# extreme bin; exact quantiles are always computed from the raw per-file bag.
_METRIC_NAMES = (
    "duration",
    "rms",
    "peak",
    "clipping_rate",
    "active_speech_ratio",
    "silence_run",
    "spectral_centroid",
    "spectral_rolloff",
    "spectral_flatness",
)
_HIST_RANGES = {
    "duration": (0.0, 60.0),
    "rms": (0.0, 1.0),
    "peak": (0.0, 1.0),
    "clipping_rate": (0.0, 1.0),
    "active_speech_ratio": (0.0, 1.0),
    "silence_run": (0.0, 500.0),
    "spectral_centroid": (0.0, 8000.0),
    "spectral_rolloff": (0.0, 8000.0),
    "spectral_flatness": (0.0, 1.0),
}


@dataclass
class AudioStats:
    """Aggregate acoustic profile over one or more audio files."""

    config: dict
    file_count: int
    total_duration_seconds: float
    metrics: dict
    sample_rate_counts: dict
    channel_counts: dict
    stationary_noise_estimate: float
    hash: str = ""
    _raw: dict | None = field(default=None, repr=False, compare=False)

    def to_json_dict(self) -> dict:
        """Serializable aggregate payload (no ``_raw``, no paths/IDs)."""
        return {
            "config": self.config,
            "file_count": self.file_count,
            "total_duration_seconds": self.total_duration_seconds,
            "metrics": self.metrics,
            "sample_rate_counts": dict(self.sample_rate_counts),
            "channel_counts": dict(self.channel_counts),
            "stationary_noise_estimate": self.stationary_noise_estimate,
            "hash": self.hash,
        }


def _load_mono(path: Path, target_sr: int) -> tuple[np.ndarray, int, int]:
    """Read audio as mono float64 resampled to ``target_sr``; return (audio, source_sr, channels)."""
    audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    if sr <= 0 or audio.shape[0] == 0:
        raise ValueError(f"audio is empty or has an invalid sample rate: {path}")
    channels = int(audio.shape[1])
    mono = np.mean(audio, axis=1, dtype=np.float64)
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"audio contains non-finite samples: {path}")
    if sr != target_sr:
        out_len = max(1, int(round(len(mono) * target_sr / sr)))
        positions = np.arange(out_len, dtype=np.float64) * (sr / target_sr)
        mono = np.interp(
            positions,
            np.arange(len(mono), dtype=np.float64),
            mono,
            left=float(mono[0]),
            right=float(mono[-1]),
        )
    return mono, int(sr), channels


def _frame_audio(audio: np.ndarray, frame_samples: int, hop_samples: int) -> np.ndarray:
    """Return a (num_frames, frame_samples) float64 view with the given hop."""
    n = len(audio)
    if n == 0:
        return np.zeros((0, frame_samples), dtype=np.float64)
    if n < frame_samples:
        padded = np.zeros(frame_samples, dtype=np.float64)
        padded[:n] = audio
        return padded[None, :]
    windows = np.lib.stride_tricks.sliding_window_view(audio, frame_samples)
    return np.ascontiguousarray(windows[::hop_samples])


def _silence_run_lengths(active_mask: np.ndarray) -> list[int]:
    runs: list[int] = []
    count = 0
    for flag in active_mask:
        if not flag:
            count += 1
        elif count > 0:
            runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return runs


def _per_file_features(
    audio: np.ndarray, sr: int, frame_samples: int, hop_samples: int
) -> dict:
    n = len(audio)
    duration = n / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if n else 0.0
    peak = float(np.max(np.abs(audio))) if n else 0.0
    clipping_rate = float(np.mean(np.abs(audio) >= CLIP_THRESHOLD)) if n else 0.0

    frames = _frame_audio(audio, frame_samples, hop_samples)
    if frames.shape[0]:
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        nonzero = frame_rms[frame_rms > 0.0]
        if nonzero.size:
            threshold = max(1e-4, 0.1 * float(np.median(nonzero)))
        else:
            threshold = 1e-4
        active_mask = frame_rms > threshold
        active_ratio = float(np.mean(active_mask))
        silence_runs = _silence_run_lengths(active_mask)
        silence_median = float(np.median(silence_runs)) if silence_runs else 0.0
        stationary = float(np.percentile(frame_rms, 10))

        window = np.hanning(frame_samples)
        spec = np.abs(np.fft.rfft(frames * window, axis=1))
        freqs = np.fft.rfftfreq(frame_samples, d=1.0 / sr)
        eps = 1e-12
        power = spec + eps
        centroid = (freqs * power).sum(axis=1) / power.sum(axis=1)
        cum = np.cumsum(power, axis=1)
        total = power.sum(axis=1, keepdims=True)
        rolloff_idx = np.argmax(cum >= ROLLOFF_PERCENTILE * total, axis=1)
        rolloff = freqs[rolloff_idx]
        flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)

        spectral_centroid = float(np.median(centroid))
        spectral_rolloff = float(np.median(rolloff))
        spectral_flatness = float(np.median(flatness))
    else:
        active_ratio = 0.0
        silence_median = 0.0
        stationary = 0.0
        spectral_centroid = 0.0
        spectral_rolloff = 0.0
        spectral_flatness = 0.0

    return {
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "clipping_rate": clipping_rate,
        "active_speech_ratio": active_ratio,
        "silence_run": silence_median,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "spectral_flatness": spectral_flatness,
        "_stationary": stationary,
    }


def _quantile_map(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {f"{q:g}": 0.0 for q in QUANTILES}
    return {f"{q:g}": float(np.quantile(arr, q)) for q in QUANTILES}


def _histogram(values: Iterable[float], lo: float, hi: float) -> list[int]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return [0] * _HIST_BINS
    edges = np.linspace(lo, hi, _HIST_BINS + 1)
    clipped = np.clip(arr, lo, hi)
    counts, _ = np.histogram(clipped, bins=edges)
    return counts.astype(int).tolist()


def _metric_summary(values: list[float], lo: float, hi: float) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        mean = minv = maxv = 0.0
    else:
        mean = float(np.mean(arr))
        minv = float(np.min(arr))
        maxv = float(np.max(arr))
    return {
        "quantiles": _quantile_map(values),
        "mean": mean,
        "min": minv,
        "max": maxv,
        "histogram": _histogram(values, lo, hi),
    }


def _quantile_from_hist(counts: list[int], lo: float, hi: float, q: float) -> float:
    arr = np.asarray(counts, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0.0:
        return 0.0
    edges = np.linspace(lo, hi, len(arr) + 1)
    cum = np.cumsum(arr)
    target = q * total
    idx = int(np.searchsorted(cum, target))
    idx = min(idx, len(arr) - 1)
    prev = float(cum[idx - 1]) if idx > 0 else 0.0
    width = float(edges[idx + 1] - edges[idx])
    frac = (target - prev) / float(arr[idx]) if arr[idx] > 0 else 0.0
    return float(edges[idx] + frac * width)


def _compute_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _config(sample_rate: int, frame_ms: int, hop_ms: int) -> dict:
    frame_samples = int(round(sample_rate * frame_ms / 1000.0))
    hop_samples = int(round(sample_rate * hop_ms / 1000.0))
    return {
        "version": PROFILE_VERSION,
        "sample_rate": int(sample_rate),
        "frame_ms": int(frame_ms),
        "hop_ms": int(hop_ms),
        "frame_samples": frame_samples,
        "hop_samples": hop_samples,
        "quantiles": [float(q) for q in QUANTILES],
        "clip_threshold": CLIP_THRESHOLD,
        "rolloff_percentile": ROLLOFF_PERCENTILE,
        "histogram_bins": _HIST_BINS,
        "histogram_ranges": {name: list(r) for name, r in _HIST_RANGES.items()},
    }


def _build_profile(
    config: dict,
    raw: dict,
    sample_rate_counts: dict,
    channel_counts: dict,
    stationary_values: list[float],
) -> AudioStats:
    metrics = {
        name: _metric_summary(raw.get(name, []), *_HIST_RANGES[name]) for name in _METRIC_NAMES
    }
    file_count = len(raw.get("duration", []))
    total_duration = float(sum(raw.get("duration", [])))
    stationary = float(np.mean(stationary_values)) if stationary_values else 0.0
    payload = {
        "config": config,
        "file_count": file_count,
        "total_duration_seconds": total_duration,
        "metrics": metrics,
        "sample_rate_counts": sample_rate_counts,
        "channel_counts": channel_counts,
        "stationary_noise_estimate": stationary,
    }
    return AudioStats(
        config=config,
        file_count=file_count,
        total_duration_seconds=total_duration,
        metrics=metrics,
        sample_rate_counts=sample_rate_counts,
        channel_counts=channel_counts,
        stationary_noise_estimate=stationary,
        hash=_compute_hash(payload),
        _raw={name: list(raw.get(name, [])) for name in _METRIC_NAMES},
    )


def profile_audio_paths(
    paths: Iterable[str | Path],
    sample_rate: int = 16_000,
    frame_ms: int = 32,
    hop_ms: int = 16,
) -> AudioStats:
    """Profile a sequence of audio files into an aggregate :class:`AudioStats`."""
    config = _config(sample_rate, frame_ms, hop_ms)
    frame_samples = config["frame_samples"]
    hop_samples = config["hop_samples"]
    raw: dict[str, list[float]] = {name: [] for name in _METRIC_NAMES}
    stationary_values: list[float] = []
    sample_rate_counts: dict[str, int] = {}
    channel_counts: dict[str, int] = {}

    for entry in paths:
        path = Path(entry)
        audio, source_sr, channels = _load_mono(path, sample_rate)
        feats = _per_file_features(audio, sample_rate, frame_samples, hop_samples)
        stationary_values.append(feats.pop("_stationary"))
        for name in _METRIC_NAMES:
            raw[name].append(feats[name])
        sr_key = str(source_sr)
        ch_key = str(channels)
        sample_rate_counts[sr_key] = sample_rate_counts.get(sr_key, 0) + 1
        channel_counts[ch_key] = channel_counts.get(ch_key, 0) + 1

    return _build_profile(config, raw, sample_rate_counts, channel_counts, stationary_values)


def _merge_counts(count_dicts: list[dict]) -> dict:
    merged: dict[str, int] = {}
    for counts in count_dicts:
        for key, value in counts.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def _build_from_histograms(profiles: list[AudioStats], config: dict) -> AudioStats:
    metrics: dict[str, dict] = {}
    for name in _METRIC_NAMES:
        lo, hi = _HIST_RANGES[name]
        merged_counts = np.zeros(_HIST_BINS, dtype=np.int64)
        mean_sum = 0.0
        weight = 0
        min_val = math.inf
        max_val = -math.inf
        for profile in profiles:
            summary = profile.metrics[name]
            counts = summary["histogram"]
            keep = min(len(counts), _HIST_BINS)
            merged_counts[:keep] += counts[:keep]
            w = profile.file_count
            if w:
                mean_sum += summary["mean"] * w
                weight += w
                min_val = min(min_val, summary["min"])
                max_val = max(max_val, summary["max"])
        metrics[name] = {
            "quantiles": {
                f"{q:g}": _quantile_from_hist(merged_counts, lo, hi, q) for q in QUANTILES
            },
            "mean": float(mean_sum / weight) if weight else 0.0,
            "min": float(min_val) if math.isfinite(min_val) else 0.0,
            "max": float(max_val) if math.isfinite(max_val) else 0.0,
            "histogram": merged_counts.astype(int).tolist(),
        }
    file_count = sum(p.file_count for p in profiles)
    total_duration = float(sum(p.total_duration_seconds for p in profiles))
    total_files = file_count
    stationary = (
        sum(p.stationary_noise_estimate * p.file_count for p in profiles) / total_files
        if total_files
        else 0.0
    )
    payload = {
        "config": config,
        "file_count": file_count,
        "total_duration_seconds": total_duration,
        "metrics": metrics,
        "sample_rate_counts": _merge_counts([p.sample_rate_counts for p in profiles]),
        "channel_counts": _merge_counts([p.channel_counts for p in profiles]),
        "stationary_noise_estimate": stationary,
    }
    return AudioStats(
        config=config,
        file_count=file_count,
        total_duration_seconds=total_duration,
        metrics=metrics,
        sample_rate_counts=payload["sample_rate_counts"],
        channel_counts=payload["channel_counts"],
        stationary_noise_estimate=stationary,
        hash=_compute_hash(payload),
        _raw=None,
    )


def profile_audio_paths_safe(
    paths: Iterable[str | Path],
    sample_rate: int = 16_000,
    frame_ms: int = 32,
    hop_ms: int = 16,
) -> tuple[AudioStats, int]:
    """Like :func:`profile_audio_paths` but skip unreadable files.

    Returns ``(profile, unreadable_count)``. Unreadable files are counted and
    omitted; no path or per-file detail is retained.
    """
    config = _config(sample_rate, frame_ms, hop_ms)
    frame_samples = config["frame_samples"]
    hop_samples = config["hop_samples"]
    raw: dict[str, list[float]] = {name: [] for name in _METRIC_NAMES}
    stationary_values: list[float] = []
    sample_rate_counts: dict[str, int] = {}
    channel_counts: dict[str, int] = {}
    unreadable = 0
    for entry in paths:
        path = Path(entry)
        try:
            audio, source_sr, channels = _load_mono(path, sample_rate)
        except Exception:
            unreadable += 1
            continue
        feats = _per_file_features(audio, sample_rate, frame_samples, hop_samples)
        stationary_values.append(feats.pop("_stationary"))
        for name in _METRIC_NAMES:
            raw[name].append(feats[name])
        sr_key = str(source_sr)
        ch_key = str(channels)
        sample_rate_counts[sr_key] = sample_rate_counts.get(sr_key, 0) + 1
        channel_counts[ch_key] = channel_counts.get(ch_key, 0) + 1
    stats = _build_profile(config, raw, sample_rate_counts, channel_counts, stationary_values)
    return stats, unreadable


def merge_profiles(profiles: Iterable[AudioStats]) -> AudioStats:
    """Merge several aggregate profiles into one.

    When every input retains its internal raw per-file bag (the in-memory case),
    the merge reproduces a direct profile of the union exactly, including the
    hash. Profiles deserialized from JSON fall back to histogram merging.
    """
    materialized = [p for p in profiles if p is not None]
    if not materialized:
        raise ValueError("merge_profiles requires at least one profile")
    config = materialized[0].config
    for profile in materialized:
        if profile.config.get("version") != config.get("version"):
            raise ValueError("cannot merge profiles with different versions")

    sample_rate_counts = _merge_counts([p.sample_rate_counts for p in materialized])
    channel_counts = _merge_counts([p.channel_counts for p in materialized])
    total_files = sum(p.file_count for p in materialized)
    stationary = (
        sum(p.stationary_noise_estimate * p.file_count for p in materialized) / total_files
        if total_files
        else 0.0
    )

    if all(p._raw is not None for p in materialized):
        raw: dict[str, list[float]] = {name: [] for name in _METRIC_NAMES}
        for profile in materialized:
            for name in _METRIC_NAMES:
                raw[name].extend(profile._raw.get(name, []))
        return _build_profile(config, raw, sample_rate_counts, channel_counts, [stationary])

    return _build_from_histograms(materialized, config)


def write_aggregate_profile(path: str | Path, profile: AudioStats) -> None:
    """Write an aggregate profile JSON (aggregate-only, no raw/paths)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = profile.to_json_dict()
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


_ALLOWED_TOP_KEYS = {
    "config",
    "file_count",
    "total_duration_seconds",
    "metrics",
    "sample_rate_counts",
    "channel_counts",
    "stationary_noise_estimate",
    "hash",
    "profiling",
}
_ALLOWED_METRIC_KEYS = {"quantiles", "mean", "min", "max", "histogram"}
_PROHIBITED_KEYS = {
    "paths", "files", "file_paths", "ids", "transcript",
    "label", "labels", "text", "prediction", "predictions",
}


def assert_aggregate_profile(payload: object) -> None:
    """Raise if a profile payload carries non-aggregate or prohibited content.

    Guards against Dataset-A paths, IDs, labels, transcripts, predictions, or
    per-file/sample-level value lists sneaking into a profile used to guide
    public generation.
    """
    if not isinstance(payload, dict):
        raise ValueError("acoustic profile must be a JSON object")
    extra = sorted(set(payload) - _ALLOWED_TOP_KEYS)
    if extra:
        raise ValueError(f"acoustic profile contains non-aggregate keys: {extra}")
    for key in _PROHIBITED_KEYS:
        if key in payload:
            raise ValueError(f"acoustic profile contains prohibited key: {key!r}")
    for sub in ("config", "profiling"):
        nested = payload.get(sub)
        if isinstance(nested, dict):
            for key in _PROHIBITED_KEYS:
                if key in nested:
                    raise ValueError(f"acoustic profile {sub!r} contains prohibited key: {key!r}")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("acoustic profile metrics must be an object")
    for name, summary in metrics.items():
        if not isinstance(summary, dict):
            raise ValueError(f"acoustic profile metric {name!r} must be an object")
        extra_metric = sorted(set(summary) - _ALLOWED_METRIC_KEYS)
        if extra_metric:
            raise ValueError(
                f"acoustic profile metric {name!r} has non-aggregate keys: {extra_metric}"
            )


def compare_profiles(reference: AudioStats, candidate: AudioStats) -> dict:
    """Per-metric normalized quantile differences between two aggregate profiles.

    Emits only aggregate comparison numbers (no paths, labels, texts, or bytes).
    ``norm_diff = (candidate - reference) / (|reference| + eps)`` so a value near
    zero means the candidate matches the reference quantile closely.
    """
    metrics: dict[str, dict] = {}
    for name in _METRIC_NAMES:
        ref_q = reference.metrics[name]["quantiles"]
        cand_q = candidate.metrics[name]["quantiles"]
        per_q: dict[str, dict] = {}
        for q in QUANTILES:
            key = f"{q:g}"
            r = float(ref_q.get(key, 0.0))
            c = float(cand_q.get(key, 0.0))
            denom = abs(r) + 1e-8
            per_q[key] = {
                "reference": r,
                "candidate": c,
                "abs_diff": c - r,
                "norm_diff": (c - r) / denom,
            }
        metrics[name] = per_q
    return {
        "reference_hash": reference.hash,
        "candidate_hash": candidate.hash,
        "reference_file_count": reference.file_count,
        "candidate_file_count": candidate.file_count,
        "metrics": metrics,
    }


def read_aggregate_profile(path: str | Path) -> AudioStats:
    """Read an aggregate profile JSON written by :func:`write_aggregate_profile`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return AudioStats(
        config=payload["config"],
        file_count=int(payload["file_count"]),
        total_duration_seconds=float(payload["total_duration_seconds"]),
        metrics=payload["metrics"],
        sample_rate_counts=payload["sample_rate_counts"],
        channel_counts=payload["channel_counts"],
        stationary_noise_estimate=float(payload["stationary_noise_estimate"]),
        hash=payload["hash"],
        _raw=None,
    )
