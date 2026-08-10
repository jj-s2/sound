"""Offline, label-free FireRedChat pVAD feature extraction."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import soundfile
from scipy.signal import resample_poly

from .firered_model_assets import FireRedModelPaths


_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
_EMA_RUN_THRESHOLDS = (0.3, 0.5, 0.7)
_INACTIVE_CROSSING_FRAME = -1
_INACTIVE_ACTIVE_SPAN_FRAMES = 0


def _threshold_name(threshold: float) -> str:
    return f"{threshold:.1f}".replace(".", "_")


def _stat_names(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_min",
        f"{prefix}_max",
        *(f"{prefix}_q{int(quantile * 100):02d}" for quantile in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)),
        *(f"{prefix}_fraction_ge_{_threshold_name(threshold)}" for threshold in _THRESHOLDS),
    )


PVAD_GATE_FEATURE_SCHEMA: tuple[str, ...] = (
    "frame_count",
    "analyzed_duration_sec",
    "dropped_tail_samples",
    "command_duration_sec",
    *_stat_names("raw"),
    *_stat_names("ema"),
    *(
        name
        for threshold in _EMA_RUN_THRESHOLDS
        for name in (
            f"ema_longest_run_ge_{_threshold_name(threshold)}_frames",
            f"ema_longest_run_ge_{_threshold_name(threshold)}_seconds",
            f"ema_first_crossing_ge_{_threshold_name(threshold)}_frame",
            f"ema_last_crossing_ge_{_threshold_name(threshold)}_frame",
            f"ema_active_span_ge_{_threshold_name(threshold)}_frames",
            f"ema_transitions_ge_{_threshold_name(threshold)}",
        )
    ),
    "enrollment_duration_sec",
    "embedding_norm_before",
    "embedding_norm_after",
)


@dataclass(frozen=True)
class PvadRuntimeConfig:
    sample_rate: int = 16000
    frame_samples: int = 160
    enrollment_cap_seconds: float = 5.0
    minimum_audio_seconds: float = 0.25
    ema_alpha: float = 0.8
    onnx_provider: str = "CPUExecutionProvider"
    ecapa_device: str = "cpu"


@dataclass(frozen=True)
class PvadUtteranceFeatures:
    sample_id: str
    values: Mapping[str, float | int]
    audit: Mapping[str, float | int | str]


def _default_ecapa_encoder(paths: FireRedModelPaths, config: PvadRuntimeConfig) -> object:
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        raise RuntimeError("speechbrain is required for FireRed ECAPA inference") from exc
    return EncoderClassifier.from_hparams(
        source=str(paths.ecapa_root), savedir=str(paths.ecapa_root), run_opts={"device": config.ecapa_device}
    )


def _default_onnx_session(paths: FireRedModelPaths, config: PvadRuntimeConfig) -> object:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for FireRed pVAD inference") from exc
    return ort.InferenceSession(str(paths.pvad_onnx), providers=[config.onnx_provider])


def _default_rss_bytes() -> int:
    try:
        import psutil
    except ImportError:
        return 0
    return int(psutil.Process().memory_info().rss)


class FireRedPvadRuntime:
    """Run one ECAPA enrollment and one fresh recurrent pVAD stream per command."""

    def __init__(
        self,
        model_paths: FireRedModelPaths,
        *,
        config: PvadRuntimeConfig = PvadRuntimeConfig(),
        ecapa_encoder: object | None = None,
        onnx_session: object | None = None,
        clock: Callable[[], float] = time.perf_counter,
        rss_bytes: Callable[[], int] = _default_rss_bytes,
        cuda_peak_bytes: Callable[[], int] | None = None,
    ) -> None:
        if config.sample_rate <= 0 or config.frame_samples <= 0:
            raise ValueError("sample rate and frame samples must be positive")
        if config.enrollment_cap_seconds <= 0 or config.minimum_audio_seconds <= 0:
            raise ValueError("audio duration limits must be positive")
        if not 0.0 <= config.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in [0, 1]")
        self.model_paths = model_paths
        self.config = config
        self._ecapa_encoder = ecapa_encoder
        self._onnx_session = onnx_session
        self._clock = clock
        self._rss_bytes = rss_bytes
        self._cuda_peak_bytes = cuda_peak_bytes

    def _encoder(self) -> object:
        if self._ecapa_encoder is None:
            self._ecapa_encoder = _default_ecapa_encoder(self.model_paths, self.config)
        return self._ecapa_encoder

    def _session(self) -> object:
        if self._onnx_session is None:
            self._onnx_session = _default_onnx_session(self.model_paths, self.config)
        return self._onnx_session

    def _read_audio(self, path: Path, kind: str) -> np.ndarray:
        try:
            samples, sample_rate = soundfile.read(path, always_2d=True)
        except Exception as exc:
            raise ValueError(f"could not decode {kind} audio {path}: {exc}") from exc
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            raise ValueError(f"{kind} audio has an invalid sample rate")
        array = np.asarray(samples, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(f"{kind} audio must be nonempty two-dimensional samples")
        if not np.isfinite(array).all():
            raise ValueError(f"{kind} audio contains non-finite samples")
        mono = array.mean(axis=1, dtype=np.float64)
        divisor = math.gcd(self.config.sample_rate, sample_rate)
        if sample_rate != self.config.sample_rate:
            mono = resample_poly(mono, self.config.sample_rate // divisor, sample_rate // divisor)
        mono = np.clip(mono, -1.0, 1.0)
        result = np.ascontiguousarray(mono, dtype=np.float32)
        if len(result) < math.ceil(self.config.minimum_audio_seconds * self.config.sample_rate):
            raise ValueError(f"{kind} audio is shorter than the configured minimum duration")
        return result

    @staticmethod
    def _as_numpy(value: object) -> np.ndarray:
        candidate = value
        if hasattr(candidate, "detach"):
            candidate = candidate.detach()
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "numpy"):
            candidate = candidate.numpy()
        return np.asarray(candidate)

    def _embedding(self, enrollment: np.ndarray) -> tuple[np.ndarray, float, float]:
        encoder = self._encoder()
        try:
            if hasattr(encoder, "encode_batch"):
                try:
                    import torch
                except ImportError as exc:
                    raise RuntimeError("torch is required for SpeechBrain ECAPA inference") from exc
                encoded = encoder.encode_batch(torch.from_numpy(enrollment).unsqueeze(0))
            elif callable(encoder):
                encoded = encoder(enrollment)
            else:
                raise TypeError("ECAPA encoder must be callable or expose encode_batch")
        except Exception as exc:
            raise ValueError(f"ECAPA enrollment extraction failed: {exc}") from exc
        vector = self._as_numpy(encoded)
        vector = np.squeeze(vector)
        if vector.shape != (192,):
            raise ValueError(f"ECAPA embedding must contain exactly 192 values, got shape {vector.shape}")
        vector = np.asarray(vector, dtype=np.float64)
        if not np.isfinite(vector).all():
            raise ValueError("ECAPA embedding must be finite")
        before = float(np.linalg.norm(vector))
        if not math.isfinite(before) or before == 0.0:
            raise ValueError("ECAPA embedding must have a finite nonzero L2 norm")
        normalized = np.ascontiguousarray((vector / before).reshape(1, 192), dtype=np.float32)
        after = float(np.linalg.norm(normalized.astype(np.float64)))
        if not math.isfinite(after) or after == 0.0:
            raise ValueError("normalized ECAPA embedding must have a finite nonzero L2 norm")
        return normalized, before, after

    @staticmethod
    def _state(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != shape or array.dtype != np.dtype(np.float32) or not np.isfinite(array).all():
            raise ValueError(f"{label} state must be finite float32 with shape {shape}")
        return np.ascontiguousarray(array)

    def _probabilities(self, command: np.ndarray, embedding: np.ndarray) -> tuple[np.ndarray, int]:
        usable = len(command) - (len(command) % self.config.frame_samples)
        tail = len(command) - usable
        if usable == 0:
            raise ValueError("command audio has no complete 160-sample frame")
        mel = np.zeros((1, 80, 15), dtype=np.float32)
        gru = np.zeros((2, 1, 256), dtype=np.float32)
        probabilities: list[float] = []
        session = self._session()
        for start in range(0, usable, self.config.frame_samples):
            frame = np.ascontiguousarray(command[start : start + self.config.frame_samples].reshape(1, -1), dtype=np.float32)
            outputs = session.run(None, {"input_audio": frame, "spkemb": embedding, "mel_buffer": mel, "gru_buffer": gru})
            if not isinstance(outputs, (list, tuple)) or len(outputs) < 4:
                raise ValueError("ONNX pVAD inference must return at least four outputs")
            probability_array = np.asarray(outputs[1])
            if probability_array.size != 1:
                raise ValueError("ONNX pVAD probability output must be scalar")
            probability = float(probability_array.reshape(-1)[0])
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("ONNX pVAD probability must be finite and in [0, 1]")
            mel = self._state(outputs[2], (1, 80, 15), "mel")
            gru = self._state(outputs[3], (2, 1, 256), "GRU")
            probabilities.append(probability)
        return np.asarray(probabilities, dtype=np.float64), tail

    def extract(self, sample_id: str, wake_path: Path, command_path: Path) -> PvadUtteranceFeatures:
        """Extract fixed pVAD aggregates without accessing labels or text metadata."""

        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample_id must be a nonempty string")
        started = float(self._clock())
        rss_before = int(self._rss_bytes())
        wake = self._read_audio(Path(wake_path), "wake")
        command = self._read_audio(Path(command_path), "command")
        cap = int(self.config.enrollment_cap_seconds * self.config.sample_rate)
        enrollment = wake[:cap]
        embedding, norm_before, norm_after = self._embedding(enrollment)
        raw, tail = self._probabilities(command, embedding)
        ema = np.empty_like(raw)
        ema[0] = raw[0]
        for index in range(1, len(raw)):
            ema[index] = self.config.ema_alpha * ema[index - 1] + (1.0 - self.config.ema_alpha) * raw[index]
        values = self._aggregate(raw, ema, len(command), tail, len(enrollment), norm_before, norm_after)
        elapsed = float(self._clock()) - started
        audio_seconds = (len(wake) + len(command)) / self.config.sample_rate
        audit: dict[str, float | int | str] = {
            "elapsed_seconds": elapsed,
            "audio_seconds": audio_seconds,
            "rtf": elapsed / audio_seconds if audio_seconds else 0.0,
            "rss_delta_bytes": int(self._rss_bytes()) - rss_before,
            "dropped_tail_samples": tail,
            "onnx_provider": self.config.onnx_provider,
            "ecapa_device": self.config.ecapa_device,
        }
        if self._cuda_peak_bytes is not None:
            audit["cuda_peak_bytes"] = int(self._cuda_peak_bytes())
        return PvadUtteranceFeatures(sample_id, values, audit)

    def _aggregate(self, raw: np.ndarray, ema: np.ndarray, command_samples: int, tail: int, enrollment_samples: int, norm_before: float, norm_after: float) -> Mapping[str, float | int]:
        frame_seconds = self.config.frame_samples / self.config.sample_rate
        values: OrderedDict[str, float | int] = OrderedDict()
        values["frame_count"] = int(len(raw))
        values["analyzed_duration_sec"] = float(len(raw) * frame_seconds)
        values["dropped_tail_samples"] = int(tail)
        values["command_duration_sec"] = float(command_samples / self.config.sample_rate)
        for name, array in (("raw", raw), ("ema", ema)):
            values[f"{name}_mean"] = float(array.mean())
            values[f"{name}_std"] = float(array.std())
            values[f"{name}_min"] = float(array.min())
            values[f"{name}_max"] = float(array.max())
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
                values[f"{name}_q{int(quantile * 100):02d}"] = float(np.quantile(array, quantile))
            for threshold in _THRESHOLDS:
                values[f"{name}_fraction_ge_{_threshold_name(threshold)}"] = float(np.mean(array >= threshold))
        for threshold in _EMA_RUN_THRESHOLDS:
            active = ema >= threshold
            indices = np.flatnonzero(active)
            longest = current = 0
            transitions = 0
            previous = False
            for flag in active:
                current = current + 1 if flag else 0
                longest = max(longest, current)
                if bool(flag) != previous:
                    transitions += 1
                previous = bool(flag)
            if indices.size:
                first, last = int(indices[0]) + 1, int(indices[-1]) + 1
                span = last - first + 1
            else:
                first = last = _INACTIVE_CROSSING_FRAME
                span = _INACTIVE_ACTIVE_SPAN_FRAMES
            threshold_name = _threshold_name(threshold)
            values[f"ema_longest_run_ge_{threshold_name}_frames"] = int(longest)
            values[f"ema_longest_run_ge_{threshold_name}_seconds"] = float(longest * frame_seconds)
            values[f"ema_first_crossing_ge_{threshold_name}_frame"] = first
            values[f"ema_last_crossing_ge_{threshold_name}_frame"] = last
            values[f"ema_active_span_ge_{threshold_name}_frames"] = span
            values[f"ema_transitions_ge_{threshold_name}"] = transitions
        values["enrollment_duration_sec"] = float(enrollment_samples / self.config.sample_rate)
        values["embedding_norm_before"] = norm_before
        values["embedding_norm_after"] = norm_after
        if tuple(values) != PVAD_GATE_FEATURE_SCHEMA:
            raise RuntimeError("pVAD aggregate schema implementation disagrees with its frozen schema")
        return values
