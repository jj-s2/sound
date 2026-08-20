"""Lightweight WAV metadata and simple quality features."""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    frames: int
    duration_sec: float
    rms: float
    valid: bool


def read_wav_info(path: str | Path) -> AudioInfo:
    path = Path(path)
    if not path.exists():
        return AudioInfo(0, 0, 0, 0.0, 0.0, False)
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            sr = wf.getframerate()
            frames = wf.getnframes()
            raw = wf.readframes(min(frames, sr * 10))
            rms = _rms(raw, width) if raw else 0.0
            duration = frames / sr if sr else 0.0
            return AudioInfo(sr, channels, frames, duration, rms, True)
    except (wave.Error, OSError, EOFError):
        return AudioInfo(0, 0, 0, 0.0, 0.0, False)


def _rms(raw: bytes, sample_width: int) -> float:
    if sample_width == 1:
        values = [b - 128 for b in raw]
    elif sample_width == 2:
        n = len(raw) // 2
        values = struct.unpack("<" + "h" * n, raw[: n * 2])
    elif sample_width == 4:
        n = len(raw) // 4
        values = struct.unpack("<" + "i" * n, raw[: n * 4])
    else:
        return 0.0
    if not values:
        return 0.0
    return math.sqrt(sum(float(v) * float(v) for v in values) / len(values))
