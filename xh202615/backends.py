"""Replaceable ASR, speaker-score and separator backends.

The default implementations are safe scaffolding, not competitive models.
Attach FunASR, WeSpeaker and TSE/SepReformer by implementing these interfaces.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .contracts import BackendMetadata
from .data import read_jsonl #.data说明从同一个包里导入


@dataclass(frozen=True)
class AsrResult:
    text: str
    confidence: float | None  #asr结果置信度
    latency_ms: float
    backend: str
    metadata: BackendMetadata | None = None
    error: str | None = None


@dataclass(frozen=True)
class SpeakerScores:
    target_probability: float | None = None
    global_similarity: float | None = None
    topk_similarity: float | None = None
    target_frame_ratio: float | None = None
    noise_score: float | None = None
    overlap_probability: float | None = None
    audio_quality: float | None = None
    backend: str = "none"


class NoopAsrBackend:
    """ASR placeholder that returns empty text.

    Use only for smoke tests. For real V0, replace this with FunASR output or
    pass an external prediction map.
    """

    name = "noop"

    def transcribe(self, sample) -> AsrResult:
        start = time.perf_counter()
        return AsrResult("", None, (time.perf_counter() - start) * 1000, self.name)


class TranscriptMapAsrBackend:
    """Read precomputed ASR predictions from JSONL/CSV by sample id."""

    name = "external_map"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.predictions = self._load(self.path)

    def _load(self, path: Path) -> dict[str, str]:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                return {
                    str(row.get("id", row.get("sample_id", ""))): row.get("text", row.get("recognition_text", ""))
                    for row in reader
                }
        values = {}
        for row in read_jsonl(path):
            sample_id = str(row.get("id", row.get("sample_id", "")))
            text = row.get("text", row.get("recognition_text", row.get("识别文本", "")))
            values[sample_id] = "" if text is None else str(text)
        return values

    def transcribe(self, sample) -> AsrResult:
        start = time.perf_counter()
        text = self.predictions.get(str(sample.id), "")
        return AsrResult(text, None, (time.perf_counter() - start) * 1000, self.name)


class ScoreCsvSpeakerBackend:
    """Read speaker and difficulty scores from CSV by sample id."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.rows = self._load(self.path) if self.path else {}

    def _load(self, path: Path) -> dict[str, dict]:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return {str(row["id"]): row for row in csv.DictReader(f)}

    @staticmethod
    def _float(row: dict, key: str) -> float | None:
        val = row.get(key)
        if val in (None, ""):
            return None
        try:
            return float(val)
        except ValueError:
            return None

    def score(self, sample) -> SpeakerScores:
        row = self.rows.get(str(sample.id), {})
        return SpeakerScores(
            target_probability=self._float(row, "target_probability"),
            global_similarity=self._float(row, "global_similarity"),
            topk_similarity=self._float(row, "topk_similarity"),
            target_frame_ratio=self._float(row, "target_frame_ratio"),
            noise_score=self._float(row, "noise_score"),
            overlap_probability=self._float(row, "overlap_probability"),
            audio_quality=self._float(row, "audio_quality"),
            backend="score_csv" if row else "none",
        )


def make_asr_backend(asr_map: str | Path | None):
    return TranscriptMapAsrBackend(asr_map) if asr_map else NoopAsrBackend()

