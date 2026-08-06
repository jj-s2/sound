"""Replay adapters for frozen ASR and speaker-score artifacts."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Mapping

from .backends import AsrResult
from .contracts import (
    BackendMetadata,
    EvidenceWindow,
    TemporalSpeakerEvidence,
)
from .data import Sample, read_jsonl


def _insert_unique(rows: dict[str, Any], sample_id: str, source: Path) -> None:
    if sample_id in rows:
        raise ValueError(f"duplicate id '{sample_id}' in {source}")


def _row_id(row: Mapping[str, object], source: Path) -> str:
    value = row.get("id", row.get("sample_id", row.get("utt_id")))
    if value is None or value == "":
        raise ValueError(f"missing id in {source}")
    return str(value)


def _float_or_none(row: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric field '{key}' value {value!r}") from exc
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _source_paths(sample: Sample) -> tuple[str, str]:
    return str(sample.wakeup_audio), str(sample.command_audio)


def _replay_metadata(name: str, model_id: str = "unknown") -> BackendMetadata:
    return BackendMetadata(name=name, model_id=model_id, version=None, replay=True)


class TranscriptReplayBackend:
    """Replay ASR text from an existing JSONL or CSV prediction artifact."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata = _replay_metadata("transcript-replay")
        self.predictions: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.suffix.lower() == ".csv":
            rows = _read_csv(self.path)
        else:
            rows = list(read_jsonl(self.path))
        for row in rows:
            sample_id = _row_id(row, self.path)
            _insert_unique(self.predictions, sample_id, self.path)
            text = row.get(
                "text",
                row.get("recognition_text", row.get("识别文本", "")),
            )
            self.predictions[sample_id] = "" if text is None else str(text)

    def load(self) -> None:
        return None

    def transcribe(self, sample: Sample) -> AsrResult:
        start = time.perf_counter()
        sample_id = str(sample.id)
        text = self.predictions.get(sample_id, "")
        error = None if sample_id in self.predictions else "missing_prediction"
        elapsed_ms = (time.perf_counter() - start) * 1000
        return AsrResult(
            text,
            None,
            elapsed_ms,
            self.metadata.name,
            self.metadata,
            error,
        )


class GlobalScoreReplayBackend:
    """Replay global speaker scores as explicitly degenerate windows.

    The one window represents the global artifact only.  It is not a claim that
    the source scorer performed temporal segmentation.
    """

    replay_window_note = "global score represented as replay-only degenerate window"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata = _replay_metadata("wespeaker-replay")
        self.rows: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        for row in _read_csv(self.path):
            sample_id = _row_id(row, self.path)
            _insert_unique(self.rows, sample_id, self.path)
            self.rows[sample_id] = row

    def load(self) -> None:
        return None

    def score(self, sample: Sample) -> TemporalSpeakerEvidence:
        sample_id = str(sample.id)
        row = self.rows.get(sample_id)
        enrollment_source, command_source = _source_paths(sample)
        if row is None:
            return TemporalSpeakerEvidence(
                id=sample_id,
                backend=self.metadata,
                enrollment_source=enrollment_source,
                command_source=command_source,
                error="missing_evidence",
            )

        global_similarity = _float_or_none(row, "global_similarity", "similarity")
        duration = _float_or_none(
            row,
            "duration",
            "duration_sec",
            "command_duration_sec",
        )
        end_sec = duration if duration is not None and duration > 0 else 1.0
        windows = ()
        if global_similarity is not None:
            windows = (EvidenceWindow(0.0, end_sec, global_similarity),)

        return TemporalSpeakerEvidence(
            id=sample_id,
            backend=self.metadata,
            enrollment_source=enrollment_source,
            command_source=command_source,
            windows=windows,
            global_similarity=global_similarity,
            topk_similarity=_float_or_none(row, "topk_similarity")
            if _float_or_none(row, "topk_similarity") is not None
            else global_similarity,
            temporal_coverage=None,
            consistency=None,
            target_probability=_float_or_none(row, "target_probability"),
            overlap_probability=_float_or_none(row, "overlap_probability"),
            quality=_float_or_none(row, "audio_quality", "quality"),
            latency_ms=_float_or_none(row, "latency_ms") or 0.0,
            error=row.get("error") or None,
        )


class TemporalEvidenceReplayBackend:
    """Replay strict temporal speaker-evidence JSONL records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata = _replay_metadata("temporal-evidence-replay")
        self.evidence: dict[str, TemporalSpeakerEvidence] = {}
        self._load()

    def _load(self) -> None:
        for row in read_jsonl(self.path):
            preliminary_id = row.get("id") if isinstance(row, dict) else None
            sample_id = str(preliminary_id) if preliminary_id is not None else "<missing>"
            _insert_unique(self.evidence, sample_id, self.path)
            parsed = TemporalSpeakerEvidence.from_dict(row)
            replay_backend = BackendMetadata(
                name=parsed.backend.name,
                model_id=parsed.backend.model_id,
                version=parsed.backend.version,
                replay=True,
                config_hash=parsed.backend.config_hash,
            )
            self.evidence[sample_id] = TemporalSpeakerEvidence(
                id=parsed.id,
                backend=replay_backend,
                enrollment_source=parsed.enrollment_source,
                command_source=parsed.command_source,
                windows=parsed.windows,
                global_similarity=parsed.global_similarity,
                topk_similarity=parsed.topk_similarity,
                temporal_coverage=parsed.temporal_coverage,
                consistency=parsed.consistency,
                target_probability=parsed.target_probability,
                overlap_probability=parsed.overlap_probability,
                quality=parsed.quality,
                latency_ms=parsed.latency_ms,
                error=parsed.error,
            )

    def load(self) -> None:
        return None

    def score(self, sample: Sample) -> TemporalSpeakerEvidence:
        evidence = self.evidence.get(str(sample.id))
        if evidence is not None:
            return evidence
        enrollment_source, command_source = _source_paths(sample)
        return TemporalSpeakerEvidence(
            id=str(sample.id),
            backend=self.metadata,
            enrollment_source=enrollment_source,
            command_source=command_source,
            error="missing_evidence",
        )
