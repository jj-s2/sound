"""Serializable contracts for replay and real inference backends."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Mapping


_MISSING = object()


def _serialize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if is_dataclass(value):
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(f"{type(value).__name__} does not implement to_dict()")
        return to_dict()
    return value


def _to_dict(value: object) -> dict:
    return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}


def _context(contract: str, field: str, sample_id: str | None = None) -> str:
    sample = f" sample {sample_id!r}" if sample_id is not None else ""
    return f"{contract}{sample} field {field!r}"


def _mapping(value: object, contract: str, sample_id: str | None = None) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{_context(contract, '<root>', sample_id)} must be a dict")
    return value


def _check_keys(
    value: Mapping[str, object],
    contract: str,
    allowed: set[str],
    sample_id: str | None = None,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"{_context(contract, unknown[0], sample_id)} is not a recognized field"
        )


def _get(
    value: Mapping[str, object],
    contract: str,
    field: str,
    default: object = _MISSING,
    sample_id: str | None = None,
) -> object:
    if field in value:
        return value[field]
    if default is not _MISSING:
        return default
    raise ValueError(f"{_context(contract, field, sample_id)} is missing")


def _string(
    value: object,
    contract: str,
    field: str,
    sample_id: str | None = None,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{_context(contract, field, sample_id)} must be a string")
    return value


def _optional_string(
    value: object,
    contract: str,
    field: str,
    sample_id: str | None = None,
) -> str | None:
    if value is None:
        return None
    return _string(value, contract, field, sample_id)


def _number(
    value: object,
    contract: str,
    field: str,
    sample_id: str | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{_context(contract, field, sample_id)} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{_context(contract, field, sample_id)} must be a finite number")
    return result


def _optional_number(
    value: object,
    contract: str,
    field: str,
    sample_id: str | None = None,
) -> float | None:
    if value is None:
        return None
    return _number(value, contract, field, sample_id)


def _integer(
    value: object,
    contract: str,
    field: str,
    sample_id: str | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{_context(contract, field, sample_id)} must be an integer")
    return value


def _boolean(
    value: object,
    contract: str,
    field: str,
    sample_id: str | None = None,
) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{_context(contract, field, sample_id)} must be a boolean")
    return value


@dataclass(frozen=True)
class BackendMetadata:
    name: str
    model_id: str
    version: str | None = None
    replay: bool = False
    config_hash: str | None = None

    def to_dict(self) -> dict:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "BackendMetadata":
        contract = cls.__name__
        mapping = _mapping(value, contract)
        _check_keys(mapping, contract, {field.name for field in fields(cls)})
        return cls(
            name=_string(_get(mapping, contract, "name"), contract, "name"),
            model_id=_string(_get(mapping, contract, "model_id"), contract, "model_id"),
            version=_optional_string(
                _get(mapping, contract, "version", None), contract, "version"
            ),
            replay=_boolean(_get(mapping, contract, "replay", False), contract, "replay"),
            config_hash=_optional_string(
                _get(mapping, contract, "config_hash", None), contract, "config_hash"
            ),
        )


@dataclass(frozen=True)
class EvidenceWindow:
    start_sec: float
    end_sec: float
    similarity: float
    quality: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("start_sec", "end_sec", "similarity"):
            _number(getattr(self, field_name), type(self).__name__, field_name)
        if self.quality is not None:
            _number(self.quality, type(self).__name__, "quality")
        if self.end_sec <= self.start_sec:
            raise ValueError("EvidenceWindow field 'end_sec' must be greater than start_sec")

    def to_dict(self) -> dict:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "EvidenceWindow":
        contract = cls.__name__
        mapping = _mapping(value, contract)
        _check_keys(mapping, contract, {field.name for field in fields(cls)})
        return cls(
            start_sec=_number(
                _get(mapping, contract, "start_sec"), contract, "start_sec"
            ),
            end_sec=_number(_get(mapping, contract, "end_sec"), contract, "end_sec"),
            similarity=_number(
                _get(mapping, contract, "similarity"), contract, "similarity"
            ),
            quality=_optional_number(
                _get(mapping, contract, "quality", None), contract, "quality"
            ),
        )


@dataclass(frozen=True)
class TemporalSpeakerEvidence:
    id: str
    backend: BackendMetadata
    enrollment_source: str
    command_source: str
    windows: tuple[EvidenceWindow, ...] = ()
    global_similarity: float | None = None
    topk_similarity: float | None = None
    temporal_coverage: float | None = None
    consistency: float | None = None
    target_probability: float | None = None
    overlap_probability: float | None = None
    quality: float | None = None
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "TemporalSpeakerEvidence":
        contract = cls.__name__
        mapping = _mapping(value, contract)
        preliminary_id = mapping.get("id")
        sample_id = preliminary_id if isinstance(preliminary_id, str) else None
        _check_keys(
            mapping, contract, {field.name for field in fields(cls)}, sample_id
        )
        parsed_id = _string(
            _get(mapping, contract, "id", sample_id=sample_id),
            contract,
            "id",
            sample_id,
        )
        sample_id = parsed_id

        backend_value = _get(mapping, contract, "backend", sample_id=sample_id)
        if not isinstance(backend_value, dict):
            raise ValueError(
                f"{_context(contract, 'backend', sample_id)} must be a dict"
            )
        try:
            backend = BackendMetadata.from_dict(backend_value)
        except ValueError as exc:
            raise ValueError(
                f"{_context(contract, 'backend', sample_id)} is malformed: {exc}"
            ) from exc

        windows_value = _get(mapping, contract, "windows", [], sample_id)
        if not isinstance(windows_value, list):
            raise ValueError(
                f"{_context(contract, 'windows', sample_id)} must be a list"
            )
        windows = []
        for index, window_value in enumerate(windows_value):
            try:
                windows.append(EvidenceWindow.from_dict(window_value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{_context(contract, f'windows[{index}]', sample_id)} "
                    f"is malformed: {exc}"
                ) from exc

        optional_numbers = {}
        for field_name in (
            "global_similarity",
            "topk_similarity",
            "temporal_coverage",
            "consistency",
            "target_probability",
            "overlap_probability",
            "quality",
        ):
            optional_numbers[field_name] = _optional_number(
                _get(mapping, contract, field_name, None, sample_id),
                contract,
                field_name,
                sample_id,
            )

        return cls(
            id=parsed_id,
            backend=backend,
            enrollment_source=_string(
                _get(mapping, contract, "enrollment_source", sample_id=sample_id),
                contract,
                "enrollment_source",
                sample_id,
            ),
            command_source=_string(
                _get(mapping, contract, "command_source", sample_id=sample_id),
                contract,
                "command_source",
                sample_id,
            ),
            windows=tuple(windows),
            **optional_numbers,
            latency_ms=_number(
                _get(mapping, contract, "latency_ms", 0.0, sample_id),
                contract,
                "latency_ms",
                sample_id,
            ),
            error=_optional_string(
                _get(mapping, contract, "error", None, sample_id),
                contract,
                "error",
                sample_id,
            ),
        )


class RouteAction(str, Enum):
    REJECT = "reject"
    RAW = "raw"
    ENHANCED = "enhanced"


@dataclass(frozen=True)
class RouteDecision:
    id: str
    action: RouteAction
    reason_code: str
    policy_version: str
    evidence_version: str
    estimated_target_probability: float | None = None
    estimated_raw_risk: float | None = None
    estimated_enhanced_risk: float | None = None
    estimated_incremental_cost_ms: float | None = None

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(frozen=True)
class StageTiming:
    stage: str
    elapsed_ms: float
    replay: bool

    def to_dict(self) -> dict:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "StageTiming":
        contract = cls.__name__
        mapping = _mapping(value, contract)
        _check_keys(mapping, contract, {field.name for field in fields(cls)})
        return cls(
            stage=_string(_get(mapping, contract, "stage"), contract, "stage"),
            elapsed_ms=_number(
                _get(mapping, contract, "elapsed_ms"), contract, "elapsed_ms"
            ),
            replay=_boolean(_get(mapping, contract, "replay"), contract, "replay"),
        )


@dataclass(frozen=True)
class RunTrace:
    run_id: str
    measurement_mode: str
    device: str
    batch_size: int
    warmup_count: int
    cuda_synchronized: bool
    model_load_sec: float
    inference_sec: float
    total_sec: float
    mean_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    peak_gpu_memory_mb: float | None
    peak_cpu_rss_mb: float | None
    sample_count: int
    stages: tuple[StageTiming, ...] = ()
    capability_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.measurement_mode not in {"replay", "real"}:
            raise ValueError(
                "RunTrace field 'measurement_mode' must be 'replay' or 'real'"
            )

    def to_dict(self) -> dict:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RunTrace":
        contract = cls.__name__
        mapping = _mapping(value, contract)
        _check_keys(mapping, contract, {field.name for field in fields(cls)})
        run_id_value = mapping.get("run_id")
        run_id = run_id_value if isinstance(run_id_value, str) else None
        parsed_run_id = _string(
            _get(mapping, contract, "run_id", sample_id=run_id),
            contract,
            "run_id",
            run_id,
        )
        measurement_mode = _string(
            _get(mapping, contract, "measurement_mode", sample_id=run_id),
            contract,
            "measurement_mode",
            run_id,
        )

        stages_value = _get(mapping, contract, "stages", [], run_id)
        if not isinstance(stages_value, list):
            raise ValueError(f"{_context(contract, 'stages', run_id)} must be a list")
        stages = []
        for index, stage_value in enumerate(stages_value):
            try:
                stages.append(StageTiming.from_dict(stage_value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{_context(contract, f'stages[{index}]', run_id)} "
                    f"is malformed: {exc}"
                ) from exc

        notes_value = _get(mapping, contract, "capability_notes", [], run_id)
        if not isinstance(notes_value, list):
            raise ValueError(
                f"{_context(contract, 'capability_notes', run_id)} must be a list"
            )
        notes = tuple(
            _string(item, contract, f"capability_notes[{index}]", run_id)
            for index, item in enumerate(notes_value)
        )

        optional_numbers = {}
        for field_name in (
            "mean_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "peak_gpu_memory_mb",
            "peak_cpu_rss_mb",
        ):
            optional_numbers[field_name] = _optional_number(
                _get(mapping, contract, field_name, sample_id=run_id),
                contract,
                field_name,
                run_id,
            )

        return cls(
            run_id=parsed_run_id,
            measurement_mode=measurement_mode,
            device=_string(
                _get(mapping, contract, "device", sample_id=run_id),
                contract,
                "device",
                run_id,
            ),
            batch_size=_integer(
                _get(mapping, contract, "batch_size", sample_id=run_id),
                contract,
                "batch_size",
                run_id,
            ),
            warmup_count=_integer(
                _get(mapping, contract, "warmup_count", sample_id=run_id),
                contract,
                "warmup_count",
                run_id,
            ),
            cuda_synchronized=_boolean(
                _get(mapping, contract, "cuda_synchronized", sample_id=run_id),
                contract,
                "cuda_synchronized",
                run_id,
            ),
            model_load_sec=_number(
                _get(mapping, contract, "model_load_sec", sample_id=run_id),
                contract,
                "model_load_sec",
                run_id,
            ),
            inference_sec=_number(
                _get(mapping, contract, "inference_sec", sample_id=run_id),
                contract,
                "inference_sec",
                run_id,
            ),
            total_sec=_number(
                _get(mapping, contract, "total_sec", sample_id=run_id),
                contract,
                "total_sec",
                run_id,
            ),
            **optional_numbers,
            sample_count=_integer(
                _get(mapping, contract, "sample_count", sample_id=run_id),
                contract,
                "sample_count",
                run_id,
            ),
            stages=tuple(stages),
            capability_notes=notes,
        )


@dataclass(frozen=True)
class EnhancedAudioResult:
    audio_path: str | None
    backend: BackendMetadata
    latency_ms: float
    used_raw_fallback: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict:
        return _to_dict(self)
