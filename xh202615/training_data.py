"""Contracts and validation for public and synthetic training manifests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Mapping


_ALLOWED_SPLITS = {"train", "val", "test", "internal_test"}
_MISSING = object()


def _context(field: str, row_id: str | None = None) -> str:
    row = f" row {row_id!r}" if row_id is not None else ""
    return f"TrainingManifestRow{row} field {field!r}"


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("TrainingManifestRow must be a dict")
    return value


def _get(
    value: Mapping[str, object],
    field: str,
    default: object = _MISSING,
    row_id: str | None = None,
) -> object:
    if field in value:
        return value[field]
    if default is not _MISSING:
        return default
    raise ValueError(f"{_context(field, row_id)} is missing")


def _string(value: object, field: str, row_id: str | None = None) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{_context(field, row_id)} must be a string")
    return value


def _optional_string(
    value: object, field: str, row_id: str | None = None
) -> str | None:
    if value is None:
        return None
    return _string(value, field, row_id)


def _path(value: object, field: str, row_id: str | None = None) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{_context(field, row_id)} must be a path string")
    return Path(value)


def _optional_path(
    value: object, field: str, row_id: str | None = None
) -> Path | None:
    if value is None:
        return None
    return _path(value, field, row_id)


def _number(value: object, field: str, row_id: str | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{_context(field, row_id)} must be a number")
    return float(value)


def _optional_number(
    value: object, field: str, row_id: str | None = None
) -> float | None:
    if value is None:
        return None
    return _number(value, field, row_id)


def _boolean(value: object, field: str, row_id: str | None = None) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{_context(field, row_id)} must be a boolean")
    return value


def _optional_integer(
    value: object, field: str, row_id: str | None = None
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{_context(field, row_id)} must be an integer or null")
    return value


def _resolve_manifest_path(value: Path, base_dir: Path | None) -> Path:
    if base_dir is None or value.is_absolute() or str(value).strip() in {"", "."}:
        return value
    return base_dir.resolve(strict=False) / value


@dataclass(frozen=True)
class TrainingManifestRow:
    row_id: str
    split: str
    source: str
    enrollment_audio: Path
    target_audio: Path
    mixture_audio: Path | None
    target_speaker_id: str
    interferer_speaker_id: str | None
    target_present: bool
    overlap_ratio: float
    snr_db: float | None
    sir_db: float | None
    text: str | None = None
    seed: int | None = None

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "split": self.split,
            "source": self.source,
            "enrollment_audio": self.enrollment_audio.as_posix(),
            "target_audio": self.target_audio.as_posix(),
            "mixture_audio": (
                self.mixture_audio.as_posix() if self.mixture_audio is not None else None
            ),
            "target_speaker_id": self.target_speaker_id,
            "interferer_speaker_id": self.interferer_speaker_id,
            "target_present": self.target_present,
            "overlap_ratio": self.overlap_ratio,
            "snr_db": self.snr_db,
            "sir_db": self.sir_db,
            "text": self.text,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(
        cls, value: dict, *, base_dir: Path | None = None
    ) -> "TrainingManifestRow":
        mapping = _mapping(value)
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(mapping) - allowed)
        preliminary_id = mapping.get("row_id")
        row_id = preliminary_id if isinstance(preliminary_id, str) else None
        if unknown:
            raise ValueError(f"{_context(unknown[0], row_id)} is not a recognized field")

        parsed_row_id = _string(_get(mapping, "row_id", row_id=row_id), "row_id", row_id)
        row_id = parsed_row_id
        enrollment_audio = _path(
            _get(mapping, "enrollment_audio", row_id=row_id),
            "enrollment_audio",
            row_id,
        )
        target_audio = _path(
            _get(mapping, "target_audio", row_id=row_id), "target_audio", row_id
        )
        mixture_audio = _optional_path(
            _get(mapping, "mixture_audio", None, row_id), "mixture_audio", row_id
        )

        return cls(
            row_id=parsed_row_id,
            split=_string(_get(mapping, "split", row_id=row_id), "split", row_id),
            source=_string(_get(mapping, "source", row_id=row_id), "source", row_id),
            enrollment_audio=_resolve_manifest_path(enrollment_audio, base_dir),
            target_audio=_resolve_manifest_path(target_audio, base_dir),
            mixture_audio=(
                _resolve_manifest_path(mixture_audio, base_dir)
                if mixture_audio is not None
                else None
            ),
            target_speaker_id=_string(
                _get(mapping, "target_speaker_id", row_id=row_id),
                "target_speaker_id",
                row_id,
            ),
            interferer_speaker_id=_optional_string(
                _get(mapping, "interferer_speaker_id", None, row_id),
                "interferer_speaker_id",
                row_id,
            ),
            target_present=_boolean(
                _get(mapping, "target_present", row_id=row_id),
                "target_present",
                row_id,
            ),
            overlap_ratio=_number(
                _get(mapping, "overlap_ratio", row_id=row_id),
                "overlap_ratio",
                row_id,
            ),
            snr_db=_optional_number(
                _get(mapping, "snr_db", None, row_id), "snr_db", row_id
            ),
            sir_db=_optional_number(
                _get(mapping, "sir_db", None, row_id), "sir_db", row_id
            ),
            text=_optional_string(_get(mapping, "text", None, row_id), "text", row_id),
            seed=_optional_integer(_get(mapping, "seed", None, row_id), "seed", row_id),
        )


@dataclass(frozen=True)
class ManifestIssue:
    code: str
    row_id: str | None
    message: str
    severity: str = "error"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "row_id": self.row_id,
            "message": self.message,
            "severity": self.severity,
        }


def read_training_manifest(path: str | Path) -> tuple[TrainingManifestRow, ...]:
    manifest_path = Path(path)
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Training manifest {manifest_path} line {line_number} contains "
                    f"malformed JSON: {exc.msg}"
                ) from exc
            try:
                rows.append(
                    TrainingManifestRow.from_dict(
                        value, base_dir=manifest_path.resolve(strict=False).parent
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Training manifest {manifest_path} line {line_number} is malformed: {exc}"
                ) from exc
    return tuple(rows)


def _empty_path(value: Path) -> bool:
    return str(value).strip() in {"", "."}


def _resolved_audio_path(value: Path, manifest_path: str | Path | None) -> Path:
    if value.is_absolute() or manifest_path is None:
        return value.resolve(strict=False)
    base_dir = Path(manifest_path).resolve(strict=False).parent
    return (base_dir / value).resolve(strict=False)


def _is_under(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def validate_training_manifest(
    rows: Iterable[TrainingManifestRow],
    *,
    manifest_path: str | Path | None = None,
    forbidden_roots: Iterable[str | Path] = (),
) -> tuple[ManifestIssue, ...]:
    materialized = tuple(rows)
    issues = []
    seen_row_ids: dict[str, int] = {}
    speaker_splits: dict[str, tuple[str, str]] = {}
    resolved_forbidden_roots = tuple(
        Path(root).resolve(strict=False) for root in forbidden_roots
    )

    def add(code: str, row: TrainingManifestRow, message: str) -> None:
        issues.append(ManifestIssue(code, row.row_id or None, message))

    for index, row in enumerate(materialized):
        if not isinstance(row, TrainingManifestRow):
            raise ValueError(
                f"Training manifest row {index + 1} must be a TrainingManifestRow"
            )

        if not row.row_id.strip():
            add("empty_row_id", row, "row_id must be a non-empty string")
        elif row.row_id in seen_row_ids:
            first_index = seen_row_ids[row.row_id]
            add(
                "duplicate_row_id",
                row,
                f"row_id {row.row_id!r} duplicates row {first_index + 1}",
            )
        else:
            seen_row_ids[row.row_id] = index

        if row.split not in _ALLOWED_SPLITS:
            add(
                "invalid_split",
                row,
                f"split must be one of {sorted(_ALLOWED_SPLITS)}",
            )
        if not row.source.strip():
            add("empty_source", row, "source must be a non-empty string")
        if not row.target_speaker_id.strip():
            add(
                "empty_target_speaker_id",
                row,
                "target_speaker_id must be a non-empty string",
            )
        if row.interferer_speaker_id is not None:
            if not row.interferer_speaker_id.strip():
                add(
                    "empty_interferer_speaker_id",
                    row,
                    "interferer_speaker_id must be null or a non-empty string",
                )
            elif row.interferer_speaker_id == row.target_speaker_id:
                add(
                    "same_speaker_id",
                    row,
                    "target and interferer speaker IDs must differ",
                )

        for field_name in ("enrollment_audio", "target_audio"):
            path_value = getattr(row, field_name)
            if _empty_path(path_value):
                add(
                    f"empty_{field_name}",
                    row,
                    f"{field_name} must be a non-empty path",
                )
        if row.mixture_audio is not None and _empty_path(row.mixture_audio):
            add(
                "empty_mixture_audio",
                row,
                "mixture_audio must be null or a non-empty path",
            )

        if not math.isfinite(row.overlap_ratio) or not 0.0 <= row.overlap_ratio <= 1.0:
            add(
                "invalid_overlap_ratio",
                row,
                "overlap_ratio must be finite and between 0 and 1 inclusive",
            )
        for field_name in ("snr_db", "sir_db"):
            number = getattr(row, field_name)
            if number is not None and not math.isfinite(number):
                add(
                    f"invalid_{field_name}",
                    row,
                    f"{field_name} must be finite when provided",
                )

        if not row.target_present and row.text is not None:
            add(
                "target_absent_text",
                row,
                "text must be null when target_present is false",
            )
        if (
            row.overlap_ratio > 0.0 or row.snr_db is not None or row.sir_db is not None
        ) and row.mixture_audio is None:
            add(
                "mixture_audio_required",
                row,
                "mixture_audio is required for overlap or noise rows",
            )

        for field_name in ("enrollment_audio", "target_audio", "mixture_audio"):
            path_value = getattr(row, field_name)
            if path_value is None or _empty_path(path_value):
                continue
            resolved_path = _resolved_audio_path(path_value, manifest_path)
            for forbidden_root in resolved_forbidden_roots:
                if _is_under(resolved_path, forbidden_root):
                    add(
                        "forbidden_path",
                        row,
                        f"{field_name} resolves under forbidden root {forbidden_root}",
                    )
                    break

        if row.split in _ALLOWED_SPLITS:
            speaker_roles = ((row.target_speaker_id, "target"),)
            if row.interferer_speaker_id is not None:
                speaker_roles += ((row.interferer_speaker_id, "interferer"),)
            for speaker_id, role in speaker_roles:
                if not speaker_id.strip():
                    continue
                previous = speaker_splits.get(speaker_id)
                if previous is None:
                    speaker_splits[speaker_id] = (row.split, row.row_id)
                elif previous[0] != row.split:
                    add(
                        "speaker_split_leakage",
                        row,
                        f"{role} speaker {speaker_id!r} occurs in split {row.split!r} "
                        f"and split {previous[0]!r} (row {previous[1]!r})",
                    )

    return tuple(issues)


def assert_valid_training_manifest(
    rows: Iterable[TrainingManifestRow],
    *,
    manifest_path: str | Path | None = None,
    forbidden_roots: Iterable[str | Path] = (),
) -> tuple[TrainingManifestRow, ...]:
    materialized = tuple(rows)
    issues = validate_training_manifest(
        materialized,
        manifest_path=manifest_path,
        forbidden_roots=forbidden_roots,
    )
    errors = tuple(issue for issue in issues if issue.severity == "error")
    if errors:
        details = "; ".join(
            f"{issue.code} ({issue.row_id or '<unknown>'}): {issue.message}"
            for issue in errors
        )
        raise ValueError(f"Training manifest validation failed: {details}")
    return materialized
