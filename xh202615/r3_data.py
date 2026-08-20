"""R3 domain-matched TSE manifest contract and leakage validation.

This module defines the leakage-safe data contract for the R3 renderer / TSE
pilot. It never reads Dataset-A; ``assert_r3_manifest_safe`` performs canonical
containment checks for every audio path before callers create output or
initialize models.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Mapping


_ALLOWED_SPLITS = {"train", "val", "test"}
_MISSING = object()

# Nuisance fields that must be identical across a counterfactual pair. Output
# paths (row_id, mixture_audio, clean_target_audio) and target_present are
# allowed to differ between the positive and negative siblings.
_PAIR_NUISANCE_FIELDS = (
    "enrollment_audio",
    "target_source_id",
    "interferer_source_ids",
    "noise_source_id",
    "target_rir_id",
    "interferer_rir_ids",
    "renderer_family",
    "snr_db",
    "sir_db",
    "overlap_ratio",
    "codec",
    "clip_threshold",
    "nuisance_fingerprint",
)


def _context(field: str, row_id: str | None = None) -> str:
    row = f" row {row_id!r}" if row_id is not None else ""
    return f"R3MixtureRow{row} field {field!r}"


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("R3MixtureRow must be a dict")
    return value


def _get(value: Mapping[str, object], field: str, default=_MISSING, row_id: str | None = None):
    if field in value:
        return value[field]
    if default is not _MISSING:
        return default
    raise ValueError(f"{_context(field, row_id)} is missing")


def _string(value: object, field: str, row_id: str | None = None) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{_context(field, row_id)} must be a string")
    return value


def _optional_string(value: object, field: str, row_id: str | None = None) -> str | None:
    if value is None:
        return None
    return _string(value, field, row_id)


def _path(value: object, field: str, row_id: str | None = None) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{_context(field, row_id)} must be a path string")
    return Path(value)


def _boolean(value: object, field: str, row_id: str | None = None) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{_context(field, row_id)} must be a boolean")
    return value


def _number(value: object, field: str, row_id: str | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{_context(field, row_id)} must be a number")
    return float(value)


def _optional_number(value: object, field: str, row_id: str | None = None) -> float | None:
    if value is None:
        return None
    return _number(value, field, row_id)


def _string_tuple(value: object, field: str, row_id: str | None = None) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{_context(field, row_id)} must be a list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{_context(field, row_id)} must contain non-empty strings")
        result.append(item)
    return tuple(result)


def _resolve_manifest_path(value: Path, base_dir: Path | None) -> Path:
    if base_dir is None or value.is_absolute() or str(value).strip() in {"", "."}:
        return value
    return base_dir.resolve(strict=False) / value


def _empty_path(value: Path) -> bool:
    return str(value).strip() in {"", "."}


def _is_under(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


@dataclass(frozen=True)
class R3MixtureRow:
    """One R3 mixture row with full provenance for leakage auditing."""

    row_id: str
    pair_id: str
    split: str
    target_present: bool
    enrollment_audio: Path
    mixture_audio: Path
    clean_target_audio: Path
    target_source_id: str
    interferer_source_ids: tuple[str, ...]
    noise_source_id: str | None
    target_rir_id: str | None
    interferer_rir_ids: tuple[str, ...]
    renderer_family: str
    snr_db: float | None
    sir_db: float | None
    overlap_ratio: float
    codec: str
    clip_threshold: float
    nuisance_fingerprint: str

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "pair_id": self.pair_id,
            "split": self.split,
            "target_present": self.target_present,
            "enrollment_audio": self.enrollment_audio.as_posix(),
            "mixture_audio": self.mixture_audio.as_posix(),
            "clean_target_audio": self.clean_target_audio.as_posix(),
            "target_source_id": self.target_source_id,
            "interferer_source_ids": list(self.interferer_source_ids),
            "noise_source_id": self.noise_source_id,
            "target_rir_id": self.target_rir_id,
            "interferer_rir_ids": list(self.interferer_rir_ids),
            "renderer_family": self.renderer_family,
            "snr_db": self.snr_db,
            "sir_db": self.sir_db,
            "overlap_ratio": self.overlap_ratio,
            "codec": self.codec,
            "clip_threshold": self.clip_threshold,
            "nuisance_fingerprint": self.nuisance_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: dict, *, base_dir: Path | None = None) -> "R3MixtureRow":
        mapping = _mapping(value)
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(mapping) - allowed)
        preliminary_id = mapping.get("row_id")
        row_id = preliminary_id if isinstance(preliminary_id, str) else None
        if unknown:
            raise ValueError(f"{_context(unknown[0], row_id)} is not a recognized field")

        parsed_row_id = _string(_get(mapping, "row_id", row_id=row_id), "row_id", row_id)
        row_id = parsed_row_id
        return cls(
            row_id=parsed_row_id,
            pair_id=_string(_get(mapping, "pair_id", row_id=row_id), "pair_id", row_id),
            split=_string(_get(mapping, "split", row_id=row_id), "split", row_id),
            target_present=_boolean(
                _get(mapping, "target_present", row_id=row_id), "target_present", row_id
            ),
            enrollment_audio=_resolve_manifest_path(
                _path(_get(mapping, "enrollment_audio", row_id=row_id), "enrollment_audio", row_id),
                base_dir,
            ),
            mixture_audio=_resolve_manifest_path(
                _path(_get(mapping, "mixture_audio", row_id=row_id), "mixture_audio", row_id),
                base_dir,
            ),
            clean_target_audio=_resolve_manifest_path(
                _path(
                    _get(mapping, "clean_target_audio", row_id=row_id),
                    "clean_target_audio",
                    row_id,
                ),
                base_dir,
            ),
            target_source_id=_string(
                _get(mapping, "target_source_id", row_id=row_id), "target_source_id", row_id
            ),
            interferer_source_ids=_string_tuple(
                _get(mapping, "interferer_source_ids", row_id=row_id),
                "interferer_source_ids",
                row_id,
            ),
            noise_source_id=_optional_string(
                _get(mapping, "noise_source_id", None, row_id), "noise_source_id", row_id
            ),
            target_rir_id=_optional_string(
                _get(mapping, "target_rir_id", None, row_id), "target_rir_id", row_id
            ),
            interferer_rir_ids=_string_tuple(
                _get(mapping, "interferer_rir_ids", row_id=row_id),
                "interferer_rir_ids",
                row_id,
            ),
            renderer_family=_string(
                _get(mapping, "renderer_family", row_id=row_id), "renderer_family", row_id
            ),
            snr_db=_optional_number(_get(mapping, "snr_db", None, row_id), "snr_db", row_id),
            sir_db=_optional_number(_get(mapping, "sir_db", None, row_id), "sir_db", row_id),
            overlap_ratio=_number(
                _get(mapping, "overlap_ratio", row_id=row_id), "overlap_ratio", row_id
            ),
            codec=_string(_get(mapping, "codec", row_id=row_id), "codec", row_id),
            clip_threshold=_number(
                _get(mapping, "clip_threshold", row_id=row_id), "clip_threshold", row_id
            ),
            nuisance_fingerprint=_string(
                _get(mapping, "nuisance_fingerprint", row_id=row_id),
                "nuisance_fingerprint",
                row_id,
            ),
        )


@dataclass(frozen=True)
class R3ManifestIssue:
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


def _row_entities(row: R3MixtureRow) -> Iterable[tuple[str, str]]:
    yield ("speaker_source_id", row.target_source_id)
    for source_id in row.interferer_source_ids:
        yield ("speaker_source_id", source_id)
    if row.noise_source_id is not None:
        yield ("noise_source_id", row.noise_source_id)
    if row.target_rir_id is not None:
        yield ("rir_id", row.target_rir_id)
    for rir_id in row.interferer_rir_ids:
        yield ("rir_id", rir_id)
    yield ("renderer_family", row.renderer_family)


def validate_r3_manifest(rows: Iterable[R3MixtureRow]) -> tuple[R3ManifestIssue, ...]:
    """Return all validation issues for the R3 manifest without raising."""
    materialized = tuple(rows)
    issues: list[R3ManifestIssue] = []
    seen_row_ids: dict[str, int] = {}
    entity_splits: dict[tuple[str, str], tuple[str, str]] = {}
    pairs: dict[str, list[R3MixtureRow]] = {}

    def add(code: str, row: object, message: str) -> None:
        issues.append(R3ManifestIssue(code, getattr(row, "row_id", None), message))

    for index, row in enumerate(materialized):
        if not isinstance(row, R3MixtureRow):
            raise ValueError(f"R3 manifest row {index + 1} must be an R3MixtureRow")

        if not row.row_id.strip():
            add("empty_row_id", row, "row_id must be a non-empty string")
        elif row.row_id in seen_row_ids:
            add(
                "duplicate_row_id",
                row,
                f"row_id {row.row_id!r} duplicates row {seen_row_ids[row.row_id] + 1}",
            )
        else:
            seen_row_ids[row.row_id] = index

        if not row.pair_id.strip():
            add("empty_pair_id", row, "pair_id must be a non-empty string")

        if row.split not in _ALLOWED_SPLITS:
            add("invalid_split", row, f"split must be one of {sorted(_ALLOWED_SPLITS)}")

        if not row.target_source_id.strip():
            add("empty_target_source_id", row, "target_source_id must be non-empty")

        for source_id in row.interferer_source_ids:
            if not isinstance(source_id, str) or not source_id.strip():
                add(
                    "empty_interferer_source_id",
                    row,
                    "interferer_source_ids must contain non-empty strings",
                )
                break

        if row.noise_source_id is not None and not row.noise_source_id.strip():
            add("empty_noise_source_id", row, "noise_source_id must be null or non-empty")

        if row.target_rir_id is not None and not row.target_rir_id.strip():
            add("empty_target_rir_id", row, "target_rir_id must be null or non-empty")

        for rir_id in row.interferer_rir_ids:
            if not isinstance(rir_id, str) or not rir_id.strip():
                add(
                    "empty_interferer_rir_id",
                    row,
                    "interferer_rir_ids must contain non-empty strings",
                )
                break
        if len(row.interferer_rir_ids) not in (0, len(row.interferer_source_ids)):
            add(
                "interferer_rir_count_mismatch",
                row,
                "interferer_rir_ids length must be 0 or match interferer_source_ids length",
            )

        if not row.renderer_family.strip():
            add("empty_renderer_family", row, "renderer_family must be non-empty")

        for field_name in ("enrollment_audio", "mixture_audio", "clean_target_audio"):
            if _empty_path(getattr(row, field_name)):
                add(f"empty_{field_name}", row, f"{field_name} must be a non-empty path")

        if not math.isfinite(row.overlap_ratio) or not 0.0 <= row.overlap_ratio <= 1.0:
            add("invalid_overlap_ratio", row, "overlap_ratio must be finite and in [0, 1]")
        if row.snr_db is not None and not math.isfinite(row.snr_db):
            add("invalid_snr_db", row, "snr_db must be finite when provided")
        if row.sir_db is not None and not math.isfinite(row.sir_db):
            add("invalid_sir_db", row, "sir_db must be finite when provided")
        if not math.isfinite(row.clip_threshold) or not 0.0 < row.clip_threshold <= 1.0:
            add("invalid_clip_threshold", row, "clip_threshold must be finite and in (0, 1]")

        if not row.codec.strip():
            add("empty_codec", row, "codec must be non-empty")
        if not row.nuisance_fingerprint.strip():
            add("empty_nuisance_fingerprint", row, "nuisance_fingerprint must be non-empty")

        pairs.setdefault(row.pair_id, []).append(row)

        if row.split in _ALLOWED_SPLITS:
            for entity_type, entity_value in _row_entities(row):
                key = (entity_type, entity_value)
                previous = entity_splits.get(key)
                if previous is None:
                    entity_splits[key] = (row.split, row.row_id)
                elif previous[0] != row.split:
                    add(
                        "entity_split_leakage",
                        row,
                        f"{entity_type} {entity_value!r} appears in split {row.split!r} "
                        f"and split {previous[0]!r}",
                    )

    for pair_id, group in pairs.items():
        if len(group) != 2:
            add(
                "pair_size",
                group[0],
                f"pair {pair_id!r} must contain exactly two rows; found {len(group)}",
            )
            continue
        positives = sum(1 for member in group if member.target_present)
        if positives != 1:
            add(
                "pair_polarity_imbalance",
                group[0],
                f"pair {pair_id!r} must have one target-present and one target-absent row",
            )
            continue
        splits = {member.split for member in group}
        if len(splits) != 1:
            add(
                "pair_split_mismatch",
                group[0],
                f"pair {pair_id!r} rows must share a split; found {sorted(splits)}",
            )
            continue
        for field_name in _PAIR_NUISANCE_FIELDS:
            if getattr(group[0], field_name) != getattr(group[1], field_name):
                add(
                    "counterfactual_nuisance_mismatch",
                    group[0],
                    f"pair {pair_id!r} {field_name} differs between counterfactual rows",
                )
                break

    return tuple(issues)


def read_r3_manifest(path: str | Path) -> tuple[R3MixtureRow, ...]:
    manifest_path = Path(path)
    rows: list[R3MixtureRow] = []
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"R3 manifest {manifest_path} line {line_number} contains "
                    f"malformed JSON: {exc.msg}"
                ) from exc
            try:
                rows.append(
                    R3MixtureRow.from_dict(
                        value, base_dir=manifest_path.resolve(strict=False).parent
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"R3 manifest {manifest_path} line {line_number} is malformed: {exc}"
                ) from exc
    return tuple(rows)


def write_r3_manifest(path: str | Path, rows: Iterable[R3MixtureRow]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True)
        for row in rows
    ]
    manifest_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def assert_r3_manifest_safe(
    rows: Iterable[R3MixtureRow], dataset_a_root: str | Path
) -> tuple[R3MixtureRow, ...]:
    """Containment + validity guard run before output creation or model init."""
    materialized = tuple(rows)
    root = Path(dataset_a_root).resolve(strict=False)
    for row in materialized:
        if not isinstance(row, R3MixtureRow):
            raise ValueError("R3 manifest row must be an R3MixtureRow")
        for field_name in ("enrollment_audio", "mixture_audio", "clean_target_audio"):
            resolved = Path(getattr(row, field_name)).resolve(strict=False)
            if _is_under(resolved, root):
                raise ValueError(
                    f"Dataset-A containment violation: {field_name} for row "
                    f"{row.row_id!r} resolves under Dataset-A root {root}"
                )
    issues = validate_r3_manifest(materialized)
    errors = tuple(issue for issue in issues if issue.severity == "error")
    if errors:
        details = "; ".join(
            f"{issue.code} ({issue.row_id or '<unknown>'}): {issue.message}"
            for issue in errors
        )
        raise ValueError(f"R3 manifest validation failed: {details}")
    return materialized
