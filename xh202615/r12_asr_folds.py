"""Label-free parent-group outer fold manifest for R12 M0 ASR out-of-fold inference.

Every train-role lineage row (positive and negative, original and augmented) is
assigned to a single outer fold keyed by its parent wake group, so downstream
ASR out-of-fold inference can cover both classes without ever crossing a
parent/wake group boundary.  The manifest serializes only structural identity
(``id``, ``parent_id``, ``group``, ``source_split``, ``augmentation_id``) plus
the ``outer_fold``; no label, target text, or audio path is ever written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .r12_dataa_augmentation import LineageRow


_SCHEMA_VERSION = "r12_asr_folds_v1"
_DEFAULT_SEED = 20260814
_DEFAULT_FOLD_COUNT = 3
_TRAIN_ROLE = "train"
_SERIAL_KEYS = frozenset({"schema_version", "seed", "fold_count", "rows", "manifest_sha256"})
_ROW_KEYS = frozenset({"id", "parent_id", "group", "source_split", "augmentation_id", "outer_fold"})
_SOURCE_CLASSES = {"neg": 0, "pos": 1}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class AsrFoldRow:
    id: str
    parent_id: str
    group: str
    source_split: str
    augmentation_id: str
    outer_fold: int


@dataclass(frozen=True)
class AsrFoldManifest:
    schema_version: str
    seed: int
    fold_count: int
    fold_by_id: Mapping[str, int]
    rows: tuple[AsrFoldRow, ...]
    manifest_sha256: str


def _row_to_dict(row: AsrFoldRow) -> dict[str, object]:
    return {
        "id": row.id,
        "parent_id": row.parent_id,
        "group": row.group,
        "source_split": row.source_split,
        "augmentation_id": row.augmentation_id,
        "outer_fold": row.outer_fold,
    }


def _manifest_sha256(data: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in data.items() if key != "manifest_sha256"}
    return _sha256_hex(_canonical(payload))


def _validate_parameters(fold_count: int, seed: int) -> None:
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ValueError(f"fold_count must be an integer >= 2, got {fold_count!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")


def _train_rows(lineage: Mapping[str, LineageRow]) -> list[LineageRow]:
    rows = [row for row in lineage.values() if row.role == _TRAIN_ROLE]
    if not rows:
        raise ValueError("lineage has no train-role rows")
    parent_groups: dict[str, str] = {}
    for row in rows:
        if not isinstance(row.group, str) or not row.group:
            raise ValueError(f"empty group for id {row.id!r}")
        if not isinstance(row.parent_id, str) or not row.parent_id:
            raise ValueError(f"empty parent_id for id {row.id!r}")
        existing = parent_groups.get(row.parent_id)
        if existing is not None and existing != row.group:
            raise ValueError("parent rows span multiple groups")
        parent_groups[row.parent_id] = row.group
    return rows


def _group_rows(rows: list[LineageRow]) -> dict[str, list[LineageRow]]:
    groups: dict[str, list[LineageRow]] = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)
    return groups


def _group_label(rows: list[LineageRow]) -> int | None:
    splits = {row.source_split for row in rows}
    if len(splits) != 1:
        return None
    return _SOURCE_CLASSES.get(next(iter(splits)))


def _stratified_group_folds(
    group_names: list[str], labels: dict[str, int], fold_count: int, seed: int,
) -> dict[str, int] | None:
    """Return a stratified group->fold assignment, or None when infeasible."""
    counts = {0: 0, 1: 0}
    for name in group_names:
        label = labels[name]
        if label not in counts:
            return None
        counts[label] += 1
    if counts[0] < fold_count or counts[1] < fold_count or len(group_names) < fold_count:
        return None

    y = np.asarray([labels[name] for name in group_names], dtype=np.int64)
    group_array = np.asarray(group_names, dtype=object)
    x = np.zeros((len(group_names), 1), dtype=np.float64)
    splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    try:
        splits = list(splitter.split(x, y, group_array))
    except ValueError:
        return None
    if len(splits) != fold_count:
        return None

    assigned = np.full(len(group_names), -1, dtype=np.int64)
    for fold_index, (_, test_indices) in enumerate(splits):
        test_indices = np.asarray(test_indices)
        if np.any(assigned[test_indices] >= 0):
            return None
        assigned[test_indices] = fold_index
    if np.any(assigned < 0):
        return None
    return {name: int(assigned[index]) for index, name in enumerate(group_names)}


def _fallback_group_folds(group_names: list[str], fold_count: int, seed: int) -> dict[str, int]:
    """Deterministic SHA-256 group round-robin preserving group disjointness."""
    ordered = sorted(
        group_names,
        key=lambda name: _sha256_hex(f"{seed}\0{name}".encode("utf-8")),
    )
    return {name: index % fold_count for index, name in enumerate(ordered)}


def build_asr_folds(
    lineage: Mapping[str, LineageRow],
    *,
    fold_count: int = _DEFAULT_FOLD_COUNT,
    seed: int = _DEFAULT_SEED,
) -> AsrFoldManifest:
    """Assign every train-role lineage row to its parent's deterministic outer fold."""
    _validate_parameters(fold_count, seed)
    groups = _group_rows(_train_rows(lineage))
    group_names = sorted(groups)

    labels = {name: _group_label(groups[name]) for name in group_names}
    group_fold = None if any(label is None for label in labels.values()) else _stratified_group_folds(
        group_names, labels, fold_count, seed  # type: ignore[arg-type]
    )
    if group_fold is None:
        group_fold = _fallback_group_folds(group_names, fold_count, seed)

    rows = tuple(
        AsrFoldRow(
            id=row.id,
            parent_id=row.parent_id,
            group=row.group,
            source_split=row.source_split,
            augmentation_id=row.augmentation_id,
            outer_fold=group_fold[row.group],
        )
        for row in sorted(_train_rows(lineage), key=lambda r: r.id)
    )
    fold_by_id: dict[str, int] = {row.id: row.outer_fold for row in rows}

    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "seed": seed,
        "fold_count": fold_count,
        "rows": [_row_to_dict(row) for row in rows],
    }
    manifest_sha256 = _manifest_sha256(payload)
    return AsrFoldManifest(
        schema_version=_SCHEMA_VERSION,
        seed=seed,
        fold_count=fold_count,
        fold_by_id=fold_by_id,
        rows=rows,
        manifest_sha256=manifest_sha256,
    )


def write_asr_folds(path: Path, manifest: AsrFoldManifest) -> None:
    """Write ``manifest`` to ``path`` as deterministic canonical JSON bytes."""
    payload: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "seed": manifest.seed,
        "fold_count": manifest.fold_count,
        "rows": [_row_to_dict(row) for row in manifest.rows],
    }
    recomputed = _manifest_sha256(payload)
    if recomputed != manifest.manifest_sha256:
        raise ValueError("manifest SHA-256 does not match recomputed payload")
    payload["manifest_sha256"] = manifest.manifest_sha256
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload) + b"\n")


def _expected_train_ids(lineage: Mapping[str, LineageRow]) -> set[str]:
    return {sample_id for sample_id, row in lineage.items() if row.role == _TRAIN_ROLE}


def _validate_rows(
    rows: list[dict[str, Any]], fold_count: int, lineage: Mapping[str, LineageRow],
) -> tuple[dict[str, int], tuple[AsrFoldRow, ...]]:
    expected_ids = _expected_train_ids(lineage)
    seen_ids: set[str] = set()
    seen_parents: set[tuple[str, str]] = set()
    group_folds: dict[str, int] = {}
    parent_folds: dict[str, int] = {}
    fold_by_id: dict[str, int] = {}
    parsed: list[AsrFoldRow] = []

    for row in rows:
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            raise ValueError("fold row has invalid fields")
        sample_id = row["id"]
        if not isinstance(sample_id, str) or sample_id in seen_ids:
            raise ValueError("fold row has invalid or duplicate id")
        if sample_id not in expected_ids:
            raise ValueError(f"fold manifest references non-train id {sample_id!r}")
        lineage_row = lineage[sample_id]
        for field in ("parent_id", "group", "source_split", "augmentation_id"):
            if row[field] != getattr(lineage_row, field):
                raise ValueError(f"fold row {sample_id!r} does not match lineage")
        outer_fold = row["outer_fold"]
        if isinstance(outer_fold, bool) or not isinstance(outer_fold, int) or not 0 <= outer_fold < fold_count:
            raise ValueError(f"invalid outer_fold for id {sample_id!r}")
        if (lineage_row.parent_id, lineage_row.augmentation_id) in seen_parents:
            raise ValueError("duplicate lineage parent/augmentation")
        seen_parents.add((lineage_row.parent_id, lineage_row.augmentation_id))

        prior_group = group_folds.setdefault(lineage_row.group, outer_fold)
        if prior_group != outer_fold:
            raise ValueError(f"group {lineage_row.group!r} spans multiple folds")
        prior_parent = parent_folds.setdefault(lineage_row.parent_id, outer_fold)
        if prior_parent != outer_fold:
            raise ValueError(f"parent {lineage_row.parent_id!r} spans multiple folds")

        seen_ids.add(sample_id)
        fold_by_id[sample_id] = outer_fold
        parsed.append(
            AsrFoldRow(
                id=sample_id,
                parent_id=lineage_row.parent_id,
                group=lineage_row.group,
                source_split=lineage_row.source_split,
                augmentation_id=lineage_row.augmentation_id,
                outer_fold=outer_fold,
            )
        )

    if set(fold_by_id) != expected_ids:
        missing = expected_ids - set(fold_by_id)
        raise ValueError(f"fold manifest is missing train ids: {sorted(missing)}")
    return fold_by_id, tuple(parsed)


def load_asr_folds(path: Path, expected_lineage: Mapping[str, LineageRow]) -> AsrFoldManifest:
    """Load and validate a fold manifest against the expected train lineage."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid fold manifest: {exc}") from exc

    if not isinstance(data, dict) or set(data) != _SERIAL_KEYS:
        raise ValueError("fold manifest has invalid keys")
    if data.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {data.get('schema_version')!r}")
    seed = data.get("seed")
    fold_count = data.get("fold_count")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"invalid seed {seed!r}")
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ValueError(f"invalid fold_count {fold_count!r}")

    manifest_sha256 = data.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or manifest_sha256 != _manifest_sha256(data):
        raise ValueError("fold manifest digest mismatch")

    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError("fold manifest rows must be a list")
    fold_by_id, parsed_rows = _validate_rows(rows, fold_count, expected_lineage)

    return AsrFoldManifest(
        schema_version=_SCHEMA_VERSION,
        seed=seed,
        fold_count=fold_count,
        fold_by_id=fold_by_id,
        rows=parsed_rows,
        manifest_sha256=manifest_sha256,
    )


def role_for_outer_fold(
    manifest: AsrFoldManifest, sample_id: str, outer_fold: int,
) -> Literal["fit", "holdout"]:
    """Return ``holdout`` when ``sample_id`` lands in ``outer_fold``, else ``fit``."""
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, int) or not 0 <= outer_fold < manifest.fold_count:
        raise ValueError(f"outer_fold must be an integer in [0, {manifest.fold_count})")
    if sample_id not in manifest.fold_by_id:
        raise ValueError(f"unknown sample id {sample_id!r}")
    return "holdout" if manifest.fold_by_id[sample_id] == outer_fold else "fit"
