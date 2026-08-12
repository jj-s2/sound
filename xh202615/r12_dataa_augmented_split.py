"""Frozen Dataset-A 70/15/15 stratified wake-group split manifest.

The manifest deliberately serializes no labels or recognition/reference text.
Labels are consumed only while assigning stratified roles.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


Role = Literal["train", "validation", "internal_test"]
_SCHEMA_VERSION = "r12_dataa_augmented_split_v1"
_SEED = 20260812
_N_SPLITS = 20
_ROLES: tuple[Role, ...] = ("train", "validation", "internal_test")
_FOLD_TO_ROLE: tuple[Role, ...] = (
    *("train" for _ in range(14)),
    *("validation" for _ in range(3)),
    *("internal_test" for _ in range(3)),
)
_SERIAL_KEYS = frozenset({
    "schema_version", "seed", "roles_by_id", "groups_by_id", "role_counts",
    "group_counts", "source_digests", "manifest_sha256",
})
_SOURCE_KEYS = frozenset({"ids", "annotations", "groups"})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class AugmentedInternalSplitManifest:
    schema_version: str
    seed: int
    roles_by_id: Mapping[str, Role]
    groups_by_id: Mapping[str, str]
    source_digests: Mapping[str, str]
    manifest_sha256: str


def manifest_sha256(data: Mapping[str, object]) -> str:
    """Return the digest over an already-serialized manifest payload."""
    return _sha({key: value for key, value in data.items() if key != "manifest_sha256"})


def _validate_inputs(
    ids: Sequence[str], labels: Mapping[str, str | None], groups: Mapping[str, str]
) -> None:
    if len(ids) != len(set(ids)):
        raise ValueError("IDs must be unique")
    expected = set(ids)
    if set(labels) != expected:
        raise ValueError("label keys must exactly cover IDs")
    if set(groups) != expected:
        raise ValueError("group keys must exactly cover IDs")
    for sample_id in ids:
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("IDs must be nonempty strings")
        if labels[sample_id] is not None and not isinstance(labels[sample_id], str):
            raise ValueError(f"label for {sample_id!r} must be str or None")
        if not isinstance(groups[sample_id], str) or not groups[sample_id]:
            raise ValueError(f"group for {sample_id!r} must be a nonempty string")


def _counts(values: Mapping[str, str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values.values():
        result[value] = result.get(value, 0) + 1
    return result


def _source_digests(ids: Sequence[str], labels: Mapping[str, str | None], groups: Mapping[str, str]) -> dict[str, str]:
    return {
        "ids": _sha(list(ids)),
        "annotations": _sha({sample_id: labels[sample_id] for sample_id in ids}),
        "groups": _sha({sample_id: groups[sample_id] for sample_id in ids}),
    }


def _validate_roles(
    roles: Mapping[str, Role], labels: Mapping[str, str | None], groups: Mapping[str, str]
) -> None:
    grouped_roles: dict[str, set[Role]] = {}
    for sample_id, role in roles.items():
        if role not in _ROLES:
            raise ValueError(f"invalid role {role!r}")
        grouped_roles.setdefault(groups[sample_id], set()).add(role)
    if leaked := sorted(group for group, assigned in grouped_roles.items() if len(assigned) != 1):
        raise ValueError(f"group spans multiple roles: {leaked[:5]}")
    for role in _ROLES:
        role_labels = {labels[sample_id] is not None for sample_id, assigned in roles.items() if assigned == role}
        if role_labels != {False, True}:
            raise ValueError(f"{role} role lacks both classes")


def build_augmented_internal_split(
    ids: Sequence[str], labels: Mapping[str, str | None], groups: Mapping[str, str], *, seed: int = _SEED
) -> AugmentedInternalSplitManifest:
    """Build the fixed 14/3/3-fold Dataset-A role assignment."""
    if seed != _SEED:
        raise ValueError(f"seed must be {_SEED}")
    _validate_inputs(ids, labels, groups)
    if len(ids) < _N_SPLITS or len(set(groups.values())) < _N_SPLITS:
        raise ValueError(f"at least {_N_SPLITS} samples and groups are required")
    target = np.asarray([labels[sample_id] is not None for sample_id in ids], dtype=np.int64)
    if set(target.tolist()) != {0, 1}:
        raise ValueError("both target classes are required")
    group_array = np.asarray([groups[sample_id] for sample_id in ids], dtype=object)
    splitter = StratifiedGroupKFold(n_splits=_N_SPLITS, shuffle=True, random_state=_SEED)
    folds = list(splitter.split(np.zeros((len(ids), 1)), target, group_array))
    assigned = np.full(len(ids), -1, dtype=np.int64)
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        if set(target[np.asarray(train_indices)].tolist()) != {0, 1}:
            raise ValueError(f"fold {fold_index} train partition lacks both classes")
        if set(target[np.asarray(test_indices)].tolist()) != {0, 1}:
            raise ValueError(f"fold {fold_index} test partition lacks both classes")
        if np.any(assigned[np.asarray(test_indices)] >= 0):
            raise ValueError("sample assigned to multiple folds")
        assigned[np.asarray(test_indices)] = fold_index
    if np.any(assigned < 0):
        raise ValueError("some IDs were not assigned to a fold")
    roles: dict[str, Role] = {
        sample_id: _FOLD_TO_ROLE[fold] for sample_id, fold in zip(ids, assigned.tolist())
    }
    _validate_roles(roles, labels, groups)
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "seed": _SEED,
        "roles_by_id": {sample_id: roles[sample_id] for sample_id in ids},
        "groups_by_id": {sample_id: groups[sample_id] for sample_id in ids},
        "role_counts": _counts(roles),
        "group_counts": _counts(groups),
        "source_digests": _source_digests(ids, labels, groups),
    }
    digest = manifest_sha256(payload)
    return AugmentedInternalSplitManifest(
        schema_version=_SCHEMA_VERSION, seed=_SEED, roles_by_id=payload["roles_by_id"],
        groups_by_id=payload["groups_by_id"], source_digests=payload["source_digests"],
        manifest_sha256=digest,
    )


def _serial_form(manifest: AugmentedInternalSplitManifest) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": manifest.schema_version,
        "seed": manifest.seed,
        "roles_by_id": dict(manifest.roles_by_id),
        "groups_by_id": dict(manifest.groups_by_id),
        "role_counts": _counts(manifest.roles_by_id),
        "group_counts": _counts(manifest.groups_by_id),
        "source_digests": dict(manifest.source_digests),
    }
    data["manifest_sha256"] = manifest_sha256(data)
    return data


def write_augmented_internal_split(path: Path, manifest: AugmentedInternalSplitManifest) -> None:
    data = _serial_form(manifest)
    if data["manifest_sha256"] != manifest.manifest_sha256:
        raise ValueError("manifest digest mismatch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(data) + b"\n")


def load_augmented_internal_split(path: Path, expected_ids: Sequence[str]) -> AugmentedInternalSplitManifest:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid split manifest: {exc}") from exc
    if not isinstance(data, dict) or set(data) != _SERIAL_KEYS:
        raise ValueError("split manifest has invalid keys")
    if data["schema_version"] != _SCHEMA_VERSION or data["seed"] != _SEED:
        raise ValueError("split manifest schema or seed is invalid")
    if data["manifest_sha256"] != manifest_sha256(data):
        raise ValueError("split manifest digest mismatch")
    roles, groups, source_digests = data["roles_by_id"], data["groups_by_id"], data["source_digests"]
    if not isinstance(roles, dict) or not isinstance(groups, dict) or set(roles) != set(expected_ids) or set(groups) != set(expected_ids):
        raise ValueError("split manifest IDs do not exactly match expected IDs")
    if not isinstance(source_digests, dict) or set(source_digests) != _SOURCE_KEYS or any(not isinstance(value, str) or not _HEX64.fullmatch(value) for value in source_digests.values()):
        raise ValueError("split manifest source digests are invalid")
    if any(role not in _ROLES for role in roles.values()) or any(not isinstance(group, str) or not group for group in groups.values()):
        raise ValueError("split manifest role or group is invalid")
    if data["role_counts"] != _counts(roles) or data["group_counts"] != _counts(groups):
        raise ValueError("split manifest counts are invalid")
    grouped_roles: dict[str, set[str]] = {}
    for sample_id, group in groups.items():
        grouped_roles.setdefault(group, set()).add(roles[sample_id])
    if any(len(value) != 1 for value in grouped_roles.values()):
        raise ValueError("group spans multiple roles")
    return AugmentedInternalSplitManifest(
        schema_version=data["schema_version"], seed=data["seed"], roles_by_id=roles,
        groups_by_id=groups, source_digests=source_digests, manifest_sha256=data["manifest_sha256"],
    )
