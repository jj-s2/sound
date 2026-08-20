"""Frozen R12 60/20/20 stratified group split manifest.

The split is produced once from a zero-feature StratifiedGroupKFold with
``n_splits=5, shuffle=True, random_state=20260807``. Folds 0/1/2 become the
train set, fold 3 becomes validation, and fold 4 becomes the held-out test set.
The manifest serializes only schema version, seed, IDs, role mapping, groups,
counts, and source digests; labels and reference text are never written.
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


_SCHEMA_VERSION = "r12_split_v1"
_FIXED_SEED = 20260807
_N_SPLITS = 5
_ROLES: tuple[str, ...] = ("train", "validation", "held_out_test")
_FOLD_TO_ROLE: tuple[Literal["train", "validation", "held_out_test"], ...] = (
    "train",
    "train",
    "train",
    "validation",
    "held_out_test",
)
_ALLOWED_SERIAL_KEYS = frozenset(
    {
        "schema_version",
        "seed",
        "roles_by_id",
        "groups_by_id",
        "role_counts",
        "group_counts",
        "source_digests",
        "manifest_sha256",
    }
)
_PRIVATE_FIELD_HINTS = frozenset(
    {"label", "reference", "target", "text", "recognition_text", "wakeup_text"}
)
_SOURCE_DIGEST_KEYS = frozenset({"ids", "annotations", "groups"})
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen.add(key)
        result[key] = value
    return result


@dataclass(frozen=True)
class R12SplitManifest:
    schema_version: str
    seed: int
    roles_by_id: Mapping[str, Literal["train", "validation", "held_out_test"]]
    groups_by_id: Mapping[str, str]
    source_digests: Mapping[str, str]
    manifest_sha256: str


def _ordered_id_mapping(
    ids_in_order: Sequence[str], mapping: Mapping[str, Any]
) -> dict[str, Any]:
    return {sid: mapping[sid] for sid in ids_in_order}


def _role_counts(
    ids_in_order: Sequence[str],
    roles_by_id: Mapping[str, Literal["train", "validation", "held_out_test"]],
) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "held_out_test": 0}
    for sid in ids_in_order:
        counts[roles_by_id[sid]] += 1
    return counts


def _group_counts(groups_by_id: Mapping[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in groups_by_id.values():
        counts[group] = counts.get(group, 0) + 1
    return counts


def _compute_source_digests(
    ids_in_order: Sequence[str],
    labels: Mapping[str, str | None],
    groups: Mapping[str, str],
) -> dict[str, str]:
    return {
        "ids": _sha256_hex(_canonical_json(list(ids_in_order))),
        "annotations": _sha256_hex(
            _canonical_json({sid: labels[sid] for sid in ids_in_order})
        ),
        "groups": _sha256_hex(_canonical_json({sid: groups[sid] for sid in ids_in_order})),
    }


def _manifest_payload(manifest_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifest_dict.items() if k != "manifest_sha256"}


def _compute_manifest_sha256(manifest_dict: Mapping[str, Any]) -> str:
    return _sha256_hex(_canonical_json(_manifest_payload(manifest_dict)))


def _build_roles_by_id(
    ids_in_order: Sequence[str],
    labels: Mapping[str, str | None],
    groups: Mapping[str, str],
) -> dict[str, Literal["train", "validation", "held_out_test"]]:
    n_samples = len(ids_in_order)
    if n_samples < _N_SPLITS:
        raise ValueError(f"need at least {_N_SPLITS} samples, got {n_samples}")

    y = np.array([0 if labels[sid] is None else 1 for sid in ids_in_order], dtype=np.int64)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("both target classes are required")

    group_array = np.array([groups[sid] for sid in ids_in_order], dtype=object)
    if len(set(group_array.tolist())) < _N_SPLITS:
        raise ValueError(f"need at least {_N_SPLITS} groups")

    X = np.zeros((n_samples, 1), dtype=np.float64)
    splitter = StratifiedGroupKFold(
        n_splits=_N_SPLITS, shuffle=True, random_state=_FIXED_SEED
    )
    folds = list(splitter.split(X, y, group_array))
    if len(folds) != _N_SPLITS:
        raise ValueError(f"expected {_N_SPLITS} folds, got {len(folds)}")

    fold_assignments = np.full(n_samples, -1, dtype=np.int64)
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        test_indices = np.asarray(test_indices)
        train_indices = np.asarray(train_indices)
        if np.unique(y[test_indices]).size != 2:
            raise ValueError(f"fold {fold_index} test set lacks both classes")
        if np.unique(y[train_indices]).size != 2:
            raise ValueError(f"fold {fold_index} train set lacks both classes")
        if np.any(fold_assignments[test_indices] != -1):
            raise ValueError("sample assigned to multiple folds")
        fold_assignments[test_indices] = fold_index

    if np.any(fold_assignments < 0):
        raise ValueError("not all samples were assigned to a fold")

    roles_by_id: dict[str, Literal["train", "validation", "held_out_test"]] = {}
    for sid, fold_index in zip(ids_in_order, fold_assignments.tolist()):
        roles_by_id[sid] = _FOLD_TO_ROLE[fold_index]

    for role in _ROLES:
        role_ids = [sid for sid, r in roles_by_id.items() if r == role]
        role_classes = {labels[sid] is not None for sid in role_ids}
        if role_classes != {True, False}:
            raise ValueError(f"role {role} lacks both classes")

    role_groups: dict[str, set[str]] = {role: set() for role in _ROLES}
    for sid, role in roles_by_id.items():
        role_groups[role].add(groups[sid])

    group_role_count: dict[str, int] = {}
    for role, role_group_set in role_groups.items():
        for group in role_group_set:
            group_role_count[group] = group_role_count.get(group, 0) + 1
    leaked = [group for group, count in group_role_count.items() if count > 1]
    if leaked:
        raise ValueError(f"groups span multiple roles: {sorted(leaked)[:5]}")

    return roles_by_id


def _validate_inputs(
    ids_in_order: Sequence[str],
    labels: Mapping[str, str | None],
    groups: Mapping[str, str],
) -> None:
    ids_set = set(ids_in_order)
    if len(ids_set) != len(ids_in_order):
        raise ValueError("duplicate IDs in ids_in_order")

    if set(labels) != ids_set:
        raise ValueError("labels keys do not match IDs exactly")
    if set(groups) != ids_set:
        raise ValueError("groups keys do not match IDs exactly")

    for sid in ids_in_order:
        label = labels[sid]
        if label is not None and not isinstance(label, str):
            raise ValueError(f"label for id {sid!r} must be str or None")
        group = groups[sid]
        if not isinstance(group, str) or group == "":
            raise ValueError(f"empty or invalid group for id {sid!r}")


def build_r12_split(
    ids_in_order: Sequence[str],
    labels: Mapping[str, str | None],
    groups: Mapping[str, str],
    *,
    seed: int = _FIXED_SEED,
) -> R12SplitManifest:
    """Build the frozen R12 stratified group split manifest."""

    if seed != _FIXED_SEED:
        raise ValueError(f"seed must be {_FIXED_SEED}, got {seed}")

    _validate_inputs(ids_in_order, labels, groups)
    roles_by_id = _build_roles_by_id(ids_in_order, labels, groups)
    source_digests = _compute_source_digests(ids_in_order, labels, groups)

    manifest_dict: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "seed": seed,
        "roles_by_id": _ordered_id_mapping(ids_in_order, roles_by_id),
        "groups_by_id": _ordered_id_mapping(ids_in_order, groups),
        "role_counts": _role_counts(ids_in_order, roles_by_id),
        "group_counts": _group_counts(groups),
        "source_digests": source_digests,
    }
    manifest_dict["manifest_sha256"] = _compute_manifest_sha256(manifest_dict)

    return R12SplitManifest(
        schema_version=manifest_dict["schema_version"],
        seed=manifest_dict["seed"],
        roles_by_id=manifest_dict["roles_by_id"],
        groups_by_id=manifest_dict["groups_by_id"],
        source_digests=manifest_dict["source_digests"],
        manifest_sha256=manifest_dict["manifest_sha256"],
    )


def write_r12_split(path: Path, manifest: R12SplitManifest) -> None:
    """Write ``manifest`` to ``path`` as deterministic canonical JSON bytes."""

    manifest_dict: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "seed": manifest.seed,
        "roles_by_id": dict(manifest.roles_by_id),
        "groups_by_id": dict(manifest.groups_by_id),
        "role_counts": _role_counts(
            list(manifest.roles_by_id.keys()), manifest.roles_by_id
        ),
        "group_counts": _group_counts(manifest.groups_by_id),
        "source_digests": dict(manifest.source_digests),
        "manifest_sha256": manifest.manifest_sha256,
    }
    expected_sha256 = _compute_manifest_sha256(manifest_dict)
    if manifest.manifest_sha256 != expected_sha256:
        raise ValueError("manifest SHA-256 does not match recomputed payload")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(manifest_dict) + b"\n")


def load_r12_split(path: Path, expected_ids: Sequence[str]) -> R12SplitManifest:
    """Load and validate an R12 split manifest from ``path``."""

    path = Path(path)
    try:
        data = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")

    extra_keys = set(data) - _ALLOWED_SERIAL_KEYS
    if extra_keys:
        for key in extra_keys:
            lower = key.lower()
            if any(hint in lower for hint in _PRIVATE_FIELD_HINTS):
                raise ValueError(f"manifest contains private field {key!r}")
        raise ValueError(f"manifest contains unexpected keys: {sorted(extra_keys)}")

    if data.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version {data.get('schema_version')!r}, "
            f"expected {_SCHEMA_VERSION!r}"
        )
    if data.get("seed") != _FIXED_SEED:
        raise ValueError(
            f"unsupported seed {data.get('seed')!r}, expected {_FIXED_SEED}"
        )

    roles_by_id = data.get("roles_by_id")
    groups_by_id = data.get("groups_by_id")
    if not isinstance(roles_by_id, dict) or not isinstance(groups_by_id, dict):
        raise ValueError("roles_by_id and groups_by_id must be mappings")

    loaded_ids = set(roles_by_id)
    expected_set = set(expected_ids)
    if len(expected_ids) != len(expected_set):
        raise ValueError("expected_ids contains duplicates")
    if loaded_ids != expected_set:
        missing = expected_set - loaded_ids
        extra = loaded_ids - expected_set
        raise ValueError(f"ID mismatch: missing={sorted(missing)}, extra={sorted(extra)}")

    if set(groups_by_id) != expected_set:
        missing = expected_set - set(groups_by_id)
        extra = set(groups_by_id) - expected_set
        raise ValueError(
            f"group ID mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )

    for sid in expected_ids:
        role = roles_by_id[sid]
        if role not in _ROLES:
            raise ValueError(f"invalid role {role!r} for id {sid!r}")

    group_roles: dict[str, set[str]] = {}
    for sid in expected_ids:
        group = groups_by_id.get(sid)
        if not isinstance(group, str) or group == "":
            raise ValueError(f"missing or empty group for id {sid!r}")
        group_roles.setdefault(group, set()).add(roles_by_id[sid])
    leaked = [group for group, roles in group_roles.items() if len(roles) > 1]
    if leaked:
        raise ValueError(f"groups span multiple roles: {sorted(leaked)[:5]}")

    role_counts = data.get("role_counts")
    if not isinstance(role_counts, dict):
        raise ValueError("role_counts must be a mapping")
    expected_role_counts = _role_counts(expected_ids, roles_by_id)
    if role_counts != expected_role_counts:
        raise ValueError(
            f"role_counts mismatch: stored={role_counts}, expected={expected_role_counts}"
        )

    group_counts = data.get("group_counts")
    if not isinstance(group_counts, dict):
        raise ValueError("group_counts must be a mapping")
    expected_group_counts = _group_counts(groups_by_id)
    if group_counts != expected_group_counts:
        raise ValueError(
            f"group_counts mismatch: stored={group_counts}, expected={expected_group_counts}"
        )

    source_digests = data.get("source_digests")
    if not isinstance(source_digests, dict):
        raise ValueError("source_digests must be a mapping")
    if set(source_digests) != _SOURCE_DIGEST_KEYS:
        raise ValueError(
            f"source_digests must contain exactly {_SOURCE_DIGEST_KEYS}, got {set(source_digests)}"
        )
    for key, digest in source_digests.items():
        if not isinstance(digest, str) or not _HEX64_RE.match(digest):
            raise ValueError(f"source_digests[{key!r}] is not a 64-char lowercase hex SHA-256")

    manifest_sha256 = data.get("manifest_sha256")
    if not isinstance(manifest_sha256, str):
        raise ValueError("manifest_sha256 must be a string")
    recomputed = _compute_manifest_sha256(data)
    if manifest_sha256 != recomputed:
        raise ValueError("manifest SHA-256 mismatch")

    return R12SplitManifest(
        schema_version=data["schema_version"],
        seed=data["seed"],
        roles_by_id=roles_by_id,
        groups_by_id=groups_by_id,
        source_digests=source_digests,
        manifest_sha256=manifest_sha256,
    )
