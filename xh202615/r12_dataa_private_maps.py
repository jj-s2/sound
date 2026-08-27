"""Recreate private Dataset-A label/group maps from raw JSONL and frozen groups."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .data import FIELD_ALIASES


def _first_present(row: dict[str, object], canonical: str) -> object | None:
    for key in FIELD_ALIASES[canonical]:
        if key in row:
            return row[key]
    return None


@dataclass(frozen=True)
class PrivateMapSummary:
    labels_output: Path
    groups_output: Path
    count: int
    labels_sha256: str
    groups_sha256: str


def _raw_labels(dataset_root: Path) -> dict[str, str | None]:
    labels: dict[str, str | None] = {}
    for split in ("pos", "neg"):
        path = dataset_root / f"{split}.jsonl"
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), (str, int)):
                raise ValueError(f"{path}:{number} has invalid id")
            sample_id = str(row["id"])
            if sample_id in labels:
                raise ValueError(f"raw Dataset-A has duplicate ID {sample_id!r}")
            label = _first_present(row, "label")
            if label is not None and not isinstance(label, str):
                raise ValueError(f"{path}:{number} label must be string or null")
            if split == "pos" and label is None:
                raise ValueError(f"{path}:{number} positive row has no label")
            if split == "neg" and label is not None:
                raise ValueError(f"{path}:{number} negative row has a label")
            labels[sample_id] = label
    return labels


def _frozen_groups(path: Path, raw_labels: dict[str, str | None]) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("group manifest rows must be a list")
    groups: dict[str, str] = {}
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not isinstance(row.get("id"), (str, int)):
            raise ValueError(f"group manifest row {number} has invalid id")
        sample_id = str(row["id"])
        component = row.get("wake_component")
        if sample_id in groups or not isinstance(component, str) or not component:
            raise ValueError(f"group manifest row {number} has invalid group")
        if row.get("label") != raw_labels.get(sample_id):
            raise ValueError(f"group manifest label differs from raw Dataset-A for {sample_id}")
        groups[sample_id] = component
    if set(groups) != set(raw_labels):
        raise ValueError("group manifest IDs must exactly cover raw Dataset-A")
    return groups


def _write_private_mapping(path: Path, value: dict[str, object]) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(rendered, encoding="utf-8", newline="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_private_maps(
    dataset_root: Path, group_manifest: Path, labels_output: Path, groups_output: Path
) -> PrivateMapSummary:
    """Write independent private maps without changing raw Dataset-A inputs."""
    labels_output, groups_output = Path(labels_output), Path(groups_output)
    if labels_output.resolve(strict=False) == groups_output.resolve(strict=False):
        raise ValueError("labels and groups outputs must differ")
    if labels_output.exists():
        raise FileExistsError(labels_output)
    if groups_output.exists():
        raise FileExistsError(groups_output)
    labels = _raw_labels(Path(dataset_root))
    groups = _frozen_groups(Path(group_manifest), labels)
    labels_digest = _write_private_mapping(labels_output, dict(sorted(labels.items())))
    groups_digest = _write_private_mapping(groups_output, dict(sorted(groups.items())))
    return PrivateMapSummary(labels_output, groups_output, len(labels), labels_digest, groups_digest)
