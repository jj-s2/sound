"""Export exact role-scoped private labels from a complete private mapping."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .r12_dataa_augmented_split import load_augmented_internal_split


_ROLES = frozenset({"train", "validation", "internal_test"})


@dataclass(frozen=True)
class RoleLabelSummary:
    role: str
    output: Path
    count: int
    sha256: str


def _read_labels(path: Path) -> dict[str, str | None]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("private labels must be a JSON object")
    if any(not isinstance(key, str) or item is not None and not isinstance(item, str) for key, item in value.items()):
        raise ValueError("private labels must map IDs to strings or null")
    return value


def export_role_labels(
    labels_path: Path, split_path: Path, role: str, output: Path
) -> RoleLabelSummary:
    """Write one role's raw-ID labels without adding split metadata to it."""
    if role not in _ROLES:
        raise ValueError("role must be train, validation, or internal_test")
    labels = _read_labels(labels_path)
    split = load_augmented_internal_split(split_path, list(labels))
    selected = {
        sample_id: labels[sample_id]
        for sample_id, assigned in split.roles_by_id.items()
        if assigned == role
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(selected, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    output.write_text(rendered, encoding="utf-8", newline="\n")
    return RoleLabelSummary(
        role=role,
        output=output,
        count=len(selected),
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    )
