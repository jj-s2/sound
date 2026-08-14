"""Private Dataset-A-train ASR manifest builder for R12 M0 Paraformer preparation.

The manifest carries ASR supervision only for train-role lineage rows whose
parent label is a nonempty string.  Private target text and audio paths are
written exclusively below ``output_root / "private"``, while the public summary
records IDs, counts, and SHA-256 digests only.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Mapping

from .r12_dataa_augmentation import LineageRow, load_lineage


_RICH_TAG_RE = re.compile(r"<\|[^>]*\|>")
_ARTIFACT_KIND = "r12_asr_manifest"
_SCHEMA_VERSION = "v1"
_AUGMENTATION_IDS = frozenset({"original", "aug_a", "aug_b"})
_PRIVATE_DIR = "private"
_TRAIN_FILENAME = "asr_train.jsonl"
_INNER_VALID_FILENAME = "asr_inner_valid.jsonl"
_SUMMARY_FILENAME = "asr_manifest_summary.json"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_jsonl(rows: list[dict[str, str]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


@dataclass(frozen=True)
class AsrManifestSummary:
    train_jsonl: Path
    inner_valid_jsonl: Path
    public_summary: Path
    train_rows: int
    inner_valid_rows: int
    manifest_sha256: str


def normalize_asr_target(value: str) -> str:
    cleaned = _RICH_TAG_RE.sub("", unicodedata.normalize("NFKC", value)).strip()
    if not cleaned:
        raise ValueError("ASR target is empty after normalization")
    return cleaned


def _read_train_labels(path: Path) -> dict[str, str | None]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("train labels must be a JSON object")
    if any(not isinstance(key, str) or value is not None and not isinstance(value, str) for key, value in raw.items()):
        raise ValueError("train labels must map IDs to strings or null")
    return raw


def _expand_train_labels(
    parent_labels: Mapping[str, str | None], lineage: Mapping[str, LineageRow]
) -> dict[str, str | None]:
    """Inherit private labels only from train-role parent samples (local semantics)."""
    result: dict[str, str | None] = {}
    for sample_id, row in lineage.items():
        if row.role != "train":
            continue
        if row.parent_id not in parent_labels:
            raise ValueError("train child has no train parent label")
        if row.augmentation_id not in _AUGMENTATION_IDS:
            raise ValueError("invalid train augmentation ID")
        result[sample_id] = parent_labels[row.parent_id]
    if set(parent_labels) != {
        row.parent_id for row in lineage.values() if row.role == "train" and row.augmentation_id == "original"
    }:
        raise ValueError("train labels must exactly cover original train parents")
    return result


def _private_row(row: LineageRow, target: str) -> dict[str, str]:
    return {
        "key": row.id,
        "source": row.command_audio,
        "target": target,
        "parent_id": row.parent_id,
        "augmentation_id": row.augmentation_id,
    }


def prepare_asr_manifests(
    lineage_path: Path, train_labels_path: Path, output_root: Path,
    *, inner_valid_parent_ids: Collection[str] = (),
) -> AsrManifestSummary:
    """Validate all inputs, stage private JSONL and public digest summary, then publish once."""
    lineage = load_lineage(lineage_path)
    parent_labels = _read_train_labels(train_labels_path)
    expanded = _expand_train_labels(parent_labels, lineage)

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    private_dir = output_root / _PRIVATE_DIR
    train_jsonl = private_dir / _TRAIN_FILENAME
    inner_valid_jsonl = private_dir / _INNER_VALID_FILENAME
    public_summary = output_root / _SUMMARY_FILENAME

    resolved_private = private_dir.resolve(strict=False)
    for private_path in (train_jsonl, inner_valid_jsonl):
        try:
            private_path.resolve(strict=False).relative_to(resolved_private)
        except ValueError as exc:
            raise ValueError("private manifest path escapes private directory") from exc

    inner_valid = set(inner_valid_parent_ids)
    train_rows: list[dict[str, str]] = []
    valid_rows: list[dict[str, str]] = []
    for sample_id, row in lineage.items():
        if row.role != "train":
            continue
        label = expanded[sample_id]
        if label is None or not label.strip():
            continue
        rendered = _private_row(row, normalize_asr_target(label))
        if row.parent_id in inner_valid:
            valid_rows.append(rendered)
        else:
            train_rows.append(rendered)

    private_dir.mkdir(parents=True, exist_ok=False)
    train_jsonl.write_text(_render_jsonl(train_rows), encoding="utf-8", newline="\n")
    inner_valid_jsonl.write_text(_render_jsonl(valid_rows), encoding="utf-8", newline="\n")

    summary_fields = {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "train_keys": [row["key"] for row in train_rows],
        "inner_valid_keys": [row["key"] for row in valid_rows],
        "train_rows": len(train_rows),
        "inner_valid_rows": len(valid_rows),
        "train_jsonl_sha256": _sha_file(train_jsonl),
        "inner_valid_jsonl_sha256": _sha_file(inner_valid_jsonl),
        "train_labels_sha256": _sha_file(train_labels_path),
        "lineage_sha256": _sha_file(lineage_path),
    }
    manifest_sha256 = hashlib.sha256(_canonical(summary_fields)).hexdigest()
    summary_fields["manifest_sha256"] = manifest_sha256
    public_summary.write_text(
        json.dumps(summary_fields, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return AsrManifestSummary(
        train_jsonl=train_jsonl,
        inner_valid_jsonl=inner_valid_jsonl,
        public_summary=public_summary,
        train_rows=len(train_rows),
        inner_valid_rows=len(valid_rows),
        manifest_sha256=manifest_sha256,
    )
