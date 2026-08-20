"""Tests for role-scoped private Dataset-A label exports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _split(path: Path) -> None:
    from xh202615.r12_dataa_augmented_split import build_augmented_internal_split, write_augmented_internal_split

    ids, labels, groups = [], {}, {}
    for index in range(20):
        for suffix, label in (("p", f"文本-{index}"), ("n", None)):
            sample_id = f"{index:02d}-{suffix}"
            ids.append(sample_id)
            labels[sample_id] = label
            groups[sample_id] = f"wake-{index:02d}"
    write_augmented_internal_split(path, build_augmented_internal_split(ids, labels, groups))


def test_export_role_labels_is_exact_and_private(tmp_path: Path) -> None:
    from xh202615.r12_dataa_role_labels import export_role_labels

    split = tmp_path / "split.json"
    _split(split)
    manifest = json.loads(split.read_text(encoding="utf-8"))
    labels = {sample_id: (f"文本-{sample_id}" if sample_id.endswith("-p") else None) for sample_id in manifest["roles_by_id"]}
    labels_path, output = tmp_path / "labels.json", tmp_path / "train.json"
    labels_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    summary = export_role_labels(labels_path, split, "train", output)

    result = json.loads(output.read_text(encoding="utf-8"))
    assert summary.count == len(result)
    assert set(result) == {sample_id for sample_id, role in manifest["roles_by_id"].items() if role == "train"}
    assert all("role" not in row for row in result.values() if isinstance(row, dict))


def test_export_role_labels_rejects_wrong_coverage_or_existing_output(tmp_path: Path) -> None:
    from xh202615.r12_dataa_role_labels import export_role_labels

    split = tmp_path / "split.json"
    _split(split)
    manifest = json.loads(split.read_text(encoding="utf-8"))
    labels = {sample_id: None for sample_id in manifest["roles_by_id"]}
    labels_path, output = tmp_path / "labels.json", tmp_path / "validation.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    del labels[next(iter(labels))]
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly match"):
        export_role_labels(labels_path, split, "validation", output)

    labels = {sample_id: None for sample_id in manifest["roles_by_id"]}
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_role_labels(labels_path, split, "validation", output)
