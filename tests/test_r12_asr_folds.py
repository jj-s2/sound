"""Tests for the label-free R12 M0 ASR parent-group outer fold manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xh202615.r12_asr_folds import (
    build_asr_folds,
    load_asr_folds,
    role_for_outer_fold,
    write_asr_folds,
)
from xh202615.r12_dataa_augmentation import LineageRow


def _row(
    sample_id: str, parent_id: str, augmentation_id: str, role: str, source_split: str,
) -> LineageRow:
    return LineageRow(
        id=sample_id,
        parent_id=parent_id,
        augmentation_id=augmentation_id,
        role=role,
        group=f"wake-{parent_id}",
        source_split=source_split,
        command_audio_sha256="a" * 64,
        wake_audio_sha256="b" * 64,
        wakeup_audio=f"{source_split}/wake-{sample_id}.wav",
        command_audio=f"{source_split}/cmd-{sample_id}.wav",
        parameters={},
    )


def _lineage(*rows: LineageRow) -> dict[str, LineageRow]:
    return {row.id: row for row in rows}


def _standard_lineage() -> dict[str, LineageRow]:
    return _lineage(
        _row("p", "p", "original", "train", "pos"),
        _row("p__aug_a", "p", "aug_a", "train", "pos"),
        _row("p__aug_b", "p", "aug_b", "train", "pos"),
        _row("n", "n", "original", "train", "neg"),
        _row("v", "v", "original", "validation", "pos"),
    )


def _balanced_lineage() -> dict[str, LineageRow]:
    rows: list[LineageRow] = []
    for index in range(3):
        rows.append(_row(f"p{index}", f"p{index}", "original", "train", "pos"))
        rows.append(_row(f"p{index}__aug_a", f"p{index}", "aug_a", "train", "pos"))
        rows.append(_row(f"p{index}__aug_b", f"p{index}", "aug_b", "train", "pos"))
    for index in range(3):
        rows.append(_row(f"n{index}", f"n{index}", "original", "train", "neg"))
        rows.append(_row(f"n{index}__aug_a", f"n{index}", "aug_a", "train", "neg"))
        rows.append(_row(f"n{index}__aug_b", f"n{index}", "aug_b", "train", "neg"))
    return _lineage(*rows)


def _groups_by_fold(manifest) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for row in manifest.rows:
        result.setdefault(row.outer_fold, set()).add(row.group)
    return result


def test_fold_manifest_keeps_all_children_of_a_parent_in_one_fold() -> None:
    manifest = build_asr_folds(_standard_lineage(), fold_count=3, seed=20260814)
    assert len({manifest.fold_by_id[sid] for sid in ("p", "p__aug_a", "p__aug_b")}) == 1


def test_fold_manifest_includes_train_negatives_but_not_other_roles() -> None:
    manifest = build_asr_folds(_standard_lineage(), fold_count=3, seed=20260814)
    assert set(manifest.fold_by_id) == {"p", "p__aug_a", "p__aug_b", "n"}


def test_each_group_maps_to_a_single_fold() -> None:
    manifest = build_asr_folds(_balanced_lineage(), fold_count=3, seed=20260814)
    group_folds: dict[str, set[int]] = {}
    for row in manifest.rows:
        group_folds.setdefault(row.group, set()).add(row.outer_fold)
    assert group_folds
    assert all(len(folds) == 1 for folds in group_folds.values())


def test_stratified_folds_balance_source_split_when_both_classes_permit() -> None:
    manifest = build_asr_folds(_balanced_lineage(), fold_count=3, seed=20260814)
    groups_by_fold = _groups_by_fold(manifest)
    group_split = {row.group: row.source_split for row in manifest.rows}
    assert set(groups_by_fold) == {0, 1, 2}
    for fold in groups_by_fold:
        groups = groups_by_fold[fold]
        splits = {group_split[group] for group in groups}
        assert splits == {"pos", "neg"}
        assert len(groups) == 2


def test_mapping_is_deterministic_for_same_seed() -> None:
    lineage = _balanced_lineage()
    first = build_asr_folds(lineage, fold_count=3, seed=20260814)
    second = build_asr_folds(lineage, fold_count=3, seed=20260814)
    assert first.fold_by_id == second.fold_by_id
    assert first.manifest_sha256 == second.manifest_sha256


def test_invalid_fold_count_or_seed_fails() -> None:
    lineage = _standard_lineage()
    with pytest.raises(ValueError):
        build_asr_folds(lineage, fold_count=1)
    with pytest.raises(ValueError):
        build_asr_folds(lineage, fold_count=0)
    with pytest.raises(ValueError):
        build_asr_folds(lineage, fold_count=2.5)
    with pytest.raises(ValueError):
        build_asr_folds(lineage, seed=-1)


def test_manifest_roundtrip_preserves_folds_and_digest(tmp_path: Path) -> None:
    lineage = _balanced_lineage()
    manifest = build_asr_folds(lineage, fold_count=3, seed=20260814)
    path = tmp_path / "asr_folds.json"
    write_asr_folds(path, manifest)
    loaded = load_asr_folds(path, lineage)
    assert loaded.schema_version == manifest.schema_version
    assert loaded.seed == manifest.seed
    assert loaded.fold_count == manifest.fold_count
    assert loaded.fold_by_id == manifest.fold_by_id
    assert loaded.manifest_sha256 == manifest.manifest_sha256


def test_serialized_manifest_is_text_and_path_free(tmp_path: Path) -> None:
    lineage = _standard_lineage()
    manifest = build_asr_folds(lineage, fold_count=3, seed=20260814)
    path = tmp_path / "asr_folds.json"
    write_asr_folds(path, manifest)
    rendered = path.read_text(encoding="utf-8")
    assert ".wav" not in rendered
    assert "cmd-" not in rendered


def test_loader_rejects_tampered_child_fold(tmp_path: Path) -> None:
    lineage = _balanced_lineage()
    manifest = build_asr_folds(lineage, fold_count=3, seed=20260814)
    path = tmp_path / "asr_folds.json"
    write_asr_folds(path, manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data["rows"]:
        if row["id"] == "p0__aug_a":
            row["outer_fold"] = (row["outer_fold"] + 1) % 3
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="digest|SHA-256"):
        load_asr_folds(path, lineage)


def test_loader_rejects_non_train_id(tmp_path: Path) -> None:
    lineage = _standard_lineage()
    manifest = build_asr_folds(lineage, fold_count=3, seed=20260814)
    path = tmp_path / "asr_folds.json"
    write_asr_folds(path, manifest)
    # Re-mark the negative train parent as validation: the manifest still lists
    # it, so the loader must reject the non-train ID against expected lineage.
    tampered = _lineage(
        _row("p", "p", "original", "train", "pos"),
        _row("p__aug_a", "p", "aug_a", "train", "pos"),
        _row("p__aug_b", "p", "aug_b", "train", "pos"),
        _row("n", "n", "original", "validation", "neg"),
    )
    with pytest.raises(ValueError, match="train|non-train"):
        load_asr_folds(path, tampered)


def test_role_for_outer_fold_returns_holdout_only_for_selected_fold() -> None:
    manifest = build_asr_folds(_balanced_lineage(), fold_count=3, seed=20260814)
    sample_id = "p0"
    selected = manifest.fold_by_id[sample_id]
    other = (selected + 1) % manifest.fold_count
    assert role_for_outer_fold(manifest, sample_id, selected) == "holdout"
    assert role_for_outer_fold(manifest, sample_id, other) == "fit"


def test_cli_prepares_fold_manifest(tmp_path: Path) -> None:
    from scripts.r12_asr_prepare_folds import main

    lineage_path = tmp_path / "lineage.jsonl"
    lineage_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": row.id,
                    "parent_id": row.parent_id,
                    "augmentation_id": row.augmentation_id,
                    "role": row.role,
                    "group": row.group,
                    "source_split": row.source_split,
                    "command_audio_sha256": row.command_audio_sha256,
                    "wake_audio_sha256": row.wake_audio_sha256,
                    "wakeup_audio": row.wakeup_audio,
                    "command_audio": row.command_audio,
                    "parameters": row.parameters,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for row in _standard_lineage().values()
        ),
        encoding="utf-8",
    )
    output = tmp_path / "asr_folds.json"
    assert main(["--lineage", str(lineage_path), "--output", str(output)]) == 0
    assert output.is_file()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["fold_count"] == 3
    assert data["seed"] == 20260814
    assert len(data["rows"]) == 4
