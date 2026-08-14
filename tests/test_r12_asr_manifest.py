"""Tests for the private Dataset-A-train ASR manifest builder (R12 M0)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xh202615.r12_asr_manifest import prepare_asr_manifests


def _row(
    sample_id: str, parent_id: str, augmentation_id: str, role: str,
    source_split: str, command_audio: str,
) -> dict[str, object]:
    return {
        "id": sample_id,
        "parent_id": parent_id,
        "augmentation_id": augmentation_id,
        "role": role,
        "group": f"wake-{parent_id}",
        "source_split": source_split,
        "command_audio_sha256": "a" * 64,
        "wake_audio_sha256": "b" * 64,
        "wakeup_audio": f"{source_split}/wake-{sample_id}.wav",
        "command_audio": command_audio,
        "parameters": {},
    }


def _standard_lineage() -> list[dict[str, object]]:
    return [
        _row("p", "p", "original", "train", "pos", "pos/cmd-p.wav"),
        _row("p__aug_a", "p", "aug_a", "train", "pos", "pos/cmd-p__aug_a.wav"),
        _row("p__aug_b", "p", "aug_b", "train", "pos", "pos/cmd-p__aug_b.wav"),
        _row("n", "n", "original", "train", "neg", "neg/cmd-n.wav"),
        _row("v", "v", "original", "validation", "pos", "pos/cmd-v.wav"),
    ]


def _write_lineage(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "lineage.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _write_labels(tmp_path: Path, labels: dict[str, str | None]) -> Path:
    path = tmp_path / "train_labels.json"
    path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    return path


def _read_rows(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_asr_manifests_inherits_train_positive_labels_only(tmp_path: Path) -> None:
    lineage_path = _write_lineage(tmp_path, _standard_lineage())
    train_labels_path = _write_labels(tmp_path, {"p": "开灯", "n": None})
    summary = prepare_asr_manifests(lineage_path, train_labels_path, tmp_path / "m0")
    rows = _read_rows(summary.train_jsonl)
    assert [row["key"] for row in rows] == ["p", "p__aug_a", "p__aug_b"]
    assert [row["target"] for row in rows] == ["开灯", "开灯", "开灯"]


def test_public_manifest_summary_cannot_contain_private_targets_or_paths(tmp_path: Path) -> None:
    lineage_path = _write_lineage(tmp_path, _standard_lineage())
    train_labels_path = _write_labels(tmp_path, {"p": "开灯", "n": None})
    summary = prepare_asr_manifests(lineage_path, train_labels_path, tmp_path / "m0")
    rendered = summary.public_summary.read_text(encoding="utf-8")
    assert "开灯" not in rendered
    assert "pos/cmd-p.wav" not in rendered


def test_non_train_lineage_id_in_train_labels_is_rejected(tmp_path: Path) -> None:
    lineage_path = _write_lineage(tmp_path, _standard_lineage())
    # "v" is a validation parent; its ID must never be accepted as a train parent.
    train_labels_path = _write_labels(tmp_path, {"p": "开灯", "n": None, "v": "打开空调"})
    with pytest.raises(ValueError, match="exactly cover original train parents"):
        prepare_asr_manifests(lineage_path, train_labels_path, tmp_path / "m0")


def test_missing_train_parent_label_fails(tmp_path: Path) -> None:
    lineage_path = _write_lineage(tmp_path, _standard_lineage())
    train_labels_path = _write_labels(tmp_path, {"n": None})  # "p" is missing
    with pytest.raises(ValueError, match="no train parent label"):
        prepare_asr_manifests(lineage_path, train_labels_path, tmp_path / "m0")


def test_null_and_empty_train_labels_are_excluded(tmp_path: Path) -> None:
    lineage = _standard_lineage()
    lineage.append(_row("e", "e", "original", "train", "pos", "pos/cmd-e.wav"))
    lineage_path = _write_lineage(tmp_path, lineage)
    train_labels_path = _write_labels(tmp_path, {"p": "开灯", "n": None, "e": ""})
    summary = prepare_asr_manifests(lineage_path, train_labels_path, tmp_path / "m0")
    rows = _read_rows(summary.train_jsonl)
    assert [row["key"] for row in rows] == ["p", "p__aug_a", "p__aug_b"]


def test_normalize_asr_target_applies_nfkc_then_strips_rich_tags() -> None:
    from xh202615.r12_asr_manifest import normalize_asr_target

    assert normalize_asr_target("ＡＢＣ<|zh|>") == "ABC"
    assert normalize_asr_target("  打开空调 <|en|>  ") == "打开空调"
    with pytest.raises(ValueError, match="empty after normalization"):
        normalize_asr_target("<|zh|>")


def test_manifest_target_applies_nfkc_and_rich_tag_removal(tmp_path: Path) -> None:
    lineage_path = _write_lineage(tmp_path, _standard_lineage())
    train_labels_path = _write_labels(tmp_path, {"p": "ＡＢＣ<|zh|>", "n": None})
    summary = prepare_asr_manifests(lineage_path, train_labels_path, tmp_path / "m0")
    rows = _read_rows(summary.train_jsonl)
    assert [row["target"] for row in rows] == ["ABC", "ABC", "ABC"]


def test_inner_valid_parent_ids_move_all_children_to_private_valid_jsonl(tmp_path: Path) -> None:
    lineage_path = _write_lineage(tmp_path, _standard_lineage())
    train_labels_path = _write_labels(tmp_path, {"p": "开灯", "n": None})
    summary = prepare_asr_manifests(
        lineage_path, train_labels_path, tmp_path / "m0", inner_valid_parent_ids=["p"],
    )
    valid_rows = _read_rows(summary.inner_valid_jsonl)
    assert [row["key"] for row in valid_rows] == ["p", "p__aug_a", "p__aug_b"]
    assert [row["target"] for row in valid_rows] == ["开灯", "开灯", "开灯"]
    assert _read_rows(summary.train_jsonl) == []
    assert summary.train_rows == 0
    assert summary.inner_valid_rows == 3


def test_existing_output_root_fails_closed(tmp_path: Path) -> None:
    lineage_path = _write_lineage(tmp_path, _standard_lineage())
    train_labels_path = _write_labels(tmp_path, {"p": "开灯", "n": None})
    output_root = tmp_path / "m0"
    output_root.mkdir()
    with pytest.raises(FileExistsError):
        prepare_asr_manifests(lineage_path, train_labels_path, output_root)


def test_validation_error_writes_nothing(tmp_path: Path) -> None:
    lineage_path = _write_lineage(tmp_path, _standard_lineage())
    train_labels_path = _write_labels(tmp_path, {"p": "开灯", "n": None, "v": "打开空调"})
    output_root = tmp_path / "m0"
    with pytest.raises(ValueError):
        prepare_asr_manifests(lineage_path, train_labels_path, output_root)
    assert not output_root.exists()


def test_cli_prepares_private_manifests(tmp_path: Path) -> None:
    from scripts.r12_asr_prepare_manifest import main

    lineage_path = _write_lineage(tmp_path, _standard_lineage())
    train_labels_path = _write_labels(tmp_path, {"p": "开灯", "n": None})
    output_root = tmp_path / "m0"
    assert main([
        "--lineage", str(lineage_path),
        "--train-labels", str(train_labels_path),
        "--output-root", str(output_root),
    ]) == 0
    assert (output_root / "private" / "asr_train.jsonl").is_file()
    assert (output_root / "asr_manifest_summary.json").is_file()
