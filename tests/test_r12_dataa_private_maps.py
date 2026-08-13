"""Tests for recreating private Dataset-A labels/groups from raw inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "datasetA"
    root.mkdir()
    (root / "pos.jsonl").write_text(json.dumps({
        "id": "p", "wakeup_audio": "pos/wake-p.wav", "command_audio": "pos/cmd-p.wav",
        "recognition_text": "打开空调",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "neg.jsonl").write_text(json.dumps({
        "id": "n", "wakeup_audio": "neg/wake-n.wav", "command_audio": "neg/cmd-n.wav",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    return root


def test_private_maps_follow_raw_labels_and_frozen_wake_groups(tmp_path: Path) -> None:
    from xh202615.r12_dataa_private_maps import build_private_maps

    raw = _raw_root(tmp_path)
    manifest = tmp_path / "groups.json"
    manifest.write_text(json.dumps({"rows": [
        {"id": "p", "label": "打开空调", "wake_component": "wake-1"},
        {"id": "n", "label": None, "wake_component": "wake-2"},
    ]}, ensure_ascii=False), encoding="utf-8")
    labels, groups = tmp_path / "private" / "labels.json", tmp_path / "private" / "groups.json"

    summary = build_private_maps(raw, manifest, labels, groups)

    assert summary.count == 2
    assert json.loads(labels.read_text(encoding="utf-8")) == {"n": None, "p": "打开空调"}
    assert json.loads(groups.read_text(encoding="utf-8")) == {"n": "wake-2", "p": "wake-1"}


def test_private_maps_reject_group_manifest_label_mismatch_and_overwrite(tmp_path: Path) -> None:
    from xh202615.r12_dataa_private_maps import build_private_maps

    raw = _raw_root(tmp_path)
    manifest = tmp_path / "groups.json"
    manifest.write_text(json.dumps({"rows": [
        {"id": "p", "label": "错误标签", "wake_component": "wake-1"},
        {"id": "n", "label": None, "wake_component": "wake-2"},
    ]}, ensure_ascii=False), encoding="utf-8")
    labels, groups = tmp_path / "labels.json", tmp_path / "groups.json.out"
    with pytest.raises(ValueError, match="label differs"):
        build_private_maps(raw, manifest, labels, groups)

    manifest.write_text(json.dumps({"rows": [
        {"id": "p", "label": "打开空调", "wake_component": "wake-1"},
        {"id": "n", "label": None, "wake_component": "wake-2"},
    ]}, ensure_ascii=False), encoding="utf-8")
    labels.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_private_maps(raw, manifest, labels, groups)


def test_private_maps_do_not_leave_labels_when_groups_destination_exists(tmp_path: Path) -> None:
    from xh202615.r12_dataa_private_maps import build_private_maps

    raw = _raw_root(tmp_path)
    manifest = tmp_path / "groups.json"
    manifest.write_text(json.dumps({"rows": [
        {"id": "p", "label": "打开空调", "wake_component": "wake-1"},
        {"id": "n", "label": None, "wake_component": "wake-2"},
    ]}, ensure_ascii=False), encoding="utf-8")
    labels, groups = tmp_path / "labels.json", tmp_path / "groups-output.json"
    groups.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_private_maps(raw, manifest, labels, groups)

    assert not labels.exists()
    assert groups.read_text(encoding="utf-8") == "preserve"


def test_private_map_cli_creates_separate_private_outputs(tmp_path: Path) -> None:
    from scripts.r12_dataa_prepare_private_maps import main

    raw = _raw_root(tmp_path)
    manifest = tmp_path / "groups.json"
    manifest.write_text(json.dumps({"rows": [
        {"id": "p", "label": "打开空调", "wake_component": "wake-1"},
        {"id": "n", "label": None, "wake_component": "wake-2"},
    ]}, ensure_ascii=False), encoding="utf-8")
    private = tmp_path / "private"

    assert main([
        "--dataset-root", str(raw), "--group-manifest", str(manifest),
        "--labels-output", str(private / "labels.json"),
        "--groups-output", str(private / "groups.json"),
    ]) == 0
    assert (private / "labels.json").is_file()
    assert (private / "groups.json").is_file()


def test_split_cli_uses_raw_dataset_ids_not_an_old_canonical_projection(tmp_path: Path) -> None:
    from scripts.r12_dataa_prepare_split import main

    raw = tmp_path / "datasetA"
    raw.mkdir()
    labels, groups = {}, {}
    for index in range(20):
        for suffix, label in (("p", f"文本-{index}"), ("n", None)):
            sample_id = f"{index:02d}-{suffix}"
            split = "pos" if label is not None else "neg"
            labels[sample_id], groups[sample_id] = label, f"wake-{index:02d}"
            with (raw / f"{split}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": sample_id, "wakeup_audio": f"{split}/wake-{sample_id}.wav", "command_audio": f"{split}/cmd-{sample_id}.wav"}) + "\n")
    labels_path, groups_path, output = tmp_path / "labels.json", tmp_path / "groups.json", tmp_path / "split.json"
    labels_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    groups_path.write_text(json.dumps(groups, ensure_ascii=False), encoding="utf-8")

    assert main(["--dataset-root", str(raw), "--labels", str(labels_path), "--groups", str(groups_path), "--output", str(output)]) == 0
    assert set(json.loads(output.read_text(encoding="utf-8"))["roles_by_id"]) == set(labels)
