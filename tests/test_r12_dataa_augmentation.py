"""Tests for deterministic train-only Dataset-A audio augmentation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from xh202615.r12_dataa_augmented_split import build_augmented_internal_split


def _raw_dataset(root: Path) -> tuple[list[str], dict[str, str | None], dict[str, str]]:
    ids: list[str] = []
    labels: dict[str, str | None] = {}
    groups: dict[str, str] = {}
    for split in ("pos", "neg"):
        (root / split).mkdir(parents=True)
    for index in range(20):
        group = f"wake-{index:02d}"
        for suffix, split, label in (("p", "pos", f"text-{index}"), ("n", "neg", None)):
            sample_id = f"{index:02d}-{suffix}"
            ids.append(sample_id)
            labels[sample_id] = label
            groups[sample_id] = group
            wake = Path(split) / f"wake-{sample_id}.wav"
            command = Path(split) / f"cmd-{sample_id}.wav"
            wave = np.linspace(-0.2, 0.2, 160, dtype=np.float32)
            sf.write(root / wake, wave, 16_000)
            sf.write(root / command, wave[::-1], 16_000)
            with (root / f"{split}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": sample_id, "wakeup_audio": str(wake), "command_audio": str(command)}) + "\n")
    return ids, labels, groups


def _children(rows, parent_id: str) -> set[str]:
    return {row.augmentation_id for row in rows.values() if row.parent_id == parent_id}


def test_only_train_parents_receive_two_children(tmp_path: Path) -> None:
    from xh202615.r12_dataa_augmentation import build_augmented_dataset, load_lineage

    raw = tmp_path / "raw"
    ids, labels, groups = _raw_dataset(raw)
    split = build_augmented_internal_split(ids, labels, groups)
    summary = build_augmented_dataset(raw, split, tmp_path / "derived")
    rows = load_lineage(summary.lineage_path)
    train_id = next(sid for sid, role in split.roles_by_id.items() if role == "train")
    validation_id = next(sid for sid, role in split.roles_by_id.items() if role == "validation")
    test_id = next(sid for sid, role in split.roles_by_id.items() if role == "internal_test")
    assert _children(rows, train_id) == {"original", "aug_a", "aug_b"}
    assert _children(rows, validation_id) == {"original"}
    assert _children(rows, test_id) == {"original"}


def test_same_seed_gives_same_lineage_and_safe_child_audio(tmp_path: Path) -> None:
    from xh202615.r12_dataa_augmentation import build_augmented_dataset, load_lineage

    raw = tmp_path / "raw"
    ids, labels, groups = _raw_dataset(raw)
    split = build_augmented_internal_split(ids, labels, groups)
    first = build_augmented_dataset(raw, split, tmp_path / "first")
    second = build_augmented_dataset(raw, split, tmp_path / "second")
    assert first.lineage_digest == second.lineage_digest
    child = next(row for row in load_lineage(first.lineage_path).values() if row.augmentation_id == "aug_a")
    waveform, rate = sf.read(first.dataset_root / child.command_audio, dtype="float32")
    assert rate == 16_000
    assert waveform.ndim == 1 and np.isfinite(waveform).all()
    assert np.abs(waveform).max() <= 10 ** (-1 / 20) + 1e-6


def test_lineage_rejects_a_child_outside_train_role(tmp_path: Path) -> None:
    from xh202615.r12_dataa_augmentation import LineageRow, validate_lineage

    rows = {
        "x__aug_a": LineageRow("x__aug_a", "x", "aug_a", "validation", "wake-x", "pos", "a" * 64, "b" * 64, "wake.wav", "cmd.wav", {}),
    }
    with np.testing.assert_raises_regex(ValueError, "train"):
        validate_lineage(rows)
