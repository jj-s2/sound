"""Tests for the augmented-R12 label-free canonical feature join."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sources(root: Path, *, mismatch: bool = False) -> tuple[Path, dict[str, Path]]:
    lineage = root / "lineage.jsonl"
    rows = [
        {"id": "train", "parent_id": "train", "augmentation_id": "original", "role": "train", "group": "wake-train", "source_split": "pos", "command_audio_sha256": "a" * 64, "wake_audio_sha256": "b" * 64, "wakeup_audio": "wake.wav", "command_audio": "cmd.wav", "parameters": {}},
        {"id": "train__aug_a", "parent_id": "train", "augmentation_id": "aug_a", "role": "train", "group": "wake-train", "source_split": "pos", "command_audio_sha256": "c" * 64, "wake_audio_sha256": "b" * 64, "wakeup_audio": "wake.wav", "command_audio": "cmd-a.wav", "parameters": {"speed": 0.95}},
        {"id": "validation", "parent_id": "validation", "augmentation_id": "original", "role": "validation", "group": "wake-val", "source_split": "neg", "command_audio_sha256": "d" * 64, "wake_audio_sha256": "e" * 64, "wakeup_audio": "wake-v.wav", "command_audio": "cmd-v.wav", "parameters": {}},
        {"id": "test", "parent_id": "test", "augmentation_id": "original", "role": "internal_test", "group": "wake-test", "source_split": "neg", "command_audio_sha256": "f" * 64, "wake_audio_sha256": "e" * 64, "wakeup_audio": "wake-t.wav", "command_audio": "cmd-t.wav", "parameters": {}},
    ]
    _write_jsonl(lineage, rows)
    paths = {name: root / f"{name}.jsonl" for name in ("fusion", "tse", "audio", "r3")}
    for name, path in paths.items():
        values = []
        for row in rows:
            digest = row["command_audio_sha256"]
            if mismatch and name == "r3" and row["id"] == "train__aug_a":
                digest = "z" * 64
            value = {"id": row["id"], "command_audio_sha256": digest}
            if name == "fusion":
                value["candidate_texts"] = {"primary": "主文本", "energy": "能量文本"}
            elif name == "tse":
                value["text"] = "TSE文本"
            elif name == "audio":
                value.update({"presence_score": 0.2, "enhanced_cosine": 0.3, "mixture_cosine": 0.4, "max_cosine": 0.4, "latency_ms": 5.0})
            else:
                value["recognition_text"] = "R3文本"
            values.append(value)
        _write_jsonl(path, values)
    paths["pvad_manifest"] = root / "pvad_manifest.json"
    pvad_audio = {row["id"]: {"wake_sha256": row["wake_audio_sha256"], "command_sha256": "z" * 64 if mismatch and row["id"] == "train__aug_a" else row["command_audio_sha256"]} for row in rows}
    paths["pvad_manifest"].write_text(json.dumps({"source": {"per_id_audio_sha256": pvad_audio}}), encoding="utf-8")
    return lineage, paths


def test_canonical_rejects_audio_digest_mismatch(tmp_path: Path) -> None:
    from xh202615.r12_dataa_canonical import build_augmented_canonical

    lineage, paths = _sources(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="command_audio_sha256"):
        build_augmented_canonical(lineage, paths["fusion"], paths["tse"], paths["audio"], paths["r3"], paths["pvad_manifest"], tmp_path / "canonical.jsonl")


def test_canonical_has_exact_legacy_schema_and_raw_val_test(tmp_path: Path) -> None:
    from xh202615.r12_dataa_canonical import build_augmented_canonical

    lineage, paths = _sources(tmp_path)
    output = tmp_path / "canonical.jsonl"
    summary = build_augmented_canonical(lineage, paths["fusion"], paths["tse"], paths["audio"], paths["r3"], paths["pvad_manifest"], output)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary.row_count == 4
    assert all(set(row) == {"id", "split", "r3_text", "primary_text", "energy_text", "tse_text", "audio_features", "source_digest"} for row in records)
    assert all("__aug_" not in row["id"] for row in records if row["split"] != "train")
    assert all("label" not in json.dumps(row, ensure_ascii=False).lower() for row in records)


def test_attestation_binds_original_source_audio_not_tse_enhanced_audio(tmp_path: Path) -> None:
    from xh202615.r12_dataa_canonical import attest_source_command_audio

    original = tmp_path / "derived-command.wav"
    enhanced = tmp_path / "tse-command.wav"
    original.write_bytes(b"derived-source-audio")
    enhanced.write_bytes(b"enhanced-audio-is-not-the-source")
    source_digest = __import__("hashlib").sha256(original.read_bytes()).hexdigest()
    lineage = tmp_path / "lineage.jsonl"
    _write_jsonl(lineage, [{
        "id": "sample", "parent_id": "sample", "augmentation_id": "original",
        "role": "train", "group": "wake", "source_split": "pos",
        "command_audio_sha256": source_digest, "wake_audio_sha256": "b" * 64,
        "wakeup_audio": "wake.wav", "command_audio": str(original), "parameters": {},
    }])
    source = tmp_path / "tse_asr.jsonl"
    _write_jsonl(source, [{
        "id": "sample", "original_command_audio": str(original),
        "command_audio": str(enhanced), "recognition_text": "文本",
    }])
    output = tmp_path / "tse_asr_attested.jsonl"

    summary = attest_source_command_audio(lineage, source, output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary.row_count == 1
    assert rows[0]["command_audio_sha256"] == source_digest


def test_attestation_rejects_source_audio_that_does_not_match_lineage(tmp_path: Path) -> None:
    from xh202615.r12_dataa_canonical import attest_source_command_audio

    expected = tmp_path / "expected.wav"
    different = tmp_path / "different.wav"
    expected.write_bytes(b"expected")
    different.write_bytes(b"different")
    lineage = tmp_path / "lineage.jsonl"
    _write_jsonl(lineage, [{
        "id": "sample", "parent_id": "sample", "augmentation_id": "original",
        "role": "train", "group": "wake", "source_split": "pos",
        "command_audio_sha256": __import__("hashlib").sha256(expected.read_bytes()).hexdigest(),
        "wake_audio_sha256": "b" * 64, "wakeup_audio": "wake.wav",
        "command_audio": str(expected), "parameters": {},
    }])
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"id": "sample", "command_audio": str(different)}])

    with pytest.raises(ValueError, match="does not match lineage"):
        attest_source_command_audio(lineage, source, tmp_path / "attested.jsonl")


def test_rebuild_cli_attest_subcommand_writes_digest_bound_source(tmp_path: Path) -> None:
    from scripts.r12_dataa_rebuild_features import main

    command = tmp_path / "command.wav"
    command.write_bytes(b"derived-command")
    digest = __import__("hashlib").sha256(command.read_bytes()).hexdigest()
    lineage = tmp_path / "lineage.jsonl"
    _write_jsonl(lineage, [{
        "id": "sample", "parent_id": "sample", "augmentation_id": "original",
        "role": "train", "group": "wake", "source_split": "pos",
        "command_audio_sha256": digest, "wake_audio_sha256": "b" * 64,
        "wakeup_audio": "wake.wav", "command_audio": str(command), "parameters": {},
    }])
    source, output = tmp_path / "source.jsonl", tmp_path / "attested.jsonl"
    _write_jsonl(source, [{"id": "sample", "command_audio": str(command), "text": "文本"}])

    assert main(["attest", "--lineage", str(lineage), "--source", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["command_audio_sha256"] == digest
