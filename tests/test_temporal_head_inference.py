"""Tests for label-free temporal candidate inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import soundfile as sf


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    import numpy as np

    input_path, candidates = tmp_path / "input.jsonl", tmp_path / "candidates.jsonl"
    rows = []
    for sample_id in ("a", "b"):
        wake, command = tmp_path / f"wake-{sample_id}.wav", tmp_path / f"cmd-{sample_id}.wav"
        sf.write(wake, np.zeros(160, dtype=np.float32), 16_000)
        sf.write(command, np.full(160, 0.05, dtype=np.float32), 16_000)
        rows.append({"id": sample_id, "split": "pos", "wakeup_audio": str(wake), "command_audio": str(command)})
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    candidates.write_text("".join(json.dumps({"id": sample_id, "recognition_text": f"文本-{sample_id}"}) + "\n" for sample_id in ("a", "b")), encoding="utf-8")
    return input_path, candidates


def test_inference_reads_only_input_and_candidate_map(tmp_path: Path) -> None:
    from scripts.run_temporal_head_inference import run_inference

    input_path, candidate_path = _inputs(tmp_path)
    rows = run_inference(input_path, candidate_path, threshold=0.5, probability_for=lambda row: 0.75 if row.sample_id == "a" else 0.25)

    assert [row["id"] for row in rows] == ["a", "b"]
    assert rows[0]["recognition_text"] == "文本-a"
    assert rows[1]["recognition_text"] == ""
    assert rows[0]["command_audio_sha256"] == _sha(Path(json.loads(input_path.read_text().splitlines()[0])["command_audio"]))
    assert "label" not in json.dumps(rows, ensure_ascii=False).lower()


def test_missing_or_duplicate_candidate_ids_fail_closed(tmp_path: Path) -> None:
    from scripts.run_temporal_head_inference import run_inference

    input_path, candidate_path = _inputs(tmp_path)
    candidate_path.write_text(json.dumps({"id": "a", "recognition_text": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cover"):
        run_inference(input_path, candidate_path, threshold=0.5, probability_for=lambda row: 1.0)
    candidate_path.write_text("\n".join([json.dumps({"id": "a", "recognition_text": "a"})] * 2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        run_inference(input_path, candidate_path, threshold=0.5, probability_for=lambda row: 1.0)


def test_inference_combines_two_label_free_manifests_without_duplicate_ids(tmp_path: Path) -> None:
    from scripts.run_temporal_head_inference import run_inference_many

    first, candidates = _inputs(tmp_path)
    second = tmp_path / "second.jsonl"
    command, wake = tmp_path / "cmd-c.wav", tmp_path / "wake-c.wav"
    sf.write(command, [0.05] * 160, 16_000)
    sf.write(wake, [0.0] * 160, 16_000)
    second.write_text(json.dumps({"id": "c", "split": "neg", "wakeup_audio": str(wake), "command_audio": str(command)}) + "\n", encoding="utf-8")
    with candidates.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "c", "recognition_text": "文本-c"}, ensure_ascii=False) + "\n")

    rows = run_inference_many([first, second], candidates, threshold=0.5, probability_for=lambda _row: 1.0)

    assert [row["id"] for row in rows] == ["a", "b", "c"]
