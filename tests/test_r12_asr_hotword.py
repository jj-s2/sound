"""Tests for the train-only private domain-hotword builder (R12 M0)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from xh202615.r12_asr_hotword import prepare_hotword_candidates, rank_hotword_phrases


def _write_labels(tmp_path: Path, labels: dict[str, str | None]) -> Path:
    path = tmp_path / "train_labels.json"
    path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_hotword_candidates_rank_by_frequency_then_unicode_order(tmp_path: Path) -> None:
    labels_path = _write_labels(tmp_path, {"a": "开灯", "b": "开灯", "c": "关灯"})
    result = prepare_hotword_candidates(labels_path, {"a", "b", "c"}, tmp_path / "m0", capacities=(1, 2))
    private = json.loads(result.private_hotwords.read_text(encoding="utf-8"))
    assert private == {"1": "开灯", "2": "开灯 关灯"}


def test_public_hotword_summary_contains_digests_but_not_phrase_text(tmp_path: Path) -> None:
    labels_path = _write_labels(tmp_path, {"a": "开灯", "b": "开灯", "c": "关灯"})
    result = prepare_hotword_candidates(labels_path, {"a", "b", "c"}, tmp_path / "m0", capacities=(1,))
    rendered = result.public_summary.read_text(encoding="utf-8")
    assert "开灯" not in rendered
    assert "关灯" not in rendered


def test_public_summary_records_capacity_token_count_and_digests(tmp_path: Path) -> None:
    labels_path = _write_labels(tmp_path, {"a": "开灯", "b": "开灯", "c": "关灯"})
    result = prepare_hotword_candidates(labels_path, {"a", "b", "c"}, tmp_path / "m0", capacities=(1, 2))
    summary = json.loads(result.public_summary.read_text(encoding="utf-8"))
    assert summary["phrase_count"] == 2
    assert summary["source_labels_sha256"] == _sha256(labels_path.read_bytes())
    assert [entry["capacity"] for entry in summary["hotwords"]] == [1, 2]
    assert [entry["token_count"] for entry in summary["hotwords"]] == [1, 2]
    assert summary["hotwords"][0]["hotword_sha256"] == _sha256("开灯".encode("utf-8"))
    assert summary["hotwords"][1]["hotword_sha256"] == _sha256("开灯 关灯".encode("utf-8"))
    assert summary["summary_sha256"] == result.summary_sha256


def test_rank_hotword_phrases_orders_by_frequency_then_utf8() -> None:
    assert rank_hotword_phrases({"a": "开灯", "b": "开灯", "c": "关灯"}) == ("开灯", "关灯")
    # Equal frequency falls back to ascending UTF-8 byte order: 关 (U+5173) before 开 (U+5F00).
    assert rank_hotword_phrases({"a": "开灯", "b": "关灯"}) == ("关灯", "开灯")


def test_labels_must_exactly_cover_declared_train_parents(tmp_path: Path) -> None:
    labels_path = _write_labels(tmp_path, {"a": "开灯", "b": "关灯"})
    with pytest.raises(ValueError, match="exactly cover declared train parents"):
        prepare_hotword_candidates(labels_path, {"a", "b", "c"}, tmp_path / "m0", capacities=(1,))
    labels_path = _write_labels(tmp_path, {"a": "开灯", "b": "关灯", "c": "开灯", "extra": "打开空调"})
    with pytest.raises(ValueError, match="exactly cover declared train parents"):
        prepare_hotword_candidates(labels_path, {"a", "b", "c"}, tmp_path / "m0", capacities=(1,))


def test_null_and_whitespace_only_labels_are_excluded(tmp_path: Path) -> None:
    labels_path = _write_labels(tmp_path, {"a": "开灯", "b": None, "c": "   "})
    result = prepare_hotword_candidates(labels_path, {"a", "b", "c"}, tmp_path / "m0", capacities=(1,))
    private = json.loads(result.private_hotwords.read_text(encoding="utf-8"))
    assert private == {"1": "开灯"}
    assert result.phrase_count == 1


def test_duplicate_zero_and_negative_capacities_fail(tmp_path: Path) -> None:
    labels_path = _write_labels(tmp_path, {"a": "开灯", "b": "开灯", "c": "关灯"})
    with pytest.raises(ValueError, match="duplicate capacity"):
        prepare_hotword_candidates(labels_path, {"a", "b", "c"}, tmp_path / "m0", capacities=(1, 1))
    with pytest.raises(ValueError, match="positive integer"):
        prepare_hotword_candidates(labels_path, {"a", "b", "c"}, tmp_path / "m0", capacities=(0,))
    with pytest.raises(ValueError, match="positive integer"):
        prepare_hotword_candidates(labels_path, {"a", "b", "c"}, tmp_path / "m0", capacities=(-1,))


def test_capacity_exceeding_ranked_phrase_count_fails(tmp_path: Path) -> None:
    labels_path = _write_labels(tmp_path, {"a": "开灯", "b": "开灯", "c": "关灯"})
    with pytest.raises(ValueError, match="exceeds ranked phrase count"):
        prepare_hotword_candidates(labels_path, {"a", "b", "c"}, tmp_path / "m0", capacities=(3,))


def test_existing_output_root_fails_closed(tmp_path: Path) -> None:
    labels_path = _write_labels(tmp_path, {"a": "开灯", "b": "开灯", "c": "关灯"})
    output_root = tmp_path / "m0"
    output_root.mkdir()
    with pytest.raises(FileExistsError):
        prepare_hotword_candidates(labels_path, {"a", "b", "c"}, output_root, capacities=(1,))


def test_unicode_normalization_applies_before_counting(tmp_path: Path) -> None:
    labels_path = _write_labels(tmp_path, {"a": "ＡＢＣ<|zh|>", "b": "ABC", "c": "开灯"})
    result = prepare_hotword_candidates(labels_path, {"a", "b", "c"}, tmp_path / "m0", capacities=(1, 2))
    private = json.loads(result.private_hotwords.read_text(encoding="utf-8"))
    # Fullwidth "ＡＢＣ<|zh|>" NFKC-normalizes and strips the rich tag to "ABC", so "ABC" is the top phrase.
    assert private == {"1": "ABC", "2": "ABC 开灯"}
    assert result.phrase_count == 2


def test_validation_error_writes_nothing(tmp_path: Path) -> None:
    labels_path = _write_labels(tmp_path, {"a": "开灯", "b": "开灯"})
    output_root = tmp_path / "m0"
    with pytest.raises(ValueError):
        prepare_hotword_candidates(labels_path, {"a", "b", "c"}, output_root, capacities=(1,))
    assert not output_root.exists()


def test_cli_prepares_private_hotwords(tmp_path: Path) -> None:
    from scripts.r12_asr_prepare_hotwords import main

    labels_path = _write_labels(tmp_path, {"a": "开灯", "b": "开灯", "c": "关灯"})
    output_root = tmp_path / "m0"
    assert main([
        "--train-labels", str(labels_path),
        "--parent-ids", "a,b,c",
        "--output-root", str(output_root),
        "--capacities", "1,2",
    ]) == 0
    assert (output_root / "private" / "domain_hotwords.json").is_file()
    assert (output_root / "hotword_summary.json").is_file()
