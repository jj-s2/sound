from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.r11_pvad_oracle_oof import (
    _REJECT_ALL,
    _reject_private,
    load_canonical_rows,
    main,
)
from tests.test_r11_pvad_oracle import _fixture


def _write_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    rows, labels, groups, cache, manifest = _fixture()
    canonical = root / "canonical.jsonl"
    canonical.write_text("".join(json.dumps({"id": row.id, "split": row.split, "r3_text": row.r3_text, "primary_text": row.primary_text, "energy_text": row.energy_text, "tse_text": row.tse_text, "audio_features": row.audio_features, "source_digest": row.source_digest}, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (root / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    (root / "groups.json").write_text(json.dumps(groups, ensure_ascii=False), encoding="utf-8")
    cache_root = root / "cache"
    cache_root.mkdir()
    (cache_root / "pvad_features.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in cache), encoding="utf-8")
    (cache_root / "pvad_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    (cache_root / "pvad_report.md").write_text("report", encoding="utf-8")
    return canonical, root / "labels.json", root / "groups.json", cache_root


def test_canonical_loader_rejects_private_fields_and_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"id": "a", "label": "secret"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact|label-free"):
        load_canonical_rows(path)
    path.write_text("\n".join([json.dumps({"id": "a", "split": "pos", "r3_text": "", "primary_text": "", "energy_text": "", "tse_text": "", "audio_features": {}, "source_digest": "a" * 64})] * 2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_canonical_rows(path)


def test_private_projection_and_reject_all_marker_are_safe() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        _reject_private({"nested": {"reference_text": "secret"}})
    assert _REJECT_ALL == "reject_all"


def test_cli_publishes_exact_five_files_and_no_dataset_feature_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    canonical, labels, groups, cache = _write_fixture(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "output"
    monkeypatch.chdir(tmp_path)
    arguments = ["--dataset-root", str(dataset), "--canonical-input-jsonl", str(canonical), "--labels", str(labels), "--groups", str(groups), "--cache-root", str(cache), "--output-root", str(output), "--bootstrap-count", "2", "--bootstrap-seed", "7"]
    assert main(arguments) == 0
    assert {path.name for path in output.iterdir()} == {"e2_manifest.json", "e2_oof_scores.jsonl", "e2_frontier.jsonl", "e2_summary.json", "e2_report.md"}
    assert not any(path.is_dir() for path in output.iterdir())
    manifest = json.loads((output / "e2_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "r11_e2_pvad_oracle"
    assert manifest["decision"] in {"continue_ranker", "consider_custom_pvad", "falsified_firered"}
    assert "label" not in (output / "e2_oof_scores.jsonl").read_text(encoding="utf-8").lower()
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    assert main(arguments) == 0
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first_bytes


def test_cli_rejects_non_frozen_split_seed(tmp_path: Path) -> None:
    canonical, labels, groups, cache = _write_fixture(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with pytest.raises(ValueError, match="frozen split seed"):
        main(["--dataset-root", str(dataset), "--canonical-input-jsonl", str(canonical), "--labels", str(labels), "--groups", str(groups), "--cache-root", str(cache), "--seed", "1"])
