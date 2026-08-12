from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.test_r11_pvad_oracle import _fixture


def _write_r10_sources(root: Path) -> dict[str, Path]:
    rows, labels, groups, _, _ = _fixture()
    sources = {
        "candidate_fusion": root / "candidate_fusion.jsonl",
        "tse_asr": root / "tse_asr.jsonl",
        "audio_map": root / "audio_map.jsonl",
        "r3_predictions": root / "r3_predictions.jsonl",
        "group_manifest": root / "group_manifest.json",
    }
    sources["candidate_fusion"].write_text(
        "".join(json.dumps({
            "id": row.id,
            "recognition_text": row.r3_text,
            "candidate_texts": {"primary": row.primary_text, "energy": row.energy_text},
        }, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    sources["tse_asr"].write_text(
        "".join(json.dumps({"id": row.id, "recognition_text": row.tse_text}, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    sources["audio_map"].write_text(
        "".join(json.dumps({
            "id": row.id,
            "presence_score": row.audio_features["presence_score"],
            "enhanced_cosine": row.audio_features["enhanced_cosine"],
            "mixture_cosine": row.audio_features["mixture_cosine"],
            "max_cosine": row.audio_features["max_cosine"],
        }, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    sources["r3_predictions"].write_text(
        "".join(json.dumps({"id": row.id, "recognition_text": row.r3_text}, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    sources["group_manifest"].write_text(
        json.dumps({"rows": [
            {"id": row.id, "label": labels[row.id], "wake_component": groups[row.id]}
            for row in rows
        ]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return sources


@pytest.fixture(autouse=True)
def feasible_cli_fixture(monkeypatch):
    from tests.test_r12_strict_holdout import _patch_feasible_finite

    _patch_feasible_finite(monkeypatch)


def test_raw_r10_sources_rebuild_label_free_canonical(tmp_path: Path) -> None:
    from scripts.r12_prepare_split import build_canonical_input

    sources = _write_r10_sources(tmp_path)
    output = tmp_path / "canonical_input.jsonl"
    summary = build_canonical_input(*[sources[name] for name in (
        "candidate_fusion", "tse_asr", "audio_map", "r3_predictions", "group_manifest"
    )], output)
    assert summary["row_count"] == 20
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records
    assert all("label" not in record for record in records)
    assert all(set(record) == {
        "id", "split", "r3_text", "primary_text", "energy_text", "tse_text",
        "audio_features", "source_digest",
    } for record in records)
    original_bytes = output.read_bytes()
    manifest = json.loads(sources["group_manifest"].read_text(encoding="utf-8"))
    for item in manifest["rows"]:
        item["label"] = "mutated-label" if item["label"] is not None else None
    sources["group_manifest"].write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    build_canonical_input(*[sources[name] for name in (
        "candidate_fusion", "tse_asr", "audio_map", "r3_predictions", "group_manifest"
    )], output)
    assert output.read_bytes() == original_bytes
    assert set(summary["source_digests"]) == set(sources)


def test_cpu_cache_cli_help_does_not_import_neural_runtime() -> None:
    result = subprocess.run(
        ["F:\\XH-202615\\XH-202615\\.venv\\Scripts\\python.exe", "scripts\\cache_firered_pvad_features.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--ecapa-device" in result.stdout


def test_strict_selection_binds_optional_r10_source_digests(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main
    from tests.test_r12_strict_holdout import _select_args, _write_fixture

    paths = _write_fixture(tmp_path)
    sources = {name: tmp_path / f"{name}.jsonl" for name in (
        "candidate_fusion", "tse_asr", "audio_map", "r3_predictions", "group_manifest"
    )}
    for path in sources.values():
        path.write_text(path.name, encoding="utf-8")
    selection = tmp_path / "selection.json"
    args = _select_args(paths, selection)
    for flag, name in (
        ("--candidate-fusion", "candidate_fusion"),
        ("--tse-asr", "tse_asr"),
        ("--audio-map", "audio_map"),
        ("--r3-predictions", "r3_predictions"),
        ("--group-manifest", "group_manifest"),
    ):
        args.extend([flag, str(sources[name])])
    assert main(args) == 0
    data = json.loads(selection.read_text(encoding="utf-8"))
    assert set(data["provenance"]["candidate_source_digests"]) == set(sources)
    sources["tse_asr"].write_text("changed", encoding="utf-8")
    evaluate_args = [
        "evaluate",
        "--canonical-input-jsonl", str(paths["canonical"]),
        "--groups", str(paths["groups"]),
        "--split-manifest", str(paths["split"]),
        "--train-labels", str(paths["train_labels"]),
        "--validation-labels", str(paths["validation_labels"]),
        "--cache-root", str(paths["cache"]),
        "--selection-input", str(selection),
        "--held-out-labels", str(paths["held_out_labels"]),
        "--evaluation-output", str(tmp_path / "evaluation"),
    ]
    for flag, name in (
        ("--candidate-fusion", "candidate_fusion"),
        ("--tse-asr", "tse_asr"),
        ("--audio-map", "audio_map"),
        ("--r3-predictions", "r3_predictions"),
        ("--group-manifest", "group_manifest"),
    ):
        evaluate_args.extend([flag, str(sources[name])])
    with pytest.raises(ValueError, match="candidate_source_digests"):
        main(evaluate_args)

