from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_r11_pvad_oracle import _fixture, _full_cpu_manifest
from xh202615.r12_split import build_r12_split, write_r12_split


def _patch_feasible_finite(monkeypatch) -> None:
    """Make the synthetic CLI fixture expose one deployable finite candidate."""

    import xh202615.r12_calibrated_gate as module

    def feasible_frontier(scores, contributions):
        return [
            {
                "threshold": 0.5,
                "cer": 0.0,
                "rr": 0.95,
                "overall": 0.975,
                "accepted_positives": 1.0,
                "accepted_negatives": 0.0,
            }
        ]

    def feasible_bootstrap(*args, **kwargs):
        return {
            "overall_median": 0.975,
            "rr_p05": 0.95,
            "n_boot": kwargs["n_boot"],
            "attempted_replicates": kwargs["n_boot"],
            "rejected_replicates": 0,
        }

    monkeypatch.setattr(module, "gate_oracle_frontier", feasible_frontier)
    monkeypatch.setattr(module, "_bootstrap_point_stats", feasible_bootstrap)


@pytest.fixture(autouse=True)
def feasible_cli_fixture(monkeypatch):
    _patch_feasible_finite(monkeypatch)


def _write_fixture(root: Path) -> dict[str, Path]:
    rows, labels, groups, cache, manifest = _fixture()
    canonical = root / "canonical.jsonl"
    canonical.write_text(
        "".join(
            json.dumps(
                {
                    "id": row.id,
                    "split": row.split,
                    "r3_text": row.r3_text,
                    "primary_text": row.primary_text,
                    "energy_text": row.energy_text,
                    "tse_text": row.tse_text,
                    "audio_features": row.audio_features,
                    "source_digest": row.source_digest,
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    group_path = root / "groups.json"
    group_path.write_text(json.dumps(groups, ensure_ascii=False), encoding="utf-8")
    split = build_r12_split([row.id for row in rows], labels, groups)
    split_path = root / "split.json"
    write_r12_split(split_path, split)

    cache_root = root / "cache"
    cache_root.mkdir()
    (cache_root / "pvad_features.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in cache),
        encoding="utf-8",
    )
    (cache_root / "pvad_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (cache_root / "pvad_report.md").write_text("fixture", encoding="utf-8")

    role_ids = {role: [sid for sid, value in split.roles_by_id.items() if value == role] for role in ("train", "validation", "held_out_test")}
    train_labels = root / "train_labels.json"
    train_labels.write_text(json.dumps({sid: labels[sid] for sid in role_ids["train"]}), encoding="utf-8")
    validation_labels = root / "validation_labels.json"
    validation_labels.write_text(json.dumps({sid: labels[sid] for sid in role_ids["validation"]}), encoding="utf-8")
    held_out_labels = root / "held_out_labels.json"
    held_out_labels.write_text(json.dumps({sid: labels[sid] for sid in role_ids["held_out_test"]}), encoding="utf-8")
    candidate_sources = {}
    for name in ("candidate_fusion", "tse_asr", "audio_map", "r3_predictions", "group_manifest"):
        source = root / f"{name}.source"
        source.write_text(f"fixture:{name}\n", encoding="utf-8")
        candidate_sources[name] = source
    return {
        "canonical": canonical,
        "groups": group_path,
        "split": split_path,
        "cache": cache_root,
        "train_labels": train_labels,
        "validation_labels": validation_labels,
        "held_out_labels": held_out_labels,
        "held_out_ids": role_ids["held_out_test"],
        "candidate_sources": candidate_sources,
    }


def _select_args(paths: dict[str, Path], output: Path) -> list[str]:
    args = [
        "select",
        "--canonical-input-jsonl", str(paths["canonical"]),
        "--groups", str(paths["groups"]),
        "--split-manifest", str(paths["split"]),
        "--train-labels", str(paths["train_labels"]),
        "--validation-labels", str(paths["validation_labels"]),
        "--cache-root", str(paths["cache"]),
        "--selection-output", str(output),
        "--bootstrap-count", "10",
    ]
    for flag, name in (
        ("--candidate-fusion", "candidate_fusion"),
        ("--tse-asr", "tse_asr"),
        ("--audio-map", "audio_map"),
        ("--r3-predictions", "r3_predictions"),
        ("--group-manifest", "group_manifest"),
    ):
        args.extend([flag, str(paths["candidate_sources"][name])])
    return args


def _source_args(paths: dict[str, Path]) -> list[str]:
    args: list[str] = []
    for flag, name in (
        ("--candidate-fusion", "candidate_fusion"),
        ("--tse-asr", "tse_asr"),
        ("--audio-map", "audio_map"),
        ("--r3-predictions", "r3_predictions"),
        ("--group-manifest", "group_manifest"),
    ):
        args.extend([flag, str(paths["candidate_sources"][name])])
    return args


def test_select_creates_frozen_selection_without_held_out_labels(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main

    paths = _write_fixture(tmp_path)
    selection = tmp_path / "selection.json"
    assert main(_select_args(paths, selection)) == 0
    data = json.loads(selection.read_text(encoding="utf-8"))
    assert data["artifact_kind"] == "r12_strict_selection"
    assert not any(sid in json.dumps(data) for sid in paths["held_out_ids"])
    assert '"label"' not in json.dumps(data).lower()


def test_select_serializes_router_without_labels_or_candidate_cer(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main

    paths = _write_fixture(tmp_path)
    selection = tmp_path / "selection.json"
    assert main(_select_args(paths, selection)) == 0
    router = json.loads(selection.read_text(encoding="utf-8"))["selection"]["router"]
    assert router["action_order"] == ["primary", "r3", "tse", "energy"]
    assert "label" not in json.dumps(router).lower()
    assert "candidate_cer" not in json.dumps(router).lower()


def test_select_serializes_text_presence_without_training_text_or_labels(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main

    paths = _write_fixture(tmp_path)
    selection = tmp_path / "selection.json"
    assert main(_select_args(paths, selection)) == 0
    text_presence = json.loads(selection.read_text(encoding="utf-8"))["selection"]["text_presence"]
    assert text_presence["input_fields"] == ["r3_text", "primary_text"]
    assert "label" not in json.dumps(text_presence).lower()
    assert "training" not in json.dumps(text_presence).lower()


def test_selection_loader_accepts_frozen_text_gate_fusion(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import _selection_from_dict, main

    paths = _write_fixture(tmp_path)
    selection = tmp_path / "selection.json"
    assert main(_select_args(paths, selection)) == 0
    data = json.loads(selection.read_text(encoding="utf-8"))
    data["selection"]["selected_model_name"] = "text_gate_fusion"
    data["selection"]["selected_blend_weight"] = 0.5
    data["provenance"]["selection_payload_sha256"] = __import__("hashlib").sha256(
        json.dumps(data["selection"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    data["provenance"]["provenance_payload_sha256"] = __import__("hashlib").sha256(
        json.dumps(
            {key: value for key, value in data["provenance"].items() if key != "provenance_payload_sha256"},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert _selection_from_dict(data).selected_model_name == "text_gate_fusion"


def test_evaluate_rebuilds_text_scores_for_frozen_text_gate_fusion(tmp_path: Path, monkeypatch) -> None:
    import scripts.r12_strict_holdout as strict

    original_select = strict.select_on_validation

    def select_text_fusion(*args, **kwargs):
        selection = original_select(*args, **kwargs)
        return replace(
            selection,
            selected_model_name="text_gate_fusion",
            selected_blend_weight=0.5,
            threshold=0.0,
        )

    monkeypatch.setattr(strict, "select_on_validation", select_text_fusion)
    paths = _write_fixture(tmp_path)
    selection = tmp_path / "selection.json"
    assert strict.main(_select_args(paths, selection)) == 0
    assert strict.main([
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
    ] + _source_args(paths)) == 0


def test_evaluate_rejects_router_payload_drift(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main

    paths = _write_fixture(tmp_path)
    selection = tmp_path / "selection.json"
    assert main(_select_args(paths, selection)) == 0
    data = json.loads(selection.read_text(encoding="utf-8"))
    data["selection"]["router"]["feature_schema_digest"] = "0" * 64
    data["provenance"]["selection_payload_sha256"] = __import__("hashlib").sha256(
        json.dumps(data["selection"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    data["provenance"]["provenance_payload_sha256"] = __import__("hashlib").sha256(
        json.dumps(
            {key: value for key, value in data["provenance"].items() if key != "provenance_payload_sha256"},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    selection.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="router"):
        main([
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
        ] + _source_args(paths))


def test_select_rejects_held_out_labels_flag(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main

    paths = _write_fixture(tmp_path)
    with pytest.raises(SystemExit):
        main(_select_args(paths, tmp_path / "selection.json") + [
            "--held-out-labels", str(paths["held_out_labels"])
        ])


def test_evaluate_requires_frozen_selection_and_held_out_labels(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main

    paths = _write_fixture(tmp_path)
    selection = tmp_path / "selection.json"
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="selection"):
        main([
            "evaluate",
            "--canonical-input-jsonl", str(paths["canonical"]),
            "--groups", str(paths["groups"]),
            "--split-manifest", str(paths["split"]),
            "--train-labels", str(paths["train_labels"]),
            "--validation-labels", str(paths["validation_labels"]),
            "--cache-root", str(paths["cache"]),
            "--selection-input", str(selection),
            "--held-out-labels", str(paths["held_out_labels"]),
            "--evaluation-output", str(output),
        ] + _source_args(paths))


def test_evaluate_publishes_exact_five_files(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main

    paths = _write_fixture(tmp_path)
    selection = tmp_path / "selection.json"
    assert main(_select_args(paths, selection)) == 0
    output = tmp_path / "evaluation"
    assert main([
        "evaluate",
        "--canonical-input-jsonl", str(paths["canonical"]),
        "--groups", str(paths["groups"]),
        "--split-manifest", str(paths["split"]),
        "--train-labels", str(paths["train_labels"]),
        "--validation-labels", str(paths["validation_labels"]),
        "--cache-root", str(paths["cache"]),
        "--selection-input", str(selection),
        "--held-out-labels", str(paths["held_out_labels"]),
        "--evaluation-output", str(output),
    ] + _source_args(paths)) == 0
    assert {p.name for p in output.iterdir()} == {
        "r12_manifest.json",
        "r12_selection.json",
        "r12_held_out_predictions.jsonl",
        "r12_summary.json",
        "r12_report.md",
    }
    assert "label" not in (output / "r12_held_out_predictions.jsonl").read_text(encoding="utf-8").lower()
    manifest = json.loads((output / "r12_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["output_digests"]) == {
        "r12_selection.json",
        "r12_held_out_predictions.jsonl",
        "r12_summary.json",
        "r12_report.md",
    }
    summary = json.loads((output / "r12_summary.json").read_text(encoding="utf-8"))
    assert set(summary["metrics"]) >= {"avg_cer", "avg_rr", "overall"}
    predictions = [
        json.loads(line)
        for line in (output / "r12_held_out_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(set(row) >= {"id", "group", "accepted", "threshold", "recognition_text"} for row in predictions)


def test_evaluate_rejects_changed_train_labels(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main

    paths = _write_fixture(tmp_path)
    selection = tmp_path / "selection.json"
    assert main(_select_args(paths, selection)) == 0
    changed = json.loads(paths["train_labels"].read_text(encoding="utf-8"))
    changed[next(iter(changed))] = "changed"
    paths["train_labels"].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="train_labels"):
        main([
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
        ] + _source_args(paths))


def test_evaluate_rejects_changed_validation_labels(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main

    paths = _write_fixture(tmp_path)
    selection = tmp_path / "selection.json"
    assert main(_select_args(paths, selection)) == 0
    changed = json.loads(paths["validation_labels"].read_text(encoding="utf-8"))
    changed[next(iter(changed))] = "changed"
    paths["validation_labels"].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="validation_labels"):
        main([
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
        ] + _source_args(paths))


def test_evaluate_rejects_tampered_selection_payload(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main

    paths = _write_fixture(tmp_path)
    selection = tmp_path / "selection.json"
    assert main(_select_args(paths, selection)) == 0
    data = json.loads(selection.read_text(encoding="utf-8"))
    data["selection"]["threshold"] = 0.123456
    data["provenance"]["selection_payload_sha256"] = __import__("hashlib").sha256(
        json.dumps(data["selection"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    selection.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="selection payload|selection provenance|frozen selection"):
        main([
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
        ] + _source_args(paths))


def test_stages_reject_groups_that_differ_from_frozen_split(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main

    paths = _write_fixture(tmp_path)
    groups = json.loads(paths["groups"].read_text(encoding="utf-8"))
    groups[next(iter(groups))] = "different-group"
    paths["groups"].write_text(json.dumps(groups), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen split"):
        main(_select_args(paths, tmp_path / "selection.json"))


def test_held_out_labels_cannot_change_published_predictions(tmp_path: Path) -> None:
    from scripts.r12_strict_holdout import main

    paths = _write_fixture(tmp_path)
    held_out_positive = next(sid for sid in paths["held_out_ids"] if sid.endswith("-p"))
    canonical_rows = [
        json.loads(line)
        for line in paths["canonical"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in canonical_rows:
        if row["id"] == held_out_positive:
            row["r3_text"] = "甲"
            row["primary_text"] = "错"
            row["energy_text"] = "错"
            row["tse_text"] = "错"
    paths["canonical"].write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in canonical_rows),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    assert main(_select_args(paths, selection)) == 0

    def evaluate(output: Path) -> None:
        assert main([
            "evaluate",
            "--canonical-input-jsonl", str(paths["canonical"]),
            "--groups", str(paths["groups"]),
            "--split-manifest", str(paths["split"]),
            "--train-labels", str(paths["train_labels"]),
            "--validation-labels", str(paths["validation_labels"]),
            "--cache-root", str(paths["cache"]),
            "--selection-input", str(selection),
            "--held-out-labels", str(paths["held_out_labels"]),
            "--evaluation-output", str(output),
        ] + _source_args(paths)) == 0

    first = tmp_path / "evaluation-first"
    evaluate(first)
    held_out_labels = json.loads(paths["held_out_labels"].read_text(encoding="utf-8"))
    held_out_labels[held_out_positive] = "错"
    paths["held_out_labels"].write_text(json.dumps(held_out_labels), encoding="utf-8")
    second = tmp_path / "evaluation-second"
    evaluate(second)
    assert (first / "r12_held_out_predictions.jsonl").read_bytes() == (
        second / "r12_held_out_predictions.jsonl"
    ).read_bytes()
