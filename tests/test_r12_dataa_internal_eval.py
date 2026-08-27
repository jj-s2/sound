"""Protocol tests for R12 augmented internal evaluation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def test_train_child_labels_are_derived_only_from_train_parent() -> None:
    from scripts.r12_dataa_internal_eval import expand_train_labels

    rows = {
        "p": ("p", "original", "train"),
        "p__aug_a": ("p", "aug_a", "train"),
        "p__aug_b": ("p", "aug_b", "train"),
        "n": ("n", "original", "train"),
    }
    assert expand_train_labels({"p": "文本", "n": None}, rows) == {
        "p": "文本", "p__aug_a": "文本", "p__aug_b": "文本", "n": None,
    }


def test_validation_and_test_reject_augmentation_ids() -> None:
    from scripts.r12_dataa_internal_eval import validate_role_ids

    validate_role_ids("validation", ["v1", "v2"])
    with pytest.raises(ValueError, match="original"):
        validate_role_ids("internal_test", ["t1__aug_a"])


def test_selection_never_accepts_internal_test_label_path() -> None:
    from scripts.r12_dataa_internal_eval import parse_args

    with pytest.raises(SystemExit):
        parse_args(["select", "--internal-test-labels", "private.json"])


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _full_protocol_fixture(tmp_path: Path) -> dict[str, Path]:
    """Build 20 balanced wake groups plus two train-only children per parent."""
    from tests.test_r11_pvad_oracle import _full_cpu_manifest
    from xh202615.firered_pvad import PVAD_GATE_FEATURE_SCHEMA
    from xh202615.r12_dataa_augmented_split import build_augmented_internal_split, write_augmented_internal_split

    original_ids: list[str] = []
    original_labels: dict[str, str | None] = {}
    original_groups: dict[str, str] = {}
    for group_index in range(20):
        for suffix, label in (("p", "甲"), ("n", None)):
            sample_id = f"{group_index:02d}-{suffix}"
            original_ids.append(sample_id)
            original_labels[sample_id] = label
            original_groups[sample_id] = f"wake-{group_index:02d}"
    split = build_augmented_internal_split(original_ids, original_labels, original_groups)
    split_path = tmp_path / "split.json"
    write_augmented_internal_split(split_path, split)

    lineage_rows: list[dict[str, object]] = []
    canonical_rows: list[dict[str, object]] = []
    cache_rows: list[dict[str, object]] = []
    full_labels: dict[str, str | None] = {}
    for parent_id in original_ids:
        role = split.roles_by_id[parent_id]
        variants = ("original", "aug_a", "aug_b") if role == "train" else ("original",)
        for variant in variants:
            sample_id = parent_id if variant == "original" else f"{parent_id}__{variant}"
            signal = 0.9 if original_labels[parent_id] is not None else 0.1
            lineage_rows.append({
                "id": sample_id, "parent_id": parent_id, "augmentation_id": variant,
                "role": role, "group": original_groups[parent_id],
                "source_split": "pos" if original_labels[parent_id] is not None else "neg",
                "command_audio_sha256": hashlib.sha256((sample_id + "-cmd").encode()).hexdigest(),
                "wake_audio_sha256": hashlib.sha256((parent_id + "-wake").encode()).hexdigest(),
                "wakeup_audio": "wake.wav", "command_audio": "cmd.wav", "parameters": {},
            })
            canonical_rows.append({
                "id": sample_id, "split": role,
                "r3_text": "甲" if signal > 0.5 else "",
                "primary_text": "甲" if signal > 0.5 else "",
                "energy_text": "", "tse_text": "甲" if signal > 0.5 else "",
                "audio_features": {"presence_score": signal, "enhanced_cosine": signal, "mixture_cosine": signal, "max_cosine": signal, "latency_ms": 1.0, "cmd_duration_sec": 1.0, "cmd_rms": signal},
                "source_digest": hashlib.sha256((sample_id + "-source").encode()).hexdigest(),
            })
            features = {name: float(signal) for name in PVAD_GATE_FEATURE_SCHEMA}
            features.update({"frame_count": 100, "dropped_tail_samples": 0, "analyzed_duration_sec": 1.0, "command_duration_sec": 1.0, "enrollment_duration_sec": 1.0, "embedding_norm_before": 1.0, "embedding_norm_after": 1.0})
            for threshold in ("0_3", "0_5", "0_7"):
                features.update({f"ema_longest_run_ge_{threshold}_frames": 1, f"ema_longest_run_ge_{threshold}_seconds": 0.01, f"ema_first_crossing_ge_{threshold}_frame": 1, f"ema_last_crossing_ge_{threshold}_frame": 1, f"ema_active_span_ge_{threshold}_frames": 1, f"ema_transitions_ge_{threshold}": 0})
            cache_rows.append({"id": sample_id, "features": features})
            full_labels[sample_id] = original_labels[parent_id]

    lineage = tmp_path / "lineage.jsonl"
    lineage.write_text("".join(json.dumps(row) + "\n" for row in lineage_rows), encoding="utf-8")
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in canonical_rows), encoding="utf-8")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "pvad_features.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cache_rows), encoding="utf-8")
    (cache_root / "pvad_manifest.json").write_text(json.dumps(_full_cpu_manifest(cache_rows)), encoding="utf-8")
    (cache_root / "pvad_report.md").write_text("fixture", encoding="utf-8")
    train_parent_ids = [sample_id for sample_id, role in split.roles_by_id.items() if role == "train"]
    validation_ids = [sample_id for sample_id, role in split.roles_by_id.items() if role == "validation"]
    test_ids = [sample_id for sample_id, role in split.roles_by_id.items() if role == "internal_test"]
    train_labels, validation_labels, test_labels = tmp_path / "train.json", tmp_path / "validation.json", tmp_path / "test.json"
    _write_json(train_labels, {sample_id: original_labels[sample_id] for sample_id in train_parent_ids})
    _write_json(validation_labels, {sample_id: original_labels[sample_id] for sample_id in validation_ids})
    _write_json(test_labels, {sample_id: original_labels[sample_id] for sample_id in test_ids})
    return {"canonical": canonical, "lineage": lineage, "split": split_path, "cache": cache_root, "train": train_labels, "validation": validation_labels, "test": test_labels, "selection": tmp_path / "selection.json", "result": tmp_path / "result"}


def test_select_and_evaluate_enforce_staged_one_time_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import xh202615.r12_calibrated_gate as gate
    from scripts.r12_dataa_internal_eval import main

    monkeypatch.setattr(gate, "gate_oracle_frontier", lambda scores, contributions: [{"threshold": 0.5, "cer": 0.0, "rr": 0.95, "overall": 0.975, "accepted_positives": 1.0, "accepted_negatives": 0.0}])
    monkeypatch.setattr(gate, "_bootstrap_point_stats", lambda *args, **kwargs: {"overall_median": 0.975, "rr_p05": 0.95, "n_boot": kwargs["n_boot"], "attempted_replicates": 1, "rejected_replicates": 0})
    paths = _full_protocol_fixture(tmp_path)
    common = ["--canonical-input-jsonl", str(paths["canonical"]), "--lineage", str(paths["lineage"]), "--split-manifest", str(paths["split"]), "--cache-root", str(paths["cache"]), "--train-labels", str(paths["train"]), "--validation-labels", str(paths["validation"]), "--bootstrap-count", "5"]

    assert main(["select", *common, "--selection-output", str(paths["selection"])]) == 0
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    assert any("__aug_a" in sample_id for sample_id in selection["provenance"]["fit_ids"])
    assert all("__aug_" not in sample_id for sample_id in selection["provenance"]["validation_ids"])
    assert "internal_test" not in json.dumps(selection, ensure_ascii=False)

    assert main(["evaluate", *common, "--selection-input", str(paths["selection"]), "--internal-test-labels", str(paths["test"]), "--evaluation-output", str(paths["result"])]) == 0
    summary = json.loads((paths["result"] / "r12_summary.json").read_text(encoding="utf-8"))
    assert summary["internal_test_label_read_count"] == 1
    assert set(summary["metrics"]) >= {"avg_cer", "avg_rr", "overall"}
    assert {path.name for path in paths["result"].iterdir()} == {
        "r12_manifest.json", "r12_internal_predictions.jsonl", "r12_summary.json", "r12_report.md",
    }
    manifest = json.loads((paths["result"] / "r12_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "r12_dataa_augmented_internal_test"
    assert manifest["internal_test_label_read_count"] == 1
    assert "Dataset-A group-disjoint internal test" in (paths["result"] / "r12_report.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="one-time"):
        main(["evaluate", *common, "--selection-input", str(paths["selection"]), "--internal-test-labels", str(paths["test"]), "--evaluation-output", str(paths["result"])])


def test_select_accepts_canonical_order_different_from_authenticated_pvad_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pVAD integrity is order-bound, while canonical rows may be role-ordered."""
    import xh202615.r12_calibrated_gate as gate
    from scripts.r12_dataa_internal_eval import main

    monkeypatch.setattr(gate, "gate_oracle_frontier", lambda scores, contributions: [{"threshold": 0.5, "cer": 0.0, "rr": 0.95, "overall": 0.975, "accepted_positives": 1.0, "accepted_negatives": 0.0}])
    monkeypatch.setattr(gate, "_bootstrap_point_stats", lambda *args, **kwargs: {"overall_median": 0.975, "rr_p05": 0.95, "n_boot": kwargs["n_boot"], "attempted_replicates": 1, "rejected_replicates": 0})
    paths = _full_protocol_fixture(tmp_path)
    canonical_rows = [line for line in paths["canonical"].read_text(encoding="utf-8").splitlines() if line]
    paths["canonical"].write_text("\n".join(canonical_rows[::2] + canonical_rows[1::2]) + "\n", encoding="utf-8")
    common = ["--canonical-input-jsonl", str(paths["canonical"]), "--lineage", str(paths["lineage"]), "--split-manifest", str(paths["split"]), "--cache-root", str(paths["cache"]), "--train-labels", str(paths["train"]), "--validation-labels", str(paths["validation"]), "--bootstrap-count", "5"]

    assert main(["select", *common, "--selection-output", str(paths["selection"])]) == 0
