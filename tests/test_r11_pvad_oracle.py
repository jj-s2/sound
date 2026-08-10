"""Focused contract tests for grouped FireRed pVAD OOF families."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict

import pytest
import numpy as np
from unittest.mock import patch

from xh202615.firered_model_assets import _RAW_SNAPSHOT_FILES, _REQUIRED_DEPENDENCY_VERSIONS, _UPSTREAM_IDENTITY
from xh202615.firered_pvad import PVAD_GATE_FEATURE_SCHEMA, PvadRuntimeConfig
from xh202615 import pvad_cache
from xh202615.r10_selector import CandidateRow
from xh202615.r11_gate_oracle import GATE_FEATURE_SCHEMA, build_gate_feature_matrix
from xh202615.r11_pvad_oracle import (
    E0_FITTING_FEATURE_SCHEMA,
    PVAD_FITTING_FEATURE_SCHEMA,
    canonical_json,
    join_pvad_e0_rows,
    run_pvad_oracle,
)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _refresh_record_digests(cache: list[dict], manifest: dict) -> None:
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in cache]
    manifest["records_sha256"] = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    manifest["per_id_record_sha256"] = {record["id"]: hashlib.sha256(line.encode()).hexdigest() for record, line in zip(cache, lines)}


def _full_cpu_manifest(cache: list[dict]) -> dict:
    """Build a realistic canonical Task 4 CPU manifest for the oracle boundary."""
    ids = [record["id"] for record in cache]
    config = asdict(PvadRuntimeConfig())
    raw = {name: hashlib.sha256(name.encode()).hexdigest() for name in _RAW_SNAPSHOT_FILES}
    aggregate = hashlib.sha256()
    for name in sorted(raw):
        aggregate.update(name.encode())
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(raw[name]))
    onnx = {
        "sample_rate_hz": 16000, "frame_samples": 160,
        "probability_output_index": 1, "mel_state_output_index": 2,
        "gru_state_output_index": 3,
        "inputs": [{"name": name, "type": "tensor(float)", "shape": list(shape)} for name, shape in (("input_audio", (1, 160)), ("spkemb", (1, 192)), ("mel_buffer", (1, 80, 15)), ("gru_buffer", (2, 1, 256)))],
        "outputs": [{"name": name, "type": "tensor(float)", "shape": list(shape)} for name, shape in (("output", (1, 1)), ("prob", (1, 1)), ("mel_buffer_out", (1, 80, 15)), ("gru_buffer_out", (2, 1, 256)))],
    }
    model = {"manifest_sha256": "1" * 64, "aggregate_sha256": aggregate.hexdigest(), "raw_sha256": raw, "upstream": _UPSTREAM_IDENTITY, "onnx": onnx, "required_dependency_versions": _REQUIRED_DEPENDENCY_VERSIONS}
    model["identity_sha256"] = pvad_cache._digest(model)
    audio = {sample_id: {"wake_sha256": hashlib.sha256((sample_id + "-wake").encode()).hexdigest(), "command_sha256": hashlib.sha256((sample_id + "-command").encode()).hexdigest()} for sample_id in ids}
    source = {"jsonl_sha256": {"pos": "2" * 64, "neg": "3" * 64}, "per_id_audio_sha256": audio}
    source["projection_sha256"] = pvad_cache._digest({"jsonl_sha256": source["jsonl_sha256"], "per_id_audio_sha256": audio})
    coverage = {"selected": {"count": len(ids), "ids": ids, "id_sha256": _digest(ids)}, "source": {"count": len(ids), "ids": ids, "id_sha256": _digest(ids)}}
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in cache]
    percentile = {"count": len(ids), "p50": 1.0, "p95": 1.0, "max": 1.0}
    return {
        "artifact_kind": "r11_e2_firered_cache", "schema_version": "v1", "feature_schema": list(PVAD_GATE_FEATURE_SCHEMA),
        "feature_schema_sha256": pvad_cache._schema_digest(),
        "digest_algorithms": {"feature_schema_sha256": "sha256(UTF-8 ordered schema names joined and terminated by literal backslash-n bytes)", "records_sha256": "sha256(UTF-8 canonical JSONL bytes)", "per_id_record_sha256": "sha256(UTF-8 canonical JSON record bytes)", "joined_state_sha256": "sha256(UTF-8 canonical JSON)", "source_audio_sha256": "sha256(raw wake/command audio bytes)", "source_projection_sha256": "sha256(UTF-8 canonical JSON source projection)", "model_sha256": "sha256(UTF-8 canonical JSON verified model identity)"},
        "coverage": coverage, "source": source, "model": model, "runtime_config": config, "runtime_config_sha256": pvad_cache._digest(config),
        "provider": "CPUExecutionProvider", "device": "cpu", "environment": {"python": "3.12.0", "platform": "test", "observed_dependencies": pvad_cache._observed_dependencies()},
        "reuse": {"reused": 0, "new": len(ids)}, "timing": {"cold_elapsed_seconds": percentile, "warm_elapsed_seconds": {"count": 0, "p50": None, "p95": None, "max": None}, "rtf": percentile, "peak_rss_delta_bytes": percentile, "cuda_peak_bytes": {"count": 0, "p50": None, "p95": None, "max": None}},
        "parity": {"status": "not-run", "passed": None, "max_abs_feature_delta": None}, "limit": {"value": None, "canonical": True, "reason": None},
        "records_sha256": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(), "per_id_record_sha256": {record["id"]: hashlib.sha256(line.encode()).hexdigest() for record, line in zip(cache, lines)},
        "joined_state_sha256": pvad_cache._digest({sample_id: pvad_cache._resume_expected(sample_id, audio[sample_id], model, config) for sample_id in ids}),
    }


def _fixture() -> tuple[list[CandidateRow], dict[str, str | None], dict[str, str], list[dict], dict]:
    rows: list[CandidateRow] = []
    labels: dict[str, str | None] = {}
    groups: dict[str, str] = {}
    cache: list[dict] = []
    for group_index in range(10):
        for present in (True, False):
            sid = f"{group_index:02d}-{'p' if present else 'n'}"
            label = "甲" if present else None
            signal = 0.9 if present else 0.1
            rows.append(CandidateRow(
                id=sid, split="pos" if present else "neg", label=label,
                r3_text="甲" if present else "", primary_text="甲" if present else "",
                energy_text="", tse_text="甲" if present else "",
                audio_features={"presence_score": signal, "enhanced_cosine": signal, "mixture_cosine": signal, "max_cosine": signal, "cmd_duration_sec": 1.0, "cmd_rms": signal},
                original_command_audio=None, source_digest="b" * 64, dedup_sources={},
            ))
            labels[sid] = label
            groups[sid] = f"wake-{group_index:02d}"
            features = {name: float(signal) for name in PVAD_GATE_FEATURE_SCHEMA}
            features["frame_count"] = 100
            features["dropped_tail_samples"] = 0
            features["analyzed_duration_sec"] = 1.0
            features["command_duration_sec"] = 1.0
            features["enrollment_duration_sec"] = 1.0
            features["embedding_norm_before"] = 1.0
            features["embedding_norm_after"] = 1.0
            for threshold in ("0_3", "0_5", "0_7"):
                features[f"ema_longest_run_ge_{threshold}_frames"] = 1
                features[f"ema_longest_run_ge_{threshold}_seconds"] = 0.01
                features[f"ema_first_crossing_ge_{threshold}_frame"] = 1
                features[f"ema_last_crossing_ge_{threshold}_frame"] = 1
                features[f"ema_active_span_ge_{threshold}_frames"] = 1
                features[f"ema_transitions_ge_{threshold}"] = 0
            cache.append({"id": sid, "features": features})
    ids = [row.id for row in rows]
    manifest = _full_cpu_manifest(cache)
    return rows, labels, groups, cache, manifest


def test_exact_join_coverage_and_duplicate_failures() -> None:
    rows, labels, groups, cache, manifest = _fixture()
    joined = join_pvad_e0_rows(rows, labels, groups, cache, manifest)
    assert [item.id for item in joined] == [row.id for row in rows]
    with pytest.raises(ValueError, match="duplicate"):
        join_pvad_e0_rows(rows, labels, groups, [*cache, cache[0]], manifest)
    with pytest.raises(ValueError, match="coverage|extra"):
        join_pvad_e0_rows(rows, labels, groups, cache[:-1], manifest)


def test_all_families_have_group_disjoint_once_only_oof_and_safe_records() -> None:
    rows, labels, groups, cache, manifest = _fixture()
    result = run_pvad_oracle(rows, labels, groups, cache, manifest)
    assert list(result["families"]) == ["firered_scalar", "firered_crossfit", "firered_fused_crossfit"]
    assert result["diagnostic_only"] is True
    assert result["branch_decision"] is None
    for family in result["families"].values():
        assert sorted(record["id"] for record in family["rows"]) == sorted(labels)
        assert all(record["action"] in {"accept", "reject"} for record in family["rows"])
        assert all("label" not in record and "text" not in record and "cer" not in record for record in family["rows"])
        assert all(not (set(fold["train_groups"]) & set(fold["test_groups"])) for fold in family["folds"])
        assert family["coverage"]["once_only"] is True


def test_deterministic_canonical_output_and_feature_allowlists() -> None:
    rows, labels, groups, cache, manifest = _fixture()
    first = run_pvad_oracle(rows, labels, groups, cache, manifest)
    second = run_pvad_oracle(rows, labels, groups, cache, manifest)
    assert canonical_json(first) == canonical_json(second)
    assert set(first["families"]["firered_crossfit"]["feature_allowlist"]) == set(PVAD_FITTING_FEATURE_SCHEMA)
    assert set(first["families"]["firered_fused_crossfit"]["feature_allowlist"]) == set(PVAD_FITTING_FEATURE_SCHEMA) | set(E0_FITTING_FEATURE_SCHEMA)
    forbidden = ("label", "reference", "text", "candidate", "audit", "provenance", "latency")
    assert all(token not in " ".join(PVAD_FITTING_FEATURE_SCHEMA).lower() for token in forbidden)


def test_held_out_pvad_values_do_not_change_training_fold_metadata() -> None:
    rows, labels, groups, cache, manifest = _fixture()
    baseline = run_pvad_oracle(rows, labels, groups, cache, manifest)
    changed = copy.deepcopy(cache)
    first_test_id = baseline["families"]["firered_crossfit"]["folds"][0]["test_ids"][0]
    next(item for item in changed if item["id"] == first_test_id)["features"]["raw_std"] = 0.123
    changed_manifest = copy.deepcopy(manifest)
    _refresh_record_digests(changed, changed_manifest)
    perturbed = run_pvad_oracle(rows, labels, groups, changed, changed_manifest)
    assert baseline["families"]["firered_crossfit"]["folds"][0]["training_state"] == perturbed["families"]["firered_crossfit"]["folds"][0]["training_state"]


@pytest.mark.parametrize("mutation", ["nan", "probability", "sentinel", "digest"])
def test_invalid_domains_sentinels_and_provenance_fail_closed(mutation: str) -> None:
    rows, labels, groups, cache, manifest = _fixture()
    if mutation == "nan":
        cache[0]["features"]["raw_mean"] = float("nan")
    elif mutation == "probability":
        cache[0]["features"]["raw_mean"] = 1.5
    elif mutation == "sentinel":
        cache[0]["features"]["ema_first_crossing_ge_0_3_frame"] = -2
    else:
        manifest["coverage"]["selected"]["id_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        run_pvad_oracle(rows, labels, groups, cache, manifest)


def test_infeasible_search_fails_closed() -> None:
    rows, labels, groups, cache, manifest = _fixture()
    labels = {sid: "甲" for sid in labels}
    with pytest.raises(ValueError, match="both target classes|feasible"):
        run_pvad_oracle(rows, labels, groups, cache, manifest)


@pytest.mark.parametrize("mutation", ["schema", "nonhex", "nested_private", "empty_row_source"])
def test_review_provenance_counterexamples_fail_closed(mutation: str) -> None:
    rows, labels, groups, cache, manifest = _fixture()
    if mutation == "schema":
        manifest["feature_schema"] = ["forged"]
    elif mutation == "nonhex":
        manifest["joined_state_sha256"] = "z" * 64
    elif mutation == "nested_private":
        manifest["source"]["nested"] = {"reference_text": "private"}
    else:
        rows[0] = CandidateRow(**{**rows[0].__dict__, "source_digest": ""})
    with pytest.raises(ValueError):
        join_pvad_e0_rows(rows, labels, groups, cache, manifest)


def test_fused_representation_is_exact_frozen_e0_schema_and_score_banks_are_complete() -> None:
    rows, labels, groups, cache, manifest = _fixture()
    result = run_pvad_oracle(rows, labels, groups, cache, manifest)
    fused = result["families"]["firered_fused_crossfit"]
    assert fused["feature_allowlist"] == [*PVAD_FITTING_FEATURE_SCHEMA, *E0_FITTING_FEATURE_SCHEMA]
    for family in result["families"].values():
        assert family["coverage"]["ids"] == [row.id for row in rows]
        assert len(family["score_bank"]) == len(family["model_specs"])
        for scores in family["score_bank"].values():
            assert len(scores) == len(rows)
        assert family["outer_threshold_grid"]


def test_fused_e0_values_match_every_name_in_the_frozen_full_builder_vector() -> None:
    rows, labels, groups, cache, manifest = _fixture()
    row = rows[0]
    rows[0] = CandidateRow(**{**row.__dict__, "audio_features": {"presence_score": 0.11, "enhanced_cosine": 0.22, "mixture_cosine": 0.33, "max_cosine": 0.44, "latency_ms": 55.0, "cmd_duration_sec": 6.6, "cmd_rms": 0.77}, "r3_text": "甲1", "primary_text": "乙22", "tse_text": "丙333"})
    joined = join_pvad_e0_rows(rows, labels, groups, cache, manifest)
    full = dict(zip(GATE_FEATURE_SCHEMA, build_gate_feature_matrix([rows[0]])[0].tolist()))
    assert joined[0].e0 == {name: full[name] for name in E0_FITTING_FEATURE_SCHEMA}


def test_reject_all_is_markerized_everywhere_and_is_json_safe() -> None:
    rows, labels, groups, cache, manifest = _fixture()
    for index, row in enumerate(rows):
        if labels[row.id] is not None:
            rows[index] = CandidateRow(**{**row.__dict__, "r3_text": "x", "primary_text": "x", "tse_text": "x"})
    result = run_pvad_oracle(rows, labels, groups, cache, manifest)
    assert 'Infinity' not in canonical_json(result)
    assert '"reject_all"' in canonical_json(result)


@pytest.mark.parametrize("mutation", ["quantiles", "duration", "crossings", "run_seconds"])
def test_pvad_cross_field_invariants_fail_closed(mutation: str) -> None:
    rows, labels, groups, cache, manifest = _fixture()
    feature = cache[0]["features"]
    if mutation == "quantiles":
        feature["raw_q10"], feature["raw_q25"] = 0.9, 0.1
    elif mutation == "duration":
        feature["analyzed_duration_sec"] = 0.3
    elif mutation == "crossings":
        feature["ema_first_crossing_ge_0_3_frame"] = -1
    else:
        feature["ema_longest_run_ge_0_3_seconds"] = 1.0
    with pytest.raises(ValueError):
        join_pvad_e0_rows(rows, labels, groups, cache, manifest)


@pytest.mark.parametrize("prediction", [np.array([2.0]), np.array([[2.0]]), np.array([[0.0, float("nan")]]), np.array([[0.0, float("inf")]]), np.array([[0.0, -0.1]])])
def test_malformed_learned_probabilities_fail_closed(prediction: np.ndarray) -> None:
    rows, labels, groups, cache, manifest = _fixture()

    class BadPipeline:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.repeat(prediction, len(X), axis=0) if prediction.ndim == 2 else prediction

    with patch("xh202615.r11_pvad_oracle._fit_gate_pipeline", return_value=BadPipeline()):
        with pytest.raises(ValueError, match="probabilities"):
            run_pvad_oracle(rows, labels, groups, cache, manifest)


def test_round2_distinct_r10_source_digests_and_legitimate_runtime_metadata_pass() -> None:
    rows, labels, groups, cache, manifest = _fixture()
    for index, row in enumerate(rows):
        rows[index] = CandidateRow(**{**row.__dict__, "source_digest": hashlib.sha256(row.id.encode()).hexdigest()})
    assert len(join_pvad_e0_rows(rows, labels, groups, cache, manifest)) == len(rows)


@pytest.mark.parametrize("mutation", ["runtime_digest", "runtime_config", "provider", "coverage", "reuse", "parity", "limit", "timing", "minimal_source", "minimal_model", "plural_private", "nested_private", "extra_top_level"])
def test_complete_task4_manifest_contract_rejects_forged_domains(mutation: str) -> None:
    rows, labels, groups, cache, manifest = _fixture()
    if mutation == "runtime_digest":
        manifest["runtime_config_sha256"] = "not-a-sha"
    elif mutation == "runtime_config":
        manifest["runtime_config"]["sample_rate"] = 123
        manifest["runtime_config_sha256"] = pvad_cache._digest(manifest["runtime_config"])
    elif mutation == "provider":
        manifest["provider"] = "CUDAExecutionProvider"
    elif mutation == "coverage":
        manifest["coverage"]["source"]["ids"] = manifest["coverage"]["source"]["ids"][:-1]
    elif mutation == "reuse":
        manifest["reuse"] = {"reused": 99, "new": 0}
    elif mutation == "parity":
        manifest["parity"] = {"status": "passed", "passed": True, "max_abs_feature_delta": 1.0}
    elif mutation == "limit":
        manifest["limit"] = {"value": None, "canonical": "yes", "reason": None}
    elif mutation == "timing":
        manifest["timing"]["rtf"] = {"count": len(cache), "p50": None, "p95": None, "max": None}
    elif mutation == "minimal_source":
        manifest["source"] = {"projection_sha256": manifest["source"]["projection_sha256"]}
    elif mutation == "minimal_model":
        manifest["model"] = {"identity_sha256": manifest["model"]["identity_sha256"]}
    elif mutation == "plural_private":
        manifest["labels"] = ["SECRET"]
    elif mutation == "nested_private":
        manifest["environment"]["reference_transcript"] = "SECRET"
    else:
        manifest["unexpected"] = True
    with pytest.raises(ValueError):
        join_pvad_e0_rows(rows, labels, groups, cache, manifest)


@pytest.mark.parametrize("mutation", ["inactive_run", "run_gt_span", "too_many_transitions"])
def test_round2_temporal_invariants_fail_closed(mutation: str) -> None:
    rows, labels, groups, cache, manifest = _fixture()
    feature = cache[0]["features"]
    if mutation == "inactive_run":
        feature["ema_first_crossing_ge_0_3_frame"] = -1
        feature["ema_last_crossing_ge_0_3_frame"] = -1
        feature["ema_active_span_ge_0_3_frames"] = 0
    elif mutation == "run_gt_span":
        feature["ema_active_span_ge_0_3_frames"] = 0
    else:
        feature["ema_transitions_ge_0_3"] = feature["frame_count"]
    with pytest.raises(ValueError):
        join_pvad_e0_rows(rows, labels, groups, cache, manifest)


def test_round2_float32_embedding_and_audit_exclusion_and_scalar_frontier_evidence() -> None:
    rows, labels, groups, cache, manifest = _fixture()
    cache[0]["features"]["embedding_norm_after"] = 1.0 + 5e-9
    _refresh_record_digests(cache, manifest)
    result = run_pvad_oracle(rows, labels, groups, cache, manifest)
    fused = result["families"]["firered_fused_crossfit"]
    assert "latency_ms" not in fused["feature_allowlist"]
    scalar = result["families"]["firered_scalar"]
    for fold in scalar["folds"]:
        assert fold["inner_frontier"]
        selected = [point for point in fold["inner_frontier"] if point["model"] == fold["selected_model"] and point["threshold"] == fold["selected_threshold"]]
        assert selected
