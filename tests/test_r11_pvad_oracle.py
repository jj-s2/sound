"""Focused contract tests for grouped FireRed pVAD OOF families."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import numpy as np
from unittest.mock import patch

from xh202615.firered_pvad import PVAD_GATE_FEATURE_SCHEMA
from xh202615.r10_selector import CandidateRow
from xh202615.r11_gate_oracle import GATE_FEATURE_SCHEMA
from xh202615.r11_pvad_oracle import (
    E0_FITTING_FEATURE_SCHEMA,
    PVAD_FITTING_FEATURE_SCHEMA,
    canonical_json,
    join_pvad_e0_rows,
    run_pvad_oracle,
)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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
    manifest = {
        "feature_schema": list(PVAD_GATE_FEATURE_SCHEMA),
        "feature_schema_sha256": hashlib.sha256((r"\n".join(PVAD_GATE_FEATURE_SCHEMA) + r"\n").encode("utf-8")).hexdigest(),
        "coverage": {"selected": {"ids": ids, "count": len(ids), "id_sha256": _digest(ids)}},
        "joined_state_sha256": "a" * 64,
        "source": {"projection_sha256": "b" * 64},
        "model": {"identity_sha256": "c" * 64},
    }
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
    perturbed = run_pvad_oracle(rows, labels, groups, changed, manifest)
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
    assert fused["feature_allowlist"] == [*PVAD_FITTING_FEATURE_SCHEMA, *GATE_FEATURE_SCHEMA]
    for family in result["families"].values():
        assert family["coverage"]["ids"] == [row.id for row in rows]
        assert len(family["score_bank"]) == len(family["model_specs"])
        for scores in family["score_bank"].values():
            assert len(scores) == len(rows)
        assert family["outer_frontiers"]


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
