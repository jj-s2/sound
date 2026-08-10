"""Focused contract tests for grouped FireRed pVAD OOF families."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from xh202615.firered_pvad import PVAD_GATE_FEATURE_SCHEMA
from xh202615.r10_selector import CandidateRow
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
                audio_features={name: signal for name in E0_FITTING_FEATURE_SCHEMA},
                original_command_audio=None, source_digest="source", dedup_sources={},
            ))
            labels[sid] = label
            groups[sid] = f"wake-{group_index:02d}"
            features = {name: float(signal) for name in PVAD_GATE_FEATURE_SCHEMA}
            features["frame_count"] = 100
            features["dropped_tail_samples"] = 0
            for name in PVAD_GATE_FEATURE_SCHEMA:
                if "frames" in name or "crossing" in name or "transitions" in name:
                    features[name] = 1
            cache.append({"id": sid, "features": features})
    ids = [row.id for row in rows]
    manifest = {
        "feature_schema": list(PVAD_GATE_FEATURE_SCHEMA),
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
    next(item for item in changed if item["id"] == first_test_id)["features"]["raw_mean"] = 0.123
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
