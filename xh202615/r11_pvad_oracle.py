"""Diagnostic-only grouped OOF FireRed pVAD gate families.

This module deliberately has no writer or branch-decision logic.  Its returned
records contain gates only, never references, hypotheses, CER vectors, or the
oracle candidate selected for an accepted positive.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .firered_pvad import PVAD_GATE_FEATURE_SCHEMA
from .r10_selector import CandidateRow
from .r11_gate_oracle import (
    GateModelSpec,
    GATE_FEATURE_SCHEMA,
    _fit_gate_pipeline,
    _best_model_frontier,
    build_oracle_contributions,
    build_gate_feature_matrix,
    default_model_specs,
    gate_oracle_frontier,
    select_frontier_point,
)


SEED = 20260807
RR_FLOOR = 0.93
# E2 reuses E0's builder, but excludes extraction timing from its fitting block.
E0_FITTING_FEATURE_SCHEMA = tuple(name for name in GATE_FEATURE_SCHEMA if name not in {"latency_ms", "latency_ms_missing"})
PVAD_FITTING_FEATURE_SCHEMA = PVAD_GATE_FEATURE_SCHEMA
_SCALAR_FEATURES = ("raw_mean", "ema_mean", "ema_fraction_ge_0_5")
_REJECT_ALL = "reject_all"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_TOL = 1e-9
_EMBEDDING_NORM_TOL = 1e-6
_TASK4_MANIFEST_KEYS = {
    "artifact_kind", "schema_version", "feature_schema", "feature_schema_sha256",
    "digest_algorithms", "coverage", "source", "model", "runtime_config",
    "runtime_config_sha256", "provider", "device", "environment", "reuse",
    "timing", "parity", "limit", "records_sha256", "per_id_record_sha256",
    "joined_state_sha256",
}
_TASK4_DIGEST_ALGORITHMS = {
    "feature_schema_sha256": "sha256(UTF-8 ordered schema names joined and terminated by literal backslash-n bytes)",
    "records_sha256": "sha256(UTF-8 canonical JSONL bytes)",
    "per_id_record_sha256": "sha256(UTF-8 canonical JSON record bytes)",
    "joined_state_sha256": "sha256(UTF-8 canonical JSON)",
    "source_audio_sha256": "sha256(raw wake/command audio bytes)",
    "source_projection_sha256": "sha256(UTF-8 canonical JSON source projection)",
    "model_sha256": "sha256(UTF-8 canonical JSON verified model identity)",
}
_TASK4_NESTED_KEYS = {
    "coverage": {"selected", "source"},
    "coverage.selected": {"count", "ids", "id_sha256"},
    "coverage.source": {"count", "ids", "id_sha256"},
    "source": {"jsonl_sha256", "per_id_audio_sha256", "projection_sha256"},
    "source.jsonl_sha256": {"pos", "neg"},
    "source.per_id_audio_sha256": None,
    "model": {"manifest_sha256", "aggregate_sha256", "raw_sha256", "upstream", "onnx", "required_dependency_versions", "identity_sha256"},
    "runtime_config": {"sample_rate", "frame_samples", "enrollment_cap_seconds", "minimum_audio_seconds", "ema_alpha", "onnx_provider", "ecapa_device"},
    "environment": {"python", "platform", "observed_dependencies"},
    "reuse": {"reused", "new"},
    "parity": {"status", "passed", "max_abs_feature_delta"},
    "limit": {"value", "canonical", "reason"},
    "timing": {"cold_elapsed_seconds", "warm_elapsed_seconds", "rtf", "peak_rss_delta_bytes", "cuda_peak_bytes"},
}


def canonical_json(value: object) -> str:
    """Return deterministic JSON suitable for digests and byte-stable tests."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _ordered_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _reject_private(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("manifest key must be a string")
            if key.lower() in {"label", "reference", "reference_text", "candidate_texts", "candidate_cer", "optimal_action", "recognition_text", "raw_embeddings", "frame_arrays"}:
                raise ValueError("manifest contains forbidden private field")
            _reject_private(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private(item)


def _id(value: object) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value or "\0" in value:
        raise ValueError("id must be a nonempty traversal-safe string")
    return value


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite native JSON number")
    return float(value)


def _counter(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integral counter")
    return value


def _validate_features(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or tuple(value) != PVAD_GATE_FEATURE_SCHEMA:
        raise ValueError("pVAD features disagree with the fixed feature schema")
    result = {name: _finite(value[name], f"pVAD feature {name}") for name in PVAD_GATE_FEATURE_SCHEMA}
    for name, numeric in result.items():
        if name.startswith(("raw_", "ema_")) and ("fraction" in name or name.endswith(("mean", "min", "max", "q10", "q25", "q50", "q75", "q90", "q95"))) and not 0.0 <= numeric <= 1.0:
            raise ValueError(f"pVAD probability/rate {name} must be in [0, 1]")
    frames = _counter(value["frame_count"], "frame_count", minimum=1)
    tail = _counter(value["dropped_tail_samples"], "dropped_tail_samples")
    if tail >= 160 or not math.isclose(result["analyzed_duration_sec"], frames / 100.0, rel_tol=_TOL, abs_tol=_TOL) or not math.isclose(result["command_duration_sec"], (frames * 160 + tail) / 16000.0, rel_tol=_TOL, abs_tol=_TOL):
        raise ValueError("pVAD frame/duration/tail invariant disagrees")
    if not 0 < result["enrollment_duration_sec"] <= 5.0 or result["embedding_norm_before"] <= 0 or not math.isclose(result["embedding_norm_after"], 1.0, rel_tol=_EMBEDDING_NORM_TOL, abs_tol=_EMBEDDING_NORM_TOL):
        raise ValueError("pVAD enrollment/embedding invariant disagrees")
    for prefix in ("raw", "ema"):
        ordered = [result[f"{prefix}_{name}"] for name in ("min", "q10", "q25", "q50", "q75", "q90", "q95", "max")]
        rates = [result[f"{prefix}_fraction_ge_0_{threshold}"] for threshold in ("1", "3", "5", "7", "9")]
        if ordered != sorted(ordered) or not all(0 <= item <= 1 for item in ordered + rates) or rates != sorted(rates, reverse=True) or result[f"{prefix}_std"] < 0 or not ordered[0] <= result[f"{prefix}_mean"] <= ordered[-1]:
            raise ValueError("pVAD statistics/rates invariant disagrees")
    for name in PVAD_GATE_FEATURE_SCHEMA:
        if name.endswith(("_frames", "_frame", "_transitions")):
            if "first_crossing" in name or "last_crossing" in name:
                counter = _counter(value[name], name, minimum=-1)
                if counter > frames:
                    raise ValueError(f"{name} exceeds frame_count")
            else:
                counter = _counter(value[name], name)
                if counter > frames:
                    raise ValueError(f"{name} exceeds frame_count")
    for threshold in ("0_3", "0_5", "0_7"):
        run = _counter(value[f"ema_longest_run_ge_{threshold}_frames"], "run")
        first = _counter(value[f"ema_first_crossing_ge_{threshold}_frame"], "first", minimum=-1)
        last = _counter(value[f"ema_last_crossing_ge_{threshold}_frame"], "last", minimum=-1)
        span = _counter(value[f"ema_active_span_ge_{threshold}_frames"], "span")
        transitions = _counter(value[f"ema_transitions_ge_{threshold}"], "transitions")
        if not math.isclose(result[f"ema_longest_run_ge_{threshold}_seconds"], run / 100.0, rel_tol=_TOL, abs_tol=_TOL) or run > span or transitions > frames - 1 or (first == -1) != (last == -1) or (first == -1 and (span != 0 or run != 0)) or (first != -1 and (not 1 <= first <= last <= frames or span != last - first + 1 or run == 0)):
            raise ValueError("pVAD run/crossing invariant disagrees")
    return result


def _validate_task4_manifest(manifest: Mapping[str, object], ids: list[str]) -> None:
    """Apply Task 4's complete manifest-domain contract at the in-memory join."""
    if set(manifest) != _TASK4_MANIFEST_KEYS and set(manifest) != _TASK4_MANIFEST_KEYS | {"cuda"}:
        raise ValueError("pVAD cache manifest has an invalid exact top-level schema")
    for name, keys in _TASK4_NESTED_KEYS.items():
        value: object = manifest
        for part in name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise ValueError(f"pVAD cache manifest is missing {name}")
            value = value[part]
        if keys is not None and (not isinstance(value, Mapping) or set(value) != keys):
            raise ValueError(f"pVAD cache manifest has an invalid {name} schema")
    if manifest["digest_algorithms"] != _TASK4_DIGEST_ALGORITHMS:
        raise ValueError("pVAD cache manifest digest algorithms disagree")
    source = manifest["source"]
    assert isinstance(source, Mapping)
    audio = source["per_id_audio_sha256"]
    if not isinstance(audio, Mapping) or set(audio) != set(ids):
        raise ValueError("pVAD source audio coverage disagrees")
    for item in audio.values():
        if not isinstance(item, Mapping) or set(item) != {"wake_sha256", "command_sha256"}:
            raise ValueError("pVAD source audio schema disagrees")
    source_projection = {"jsonl_sha256": source["jsonl_sha256"], "per_id_audio_sha256": audio}
    if source["projection_sha256"] != _digest(source_projection):
        raise ValueError("pVAD source projection digest disagrees")
    from . import pvad_cache
    config_value = manifest["runtime_config"]
    if not isinstance(config_value, Mapping) or set(config_value) != {"sample_rate", "frame_samples", "enrollment_cap_seconds", "minimum_audio_seconds", "ema_alpha", "onnx_provider", "ecapa_device"}:
        raise ValueError("pVAD runtime config is invalid")
    try:
        config = dict(config_value)
        from .firered_pvad import PvadRuntimeConfig
        config = asdict(PvadRuntimeConfig(**config))
    except (TypeError, ValueError) as exc:
        raise ValueError("pVAD runtime config is invalid") from exc
    if config["onnx_provider"] != "CPUExecutionProvider" and not config["onnx_provider"].endswith("ExecutionProvider"):
        raise ValueError("pVAD runtime provider is invalid")
    if manifest["runtime_config_sha256"] != _digest(config) or manifest["provider"] != config["onnx_provider"] or manifest["device"] != config["ecapa_device"]:
        raise ValueError("pVAD runtime config identity disagrees")
    try:
        pvad_cache._validate_model_identity(manifest["model"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pVAD model identity is invalid") from exc
    environment = manifest["environment"]
    if not isinstance(environment["python"], str) or not environment["python"] or not isinstance(environment["platform"], str) or not environment["platform"] or set(environment["observed_dependencies"]) != set(pvad_cache._observed_dependencies()) or any(value is not None and not isinstance(value, str) for value in environment["observed_dependencies"].values()):
        raise ValueError("pVAD environment domain is invalid")
    try:
        # Task 4 owns the complete domain contract, including CUDA manifests.
        pvad_cache._validate_manifest_domains(manifest, ids)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pVAD cache manifest domains are invalid") from exc
    reuse = manifest["reuse"]
    if any(type(value) is not int or value < 0 for value in reuse.values()) or sum(reuse.values()) != len(ids):
        raise ValueError("pVAD reuse domain is invalid")
    parity = manifest["parity"]
    if parity not in ({"status": "not-run", "passed": None, "max_abs_feature_delta": None},) and not (parity["status"] == "passed" and parity["passed"] is True and type(parity["max_abs_feature_delta"]) in (int, float) and math.isfinite(parity["max_abs_feature_delta"]) and 0 <= parity["max_abs_feature_delta"] <= 1e-4):
        raise ValueError("pVAD parity domain is invalid")
    limit = manifest["limit"]
    source_ids = manifest["coverage"]["source"]["ids"]
    if set(limit) != {"value", "canonical", "reason"} or not ((limit["value"] is None and limit["canonical"] is True and limit["reason"] is None and ids == source_ids) or (type(limit["value"]) is int and limit["value"] > 0 and limit["canonical"] is False and limit["reason"] == "explicit noncanonical partial cache" and ids == source_ids[:min(limit["value"], len(source_ids))])):
        raise ValueError("pVAD limit domain is invalid")
    timing = manifest["timing"]
    cuda_expected = config["ecapa_device"].startswith("cuda")
    for name, percentile in timing.items():
        count = percentile["count"]
        expected = len(ids) if name in {"rtf", "peak_rss_delta_bytes"} or (name == "cuda_peak_bytes" and cuda_expected) else 0 if name == "cuda_peak_bytes" else None
        if type(count) is not int or count < 0 or (expected is not None and count != expected) or (count == 0 and any(percentile[key] is not None for key in ("p50", "p95", "max"))) or (count > 0 and any(type(percentile[key]) not in (int, float) or not math.isfinite(percentile[key]) or percentile[key] < 0 for key in ("p50", "p95", "max"))) or (count > 0 and not percentile["p50"] <= percentile["p95"] <= percentile["max"]):
            raise ValueError("pVAD timing domain is invalid")
    if timing["cold_elapsed_seconds"]["count"] + timing["warm_elapsed_seconds"]["count"] != len(ids):
        raise ValueError("pVAD timing coverage disagrees")
    if manifest["joined_state_sha256"] != _digest({sample_id: pvad_cache._resume_expected(sample_id, audio[sample_id], manifest["model"], config) for sample_id in ids}):
        raise ValueError("pVAD joined-state identity disagrees")
    selected = manifest["coverage"]["selected"]
    source_coverage = manifest["coverage"]["source"]
    if selected["ids"] != ids or selected["count"] != len(ids) or selected["id_sha256"] != _digest(ids) or not set(ids) <= set(source_coverage["ids"]):
        raise ValueError("pVAD cache coverage disagrees with joined IDs")
    if source_coverage["count"] != len(source_coverage["ids"]) or source_coverage["id_sha256"] != _digest(source_coverage["ids"]):
        raise ValueError("pVAD source coverage digest disagrees")


@dataclass(frozen=True)
class JoinedPvadRow:
    id: str
    group: str
    target_present: int
    pvad: Mapping[str, float]
    e0: Mapping[str, float]
    source_digest: str


def join_pvad_e0_rows(
    rows: Sequence[CandidateRow], labels: Mapping[str, str | None], groups: Mapping[str, str],
    cache_records: Sequence[Mapping[str, object]], cache_manifest: Mapping[str, object],
) -> list[JoinedPvadRow]:
    """Validate and exactly join canonical rows, cache records, and provenance."""
    canonical: dict[str, CandidateRow] = {}
    for row in rows:
        sid = _id(row.id)
        if sid in canonical:
            raise ValueError("duplicate canonical id")
        canonical[sid] = row
    if not canonical or set(labels) != set(canonical) or set(groups) != set(canonical):
        raise ValueError("canonical labels/groups do not have exact ID coverage")
    cache: dict[str, Mapping[str, float]] = {}
    for record in cache_records:
        if not isinstance(record, Mapping) or set(record) != {"id", "features"}:
            raise ValueError("pVAD cache record must contain exactly id and features")
        sid = _id(record["id"])
        if sid in cache:
            raise ValueError("duplicate pVAD cache id")
        cache[sid] = _validate_features(record["features"])
    if set(cache) != set(canonical):
        raise ValueError("pVAD cache ID coverage has missing or extra IDs")
    if not isinstance(cache_manifest, Mapping):
        raise ValueError("pVAD cache manifest is required")
    if cache_manifest.get("artifact_kind") != "r11_e2_firered_cache" or cache_manifest.get("schema_version") != "v1":
        raise ValueError("pVAD cache manifest is not the Task 4 identity contract")
    _reject_private(cache_manifest)
    _validate_task4_manifest(cache_manifest, [row.id for row in rows])
    if cache_manifest.get("feature_schema") != list(PVAD_GATE_FEATURE_SCHEMA):
        raise ValueError("pVAD cache feature schema disagrees")
    expected_schema_digest = hashlib.sha256((r"\n".join(PVAD_GATE_FEATURE_SCHEMA) + r"\n").encode("utf-8")).hexdigest()
    schema_digest = cache_manifest.get("feature_schema_sha256")
    if schema_digest != expected_schema_digest:
        raise ValueError("pVAD cache feature schema digest disagrees")
    record_lines = [_ordered_json({"id": record["id"], "features": record["features"]}) for record in cache_records]
    record_text = "\n".join(record_lines) + "\n"
    per_id = {record["id"]: hashlib.sha256(line.encode("utf-8")).hexdigest() for record, line in zip(cache_records, record_lines)}
    if _sha(cache_manifest.get("records_sha256"), "pVAD records digest") != hashlib.sha256(record_text.encode("utf-8")).hexdigest() or cache_manifest.get("per_id_record_sha256") != per_id:
        raise ValueError("pVAD record digests disagree")
    coverage = cache_manifest.get("coverage")
    if not isinstance(coverage, Mapping) or not isinstance(coverage.get("selected"), Mapping):
        raise ValueError("pVAD cache manifest coverage is invalid")
    selected = coverage["selected"]
    ids = selected.get("ids")
    if not isinstance(ids, list) or ids != [row.id for row in rows] or selected.get("count") != len(ids) or selected.get("id_sha256") != _digest(ids):
        raise ValueError("pVAD cache manifest coverage/order/digest disagrees")
    _sha(cache_manifest.get("joined_state_sha256"), "pVAD cache joined-state identity")
    source = cache_manifest["source"]
    model = cache_manifest["model"]
    source_digest = _sha(source["projection_sha256"], "pVAD source projection")
    model_digest = _sha(model["identity_sha256"], "pVAD model identity")
    joined: list[JoinedPvadRow] = []
    for row in rows:
        group = groups[row.id]
        if not isinstance(group, str) or not group:
            raise ValueError("wake_component group must be a nonempty string")
        # R10 candidate-text identity and pVAD audio/source identity are separate domains.
        _sha(row.source_digest, "canonical row source digest")
        full_e0 = dict(zip(GATE_FEATURE_SCHEMA, build_gate_feature_matrix([row])[0].tolist()))
        e0 = {name: full_e0[name] for name in E0_FITTING_FEATURE_SCHEMA}
        joined.append(JoinedPvadRow(row.id, group, int(labels[row.id] is not None), cache[row.id], e0, row.source_digest))
    return joined


def _outer_folds(joined: Sequence[JoinedPvadRow]) -> list[tuple[np.ndarray, np.ndarray]]:
    target = np.asarray([row.target_present for row in joined], dtype=np.int64)
    groups = np.asarray([row.group for row in joined], dtype=object)
    if set(target.tolist()) != {0, 1} or len(set(groups.tolist())) < 5:
        raise ValueError("a feasible five-way grouped split requires both target classes and five groups")
    try:
        folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros((len(joined), 1)), target, groups))
    except ValueError as exc:
        raise ValueError("a feasible five-way grouped split cannot be formed") from exc
    covered = np.full(len(joined), -1, dtype=np.int64)
    for index, (train, test) in enumerate(folds):
        if np.unique(target[train]).size != 2 or np.unique(target[test]).size != 2 or set(groups[train]) & set(groups[test]) or np.any(covered[test] != -1):
            raise ValueError("outer split is not feasible and group-disjoint")
        covered[test] = index
    if np.any(covered < 0):
        raise ValueError("outer split does not cover every row exactly once")
    return folds


def _matrix(rows: Sequence[JoinedPvadRow], names: Sequence[str]) -> np.ndarray:
    return np.asarray([[row.pvad.get(name, row.e0.get(name)) for name in names] for row in rows], dtype=np.float64)


def _probabilities(pipeline: object, X: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(pipeline.predict_proba(X), dtype=np.float64)
    if values.shape != (len(X), 2):
        raise ValueError(f"{label} probabilities must have shape (n_rows, 2)")
    scores = values[:, 1]
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"{label} probabilities must be finite and in [0, 1]")
    return scores


def _safe_threshold(value: float) -> float | str:
    return _REJECT_ALL if math.isinf(value) else float(value)


def _train_state(X: np.ndarray, target: np.ndarray) -> dict[str, object]:
    return {"feature_digest": _digest(X.tolist()), "target_count": {"negative": int((target == 0).sum()), "positive": int((target == 1).sum())}}


def _inner_select(scores: Mapping[str, np.ndarray], train_rows: Sequence[CandidateRow], labels: Mapping[str, str | None]) -> tuple[str, float, list[dict[str, object]]]:
    contributions = build_oracle_contributions(train_rows, labels)
    selected, frontier = _best_model_frontier(scores, contributions, RR_FLOOR)
    safe_frontier = [
        {**point, "threshold": _safe_threshold(float(point["threshold"]))}
        for point in frontier
    ]
    return str(selected["model"]), float(selected["threshold"]), safe_frontier


def _learned_family(name: str, joined: Sequence[JoinedPvadRow], original: Mapping[str, CandidateRow], labels: Mapping[str, str | None], folds: Sequence[tuple[np.ndarray, np.ndarray]], features: Sequence[str]) -> dict[str, object]:
    X, y, groups = _matrix(joined, features), np.asarray([row.target_present for row in joined]), np.asarray([row.group for row in joined], dtype=object)
    specs = default_model_specs()
    score_bank = {spec.name: np.full(len(joined), np.nan) for spec in specs}
    records: list[dict[str, object] | None] = [None] * len(joined)
    metadata: list[dict[str, object]] = []
    for fold_index, (train, test) in enumerate(folds):
        # Inner OOF selects the threshold and model without looking at outer test labels.
        inner_groups = groups[train]
        if len(set(inner_groups.tolist())) < 3:
            raise ValueError("inner search is infeasible")
        inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + fold_index + 1).split(X[train], y[train], inner_groups))
        inner_scores = {spec.name: np.full(len(train), np.nan) for spec in specs}
        for inner_index, (inner_train, inner_test) in enumerate(inner):
            if np.unique(y[train][inner_train]).size != 2 or np.unique(y[train][inner_test]).size != 2:
                raise ValueError("inner search is infeasible")
            for spec in specs:
                pipeline = _fit_gate_pipeline(spec, X[train][inner_train], y[train][inner_train], SEED + fold_index * 10 + inner_index)
                inner_scores[spec.name][inner_test] = _probabilities(pipeline, X[train][inner_test], "inner")
        if any(not np.isfinite(score).all() for score in inner_scores.values()):
            raise ValueError("inner search produced incomplete scores")
        train_rows = [original[joined[index].id] for index in train]
        selected_model, threshold, frontier = _inner_select(inner_scores, train_rows, labels)
        for spec in specs:
            pipeline = _fit_gate_pipeline(spec, X[train], y[train], SEED + fold_index)
            score_bank[spec.name][test] = _probabilities(pipeline, X[test], "outer")
        for index, score in zip(test, score_bank[selected_model][test]):
            records[index] = {"id": joined[index].id, "group": joined[index].group, "fold": fold_index, "model": selected_model, "score": float(score), "threshold": _safe_threshold(threshold), "action": "accept" if score >= threshold else "reject"}
        metadata.append({"fold": fold_index, "train_ids": [joined[i].id for i in train], "test_ids": [joined[i].id for i in test], "train_groups": sorted(set(groups[train])), "test_groups": sorted(set(groups[test])), "group_disjoint": True, "selected_model": selected_model, "selected_threshold": _safe_threshold(threshold), "inner_search_feasible": True, "inner_frontier": frontier, "training_state": _train_state(X[train], y[train])})
    return _family_result(name, features, records, metadata, specs, score_bank)


def _scalar_family(joined: Sequence[JoinedPvadRow], original: Mapping[str, CandidateRow], labels: Mapping[str, str | None], folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict[str, object]:
    records: list[dict[str, object] | None] = [None] * len(joined)
    groups = np.asarray([row.group for row in joined], dtype=object)
    metadata: list[dict[str, object]] = []
    score_bank = {feature: np.full(len(joined), np.nan) for feature in _SCALAR_FEATURES}
    for fold_index, (train, test) in enumerate(folds):
        scores = {feature: np.asarray([joined[i].pvad[feature] for i in train], dtype=np.float64) for feature in _SCALAR_FEATURES}
        selected, threshold, frontier = _inner_select(
            scores, [original[joined[i].id] for i in train], labels
        )
        point = select_frontier_point(gate_oracle_frontier(scores[selected], build_oracle_contributions([original[joined[i].id] for i in train], labels)), RR_FLOOR)
        if point is None:
            raise ValueError("scalar search is infeasible")
        if threshold != float(point["threshold"]):
            raise ValueError("scalar frontier selection disagrees")
        for index in test:
            score = float(joined[index].pvad[selected])
            records[index] = {"id": joined[index].id, "group": joined[index].group, "fold": fold_index, "model": selected, "score": score, "threshold": _safe_threshold(threshold), "action": "accept" if score >= threshold else "reject"}
            for feature in _SCALAR_FEATURES:
                score_bank[feature][index] = joined[index].pvad[feature]
        metadata.append({"fold": fold_index, "train_ids": [joined[i].id for i in train], "test_ids": [joined[i].id for i in test], "train_groups": sorted(set(groups[train])), "test_groups": sorted(set(groups[test])), "group_disjoint": True, "selected_model": selected, "selected_threshold": _safe_threshold(threshold), "inner_search_feasible": True, "inner_frontier": frontier, "training_state": _train_state(np.asarray([scores[selected]]).T, np.asarray([joined[i].target_present for i in train]))})
    scalar_specs = tuple(GateModelSpec(feature, "scalar", ()) for feature in _SCALAR_FEATURES)
    return _family_result("firered_scalar", _SCALAR_FEATURES, records, metadata, scalar_specs, score_bank)


def _family_result(name: str, features: Sequence[str], records: Sequence[dict[str, object] | None], folds: Sequence[dict[str, object]], specs: Sequence[GateModelSpec], score_bank: Mapping[str, np.ndarray] | None = None) -> dict[str, object]:
    if any(item is None for item in records):
        raise ValueError("OOF records do not cover every row")
    public = [item for item in records if item is not None]
    bank = {key: values.tolist() for key, values in (score_bank or {}).items()}
    if any(not np.isfinite(values).all() or np.any((values < 0) | (values > 1)) for values in (score_bank or {}).values()):
        raise ValueError("outer score bank is incomplete or invalid")
    return {"name": name, "feature_allowlist": list(features), "model_specs": [{"name": spec.name, "family": spec.family, "parameters": dict(spec.parameters)} for spec in specs], "rows": public, "folds": list(folds), "score_bank": bank, "outer_threshold_grid": [{"model": key, "thresholds": [_REJECT_ALL, *sorted(set(values), reverse=True)]} for key, values in bank.items()], "coverage": {"ids": [item["id"] for item in public], "ids_sha256": _digest([item["id"] for item in public]), "n_rows_total": len(public), "n_rows_covered": len({item["id"] for item in public}), "once_only": len(public) == len({item["id"] for item in public})}}


def run_pvad_oracle(rows: Sequence[CandidateRow], labels: Mapping[str, str | None], groups: Mapping[str, str], cache_records: Sequence[Mapping[str, object]], cache_manifest: Mapping[str, object]) -> dict[str, object]:
    """Run the three fixed non-deployable grouped OOF pVAD gate families."""
    joined = join_pvad_e0_rows(rows, labels, groups, cache_records, cache_manifest)
    folds = _outer_folds(joined)
    original = {row.id: row for row in rows}
    families = {
        "firered_scalar": _scalar_family(joined, original, labels, folds),
        "firered_crossfit": _learned_family("firered_crossfit", joined, original, labels, folds, PVAD_FITTING_FEATURE_SCHEMA),
        "firered_fused_crossfit": _learned_family("firered_fused_crossfit", joined, original, labels, folds, (*PVAD_FITTING_FEATURE_SCHEMA, *E0_FITTING_FEATURE_SCHEMA)),
    }
    return {"diagnostic_only": True, "deployable": False, "branch_decision": None, "seed": SEED, "rr_floor": RR_FLOOR, "source_joined_state": {"cache_joined_state_sha256": cache_manifest["joined_state_sha256"], "cache_source_projection_sha256": cache_manifest["source"]["projection_sha256"], "cache_model_identity_sha256": cache_manifest["model"]["identity_sha256"], "joined_rows_sha256": _digest([{"id": row.id, "group": row.group, "pvad": row.pvad, "e0": row.e0, "source_digest": row.source_digest} for row in joined])}, "families": families}
