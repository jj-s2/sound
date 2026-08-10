"""Diagnostic-only grouped OOF FireRed pVAD gate families.

This module deliberately has no writer or branch-decision logic.  Its returned
records contain gates only, never references, hypotheses, CER vectors, or the
oracle candidate selected for an accepted positive.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .firered_pvad import PVAD_GATE_FEATURE_SCHEMA
from .r10_selector import CandidateRow
from .r11_gate_oracle import (
    GateModelSpec,
    _fit_gate_pipeline,
    _best_model_frontier,
    build_oracle_contributions,
    default_model_specs,
    gate_oracle_frontier,
    select_frontier_point,
)


SEED = 20260807
RR_FLOOR = 0.93
# Timing is audit-only.  These are the frozen, label-free E0 cached fields.
E0_FITTING_FEATURE_SCHEMA = (
    "presence_score", "enhanced_cosine", "mixture_cosine", "max_cosine",
    "cmd_duration_sec", "cmd_rms",
)
_AUDIT_NAMES = {"elapsed", "rtf", "rss", "memory", "peak", "latency", "device", "provider", "phase"}
PVAD_FITTING_FEATURE_SCHEMA = tuple(
    name for name in PVAD_GATE_FEATURE_SCHEMA
    if not any(token in name.lower() for token in _AUDIT_NAMES)
)
_SCALAR_FEATURES = ("raw_mean", "ema_mean", "ema_fraction_ge_0_5")
_REJECT_ALL = "reject_all"


def canonical_json(value: object) -> str:
    """Return deterministic JSON suitable for digests and byte-stable tests."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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
    _counter(value["dropped_tail_samples"], "dropped_tail_samples")
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
    return result


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
    forbidden = {key for key in cache_manifest if any(token in key.lower() for token in ("label", "reference", "candidate", "cer", "action"))}
    if forbidden:
        raise ValueError("pVAD cache manifest contains forbidden label/reference data")
    coverage = cache_manifest.get("coverage")
    if not isinstance(coverage, Mapping) or not isinstance(coverage.get("selected"), Mapping):
        raise ValueError("pVAD cache manifest coverage is invalid")
    selected = coverage["selected"]
    ids = selected.get("ids")
    if not isinstance(ids, list) or ids != [row.id for row in rows] or selected.get("count") != len(ids) or selected.get("id_sha256") != _digest(ids):
        raise ValueError("pVAD cache manifest coverage/order/digest disagrees")
    for key in ("joined_state_sha256",):
        value = cache_manifest.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("pVAD cache joined-state identity is invalid")
    source = cache_manifest.get("source")
    model = cache_manifest.get("model")
    if not isinstance(source, Mapping) or not isinstance(model, Mapping) or not isinstance(source.get("projection_sha256"), str) or not isinstance(model.get("identity_sha256"), str):
        raise ValueError("pVAD cache source/model identity is invalid")
    joined: list[JoinedPvadRow] = []
    for row in rows:
        group = groups[row.id]
        if not isinstance(group, str) or not group:
            raise ValueError("wake_component group must be a nonempty string")
        e0 = {name: _finite(row.audio_features.get(name), f"E0 feature {name}") for name in E0_FITTING_FEATURE_SCHEMA}
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


def _train_state(X: np.ndarray, target: np.ndarray) -> dict[str, object]:
    return {"feature_digest": _digest(X.tolist()), "target_count": {"negative": int((target == 0).sum()), "positive": int((target == 1).sum())}}


def _inner_select(scores: Mapping[str, np.ndarray], train_rows: Sequence[CandidateRow], labels: Mapping[str, str | None]) -> tuple[str, float, list[dict[str, object]]]:
    contributions = build_oracle_contributions(train_rows, labels)
    selected, frontier = _best_model_frontier(scores, contributions, RR_FLOOR)
    safe_frontier = [
        {**point, "threshold": _REJECT_ALL if math.isinf(float(point["threshold"])) else float(point["threshold"])}
        for point in frontier
    ]
    return str(selected["model"]), float(selected["threshold"]), safe_frontier


def _learned_family(name: str, joined: Sequence[JoinedPvadRow], original: Mapping[str, CandidateRow], labels: Mapping[str, str | None], folds: Sequence[tuple[np.ndarray, np.ndarray]], features: Sequence[str]) -> dict[str, object]:
    X, y, groups = _matrix(joined, features), np.asarray([row.target_present for row in joined]), np.asarray([row.group for row in joined], dtype=object)
    specs = default_model_specs()
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
                inner_scores[spec.name][inner_test] = pipeline.predict_proba(X[train][inner_test])[:, 1]
        if any(not np.isfinite(score).all() for score in inner_scores.values()):
            raise ValueError("inner search produced incomplete scores")
        train_rows = [original[joined[index].id] for index in train]
        selected_model, threshold, frontier = _inner_select(inner_scores, train_rows, labels)
        spec = next(item for item in specs if item.name == selected_model)
        pipeline = _fit_gate_pipeline(spec, X[train], y[train], SEED + fold_index)
        scores = pipeline.predict_proba(X[test])[:, 1]
        for index, score in zip(test, scores):
            records[index] = {"id": joined[index].id, "group": joined[index].group, "fold": fold_index, "model": selected_model, "score": float(score), "threshold": threshold, "action": "accept" if score >= threshold else "reject"}
        metadata.append({"fold": fold_index, "train_ids": [joined[i].id for i in train], "test_ids": [joined[i].id for i in test], "train_groups": sorted(set(groups[train])), "test_groups": sorted(set(groups[test])), "group_disjoint": True, "selected_model": selected_model, "selected_threshold": threshold, "inner_search_feasible": True, "inner_frontier": frontier, "training_state": _train_state(X[train], y[train])})
    return _family_result(name, features, records, metadata, specs)


def _scalar_family(joined: Sequence[JoinedPvadRow], original: Mapping[str, CandidateRow], labels: Mapping[str, str | None], folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict[str, object]:
    records: list[dict[str, object] | None] = [None] * len(joined)
    groups = np.asarray([row.group for row in joined], dtype=object)
    metadata: list[dict[str, object]] = []
    for fold_index, (train, test) in enumerate(folds):
        scores = {feature: np.asarray([joined[i].pvad[feature] for i in train], dtype=np.float64) for feature in _SCALAR_FEATURES}
        selected, threshold, _frontier = _inner_select(
            scores, [original[joined[i].id] for i in train], labels
        )
        point = select_frontier_point(gate_oracle_frontier(scores[selected], build_oracle_contributions([original[joined[i].id] for i in train], labels)), RR_FLOOR)
        if point is None:
            raise ValueError("scalar search is infeasible")
        if threshold != float(point["threshold"]):
            raise ValueError("scalar frontier selection disagrees")
        for index in test:
            score = float(joined[index].pvad[selected])
            records[index] = {"id": joined[index].id, "group": joined[index].group, "fold": fold_index, "model": selected, "score": score, "threshold": threshold, "action": "accept" if score >= threshold else "reject"}
        metadata.append({"fold": fold_index, "train_ids": [joined[i].id for i in train], "test_ids": [joined[i].id for i in test], "train_groups": sorted(set(groups[train])), "test_groups": sorted(set(groups[test])), "group_disjoint": True, "selected_model": selected, "selected_threshold": threshold, "inner_search_feasible": True, "training_state": _train_state(np.asarray([scores[selected]]).T, np.asarray([joined[i].target_present for i in train]))})
    return _family_result("firered_scalar", _SCALAR_FEATURES, records, metadata, ())


def _family_result(name: str, features: Sequence[str], records: Sequence[dict[str, object] | None], folds: Sequence[dict[str, object]], specs: Sequence[GateModelSpec]) -> dict[str, object]:
    if any(item is None for item in records):
        raise ValueError("OOF records do not cover every row")
    public = [item for item in records if item is not None]
    return {"name": name, "feature_allowlist": list(features), "model_specs": [{"name": spec.name, "family": spec.family, "parameters": dict(spec.parameters)} for spec in specs], "rows": public, "folds": list(folds), "coverage": {"n_rows_total": len(public), "n_rows_covered": len({item["id"] for item in public}), "once_only": len(public) == len({item["id"] for item in public})}}


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
