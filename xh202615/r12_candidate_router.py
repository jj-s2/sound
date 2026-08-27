"""Label-free inference features for the deployable R12 candidate router."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import pickle
from typing import Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .metrics import cer_stats
from .r10_selector import CandidateRow
from .r11_pvad_oracle import JoinedPvadRow
from .r12_personal_vad import PERSONAL_VAD_FEATURE_SCHEMA


ROUTER_ACTIONS = ("primary", "r3", "tse", "energy")
_ROUTER_PARAMETERS = {
    "max_leaf_nodes": 7,
    "l2_regularization": 1.0,
    "loss": "squared_error",
}


@dataclass(frozen=True)
class RouterRowKey:
    id: str
    action: str


@dataclass(frozen=True)
class TrainCandidateRouter:
    feature_schema: tuple[str, ...]
    model: object
    fit_row_count: int
    fit_group_count: int
    seed: int

    def to_public_dict(self) -> dict[str, object]:
        """Return a label-free, deterministic identity for the frozen router."""
        parameters = {**_ROUTER_PARAMETERS, "random_state": self.seed}
        return {
            "action_order": list(ROUTER_ACTIONS),
            "feature_schema": list(self.feature_schema),
            "feature_schema_digest": _digest(self.feature_schema),
            "parameters_digest": _digest(parameters),
            "model_digest": hashlib.sha256(pickle.dumps(self.model, protocol=5)).hexdigest(),
        }


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _similarity(left: str, right: str) -> float:
    denominator = max(len(left), len(right), 1)
    return 1.0 - cer_stats(left, right).errors / denominator


def _features(row: CandidateRow, joined: JoinedPvadRow, action: str) -> dict[str, float]:
    texts = row.texts
    text = texts[action]
    others = [texts[name] for name in ROUTER_ACTIONS if name != action]
    similarities = [_similarity(text, other) for other in others]
    lengths = [len(texts[name]) for name in ROUTER_ACTIONS]
    features: dict[str, float] = {
        **{f"action_{name}": float(name == action) for name in ROUTER_ACTIONS},
        "candidate_empty": float(not text),
        "candidate_length": float(len(text)),
        "candidate_similarity_mean": float(np.mean(similarities)),
        "candidate_similarity_max": float(np.max(similarities)),
        "candidate_agreement_count": float(sum(text == other for other in others)),
        "nonempty_candidate_count": float(sum(bool(value) for value in texts.values())),
        "candidate_length_from_mean": float(abs(len(text) - np.mean(lengths))),
        "candidate_length_range": float(max(lengths) - min(lengths)),
    }
    for prefix, mapping in (
        ("pvad", joined.pvad),
        ("e0", joined.e0),
        ("personal", joined.personal_vad),
    ):
        for name in sorted(mapping):
            features[f"{prefix}_{name}"] = _finite(mapping[name])
    for name in PERSONAL_VAD_FEATURE_SCHEMA:
        features.setdefault(f"personal_{name}", _finite(joined.personal_vad.get(name)))
    return features


def build_router_matrix(
    rows: Sequence[CandidateRow], joined_rows: Sequence[JoinedPvadRow]
) -> tuple[np.ndarray, tuple[RouterRowKey, ...], tuple[str, ...]]:
    """Expand each inference row into fixed-action, label-free feature rows."""
    if len(rows) != len(joined_rows) or [row.id for row in rows] != [row.id for row in joined_rows]:
        raise ValueError("rows and joined_rows must have matching ordered IDs")
    if not rows:
        raise ValueError("router requires at least one row")
    raw_features: list[dict[str, float]] = []
    keys: list[RouterRowKey] = []
    for row, joined in zip(rows, joined_rows):
        for action in ROUTER_ACTIONS:
            raw_features.append(_features(row, joined, action))
            keys.append(RouterRowKey(row.id, action))
    schema = tuple(raw_features[0])
    matrix = np.asarray([[feature[name] for name in schema] for feature in raw_features], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("router matrix must be finite")
    return matrix, tuple(keys), schema


def fit_train_candidate_router(
    rows: Sequence[CandidateRow],
    joined_rows: Sequence[JoinedPvadRow],
    labels: Mapping[str, str | None],
    seed: int,
) -> TrainCandidateRouter:
    """Fit candidate error prediction from train-role positive rows only."""
    if set(labels) != {row.id for row in rows}:
        raise ValueError("router labels must exactly cover train rows")
    positive_indices = [index for index, row in enumerate(rows) if labels[row.id] is not None]
    groups = {joined_rows[index].group for index in positive_indices}
    if len(groups) < 3:
        raise ValueError("router requires positive samples from at least three groups")
    positive_rows = [rows[index] for index in positive_indices]
    positive_joined = [joined_rows[index] for index in positive_indices]
    matrix, keys, schema = build_router_matrix(positive_rows, positive_joined)
    positive_by_id = {row.id: row for row in positive_rows}
    targets = np.asarray(
        [
            min(1.0, cer_stats(str(labels[key.id]), positive_by_id[key.id].texts[key.action]).cer)
            for key in keys
        ],
        dtype=np.float64,
    )
    if not np.isfinite(targets).all():
        raise ValueError("router targets must be finite")
    model = HistGradientBoostingRegressor(
        **_ROUTER_PARAMETERS, random_state=seed
    ).fit(matrix, targets)
    return TrainCandidateRouter(schema, model, len(keys), len(groups), seed)


def predict_router_actions(
    router: TrainCandidateRouter,
    rows: Sequence[CandidateRow],
    joined_rows: Sequence[JoinedPvadRow],
) -> tuple[str, ...]:
    matrix, keys, schema = build_router_matrix(rows, joined_rows)
    if schema != router.feature_schema:
        raise ValueError("router feature schema drift")
    values = np.asarray(router.model.predict(matrix), dtype=np.float64)
    if values.shape != (len(keys),) or not np.isfinite(values).all():
        raise ValueError("router predictions must be finite and complete")
    actions: list[str] = []
    for offset in range(0, len(keys), len(ROUTER_ACTIONS)):
        action_scores = values[offset : offset + len(ROUTER_ACTIONS)]
        actions.append(ROUTER_ACTIONS[int(np.argmin(action_scores))])
    return tuple(actions)
