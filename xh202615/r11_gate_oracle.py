"""Label-free R11 gate features and pooled gate-oracle metrics."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .metrics import cer_stats
from .r10_selector import CANDIDATE_ACTIONS, CandidateRow
from .text import normalize_text


_ACOUSTIC_FIELDS = (
    "presence_score",
    "enhanced_cosine",
    "mixture_cosine",
    "max_cosine",
    "latency_ms",
    "cmd_duration_sec",
    "cmd_rms",
)
_TEXT_SHAPE_FIELDS = ("length", "empty", "digit_ratio", "chinese_ratio")
_PAIRWISE_SOURCES = tuple(
    (left, right)
    for index, left in enumerate(CANDIDATE_ACTIONS)
    for right in CANDIDATE_ACTIONS[index + 1 :]
)
_DIGIT_RE = re.compile(r"\d")
_CHINESE_RE = re.compile(r"[一-鿿]")


GATE_FEATURE_SCHEMA: tuple[str, ...] = (
    *(name for field in _ACOUSTIC_FIELDS for name in (field, f"{field}_missing")),
    *(
        f"{source}_{field}"
        for source in CANDIDATE_ACTIONS
        for field in _TEXT_SHAPE_FIELDS
    ),
    *(f"distance_{left}_{right}" for left, right in _PAIRWISE_SOURCES),
    "n_nonempty_candidates",
    "n_unique_candidates",
    "candidate_length_mean",
    "candidate_length_std",
    "candidate_length_min",
    "candidate_length_max",
    "candidate_length_range",
)


def _finite_or_zero(value: object) -> tuple[float, float]:
    try:
        finite = float(value)
    except (TypeError, ValueError):
        return 0.0, 1.0
    if not math.isfinite(finite):
        return 0.0, 1.0
    return finite, 0.0


def _text_shape(text: str | None) -> dict[str, float]:
    normalized = normalize_text(text)
    length = len(normalized)
    if length == 0:
        return {"length": 0.0, "empty": 1.0, "digit_ratio": 0.0, "chinese_ratio": 0.0}
    return {
        "length": float(length),
        "empty": 0.0,
        "digit_ratio": len(_DIGIT_RE.findall(normalized)) / length,
        "chinese_ratio": len(_CHINESE_RE.findall(normalized)) / length,
    }


def _hypothesis_distance(left: str | None, right: str | None) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    denominator = max(len(left_norm), len(right_norm))
    if denominator == 0:
        return 0.0
    return cer_stats(left_norm, right_norm).errors / denominator


def build_gate_feature_matrix(rows: Sequence[CandidateRow]) -> np.ndarray:
    """Build one finite, label-free float64 feature vector per candidate row."""

    matrix = np.empty((len(rows), len(GATE_FEATURE_SCHEMA)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        features: dict[str, float] = {}
        for field in _ACOUSTIC_FIELDS:
            value, missing = _finite_or_zero(row.audio_features.get(field))
            features[field] = value
            features[f"{field}_missing"] = missing

        normalized_texts: dict[str, str] = {}
        lengths: list[float] = []
        for source in CANDIDATE_ACTIONS:
            text = row.texts.get(source, "")
            normalized_texts[source] = normalize_text(text)
            shape = _text_shape(text)
            lengths.append(shape["length"])
            for field in _TEXT_SHAPE_FIELDS:
                features[f"{source}_{field}"] = shape[field]

        for left, right in _PAIRWISE_SOURCES:
            features[f"distance_{left}_{right}"] = _hypothesis_distance(
                normalized_texts[left], normalized_texts[right]
            )

        nonempty = [text for text in normalized_texts.values() if text]
        features["n_nonempty_candidates"] = float(len(nonempty))
        features["n_unique_candidates"] = float(len(set(nonempty)))
        length_array = np.asarray(lengths, dtype=np.float64)
        features["candidate_length_mean"] = float(length_array.mean())
        features["candidate_length_std"] = float(length_array.std())
        features["candidate_length_min"] = float(length_array.min())
        features["candidate_length_max"] = float(length_array.max())
        features["candidate_length_range"] = float(length_array.max() - length_array.min())

        matrix[row_index] = [features[name] for name in GATE_FEATURE_SCHEMA]
    return matrix


@dataclass(frozen=True)
class OracleContributions:
    """Per-row integer contributions for accepted and rejected gate decisions."""

    substitutions: np.ndarray
    insertions: np.ndarray
    deletions: np.ndarray
    ref_chars: np.ndarray
    is_positive: np.ndarray
    chosen_actions: tuple[str, ...]


def build_oracle_contributions(
    rows: Sequence[CandidateRow], labels: Mapping[str, str | None]
) -> OracleContributions:
    """Build accepted-positive oracle edits and row class indicators."""

    n_rows = len(rows)
    substitutions = np.zeros(n_rows, dtype=np.int64)
    insertions = np.zeros(n_rows, dtype=np.int64)
    deletions = np.zeros(n_rows, dtype=np.int64)
    ref_chars = np.zeros(n_rows, dtype=np.int64)
    is_positive = np.zeros(n_rows, dtype=np.bool_)
    chosen_actions: list[str] = []

    for index, row in enumerate(rows):
        label = labels[row.id]
        if label is None:
            chosen_actions.append("reject")
            continue

        is_positive[index] = True
        best_action = CANDIDATE_ACTIONS[0]
        best_stats = cer_stats(label, row.texts.get(best_action, ""))
        for action in CANDIDATE_ACTIONS[1:]:
            stats = cer_stats(label, row.texts.get(action, ""))
            if stats.cer < best_stats.cer:
                best_action = action
                best_stats = stats

        substitutions[index] = best_stats.substitutions
        insertions[index] = best_stats.insertions
        deletions[index] = best_stats.deletions
        ref_chars[index] = best_stats.ref_chars
        chosen_actions.append(best_action)

    return OracleContributions(
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        ref_chars=ref_chars,
        is_positive=is_positive,
        chosen_actions=tuple(chosen_actions),
    )


def _validate_contributions(contributions: OracleContributions) -> int:
    lengths = {
        len(contributions.substitutions),
        len(contributions.insertions),
        len(contributions.deletions),
        len(contributions.ref_chars),
        len(contributions.is_positive),
        len(contributions.chosen_actions),
    }
    if len(lengths) != 1:
        raise ValueError("oracle contribution arrays must have equal lengths")
    return lengths.pop()


def gate_oracle_frontier(
    scores: Sequence[float], contributions: OracleContributions
) -> list[dict[str, float]]:
    """Evaluate reject-all and each unique ``score >= threshold`` boundary."""

    n_rows = _validate_contributions(contributions)
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1 or len(score_array) != n_rows:
        raise ValueError("scores and contributions must have equal one-dimensional lengths")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")

    positive = np.asarray(contributions.is_positive, dtype=np.bool_)
    negative = ~positive
    total_ref_chars = int(np.asarray(contributions.ref_chars, dtype=np.int64).sum())
    negative_count = int(negative.sum())

    def point(threshold: float, accepted: np.ndarray) -> dict[str, float]:
        accepted_positive = accepted & positive
        rejected_positive = ~accepted & positive
        accepted_negative = accepted & negative

        substitutions = int(contributions.substitutions[accepted_positive].sum())
        insertions = int(contributions.insertions[accepted_positive].sum())
        deletions = int(contributions.deletions[accepted_positive].sum())
        deletions += int(contributions.ref_chars[rejected_positive].sum())
        errors = substitutions + insertions + deletions
        cer = errors / total_ref_chars if total_ref_chars else 0.0
        accepted_negative_count = int(accepted_negative.sum())
        rr = (
            (negative_count - accepted_negative_count) / negative_count
            if negative_count
            else 0.0
        )
        overall = ((1.0 - cer) + rr) / 2.0
        return {
            "threshold": float(threshold),
            "cer": float(cer),
            "rr": float(rr),
            "overall": float(overall),
            "accepted_positives": float(accepted_positive.sum()),
            "accepted_negatives": float(accepted_negative_count),
        }

    points = [point(math.inf, np.zeros(n_rows, dtype=np.bool_))]
    for threshold in sorted(np.unique(score_array), reverse=True):
        points.append(point(float(threshold), score_array >= threshold))
    return points


def select_frontier_point(
    points: Sequence[dict[str, float]], rr_floor: float
) -> dict[str, float] | None:
    """Select the best feasible point, preferring RR then conservative ties."""

    feasible = [point for point in points if point["rr"] >= rr_floor]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda point: (
            point["overall"],
            point["rr"],
            -point["cer"],
            point["threshold"],
        ),
    )
