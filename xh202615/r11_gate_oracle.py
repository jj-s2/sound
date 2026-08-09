"""Label-free R11 gate features and pooled gate-oracle metrics."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
class GateModelSpec:
    """One frozen CPU gate-model configuration."""

    name: str
    family: str
    parameters: tuple[tuple[str, float | int], ...]


@dataclass(frozen=True)
class CrossFitResult:
    """Group-disjoint OOF probabilities and shared fold metadata."""

    specs: tuple[GateModelSpec, ...]
    scores_by_model: dict[str, np.ndarray]
    fold_assignments: np.ndarray
    fold_metadata: tuple[dict[str, object], ...]


def default_model_specs() -> tuple[GateModelSpec, ...]:
    """Return the predeclared E0 CPU model grid."""

    logistic = tuple(
        GateModelSpec(f"logistic_c_{C:g}", "logistic", (("C", C),))
        for C in (0.01, 0.1, 1.0, 10.0)
    )
    hist = tuple(
        GateModelSpec(
            f"hist_gradient_boosting_leaf_{max_leaf_nodes}",
            "hist_gradient_boosting",
            (
                ("max_leaf_nodes", max_leaf_nodes),
                ("learning_rate", 0.05),
                ("max_iter", 150),
                ("l2_regularization", 1.0),
            ),
        )
        for max_leaf_nodes in (3, 7)
    )
    return logistic + hist


def _fit_gate_pipeline(
    spec: GateModelSpec, X_train: np.ndarray, y_train: np.ndarray, seed: int
) -> Pipeline:
    parameters = dict(spec.parameters)
    if spec.family == "logistic":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(parameters["C"]),
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    elif spec.family == "hist_gradient_boosting":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_leaf_nodes=int(parameters["max_leaf_nodes"]),
                        learning_rate=float(parameters["learning_rate"]),
                        max_iter=int(parameters["max_iter"]),
                        l2_regularization=float(parameters["l2_regularization"]),
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"unknown gate model family: {spec.family}")
    return pipeline.fit(X_train, y_train)


def cross_fit_gate_models(
    X: np.ndarray,
    target_present: Sequence[bool | int],
    groups: Sequence[object],
    *,
    n_splits: int,
    seed: int,
    specs: Sequence[GateModelSpec],
) -> CrossFitResult:
    """Produce once-only group-disjoint OOF probabilities for every model spec."""

    matrix = np.asarray(X, dtype=np.float64)
    raw_target = np.asarray(target_present)
    group_array = np.asarray(groups)
    frozen_specs = tuple(specs)
    if matrix.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if raw_target.ndim != 1 or group_array.ndim != 1:
        raise ValueError("target_present and groups must be one-dimensional")
    if len(matrix) != len(raw_target) or len(raw_target) != len(group_array):
        raise ValueError("X, target_present, and groups must have equal lengths")
    if not np.isin(raw_target, [0, 1]).all():
        raise ValueError("target_present must be binary")
    target = raw_target.astype(np.int64)
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("target_present must contain both target classes")
    if not frozen_specs:
        raise ValueError("at least one gate model spec is required")
    if len({spec.name for spec in frozen_specs}) != len(frozen_specs):
        raise ValueError("gate model spec names must be unique")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    folds = list(splitter.split(matrix, target, group_array))
    fold_assignments = np.full(len(target), -1, dtype=np.int64)
    fold_metadata: list[dict[str, object]] = []
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        if np.unique(target[train_indices]).size != 2 or np.unique(target[test_indices]).size != 2:
            raise ValueError(f"fold {fold_index} must contain both target classes")
        train_groups = set(group_array[train_indices].tolist())
        test_groups = set(group_array[test_indices].tolist())
        if train_groups & test_groups:
            raise ValueError(f"fold {fold_index} has train/test group overlap")
        if np.any(fold_assignments[test_indices] != -1):
            raise ValueError("OOF rows must be covered exactly once")
        fold_assignments[test_indices] = fold_index
        fold_metadata.append(
            {
                "fold_index": fold_index,
                "train_indices": train_indices.tolist(),
                "test_indices": test_indices.tolist(),
                "train_groups": sorted(train_groups, key=str),
                "test_groups": sorted(test_groups, key=str),
            }
        )
    if np.any(fold_assignments < 0):
        raise ValueError("OOF rows must be covered exactly once")

    scores_by_model: dict[str, np.ndarray] = {}
    for spec in frozen_specs:
        scores = np.full(len(target), np.nan, dtype=np.float64)
        for fold_index, (train_indices, test_indices) in enumerate(folds):
            pipeline = _fit_gate_pipeline(
                spec, matrix[train_indices], target[train_indices], seed + fold_index
            )
            scores[test_indices] = pipeline.predict_proba(matrix[test_indices])[:, 1]
        if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(f"model {spec.name} produced invalid OOF probabilities")
        scores_by_model[spec.name] = scores

    return CrossFitResult(
        specs=frozen_specs,
        scores_by_model=scores_by_model,
        fold_assignments=fold_assignments,
        fold_metadata=tuple(fold_metadata),
    )


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
    """Evaluate reject-all and each atomic ``score >= threshold`` boundary."""

    n_rows = _validate_contributions(contributions)
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1 or len(score_array) != n_rows:
        raise ValueError("scores and contributions must have equal one-dimensional lengths")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")

    positive = np.asarray(contributions.is_positive, dtype=np.bool_)
    substitutions = np.asarray(contributions.substitutions, dtype=np.int64)
    insertions = np.asarray(contributions.insertions, dtype=np.int64)
    deletions = np.asarray(contributions.deletions, dtype=np.int64)
    ref_chars = np.asarray(contributions.ref_chars, dtype=np.int64)
    total_ref_chars = int(ref_chars.sum())
    negative_count = int((~positive).sum())
    errors = total_ref_chars
    accepted_positive_count = 0
    accepted_negative_count = 0

    def point(threshold: float) -> dict[str, float]:
        cer = errors / total_ref_chars if total_ref_chars else 0.0
        rr = (
            (negative_count - accepted_negative_count) / negative_count
            if negative_count
            else 0.0
        )
        return {
            "threshold": float(threshold),
            "cer": float(cer),
            "rr": float(rr),
            "overall": float(((1.0 - cer) + rr) / 2.0),
            "accepted_positives": float(accepted_positive_count),
            "accepted_negatives": float(accepted_negative_count),
        }

    points = [point(math.inf)]
    order = np.argsort(-score_array, kind="mergesort")
    start = 0
    while start < n_rows:
        threshold = score_array[order[start]]
        end = start + 1
        while end < n_rows and score_array[order[end]] == threshold:
            end += 1
        tied_indices = order[start:end]
        tied_positive = tied_indices[positive[tied_indices]]
        accepted_positive_count += len(tied_positive)
        accepted_negative_count += len(tied_indices) - len(tied_positive)
        errors += int(
            (
                substitutions[tied_positive]
                + insertions[tied_positive]
                + deletions[tied_positive]
                - ref_chars[tied_positive]
            ).sum()
        )
        points.append(point(float(threshold)))
        start = end
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


def _subset_contributions(
    contributions: OracleContributions, indices: np.ndarray
) -> OracleContributions:
    return OracleContributions(
        substitutions=np.asarray(contributions.substitutions)[indices],
        insertions=np.asarray(contributions.insertions)[indices],
        deletions=np.asarray(contributions.deletions)[indices],
        ref_chars=np.asarray(contributions.ref_chars)[indices],
        is_positive=np.asarray(contributions.is_positive)[indices],
        chosen_actions=tuple(contributions.chosen_actions[int(index)] for index in indices),
    )


def _best_model_frontier(
    scores_by_model: Mapping[str, Sequence[float]],
    contributions: OracleContributions,
    rr_floor: float,
    *,
    include_frontier: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select the pooled best point with stable model/threshold tie-breaking.

    Models are visited by ascending name, so metric-identical models keep the
    stable first name. ``select_frontier_point`` resolves within-model ties by
    RR, CER, then the more conservative higher threshold.
    """

    best: dict[str, object] | None = None
    best_key: tuple[float, float, float] | None = None
    annotated_frontier: list[dict[str, object]] = []
    for model_name in sorted(scores_by_model):
        points = gate_oracle_frontier(scores_by_model[model_name], contributions)
        if include_frontier:
            annotated_frontier.extend({"model": model_name, **point} for point in points)
        selected = select_frontier_point(points, rr_floor)
        if selected is None:
            continue
        candidate: dict[str, object] = {"model": model_name, **selected}
        candidate_key = (
            float(selected["overall"]),
            float(selected["rr"]),
            -float(selected["cer"]),
        )
        if best_key is None or candidate_key > best_key:
            best = candidate
            best_key = candidate_key
    if best is None:
        raise ValueError("no model has a feasible frontier point at the RR floor")
    return best, annotated_frontier


def group_bootstrap_best_frontier(
    scores_by_model: Mapping[str, Sequence[float]],
    contributions: OracleContributions,
    groups: Sequence[object],
    *,
    rr_floor: float,
    n_boot: int,
    seed: int,
) -> dict:
    """Group-bootstrap the best feasible model/threshold with reselection."""

    n_rows = _validate_contributions(contributions)
    group_array = np.asarray(groups)
    if group_array.ndim != 1 or len(group_array) != n_rows:
        raise ValueError("groups and contributions must have equal one-dimensional lengths")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not scores_by_model:
        raise ValueError("scores_by_model must not be empty")

    frozen_scores: dict[str, np.ndarray] = {}
    for model_name, scores in scores_by_model.items():
        score_array = np.asarray(scores, dtype=np.float64)
        if score_array.ndim != 1 or len(score_array) != n_rows:
            raise ValueError(f"scores for {model_name} must match contributions")
        if not np.isfinite(score_array).all():
            raise ValueError(f"scores for {model_name} must be finite")
        frozen_scores[model_name] = score_array

    group_to_indices: dict[object, list[int]] = {}
    for index, group in enumerate(group_array.tolist()):
        group_to_indices.setdefault(group, []).append(index)
    group_names = sorted(group_to_indices, key=str)
    if not group_names:
        raise ValueError("at least one group is required")
    group_indices = {
        group: np.asarray(indices, dtype=np.int64)
        for group, indices in group_to_indices.items()
    }
    rng = np.random.default_rng(seed)
    overall_samples: list[float] = []
    selected_models: list[str] = []
    selected_thresholds: list[float] = []

    for _ in range(n_boot):
        sampled_positions = rng.integers(0, len(group_names), size=len(group_names))
        sampled_names = [group_names[int(position)] for position in sampled_positions]
        sampled_indices = np.concatenate(
            [group_indices[group_name] for group_name in sampled_names]
        )
        sampled_contributions = _subset_contributions(contributions, sampled_indices)
        if np.unique(sampled_contributions.is_positive).size != 2:
            raise ValueError("group bootstrap replicate lost a target class")
        sampled_scores = {
            model_name: scores[sampled_indices]
            for model_name, scores in frozen_scores.items()
        }
        selected, _frontier = _best_model_frontier(
            sampled_scores,
            sampled_contributions,
            rr_floor,
            include_frontier=False,
        )
        overall_samples.append(float(selected["overall"]))
        selected_models.append(str(selected["model"]))
        selected_thresholds.append(float(selected["threshold"]))

    overall_array = np.asarray(overall_samples, dtype=np.float64)
    return {
        "n_boot": n_boot,
        "n_groups": len(group_names),
        "overall_mean": float(overall_array.mean()),
        "ci_low": float(np.quantile(overall_array, 0.025)),
        "ci_high": float(np.quantile(overall_array, 0.975)),
        "overall_samples": overall_samples,
        "selected_models": selected_models,
        "selected_thresholds": selected_thresholds,
    }


def _fixed_threshold_point(
    scores: Sequence[float],
    contributions: OracleContributions,
    threshold: float,
) -> dict[str, float]:
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1 or len(score_array) != _validate_contributions(contributions):
        raise ValueError("scores and contributions must have equal one-dimensional lengths")
    accepted = score_array >= threshold
    positive = np.asarray(contributions.is_positive, dtype=np.bool_)
    negative = ~positive
    accepted_positive = accepted & positive
    rejected_positive = ~accepted & positive
    accepted_negative = accepted & negative
    total_ref_chars = int(np.asarray(contributions.ref_chars, dtype=np.int64).sum())
    errors = int(contributions.substitutions[accepted_positive].sum())
    errors += int(contributions.insertions[accepted_positive].sum())
    errors += int(contributions.deletions[accepted_positive].sum())
    errors += int(contributions.ref_chars[rejected_positive].sum())
    cer = errors / total_ref_chars if total_ref_chars else 0.0
    negative_count = int(negative.sum())
    accepted_negative_count = int(accepted_negative.sum())
    rr = (
        (negative_count - accepted_negative_count) / negative_count
        if negative_count
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "cer": float(cer),
        "rr": float(rr),
        "overall": float(((1.0 - cer) + rr) / 2.0),
        "accepted_positives": float(accepted_positive.sum()),
        "accepted_negatives": float(accepted_negative_count),
    }


def evaluate_e0(
    rows: Sequence[CandidateRow],
    labels: Mapping[str, str | None],
    groups: Sequence[object],
    *,
    n_splits: int = 5,
    seed: int = 20260807,
    rr_floor: float = 0.93,
    n_boot: int = 2000,
) -> dict:
    """Run the optimistic, diagnostic-only E0 cached gate-oracle evaluation."""

    if len(rows) != len(groups):
        raise ValueError("rows and groups must have equal lengths")
    target_present = np.asarray(
        [labels[row.id] is not None for row in rows], dtype=np.int64
    )
    matrix = build_gate_feature_matrix(rows)
    specs = default_model_specs()
    cross_fit = cross_fit_gate_models(
        matrix,
        target_present,
        groups,
        n_splits=n_splits,
        seed=seed,
        specs=specs,
    )

    # Reference text is introduced only after all gate models have been fitted.
    contributions = build_oracle_contributions(rows, labels)
    selected, frontier = _best_model_frontier(
        cross_fit.scores_by_model, contributions, rr_floor
    )
    selected_point = {
        **selected,
        "diagnostic_only": True,
        "deployable": False,
        "threshold_scope": "global_oof_diagnostic_non_deployable",
    }

    selected_model = str(selected["model"])
    selected_threshold = float(selected["threshold"])
    fold_metrics: list[dict[str, object]] = []
    for metadata in cross_fit.fold_metadata:
        test_indices = np.asarray(metadata["test_indices"], dtype=np.int64)
        fold_contributions = _subset_contributions(contributions, test_indices)
        fold_point = _fixed_threshold_point(
            cross_fit.scores_by_model[selected_model][test_indices],
            fold_contributions,
            selected_threshold,
        )
        fold_metrics.append(
            {
                "fold_index": int(metadata["fold_index"]),
                "test_groups": list(metadata["test_groups"]),
                **fold_point,
            }
        )
    worst_fold = min(
        fold_metrics,
        key=lambda point: (float(point["overall"]), int(point["fold_index"])),
    )

    bootstrap = group_bootstrap_best_frontier(
        cross_fit.scores_by_model,
        contributions,
        groups,
        rr_floor=rr_floor,
        n_boot=n_boot,
        seed=seed,
    )
    if float(bootstrap["ci_high"]) < 0.80:
        decision = "falsified_cached"
    elif (
        float(selected["overall"]) >= 0.81
        and float(selected["rr"]) >= rr_floor
        and float(worst_fold["overall"]) >= 0.77
    ):
        decision = "continue_cached"
    else:
        decision = "proceed_pvad"

    return {
        "decision": decision,
        "diagnostic_only": True,
        "global_threshold_deployable": False,
        "selected_point": selected_point,
        "bootstrap": bootstrap,
        "worst_fold": worst_fold,
        "fold_metrics": fold_metrics,
        "frontier": frontier,
        "model_specs": cross_fit.specs,
        "scores_by_model": cross_fit.scores_by_model,
        "fold_assignments": cross_fit.fold_assignments,
        "fold_metadata": cross_fit.fold_metadata,
    }
