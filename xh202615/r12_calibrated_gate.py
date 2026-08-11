"""R12 train-only calibrated leaf7/leaf15 gate and robust validation selection."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .r10_selector import CandidateRow
from .r11_gate_oracle import (
    GateModelSpec,
    _fit_gate_pipeline,
    build_oracle_contributions,
    cross_fit_gate_models,
    gate_oracle_frontier,
)
from .r11_pvad_oracle import E0_FITTING_FEATURE_SCHEMA, JoinedPvadRow, canonical_json


BASE_MODELS = (
    "hist_gradient_boosting_leaf_7",
    "hist_gradient_boosting_leaf_15",
)
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
_BLEND_MODEL_NAME = "blend"
_REJECT_ALL_MODEL_NAME = "reject_all"
_RAW_RR_FLOOR = 0.95
_BOOTSTRAP_RR_FLOOR = 0.93
_N_SPLITS = 3


def _base_specs() -> tuple[GateModelSpec, ...]:
    return (
        GateModelSpec(
            "hist_gradient_boosting_leaf_7",
            "hist_gradient_boosting",
            (
                ("max_leaf_nodes", 7),
                ("learning_rate", 0.05),
                ("max_iter", 150),
                ("l2_regularization", 1.0),
            ),
        ),
        GateModelSpec(
            "hist_gradient_boosting_leaf_15",
            "hist_gradient_boosting",
            (
                ("max_leaf_nodes", 15),
                ("learning_rate", 0.05),
                ("max_iter", 150),
                ("l2_regularization", 1.0),
            ),
        ),
    )


def _schema_digest(schema: Sequence[str]) -> str:
    return hashlib.sha256((r"\n".join(schema) + r"\n").encode("utf-8")).hexdigest()


def _parameters_digest(specs: Sequence[GateModelSpec]) -> str:
    payload = [
        {"name": spec.name, "family": spec.family, "parameters": dict(spec.parameters)}
        for spec in specs
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _build_matrix(rows: Sequence[JoinedPvadRow], schema: Sequence[str]) -> np.ndarray:
    return np.asarray([[row.e0[name] for name in schema] for row in rows], dtype=np.float64)


def _bootstrap_point_stats(
    scores: np.ndarray,
    contributions: Any,
    groups: Sequence[object],
    threshold: float,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    n_rows = len(scores)
    group_array = np.asarray(groups)
    group_names, row_group_codes = np.unique(group_array, return_inverse=True)
    n_groups = len(group_names)

    positive = np.asarray(contributions.is_positive, dtype=np.bool_)
    negative = ~positive
    accepted = np.asarray(scores, dtype=np.float64) >= threshold

    per_row_errors = np.zeros(n_rows, dtype=np.int64)
    accepted_positive = accepted & positive
    rejected_positive = (~accepted) & positive
    per_row_errors[accepted_positive] = (
        contributions.substitutions[accepted_positive]
        + contributions.insertions[accepted_positive]
        + contributions.deletions[accepted_positive]
    )
    ref_chars = np.asarray(contributions.ref_chars, dtype=np.int64)
    per_row_errors[rejected_positive] = ref_chars[rejected_positive]

    correct_reject = ((~accepted) & negative).astype(np.int64)

    group_errors = np.bincount(row_group_codes, weights=per_row_errors, minlength=n_groups)
    group_ref_chars = np.bincount(row_group_codes, weights=ref_chars, minlength=n_groups)
    group_correct_reject = np.bincount(
        row_group_codes, weights=correct_reject, minlength=n_groups
    )
    group_negatives = np.bincount(
        row_group_codes, weights=negative.astype(np.int64), minlength=n_groups
    )
    group_positives = np.bincount(
        row_group_codes, weights=positive.astype(np.int64), minlength=n_groups
    )

    rng = np.random.default_rng(seed)
    overall_samples: list[float] = []
    rr_samples: list[float] = []
    attempted_replicates = 0
    rejected_replicates = 0
    while len(overall_samples) < n_boot:
        sampled_positions = rng.integers(0, n_groups, size=n_groups)
        attempted_replicates += 1
        multiplicities = np.bincount(sampled_positions, minlength=n_groups).astype(np.int64)
        neg_count = int(group_negatives @ multiplicities)
        pos_count = int(group_positives @ multiplicities)
        if neg_count == 0 or pos_count == 0:
            rejected_replicates += 1
            continue
        ref_count = int(group_ref_chars @ multiplicities)
        err_count = int(group_errors @ multiplicities)
        cer = err_count / ref_count if ref_count else 0.0
        rr = int(group_correct_reject @ multiplicities) / neg_count
        overall = ((1.0 - cer) + rr) / 2.0
        overall_samples.append(overall)
        rr_samples.append(rr)

    overall_array = np.asarray(overall_samples, dtype=np.float64)
    rr_array = np.asarray(rr_samples, dtype=np.float64)
    return {
        "overall_median": float(np.median(overall_array)),
        "rr_p05": float(np.quantile(rr_array, 0.05)),
        "n_boot": n_boot,
        "attempted_replicates": attempted_replicates,
        "rejected_replicates": rejected_replicates,
    }


def _safe_threshold(value: float) -> float | str:
    return _REJECT_ALL_MODEL_NAME if math.isinf(value) else float(value)


@dataclass(frozen=True)
class TrainCalibratedGate:
    """Train-side artifact: refit base models, calibrators, and OOF bank."""

    base_specs: tuple[GateModelSpec, ...]
    base_model_names: tuple[str, ...]
    feature_schema: tuple[str, ...]
    feature_schema_digest: str
    base_parameters_digest: str
    base_models: Mapping[str, Pipeline]
    calibrators: Mapping[str, LogisticRegression]
    oof_scores: Mapping[str, np.ndarray]
    calibration_inputs: Mapping[str, np.ndarray]
    calibration_targets: Mapping[str, np.ndarray]
    calibration_rows: tuple[JoinedPvadRow, ...]
    fold_assignments: np.ndarray
    fold_metadata: tuple[dict[str, object], ...]
    refit_on_whole_train: bool
    seed: int


@dataclass(frozen=True)
class FrozenGateSelection:
    """Frozen validation selection without fitted estimators or private labels."""

    base_model_names: tuple[str, ...]
    base_parameters_digest: str
    feature_schema_digest: str
    calibrators: tuple[tuple[str, float, float], ...]
    blend_definition: dict[str, object]
    selected_model_name: str
    selected_blend_weight: float
    threshold: float | str
    validation_raw_metrics: dict[str, float]
    validation_bootstrapped_metrics: dict[str, float]
    provenance: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "base_model_names": list(self.base_model_names),
            "base_parameters_digest": self.base_parameters_digest,
            "feature_schema_digest": self.feature_schema_digest,
            "calibrator": [
                {"model": name, "coefficient": float(coef), "intercept": float(intercept)}
                for name, coef, intercept in self.calibrators
            ],
            "blend_definition": self.blend_definition,
            "selected_model_name": self.selected_model_name,
            "selected_blend_weight": self.selected_blend_weight,
            "threshold": self.threshold,
            "validation_raw_metrics": self.validation_raw_metrics,
            "validation_bootstrapped_metrics": self.validation_bootstrapped_metrics,
            "provenance": self.provenance,
        }


def fit_train_calibrated_gate(
    joined_train: Sequence[JoinedPvadRow], *, seed: int
) -> TrainCalibratedGate:
    """Fit the two base models with 3-fold group-disjoint OOF and Platt scaling."""

    if len(joined_train) == 0:
        raise ValueError("at least one joined training row is required")

    feature_schema = E0_FITTING_FEATURE_SCHEMA
    X_train = _build_matrix(joined_train, feature_schema)
    target = np.asarray([row.target_present for row in joined_train], dtype=np.int64)
    groups = [row.group for row in joined_train]

    if not np.isin(target, [0, 1]).all():
        raise ValueError("target_present must be binary")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("target_present must contain both target classes")
    if len(set(groups)) < _N_SPLITS:
        raise ValueError(f"at least {_N_SPLITS} training groups are required")

    specs = _base_specs()
    cross_fit = cross_fit_gate_models(
        X_train,
        target,
        groups,
        n_splits=_N_SPLITS,
        seed=seed,
        specs=specs,
    )

    oof_scores: dict[str, np.ndarray] = {}
    calibration_inputs: dict[str, np.ndarray] = {}
    calibration_targets: dict[str, np.ndarray] = {}
    calibrators: dict[str, LogisticRegression] = {}
    base_models: dict[str, Pipeline] = {}

    for spec in specs:
        name = spec.name
        scores = np.asarray(cross_fit.scores_by_model[name], dtype=np.float64)
        oof_scores[name] = scores

        calibrator_X = scores.reshape(-1, 1)
        calibration_inputs[name] = calibrator_X
        calibration_targets[name] = target.copy()

        calibrator = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        )
        calibrator.fit(calibrator_X, target)
        calibrators[name] = calibrator

        base_models[name] = _fit_gate_pipeline(spec, X_train, target, seed)

    return TrainCalibratedGate(
        base_specs=specs,
        base_model_names=BASE_MODELS,
        feature_schema=feature_schema,
        feature_schema_digest=_schema_digest(feature_schema),
        base_parameters_digest=_parameters_digest(specs),
        base_models=base_models,
        calibrators=calibrators,
        oof_scores=oof_scores,
        calibration_inputs=calibration_inputs,
        calibration_targets=calibration_targets,
        calibration_rows=tuple(joined_train),
        fold_assignments=cross_fit.fold_assignments,
        fold_metadata=cross_fit.fold_metadata,
        refit_on_whole_train=True,
        seed=seed,
    )


def _validation_base_scores(
    trained: TrainCalibratedGate, X: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(trained.base_models[name].predict_proba(X)[:, 1], dtype=np.float64)
        for name in trained.base_model_names
    }


def _validation_calibrated_scores(
    trained: TrainCalibratedGate, base_scores: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(
            trained.calibrators[name].predict_proba(
                np.asarray(base_scores[name], dtype=np.float64).reshape(-1, 1)
            )[:, 1],
            dtype=np.float64,
        )
        for name in trained.base_model_names
    }


def _score_variants(
    calibrated_scores: Mapping[str, np.ndarray]
) -> dict[str, tuple[str, float, np.ndarray]]:
    """Return all evaluated score variants keyed by candidate name."""
    variants: dict[str, tuple[str, float, np.ndarray]] = {}
    leaf7 = calibrated_scores[BASE_MODELS[0]]
    leaf15 = calibrated_scores[BASE_MODELS[1]]
    for name in BASE_MODELS:
        variants[name] = (name, 1.0 if name == BASE_MODELS[1] else 0.0, calibrated_scores[name])
    for weight in BLEND_WEIGHTS:
        score = (1.0 - weight) * leaf7 + weight * leaf15
        variants[f"blend_{weight:.2f}"] = (_BLEND_MODEL_NAME, weight, score)
    return variants


def select_on_validation(
    trained: TrainCalibratedGate,
    joined_validation: Sequence[JoinedPvadRow],
    validation_rows: Sequence[CandidateRow],
    validation_labels: Mapping[str, str | None],
    *,
    n_boot: int,
    seed: int,
) -> FrozenGateSelection:
    """Select the best validation threshold/model/blend under RR floors."""

    if len(joined_validation) != len(validation_rows):
        raise ValueError("joined_validation and validation_rows must have equal lengths")
    if {row.id for row in joined_validation} != {row.id for row in validation_rows}:
        raise ValueError("joined_validation and validation_rows must have the same IDs")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    X_val = _build_matrix(joined_validation, trained.feature_schema)
    base_scores = _validation_base_scores(trained, X_val)
    calibrated_scores = _validation_calibrated_scores(trained, base_scores)
    variants = _score_variants(calibrated_scores)

    contributions = build_oracle_contributions(validation_rows, validation_labels)
    groups = [row.group for row in joined_validation]

    candidates: list[dict[str, object]] = []
    for candidate_name, (model_name, blend_weight, scores) in variants.items():
        frontier = gate_oracle_frontier(scores, contributions)
        for point in frontier:
            candidates.append(
                {
                    "candidate_name": candidate_name,
                    "model_name": model_name,
                    "blend_weight": blend_weight,
                    "threshold": float(point["threshold"]),
                    "raw_cer": float(point["cer"]),
                    "raw_rr": float(point["rr"]),
                    "raw_overall": float(point["overall"]),
                }
            )

    eligible = [c for c in candidates if c["raw_rr"] >= _RAW_RR_FLOOR]
    if not eligible:
        reject_all = next(c for c in candidates if math.isinf(c["threshold"]))
        eligible = [reject_all]

    bootstrapped: list[dict[str, object]] = []
    for candidate in eligible:
        variant_scores = variants[candidate["candidate_name"]][2]
        stats = _bootstrap_point_stats(
            variant_scores,
            contributions,
            groups,
            candidate["threshold"],
            n_boot=n_boot,
            seed=seed,
        )
        bootstrapped.append({**candidate, **stats})

    feasible = [c for c in bootstrapped if c["rr_p05"] >= _BOOTSTRAP_RR_FLOOR]
    if not feasible:
        feasible = [c for c in bootstrapped if math.isinf(c["threshold"])]
        if not feasible:
            feasible = bootstrapped

    def sort_key(c: dict[str, object]) -> tuple:
        threshold = c["threshold"]
        return (
            -float(c["overall_median"]),
            -float(c["raw_rr"]),
            float(c["raw_cer"]),
            0.0 if math.isinf(threshold) else float(threshold),
            str(c["model_name"]),
            float(c["blend_weight"]),
        )

    best = min(feasible, key=sort_key)

    calibrator_records = tuple(
        (
            name,
            float(trained.calibrators[name].coef_.ravel()[0]),
            float(trained.calibrators[name].intercept_.ravel()[0]),
        )
        for name in trained.base_model_names
    )

    return FrozenGateSelection(
        base_model_names=trained.base_model_names,
        base_parameters_digest=trained.base_parameters_digest,
        feature_schema_digest=trained.feature_schema_digest,
        calibrators=calibrator_records,
        blend_definition={
            "weights": list(BLEND_WEIGHTS),
            "description": "weight is leaf15; blend = (1-w)*leaf7 + w*leaf15",
        },
        selected_model_name=str(best["model_name"]),
        selected_blend_weight=float(best["blend_weight"]),
        threshold=_safe_threshold(float(best["threshold"])),
        validation_raw_metrics={
            "overall": float(best["raw_overall"]),
            "rr": float(best["raw_rr"]),
            "cer": float(best["raw_cer"]),
        },
        validation_bootstrapped_metrics={
            "overall_median": float(best["overall_median"]),
            "rr_p05": float(best["rr_p05"]),
            "n_boot": int(best["n_boot"]),
        },
        provenance={
            "train_seed": trained.seed,
            "validation_n_boot": n_boot,
            "validation_seed": seed,
            "raw_rr_floor": _RAW_RR_FLOOR,
            "bootstrap_rr_floor": _BOOTSTRAP_RR_FLOOR,
            "n_candidates_evaluated": len(candidates),
            "n_eligible_raw": len(eligible),
            "n_feasible_bootstrap": len(feasible),
        },
    )


def predict_with_selection(
    trained: TrainCalibratedGate,
    selection: FrozenGateSelection,
    joined_rows: Sequence[JoinedPvadRow],
) -> np.ndarray:
    """Return binary accept/reject predictions for ``joined_rows``."""

    X = _build_matrix(joined_rows, trained.feature_schema)
    base_scores = _validation_base_scores(trained, X)
    calibrated_scores = _validation_calibrated_scores(trained, base_scores)

    if selection.selected_model_name == _REJECT_ALL_MODEL_NAME:
        score = np.full(len(joined_rows), -1.0, dtype=np.float64)
    elif selection.selected_model_name == _BLEND_MODEL_NAME:
        weight = selection.selected_blend_weight
        score = (
            (1.0 - weight) * calibrated_scores[BASE_MODELS[0]]
            + weight * calibrated_scores[BASE_MODELS[1]]
        )
    elif selection.selected_model_name in trained.base_model_names:
        score = calibrated_scores[selection.selected_model_name]
    else:
        raise ValueError(f"unknown selected model name: {selection.selected_model_name}")

    threshold = selection.threshold
    if isinstance(threshold, str):
        if threshold == _REJECT_ALL_MODEL_NAME:
            return np.zeros(len(joined_rows), dtype=np.int64)
        raise ValueError(f"invalid threshold marker: {threshold}")
    return (score >= threshold).astype(np.int64)
