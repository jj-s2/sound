"""R12 train-only calibrated leaf7/leaf15 base-gate core (Task 2a)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .r11_gate_oracle import GateModelSpec, _fit_gate_pipeline, cross_fit_gate_models
from .r11_pvad_oracle import E0_FITTING_FEATURE_SCHEMA, JoinedPvadRow


BASE_MODELS = (
    "hist_gradient_boosting_leaf_7",
    "hist_gradient_boosting_leaf_15",
)
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


def _build_matrix(rows: Sequence[JoinedPvadRow], schema: Sequence[str]) -> np.ndarray:
    return np.asarray([[row.e0[name] for name in schema] for row in rows], dtype=np.float64)


@dataclass(frozen=True)
class TrainCalibratedGate:
    """Train-side artifact: refit base models, calibrators, and OOF bank."""

    base_specs: tuple[GateModelSpec, ...]
    base_model_names: tuple[str, ...]
    feature_schema: tuple[str, ...]
    feature_schema_digest: str
    base_models: Mapping[str, Pipeline]
    calibrators: Mapping[str, LogisticRegression]
    oof_scores: Mapping[str, np.ndarray]
    calibration_inputs: Mapping[str, np.ndarray]
    calibration_targets: Mapping[str, np.ndarray]
    fold_assignments: np.ndarray
    fold_metadata: tuple[dict[str, object], ...]
    refit_on_whole_train: bool
    seed: int
    n_splits: int


def fit_train_calibrated_gate(
    joined_train: Sequence[JoinedPvadRow], *, seed: int
) -> TrainCalibratedGate:
    """Fit two HGB base models with 3-fold group-disjoint OOF and Platt scaling."""

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
        base_models=base_models,
        calibrators=calibrators,
        oof_scores=oof_scores,
        calibration_inputs=calibration_inputs,
        calibration_targets=calibration_targets,
        fold_assignments=cross_fit.fold_assignments,
        fold_metadata=cross_fit.fold_metadata,
        refit_on_whole_train=True,
        seed=seed,
        n_splits=_N_SPLITS,
    )


def predict_calibrated(
    trained: TrainCalibratedGate, joined_rows: Sequence[JoinedPvadRow]
) -> np.ndarray:
    """Return the two calibrated score columns in [0, 1] for ``joined_rows``."""

    X = _build_matrix(joined_rows, trained.feature_schema)
    n_rows = len(joined_rows)
    n_models = len(trained.base_model_names)
    calibrated = np.empty((n_rows, n_models), dtype=np.float64)

    for col, name in enumerate(trained.base_model_names):
        base_probs = np.asarray(
            trained.base_models[name].predict_proba(X)[:, 1], dtype=np.float64
        )
        scores = np.asarray(
            trained.calibrators[name].predict_proba(base_probs.reshape(-1, 1))[:, 1],
            dtype=np.float64,
        )
        if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(f"model {name} produced invalid calibrated probabilities")
        calibrated[:, col] = scores

    return calibrated
