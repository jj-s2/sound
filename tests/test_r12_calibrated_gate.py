"""Focused contract tests for the R12 train-only calibrated gate."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from typing import Any

import numpy as np
import pytest

from tests.test_r11_pvad_oracle import _fixture, _full_cpu_manifest
from xh202615.r11_gate_oracle import GateModelSpec
from xh202615.r11_pvad_oracle import (
    E0_FITTING_FEATURE_SCHEMA,
    PVAD_FITTING_FEATURE_SCHEMA,
    JoinedPvadRow,
    join_pvad_e0_rows,
)


SEED = 20260812


def _split_fixture(n_train_groups: int = 6, n_val_groups: int = 4) -> tuple[list[Any], list[Any], list[Any], dict[str, str | None]]:
    rows, labels, groups, cache, _ = _fixture()
    all_group_names = sorted({groups[row.id] for row in rows})
    train_group_names = set(all_group_names[:n_train_groups])
    val_group_names = set(all_group_names[n_train_groups : n_train_groups + n_val_groups])

    train_rows = [row for row in rows if groups[row.id] in train_group_names]
    val_rows = [row for row in rows if groups[row.id] in val_group_names]
    train_labels = {row.id: labels[row.id] for row in train_rows}
    val_labels = {row.id: labels[row.id] for row in val_rows}
    train_groups_map = {row.id: groups[row.id] for row in train_rows}
    val_groups_map = {row.id: groups[row.id] for row in val_rows}

    train_cache = [record for record in cache if record["id"] in train_labels]
    val_cache = [record for record in cache if record["id"] in val_labels]
    train_manifest = _full_cpu_manifest(train_cache)
    val_manifest = _full_cpu_manifest(val_cache)

    joined_train = join_pvad_e0_rows(
        train_rows, train_labels, train_groups_map, train_cache, train_manifest
    )
    joined_val = join_pvad_e0_rows(
        val_rows, val_labels, val_groups_map, val_cache, val_manifest
    )
    return joined_train, joined_val, val_rows, val_labels


def _joined_with_targets(joined: list[JoinedPvadRow], targets: list[int]) -> list[JoinedPvadRow]:
    return [
        JoinedPvadRow(row.id, row.group, target, row.pvad, row.e0, row.source_digest)
        for row, target in zip(joined, targets)
    ]


def _patch_feasible_finite(module, monkeypatch, *, raw_rr: float = 0.95, rr_p05: float = 0.93):
    """Patch frontier/bootstrap so at least one finite threshold meets both RR floors."""

    def feasible_frontier(scores, contributions):
        cer = 1.0 - raw_rr
        return [
            {
                "threshold": float("inf"),
                "cer": 1.0,
                "rr": 1.0,
                "overall": 0.5,
                "accepted_positives": 0.0,
                "accepted_negatives": 0.0,
            },
            {
                "threshold": 0.5,
                "cer": cer,
                "rr": raw_rr,
                "overall": (1.0 - cer + raw_rr) / 2.0,
                "accepted_positives": 1.0,
                "accepted_negatives": 0.0,
            },
        ]

    def feasible_bootstrap(*args, **kwargs):
        return {
            "overall_median": 0.95,
            "rr_p05": rr_p05,
            "n_boot": kwargs["n_boot"],
            "attempted_replicates": kwargs["n_boot"],
            "rejected_replicates": 0,
        }

    monkeypatch.setattr(module, "gate_oracle_frontier", feasible_frontier)
    monkeypatch.setattr(module, "_bootstrap_point_stats", feasible_bootstrap)


@pytest.fixture
def split_fixture():
    return _split_fixture()


@pytest.fixture
def feasible_finite(monkeypatch):
    import xh202615.r12_calibrated_gate as module

    _patch_feasible_finite(module, monkeypatch)


class TestPublicInterface:
    def test_exports_match_brief(self):
        from xh202615 import r12_calibrated_gate as module

        assert module.BASE_MODELS == (
            "hist_gradient_boosting_leaf_7",
            "hist_gradient_boosting_leaf_15",
        )
        assert module.BLEND_WEIGHTS == (0.0, 0.25, 0.5, 0.75, 1.0)
        assert module.FrozenGateSelection is not None
        assert module.TrainCalibratedGate is not None
        assert callable(module.fit_train_calibrated_gate)
        assert callable(module.select_on_validation)
        assert callable(module.predict_with_selection)

    def test_fit_signature_takes_no_labels(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate

        sig = inspect.signature(fit_train_calibrated_gate)
        names = {p.name for p in sig.parameters.values()}
        assert "joined_train" in names
        assert "seed" in names
        assert "labels" not in names
        assert "validation" not in names


class TestTrainCalibration:
    def test_train_object_has_exact_base_specs_and_single_calibrator(self, split_fixture):
        from sklearn.linear_model import LogisticRegression
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate

        joined_train, _, _, _ = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        assert trained.base_model_names == (
            "hist_gradient_boosting_leaf_7",
            "hist_gradient_boosting_leaf_15",
        )
        assert all(isinstance(spec, GateModelSpec) for spec in trained.base_specs)
        assert {spec.name for spec in trained.base_specs} == set(trained.base_model_names)
        assert isinstance(trained.calibrator, LogisticRegression)
        assert trained.calibrator.coef_.shape == (1, 2)
        assert trained.calibrator.intercept_.shape == (1,)

    def test_oof_scores_are_once_only_group_disjoint_and_finite(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate

        joined_train, _, _, _ = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        n_train = len(joined_train)
        groups = np.array([row.group for row in joined_train], dtype=object)

        for name in trained.base_model_names:
            scores = trained.oof_scores[name]
            assert scores.shape == (n_train,)
            assert np.isfinite(scores).all()
            assert ((scores >= 0.0) & (scores <= 1.0)).all()

        fold_assignments = trained.fold_assignments
        assert fold_assignments.shape == (n_train,)
        assert np.all(fold_assignments >= 0)
        assert np.all(fold_assignments < 3)

        for fold_index in range(int(fold_assignments.max()) + 1):
            test_groups = set(groups[fold_assignments == fold_index])
            train_groups = set(groups[fold_assignments != fold_index])
            assert not (test_groups & train_groups)

    def test_calibration_uses_fused_two_column_oof_in_base_order(self, split_fixture):
        from xh202615.r12_calibrated_gate import BASE_MODELS, fit_train_calibrated_gate

        joined_train, joined_val, _, _ = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        n_train = len(joined_train)
        target = np.array([row.target_present for row in joined_train], dtype=np.int64)

        assert trained.calibration_input.shape == (n_train, 2)
        assert np.allclose(trained.calibration_input[:, 0], trained.oof_scores[BASE_MODELS[0]])
        assert np.allclose(trained.calibration_input[:, 1], trained.oof_scores[BASE_MODELS[1]])
        assert np.array_equal(trained.calibration_targets, target)

        val_ids = {row.id for row in joined_val}
        assert not any(row.id in val_ids for row in trained.calibration_rows)

    def test_base_models_are_refit_on_whole_train(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate

        joined_train, _, _, _ = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        assert trained.refit_on_whole_train is True
        for name in trained.base_model_names:
            model = trained.base_models[name]
            assert hasattr(model, "predict_proba")

    def test_feature_schema_is_fused_pvad_e0(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate

        joined_train, _, _, _ = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        expected_schema = (*PVAD_FITTING_FEATURE_SCHEMA, *E0_FITTING_FEATURE_SCHEMA)
        assert trained.feature_schema == expected_schema
        expected_digest = hashlib.sha256(
            (r"\n".join(expected_schema) + r"\n").encode("utf-8")
        ).hexdigest()
        assert trained.feature_schema_digest == expected_digest
        assert "latency_ms" not in trained.feature_schema
        assert len(trained.feature_schema_digest) == 64

    def test_pvad_feature_changes_fused_fitting_matrix(self, split_fixture):
        from xh202615.r12_calibrated_gate import _build_matrix, fit_train_calibrated_gate

        joined_train, _, _, _ = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        base_matrix = _build_matrix(joined_train, trained.feature_schema)

        mutated = []
        for i, row in enumerate(joined_train):
            pvad = dict(row.pvad)
            if i == 0:
                pvad[PVAD_FITTING_FEATURE_SCHEMA[0]] = 1.0 - pvad[PVAD_FITTING_FEATURE_SCHEMA[0]]
            mutated.append(
                JoinedPvadRow(row.id, row.group, row.target_present, pvad, row.e0, row.source_digest)
            )
        mutated_matrix = _build_matrix(mutated, trained.feature_schema)
        assert mutated_matrix.shape == base_matrix.shape
        assert not np.allclose(mutated_matrix, base_matrix)

    def test_fit_rejects_empty_train(self):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate

        with pytest.raises(ValueError):
            fit_train_calibrated_gate([], seed=SEED)

    def test_fit_rejects_nonbinary_or_single_class_target(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate

        joined_train, _, _, _ = split_fixture
        all_positive = _joined_with_targets(joined_train, [1] * len(joined_train))
        with pytest.raises(ValueError, match="target"):
            fit_train_calibrated_gate(all_positive, seed=SEED)

        all_negative = _joined_with_targets(joined_train, [0] * len(joined_train))
        with pytest.raises(ValueError, match="target"):
            fit_train_calibrated_gate(all_negative, seed=SEED)

        bad = _joined_with_targets(
            joined_train,
            [2 if i == 0 else row.target_present for i, row in enumerate(joined_train)],
        )
        with pytest.raises(ValueError, match="target"):
            fit_train_calibrated_gate(bad, seed=SEED)

    def test_fit_rejects_insufficient_groups(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate

        joined_train, _, _, _ = split_fixture
        single_group = [
            JoinedPvadRow(row.id, "only", row.target_present, row.pvad, row.e0, row.source_digest)
            for row in joined_train
        ]
        with pytest.raises(ValueError, match="groups"):
            fit_train_calibrated_gate(single_group, seed=SEED)

    def test_train_determinism(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate

        joined_train, _, _, _ = split_fixture
        a = fit_train_calibrated_gate(joined_train, seed=SEED)
        b = fit_train_calibrated_gate(joined_train, seed=SEED)
        assert a.base_model_names == b.base_model_names
        assert a.feature_schema == b.feature_schema
        assert np.array_equal(a.fold_assignments, b.fold_assignments)
        for name in a.base_model_names:
            assert np.allclose(a.oof_scores[name], b.oof_scores[name])
        assert np.allclose(a.calibrator.coef_, b.calibrator.coef_)
        assert np.isclose(a.calibrator.intercept_, b.calibrator.intercept_)


class TestValidationSelection:
    def test_selection_fails_closed_when_no_bootstrap_candidate_is_feasible(
        self, split_fixture, monkeypatch
    ):
        import xh202615.r12_calibrated_gate as module

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = module.fit_train_calibrated_gate(joined_train, seed=SEED)

        def infeasible_bootstrap(*args, **kwargs):
            return {
                "overall_median": 0.5,
                "rr_p05": 0.92,
                "n_boot": kwargs["n_boot"],
                "attempted_replicates": kwargs["n_boot"],
                "rejected_replicates": 0,
            }

        monkeypatch.setattr(module, "_bootstrap_point_stats", infeasible_bootstrap)
        with pytest.raises(
            module.BootstrapFeasibilityError, match="no validation candidate satisfies"
        ):
            module.select_on_validation(
                trained, joined_val, val_rows, val_labels, n_boot=10, seed=SEED
            )

    def test_reject_all_is_never_selected_as_fallback(self, split_fixture, monkeypatch):
        import xh202615.r12_calibrated_gate as module

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = module.fit_train_calibrated_gate(joined_train, seed=SEED)

        def finite_infeasible_reject_all_feasible(*args, **kwargs):
            threshold = kwargs["threshold"]
            if math.isinf(threshold):
                return {
                    "overall_median": 0.5,
                    "rr_p05": 0.93,
                    "n_boot": kwargs["n_boot"],
                    "attempted_replicates": kwargs["n_boot"],
                    "rejected_replicates": 0,
                }
            return {
                "overall_median": 0.5,
                "rr_p05": 0.92,
                "n_boot": kwargs["n_boot"],
                "attempted_replicates": kwargs["n_boot"],
                "rejected_replicates": 0,
            }

        monkeypatch.setattr(
            module, "_bootstrap_point_stats", finite_infeasible_reject_all_feasible
        )
        with pytest.raises(
            module.BootstrapFeasibilityError, match="no validation candidate satisfies"
        ):
            module.select_on_validation(
                trained, joined_val, val_rows, val_labels, n_boot=10, seed=SEED
            )

    def test_finite_lower_threshold_wins_tie_against_reject_all(
        self, split_fixture, feasible_finite
    ):
        import xh202615.r12_calibrated_gate as module

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = module.fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = module.select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=10, seed=SEED
        )
        assert isinstance(selection.threshold, float)
        assert math.isfinite(selection.threshold)

    def test_infinite_threshold_sorts_after_finite_threshold(self):
        import xh202615.r12_calibrated_gate as module

        assert module._threshold_sort_value(float("0.5")) < module._threshold_sort_value(float("inf"))

    def test_selection_returns_frozen_gate_selection(self, split_fixture, feasible_finite):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        assert selection.threshold is not None
        assert selection.selected_model_name is not None
        assert isinstance(selection.validation_raw_metrics, dict)
        assert isinstance(selection.validation_bootstrapped_metrics, dict)

    def test_selection_can_freeze_a_deployable_transcript_action(self, split_fixture, feasible_finite):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained,
            joined_val,
            val_rows,
            val_labels,
            n_boot=10,
            seed=SEED,
            accepted_action="primary",
        )
        assert selection.provenance["accepted_action"] == "primary"

    def test_eligible_point_has_raw_rr_at_least_floor(self, split_fixture, feasible_finite):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        assert selection.validation_raw_metrics["rr"] >= 0.95

    def test_eligible_point_has_bootstrap_rr_5th_percentile_at_least_floor(self, split_fixture, feasible_finite):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=200, seed=SEED
        )
        assert selection.validation_bootstrapped_metrics["rr_p05"] >= 0.93

    def test_selection_never_returns_reject_all(self, split_fixture, feasible_finite):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        assert isinstance(selection.threshold, float)
        assert math.isfinite(selection.threshold)

    def test_selection_is_deterministic(self, split_fixture, feasible_finite):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained_a = fit_train_calibrated_gate(joined_train, seed=SEED)
        trained_b = fit_train_calibrated_gate(joined_train, seed=SEED)
        sel_a = select_on_validation(
            trained_a, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        sel_b = select_on_validation(
            trained_b, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        assert sel_a == sel_b


class TestPrediction:
    def test_predict_matches_selected_threshold(self, split_fixture, feasible_finite):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, predict_with_selection, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        preds = predict_with_selection(trained, selection, joined_val)
        assert preds.shape == (len(joined_val),)
        assert set(np.unique(preds).tolist()) <= {0, 1}

    def test_predict_is_deterministic(self, split_fixture, feasible_finite):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, predict_with_selection, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        a = predict_with_selection(trained, selection, joined_val)
        b = predict_with_selection(trained, selection, joined_val)
        assert np.array_equal(a, b)


class TestSerializationPrivacy:
    def test_serial_form_excludes_private_fields(self, split_fixture, feasible_finite):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        serial = selection.to_dict()
        text = json.dumps(serial, ensure_ascii=False, sort_keys=True).lower()
        forbidden = (
            "label",
            "reference",
            "target",
            "candidate_cer",
            "optimal_action",
            "embedding",
            "frame_",
            "sklearn",
            "pickle",
        )
        for token in forbidden:
            assert token not in text, f"serial form leaks {token!r}"

    def test_serial_form_includes_required_fields(self, split_fixture, feasible_finite):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        serial = selection.to_dict()
        assert "base_model_names" in serial
        assert "base_parameters_digest" in serial
        assert "feature_schema_digest" in serial
        assert "calibrator" in serial
        assert "blend_definition" in serial
        assert "threshold" in serial
        assert "validation_raw_metrics" in serial
        assert "validation_bootstrapped_metrics" in serial
        assert "provenance" in serial

    def test_serial_form_calibrator_has_two_coefficients_and_intercept(self, split_fixture, feasible_finite):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        assert len(selection.calibrator_coefficients) == 2
        assert all(isinstance(c, float) for c in selection.calibrator_coefficients)
        assert isinstance(selection.calibrator_intercept, float)
        serial = selection.to_dict()
        assert serial["calibrator"]["coefficients"] == list(selection.calibrator_coefficients)
        assert serial["calibrator"]["intercept"] == selection.calibrator_intercept

