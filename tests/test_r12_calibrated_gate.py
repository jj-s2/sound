"""Focused contract tests for the R12 train-only calibrated gate."""

from __future__ import annotations

import inspect
import json
import math
from typing import Any

import numpy as np
import pytest

from tests.test_r11_pvad_oracle import _fixture, _full_cpu_manifest
from xh202615.r11_gate_oracle import GateModelSpec
from xh202615.r11_pvad_oracle import E0_FITTING_FEATURE_SCHEMA, JoinedPvadRow, join_pvad_e0_rows


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


@pytest.fixture
def split_fixture():
    return _split_fixture()


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
    def test_train_object_has_exact_base_specs_and_calibrators(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate

        joined_train, _, _, _ = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        assert trained.base_model_names == (
            "hist_gradient_boosting_leaf_7",
            "hist_gradient_boosting_leaf_15",
        )
        assert all(isinstance(spec, GateModelSpec) for spec in trained.base_specs)
        assert {spec.name for spec in trained.base_specs} == set(trained.base_model_names)
        assert set(trained.calibrators) == set(trained.base_model_names)
        assert all(
            hasattr(trained.calibrators[name], "coef_")
            and hasattr(trained.calibrators[name], "intercept_")
            for name in trained.base_model_names
        )

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

    def test_calibration_uses_only_train_oof_scores(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate

        joined_train, joined_val, _, _ = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        n_train = len(joined_train)
        target = np.array([row.target_present for row in joined_train], dtype=np.int64)

        for name in trained.base_model_names:
            assert trained.calibration_inputs[name].shape == (n_train, 1)
            assert len(trained.calibrators[name].coef_.ravel()) == 1
            assert np.array_equal(trained.calibration_targets[name], target)

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

    def test_feature_schema_is_frozen_r11_e0(self, split_fixture):
        from xh202615.r11_gate_oracle import GATE_FEATURE_SCHEMA
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate

        joined_train, _, _, _ = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        assert trained.feature_schema == E0_FITTING_FEATURE_SCHEMA
        assert all(name in GATE_FEATURE_SCHEMA for name in trained.feature_schema)
        assert "latency_ms" not in trained.feature_schema
        assert trained.feature_schema_digest is not None
        assert len(trained.feature_schema_digest) == 64

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
            assert np.allclose(a.calibrators[name].coef_, b.calibrators[name].coef_)
            assert np.isclose(a.calibrators[name].intercept_, b.calibrators[name].intercept_)


class TestValidationSelection:
    def test_selection_returns_frozen_gate_selection(self, split_fixture):
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

    def test_eligible_point_has_raw_rr_at_least_floor(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        assert selection.validation_raw_metrics["rr"] >= 0.95

    def test_eligible_point_has_bootstrap_rr_5th_percentile_at_least_floor(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=200, seed=SEED
        )
        assert selection.validation_bootstrapped_metrics["rr_p05"] >= 0.93

    def test_selection_handles_reject_all(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        if selection.threshold == "reject_all":
            assert selection.validation_raw_metrics["rr"] == 1.0
            assert selection.validation_raw_metrics["cer"] == 1.0
            assert selection.validation_raw_metrics["overall"] == 0.5
        else:
            assert isinstance(selection.threshold, float)
            assert math.isfinite(selection.threshold)

    def test_selection_is_deterministic(self, split_fixture):
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
    def test_predict_matches_selected_threshold(self, split_fixture):
        from xh202615.r12_calibrated_gate import fit_train_calibrated_gate, predict_with_selection, select_on_validation

        joined_train, joined_val, val_rows, val_labels = split_fixture
        trained = fit_train_calibrated_gate(joined_train, seed=SEED)
        selection = select_on_validation(
            trained, joined_val, val_rows, val_labels, n_boot=50, seed=SEED
        )
        preds = predict_with_selection(trained, selection, joined_val)
        assert preds.shape == (len(joined_val),)
        assert set(np.unique(preds).tolist()) <= {0, 1}

    def test_predict_is_deterministic(self, split_fixture):
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
    def test_serial_form_excludes_private_fields(self, split_fixture):
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

    def test_serial_form_includes_required_fields(self, split_fixture):
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
