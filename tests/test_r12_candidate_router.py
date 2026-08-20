"""Contracts for the train-only deployable R12 candidate router."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tests.test_r12_calibrated_gate import _split_fixture


def test_router_matrix_has_four_actions_per_sample_without_private_columns() -> None:
    from xh202615.r12_candidate_router import ROUTER_ACTIONS, build_router_matrix

    _, joined_rows, rows, _ = _split_fixture(n_train_groups=6, n_val_groups=4)
    matrix, keys, schema = build_router_matrix(rows, joined_rows)

    assert matrix.shape[0] == len(rows) * 4
    assert matrix.shape[1] == len(schema)
    assert {key.action for key in keys} == set(ROUTER_ACTIONS)
    assert not {"label", "reference", "cer", "oracle"} & set(schema)
    assert np.isfinite(matrix).all()


def test_router_tie_order_is_primary_r3_tse_energy() -> None:
    from xh202615.r12_candidate_router import (
        ROUTER_ACTIONS,
        TrainCandidateRouter,
        build_router_matrix,
        predict_router_actions,
    )

    _, joined_rows, rows, _ = _split_fixture(n_train_groups=6, n_val_groups=4)

    class EqualRegressor:
        def predict(self, matrix):
            return np.zeros(matrix.shape[0], dtype=np.float64)

    _, _, schema = build_router_matrix(rows[:1], joined_rows[:1])
    router = TrainCandidateRouter(
        feature_schema=schema,
        model=EqualRegressor(),
        fit_row_count=0,
        fit_group_count=0,
        seed=20260807,
    )
    assert ROUTER_ACTIONS == ("primary", "r3", "tse", "energy")
    assert predict_router_actions(router, rows[:1], joined_rows[:1]) == ("primary",)


def test_router_fit_uses_only_positive_train_rows() -> None:
    from xh202615.r12_candidate_router import fit_train_candidate_router

    _, joined_train, train_rows, train_labels = _split_fixture(
        n_train_groups=6, n_val_groups=4
    )
    router = fit_train_candidate_router(
        train_rows, joined_train, train_labels, seed=20260807
    )

    positive_count = sum(label is not None for label in train_labels.values())
    assert router.fit_row_count == positive_count * 4
    assert router.fit_group_count == 4


def test_router_report_marks_existing_held_out_contaminated() -> None:
    from pathlib import Path

    report = Path("docs/r12/router-development-report.md")
    text = report.read_text(encoding="utf-8")
    assert "contaminated" in text
    assert "not used for selection" in text
