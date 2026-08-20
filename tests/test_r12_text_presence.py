"""Contracts for the train-only R12 text presence gate."""

from __future__ import annotations

import numpy as np

from tests.test_r12_calibrated_gate import _split_fixture


def test_text_presence_fit_is_train_only_and_predicts_finite_scores() -> None:
    from xh202615.r12_text_presence import (
        fit_train_text_presence,
        predict_text_presence,
    )

    joined_train, joined_validation, train_rows, train_labels = _split_fixture(
        n_train_groups=6, n_val_groups=4
    )
    model = fit_train_text_presence(train_rows, train_labels, seed=20260807)
    scores = predict_text_presence(model, train_rows[:2])

    assert model.fit_row_count == len(train_rows)
    assert np.isfinite(scores).all()
    assert scores.shape == (2,)


def test_text_presence_public_payload_excludes_training_text_and_labels() -> None:
    from xh202615.r12_text_presence import fit_train_text_presence

    _, _, train_rows, train_labels = _split_fixture(n_train_groups=6, n_val_groups=4)
    payload = fit_train_text_presence(train_rows, train_labels, seed=20260807).to_public_dict()

    assert payload["input_fields"] == ["r3_text", "primary_text"]
    assert "label" not in str(payload).lower()
    assert "text" not in set(payload) - {"input_fields"}
