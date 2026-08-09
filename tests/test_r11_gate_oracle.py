"""Tests for R11 label-free gate features and oracle frontier."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from xh202615.r10_selector import CandidateRow
from xh202615.r11_gate_oracle import (
    GATE_FEATURE_SCHEMA,
    build_gate_feature_matrix,
    build_oracle_contributions,
    gate_oracle_frontier,
    select_frontier_point,
)


def _row(
    sid: str,
    label: str | None,
    *,
    r3: str = "打开空调",
    primary: str = "开空调",
    energy: str = "",
    tse: str = "打开空调",
    **audio: float,
) -> CandidateRow:
    features = {
        "presence_score": 0.5,
        "enhanced_cosine": 0.6,
        "mixture_cosine": 0.55,
        "max_cosine": 0.6,
        "latency_ms": 100.0,
        "cmd_duration_sec": 1.25,
        "cmd_rms": 0.02,
    }
    features.update(audio)
    return CandidateRow(
        id=sid,
        split="pos" if label is not None else "neg",
        label=label,
        r3_text=r3,
        primary_text=primary,
        energy_text=energy,
        tse_text=tse,
        audio_features=features,
        original_command_audio=None,
        source_digest="",
        dedup_sources={},
    )


class GateFeatureTests(unittest.TestCase):
    def test_feature_vector_is_independent_of_row_label(self):
        # Catches realistic leakage from reference text or pos/neg split metadata.
        original = _row("x", "打开空调")
        changed_label = replace(original, label=None)
        np.testing.assert_array_equal(
            build_gate_feature_matrix([original]),
            build_gate_feature_matrix([changed_label]),
        )

    def test_feature_missing_values_keep_explicit_flags(self):
        # Catches median/zero preprocessing that erases missingness information.
        missing_row = _row("missing", None, presence_score=math.nan, cmd_rms=None)
        observed_zero_row = _row("zero", None, presence_score=0.0, cmd_rms=0.0)
        matrix = build_gate_feature_matrix([missing_row, observed_zero_row])
        presence = GATE_FEATURE_SCHEMA.index("presence_score")
        presence_missing = GATE_FEATURE_SCHEMA.index("presence_score_missing")
        rms = GATE_FEATURE_SCHEMA.index("cmd_rms")
        rms_missing = GATE_FEATURE_SCHEMA.index("cmd_rms_missing")
        self.assertEqual(matrix[0, presence], 0.0)
        self.assertEqual(matrix[1, presence], 0.0)
        self.assertEqual(matrix[0, presence_missing], 1.0)
        self.assertEqual(matrix[1, presence_missing], 0.0)
        self.assertEqual(matrix[0, rms], 0.0)
        self.assertEqual(matrix[1, rms], 0.0)
        self.assertEqual(matrix[0, rms_missing], 1.0)
        self.assertEqual(matrix[1, rms_missing], 0.0)
        self.assertTrue(np.isfinite(matrix).all())

    def test_gate_feature_schema_excludes_label_and_oracle_fields(self):
        # Catches adding training-only targets to the published gate boundary.
        for name in GATE_FEATURE_SCHEMA:
            lowered = name.lower()
            self.assertNotIn("label", lowered, name)
            self.assertNotIn("reference", lowered, name)
            self.assertNotIn("oracle", lowered, name)
            self.assertNotIn("split", lowered, name)
            self.assertNotIn("candidate_cer", lowered, name)

    def test_feature_builder_emits_one_vector_per_row(self):
        # Catches accidentally reusing R10's five-action stacked representation.
        rows = [_row("a", "打开空调"), _row("b", None)]
        matrix = build_gate_feature_matrix(rows)
        self.assertEqual(matrix.dtype, np.float64)
        self.assertEqual(matrix.shape, (2, len(GATE_FEATURE_SCHEMA)))


class OracleMetricTests(unittest.TestCase):
    def test_oracle_frontier_matches_literal_two_row_example(self):
        rows = [
            _row("p", "甲", r3="甲", primary="乙", energy="", tse="丙"),
            _row("n", None, r3="", primary="", energy="", tse=""),
        ]
        contributions = build_oracle_contributions(rows, {"p": "甲", "n": None})
        points = gate_oracle_frontier([0.9, 0.8], contributions)
        by_threshold = {point["threshold"]: point for point in points}

        reject_all = by_threshold[math.inf]
        self.assertEqual(reject_all["cer"], 1.0)
        self.assertEqual(reject_all["rr"], 1.0)
        self.assertEqual(reject_all["overall"], 0.5)
        self.assertEqual(reject_all["accepted_positives"], 0.0)
        self.assertEqual(reject_all["accepted_negatives"], 0.0)

        accepted_positive = by_threshold[0.9]
        self.assertEqual(accepted_positive["cer"], 0.0)
        self.assertEqual(accepted_positive["rr"], 1.0)
        self.assertEqual(accepted_positive["overall"], 1.0)
        self.assertEqual(accepted_positive["accepted_positives"], 1.0)
        self.assertEqual(accepted_positive["accepted_negatives"], 0.0)

        accepted_both = by_threshold[0.8]
        self.assertEqual(accepted_both["cer"], 0.0)
        self.assertEqual(accepted_both["rr"], 0.0)
        self.assertEqual(accepted_both["overall"], 0.5)
        self.assertEqual(accepted_both["accepted_positives"], 1.0)
        self.assertEqual(accepted_both["accepted_negatives"], 1.0)

    def test_accepted_negative_reduces_rr_even_when_all_candidates_are_empty(self):
        row = _row("n", None, r3="", primary="", energy="", tse="")
        contributions = build_oracle_contributions([row], {"n": None})
        accepted = gate_oracle_frontier([0.5], contributions)[1]
        self.assertEqual(accepted["rr"], 0.0)
        self.assertEqual(accepted["accepted_negatives"], 1.0)

    def test_oracle_rejected_positive_charges_full_deletion(self):
        row = _row("p", "甲乙丙丁", r3="甲乙丙丁", primary="", energy="", tse="")
        contributions = build_oracle_contributions([row], {"p": "甲乙丙丁"})
        reject_all = gate_oracle_frontier([0.4], contributions)[0]
        self.assertEqual(contributions.ref_chars.tolist(), [4])
        self.assertEqual(reject_all["cer"], 1.0)

    def test_oracle_candidate_cer_ties_use_source_order(self):
        fixtures = [
            (
                _row("r3", "甲乙", r3="甲丙", primary="甲丁", energy="甲戊", tse="甲己"),
                "r3",
            ),
            (
                _row("primary", "甲乙", r3="", primary="甲丙", energy="甲丁", tse="甲戊"),
                "primary",
            ),
            (
                _row("energy", "甲乙", r3="", primary="", energy="甲丙", tse="甲丁"),
                "energy",
            ),
        ]
        labels = {row.id: "甲乙" for row, _expected in fixtures}
        contributions = build_oracle_contributions(
            [row for row, _expected in fixtures], labels
        )
        self.assertEqual(
            contributions.chosen_actions,
            tuple(expected for _row_value, expected in fixtures),
        )
        self.assertEqual(contributions.substitutions.tolist(), [1, 1, 1])
        self.assertEqual(contributions.ref_chars.tolist(), [2, 2, 2])

    def test_frontier_keeps_tied_scores_in_one_boundary(self):
        rows = [
            _row("p", "甲", r3="甲"),
            _row("n", None, r3=""),
        ]
        contributions = build_oracle_contributions(rows, {"p": "甲", "n": None})
        points = gate_oracle_frontier([0.5, 0.5], contributions)
        self.assertEqual([point["threshold"] for point in points], [math.inf, 0.5])
        self.assertEqual(points[1]["accepted_positives"], 1.0)
        self.assertEqual(points[1]["accepted_negatives"], 1.0)

    def test_oracle_pools_integer_edit_contributions(self):
        rows = [
            _row("p0", "甲乙", r3="甲丙"),
            _row("p1", "丁", r3="丁戊"),
        ]
        contributions = build_oracle_contributions(rows, {"p0": "甲乙", "p1": "丁"})
        points = gate_oracle_frontier([0.9, 0.8], contributions)
        accepted_both = points[-1]
        self.assertEqual(contributions.substitutions.tolist(), [1, 0])
        self.assertEqual(contributions.insertions.tolist(), [0, 1])
        self.assertEqual(contributions.deletions.tolist(), [0, 0])
        self.assertEqual(contributions.ref_chars.tolist(), [2, 1])
        self.assertAlmostEqual(accepted_both["cer"], 2.0 / 3.0)

    def test_select_frontier_point_enforces_floor_and_deterministic_ties(self):
        points = [
            {"threshold": 0.9, "cer": 0.2, "rr": 0.95, "overall": 0.8},
            {"threshold": 0.8, "cer": 0.3, "rr": 0.96, "overall": 0.8},
            {"threshold": 0.7, "cer": 0.0, "rr": 0.90, "overall": 0.95},
        ]
        self.assertIs(select_frontier_point(points, 0.95), points[1])
        self.assertIsNone(select_frontier_point(points, 0.99))


if __name__ == "__main__":
    unittest.main()
