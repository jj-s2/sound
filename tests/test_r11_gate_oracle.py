"""Tests for R11 label-free gate features and oracle frontier."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from xh202615.r10_selector import CandidateRow
from xh202615.r11_gate_oracle import (
    GATE_FEATURE_SCHEMA,
    CrossFitResult,
    build_gate_feature_matrix,
    build_oracle_contributions,
    cross_fit_gate_models,
    default_model_specs,
    evaluate_e0,
    gate_oracle_frontier,
    group_bootstrap_best_frontier,
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
    def test_feature_vector_is_independent_of_row_label_and_split(self):
        # Catches realistic leakage from reference text or pos/neg split metadata.
        original = _row("x", "打开空调")
        changed_label_and_split = replace(original, label=None, split="neg")
        np.testing.assert_array_equal(
            build_gate_feature_matrix([original]),
            build_gate_feature_matrix([changed_label_and_split]),
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


class GateCrossFitTests(unittest.TestCase):
    @staticmethod
    def _synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(20260810)
        groups = np.repeat([f"g{index:02d}" for index in range(10)], 2)
        target = np.tile(np.array([0, 1], dtype=np.int64), 10)
        X = rng.normal(size=(len(target), 5))
        X[:, 0] += target * 1.5
        X[::4, 2] = np.nan
        return X, target, groups

    def test_default_model_specs_match_frozen_grid(self):
        specs = default_model_specs()
        self.assertEqual(len(specs), 6)
        self.assertEqual(
            [(spec.family, dict(spec.parameters)) for spec in specs],
            [
                ("logistic", {"C": 0.01}),
                ("logistic", {"C": 0.1}),
                ("logistic", {"C": 1.0}),
                ("logistic", {"C": 10.0}),
                (
                    "hist_gradient_boosting",
                    {
                        "max_leaf_nodes": 3,
                        "learning_rate": 0.05,
                        "max_iter": 150,
                        "l2_regularization": 1.0,
                    },
                ),
                (
                    "hist_gradient_boosting",
                    {
                        "max_leaf_nodes": 7,
                        "learning_rate": 0.05,
                        "max_iter": 150,
                        "l2_regularization": 1.0,
                    },
                ),
            ],
        )
        self.assertEqual(len({spec.name for spec in specs}), len(specs))

    def test_cross_fit_has_once_only_group_disjoint_coverage(self):
        X, target, groups = self._synthetic_data()
        result = cross_fit_gate_models(
            X,
            target,
            groups,
            n_splits=5,
            seed=20260807,
            specs=default_model_specs(),
        )

        self.assertEqual(result.fold_assignments.shape, (len(target),))
        self.assertTrue(np.all(result.fold_assignments >= 0))
        covered = np.concatenate(
            [np.asarray(fold["test_indices"], dtype=np.int64) for fold in result.fold_metadata]
        )
        np.testing.assert_array_equal(np.sort(covered), np.arange(len(target)))
        self.assertEqual(len(np.unique(covered)), len(target))

        expected_names = [spec.name for spec in result.specs]
        self.assertEqual(list(result.scores_by_model), expected_names)
        for scores in result.scores_by_model.values():
            self.assertEqual(scores.shape, (len(target),))
            self.assertTrue(np.isfinite(scores).all())
            self.assertTrue(((scores >= 0.0) & (scores <= 1.0)).all())

        for fold in result.fold_metadata:
            train_groups = set(fold["train_groups"])
            test_groups = set(fold["test_groups"])
            self.assertFalse(train_groups & test_groups)
            test_indices = np.asarray(fold["test_indices"], dtype=np.int64)
            self.assertTrue(
                np.all(result.fold_assignments[test_indices] == fold["fold_index"])
            )

    def test_cross_fit_fail_closed_when_held_out_fold_loses_class(self):
        X = np.arange(24, dtype=np.float64).reshape(12, 2)
        target = np.array([0] * 4 + [0, 0, 1, 1] + [1] * 4, dtype=np.int64)
        groups = np.repeat(np.array(["g0", "g1", "g2"]), 4)
        with self.assertRaisesRegex(ValueError, "both target classes"):
            cross_fit_gate_models(
                X,
                target,
                groups,
                n_splits=3,
                seed=20260807,
                specs=default_model_specs()[:1],
            )


class GateBootstrapDecisionTests(unittest.TestCase):
    @staticmethod
    def _literal_group_fixture():
        rows: list[CandidateRow] = []
        labels: dict[str, str | None] = {}
        groups: list[str] = []
        for group_index in range(4):
            positive_id = f"g{group_index}_p"
            negative_id = f"g{group_index}_n"
            rows.extend(
                [
                    _row(positive_id, "甲", r3="甲", primary="", energy="", tse=""),
                    _row(negative_id, None, r3="", primary="", energy="", tse=""),
                ]
            )
            labels[positive_id] = "甲"
            labels[negative_id] = None
            groups.extend([f"g{group_index}", f"g{group_index}"])
        scores = {
            "model_a": np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.7, 0.4, 0.6]),
            "model_b": np.array([0.3, 0.7, 0.4, 0.6, 0.9, 0.1, 0.8, 0.2]),
        }
        return rows, labels, np.asarray(groups), scores

    def test_group_bootstrap_reselects_model_and_threshold_per_replicate(self):
        rows, labels, groups, scores = self._literal_group_fixture()
        contributions = build_oracle_contributions(rows, labels)
        first = group_bootstrap_best_frontier(
            scores,
            contributions,
            groups,
            rr_floor=0.93,
            n_boot=64,
            seed=17,
        )
        second = group_bootstrap_best_frontier(
            scores,
            contributions,
            groups,
            rr_floor=0.93,
            n_boot=64,
            seed=17,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["n_boot"], 64)
        self.assertEqual(len(first["overall_samples"]), 64)
        self.assertEqual(len(first["selected_models"]), 64)
        self.assertEqual(len(first["selected_thresholds"]), 64)
        self.assertEqual(set(first["selected_models"]), {"model_a", "model_b"})
        self.assertGreater(len(set(first["selected_thresholds"])), 1)

    def test_group_bootstrap_uses_stable_model_name_for_metric_ties(self):
        rows, labels, groups, scores = self._literal_group_fixture()
        contributions = build_oracle_contributions(rows, labels)
        tied = group_bootstrap_best_frontier(
            {"z_model": scores["model_a"], "a_model": scores["model_a"].copy()},
            contributions,
            groups,
            rr_floor=0.93,
            n_boot=8,
            seed=17,
        )
        self.assertEqual(set(tied["selected_models"]), {"a_model"})

    def test_group_bootstrap_fails_closed_if_resample_loses_target_class(self):
        rows = [
            _row("p", "甲", r3="甲", primary="", energy="", tse=""),
            _row("n", None, r3="", primary="", energy="", tse=""),
        ]
        contributions = build_oracle_contributions(rows, {"p": "甲", "n": None})
        with self.assertRaisesRegex(ValueError, "lost a target class"):
            group_bootstrap_best_frontier(
                {"model": np.array([0.9, 0.1])},
                contributions,
                np.array(["positive", "negative"]),
                rr_floor=0.93,
                n_boot=1,
                seed=11,
            )

    @classmethod
    def _patched_cross_fit(cls):
        rows, labels, groups, scores = cls._literal_group_fixture()
        spec = default_model_specs()[0]
        scores_by_model = {spec.name: scores["model_a"]}
        result = CrossFitResult(
            specs=(spec,),
            scores_by_model=scores_by_model,
            fold_assignments=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
            fold_metadata=(
                {
                    "fold_index": 0,
                    "train_indices": [4, 5, 6, 7],
                    "test_indices": [0, 1, 2, 3],
                    "train_groups": ["g2", "g3"],
                    "test_groups": ["g0", "g1"],
                },
                {
                    "fold_index": 1,
                    "train_indices": [0, 1, 2, 3],
                    "test_indices": [4, 5, 6, 7],
                    "train_groups": ["g0", "g1"],
                    "test_groups": ["g2", "g3"],
                },
            ),
        )
        return rows, labels, groups, result

    def test_decision_falsifies_when_bootstrap_ci_high_is_below_point_eight(self):
        rows, labels, groups, cross_fit = self._patched_cross_fit()
        bootstrap = {"ci_low": 0.70, "ci_high": 0.79, "n_boot": 8}
        with patch(
            "xh202615.r11_gate_oracle.cross_fit_gate_models", return_value=cross_fit
        ), patch(
            "xh202615.r11_gate_oracle.group_bootstrap_best_frontier",
            return_value=bootstrap,
        ):
            result = evaluate_e0(rows, labels, groups, n_splits=2, n_boot=8)
        self.assertEqual(result["decision"], "falsified_cached")
        self.assertFalse(result["selected_point"]["deployable"])

    def test_decision_continues_cached_only_with_pooled_and_worst_fold_headroom(self):
        rows, labels, groups, cross_fit = self._patched_cross_fit()
        model_name = next(iter(cross_fit.scores_by_model))
        cross_fit = CrossFitResult(
            specs=cross_fit.specs,
            scores_by_model={
                model_name: np.array([0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2])
            },
            fold_assignments=cross_fit.fold_assignments,
            fold_metadata=cross_fit.fold_metadata,
        )
        bootstrap = {"ci_low": 0.90, "ci_high": 1.0, "n_boot": 8}
        with patch(
            "xh202615.r11_gate_oracle.cross_fit_gate_models", return_value=cross_fit
        ), patch(
            "xh202615.r11_gate_oracle.group_bootstrap_best_frontier",
            return_value=bootstrap,
        ):
            result = evaluate_e0(rows, labels, groups, n_splits=2, n_boot=8)
        self.assertEqual(result["decision"], "continue_cached")
        self.assertEqual(result["selected_point"]["overall"], 1.0)
        self.assertEqual(result["worst_fold"]["overall"], 1.0)

    def test_decision_proceeds_pvad_when_best_point_lacks_headroom(self):
        rows, labels, groups, cross_fit = self._patched_cross_fit()
        bootstrap = {"ci_low": 0.70, "ci_high": 0.85, "n_boot": 8}
        with patch(
            "xh202615.r11_gate_oracle.cross_fit_gate_models", return_value=cross_fit
        ), patch(
            "xh202615.r11_gate_oracle.group_bootstrap_best_frontier",
            return_value=bootstrap,
        ):
            result = evaluate_e0(rows, labels, groups, n_splits=2, n_boot=8)
        self.assertEqual(result["selected_point"]["overall"], 0.75)
        self.assertEqual(result["decision"], "proceed_pvad")
        self.assertEqual(result["worst_fold"]["overall"], 0.5)


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
