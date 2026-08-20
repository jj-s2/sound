"""Tests for R11 label-free gate features and oracle frontier."""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from xh202615.data import Sample
from xh202615.evaluation import evaluate_rows
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.r11_gate_oracle_oof import (
    _ARTIFACT_KIND,
    _REJECT_ALL_THRESHOLD_MARKER,
    _SCHEMA_VERSION,
    _SENTINEL_ACCEPTED_NEGATIVE,
    _check_evaluator_parity,
    _next_branch,
    main,
    write_e0_artifacts,
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
        self.assertEqual(len(specs), 7)
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
                (
                    "hist_gradient_boosting",
                    {
                        "max_leaf_nodes": 15,
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

    @staticmethod
    def _manual_best_for_sample(
        sampled_groups: list[str],
        rows: list[CandidateRow],
        labels: dict[str, str | None],
        groups: np.ndarray,
        scores: dict[str, np.ndarray],
        rr_floor: float,
    ) -> tuple[str, float]:
        indices = [
            index
            for sampled_group in sampled_groups
            for index, group in enumerate(groups)
            if group == sampled_group
        ]
        best = None
        for model_name in sorted(scores):
            sampled_scores = [float(scores[model_name][index]) for index in indices]
            thresholds = [math.inf, *sorted(set(sampled_scores), reverse=True)]
            model_best = None
            for threshold in thresholds:
                accepted = [score >= threshold for score in sampled_scores]
                positives = [labels[rows[index].id] is not None for index in indices]
                n_negative = sum(not positive for positive in positives)
                accepted_negative = sum(
                    accept and not positive
                    for accept, positive in zip(accepted, positives)
                )
                rr = (n_negative - accepted_negative) / n_negative
                errors = sum(
                    0 if accept else 1
                    for accept, positive in zip(accepted, positives)
                    if positive
                )
                ref_chars = sum(positives)
                cer = errors / ref_chars
                overall = ((1.0 - cer) + rr) / 2.0
                if rr < rr_floor:
                    continue
                key = (overall, rr, -cer, threshold)
                if model_best is None or key > model_best[0]:
                    model_best = (key, threshold)
            candidate = (model_best[0][:3], model_name, model_best[1])
            if best is None or candidate[0] > best[0]:
                best = candidate
        return best[1], best[2]

    def test_group_bootstrap_reselects_model_and_threshold_per_replicate(self):
        rows, labels, groups, scores = self._literal_group_fixture()
        contributions = build_oracle_contributions(rows, labels)
        n_boot = 16
        seed = 17
        first = group_bootstrap_best_frontier(
            scores,
            contributions,
            groups,
            rr_floor=0.93,
            n_boot=n_boot,
            seed=seed,
        )
        second = group_bootstrap_best_frontier(
            scores,
            contributions,
            groups,
            rr_floor=0.93,
            n_boot=n_boot,
            seed=seed,
        )

        rng = np.random.default_rng(seed)
        group_names = sorted(set(groups.tolist()))
        expected = []
        while len(expected) < n_boot:
            sampled = [
                group_names[index]
                for index in rng.integers(0, len(group_names), size=len(group_names))
            ]
            sampled_indices = [
                index
                for sampled_group in sampled
                for index, group in enumerate(groups)
                if group == sampled_group
            ]
            sampled_classes = {labels[rows[index].id] is not None for index in sampled_indices}
            if len(sampled_classes) != 2:
                continue
            expected.append(
                self._manual_best_for_sample(
                    sampled, rows, labels, groups, scores, rr_floor=0.93
                )
            )

        self.assertEqual(first, second)
        self.assertEqual(first["n_boot"], n_boot)
        self.assertEqual(first["attempted_replicates"], n_boot)
        self.assertEqual(first["rejected_replicates"], 0)
        self.assertEqual(first["selected_models"], [item[0] for item in expected])
        self.assertEqual(first["selected_thresholds"], [item[1] for item in expected])

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

    def test_group_bootstrap_redraws_class_degenerate_samples(self):
        rows = [
            _row("p", "甲", r3="甲", primary="", energy="", tse=""),
            _row("n", None, r3="", primary="", energy="", tse=""),
        ]
        contributions = build_oracle_contributions(rows, {"p": "甲", "n": None})
        result = group_bootstrap_best_frontier(
            {"model": np.array([0.9, 0.1])},
            contributions,
            np.array(["positive", "negative"]),
            rr_floor=0.93,
            n_boot=1,
            seed=11,
            max_attempts=2,
        )
        self.assertEqual(result["n_boot"], 1)
        self.assertEqual(result["attempted_replicates"], 2)
        self.assertEqual(result["rejected_replicates"], 1)
        self.assertEqual(result["selected_models"], ["model"])
        self.assertEqual(result["selected_thresholds"], [0.9])

    def test_group_bootstrap_fails_closed_when_redraw_cap_is_exhausted(self):
        rows = [
            _row("p", "甲", r3="甲", primary="", energy="", tse=""),
            _row("n", None, r3="", primary="", energy="", tse=""),
        ]
        contributions = build_oracle_contributions(rows, {"p": "甲", "n": None})
        with self.assertRaisesRegex(RuntimeError, "exhausted max_attempts=1"):
            group_bootstrap_best_frontier(
                {"model": np.array([0.9, 0.1])},
                contributions,
                np.array(["positive", "negative"]),
                rr_floor=0.93,
                n_boot=1,
                seed=11,
                max_attempts=1,
            )

    def test_group_bootstrap_uses_best_point_path_without_full_frontiers(self):
        rows, labels, groups, scores = self._literal_group_fixture()
        contributions = build_oracle_contributions(rows, labels)
        with patch(
            "xh202615.r11_gate_oracle.gate_oracle_frontier",
            side_effect=AssertionError("bootstrap must not allocate full frontiers"),
        ), patch(
            "xh202615.r11_gate_oracle._subset_contributions",
            side_effect=AssertionError("bootstrap must not copy chosen actions"),
        ):
            result = group_bootstrap_best_frontier(
                scores,
                contributions,
                groups,
                rr_floor=0.93,
                n_boot=4,
                seed=17,
            )
        self.assertEqual(len(result["overall_samples"]), 4)

    def test_group_bootstrap_best_point_keeps_tied_scores_atomic(self):
        rows = [
            _row("p0", "甲", r3="甲", primary="", energy="", tse=""),
            _row("n0", None, r3="", primary="", energy="", tse=""),
            _row("p1", "乙", r3="乙", primary="", energy="", tse=""),
            _row("n1", None, r3="", primary="", energy="", tse=""),
        ]
        labels = {"p0": "甲", "n0": None, "p1": "乙", "n1": None}
        contributions = build_oracle_contributions(rows, labels)
        result = group_bootstrap_best_frontier(
            {"model": np.array([0.8, 0.8, 0.7, 0.1])},
            contributions,
            np.array(["group"] * 4),
            rr_floor=0.5,
            n_boot=1,
            seed=1,
        )
        self.assertEqual(result["selected_thresholds"], [0.7])

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


class ArtifactWriterTests(unittest.TestCase):
    def _synthetic_rows(self):
        labels = {"0": "甲", "1": None, "2": "乙", "3": None}
        groups = ["g0", "g0", "g1", "g1"]
        rows = [
            _row("0", "甲", r3="甲", primary="乙", energy="", tse="甲"),
            _row("1", None, r3="", primary="", energy="", tse=""),
            _row("2", "乙", r3="乙", primary="丙", energy="", tse="乙"),
            _row("3", None, r3="", primary="", energy="", tse=""),
        ]
        return rows, labels, groups

    def _make_paths(self, tmp_path: Path):
        dataset_root = tmp_path / "datasetA"
        dataset_root.mkdir()
        for split in ("pos", "neg"):
            (dataset_root / f"{split}.jsonl").write_text("", encoding="utf-8")
        candidate_fusion = tmp_path / "candidate_fusion.jsonl"
        tse_asr = tmp_path / "tse_asr.jsonl"
        audio_map = tmp_path / "audio_map.jsonl"
        r3_predictions = tmp_path / "r3_predictions.jsonl"
        group_manifest = tmp_path / "manifest.json"
        for path in (candidate_fusion, tse_asr, audio_map, r3_predictions):
            path.write_text(
                "".join(json.dumps({"id": sid}) + "\n" for sid in ("0", "1", "2", "3")),
                encoding="utf-8",
            )
        group_manifest.write_text(
            json.dumps(
                {
                    "rows": [
                        {"id": "0", "wake_component": "g0"},
                        {"id": "1", "wake_component": "g0"},
                        {"id": "2", "wake_component": "g1"},
                        {"id": "3", "wake_component": "g1"},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "dataset_root": dataset_root,
            "candidate_fusion": candidate_fusion,
            "tse_asr": tse_asr,
            "audio_map": audio_map,
            "r3_predictions": r3_predictions,
            "group_manifest": group_manifest,
            "n_outer": 2,
            "n_boot": 4,
            "seed": 1,
            "rr_floor": 1.0,
        }

    def _evaluate(self, rows, labels, groups):
        result = evaluate_e0(
            rows,
            labels,
            groups,
            n_splits=2,
            seed=1,
            rr_floor=1.0,
            n_boot=4,
        )
        result["official_metrics"] = _check_evaluator_parity(rows, labels, result)
        return result

    def _expected_files(self, out_root: Path):
        return {
            "manifest": out_root / "e0_manifest.json",
            "summary": out_root / "e0_summary.json",
            "scores": out_root / "e0_oof_scores.jsonl",
            "frontier": out_root / "e0_frontier.jsonl",
            "report": out_root / "e0_report.md",
        }

    def test_writes_exactly_five_artifacts_with_expected_names(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            expected = self._expected_files(out_root)
            self.assertEqual(files, expected)
            self.assertEqual(
                {p.name for p in out_root.iterdir()},
                {p.name for p in expected.values()},
            )
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_manifest_has_required_provenance_and_summary_keys(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            for key in (
                "artifact_kind",
                "schema_version",
                "config_hash",
                "source_digest",
                "feature_state_digest",
                "resolved_paths",
                "source_id_sets",
                "coverage",
                "fold_metadata",
                "official_parity",
                "model_specs",
                "feature_schema",
                "decision",
                "selected_point",
                "worst_fold",
                "fold_metrics",
                "bootstrap_summary",
            ):
                self.assertIn(key, manifest, key)
            self.assertEqual(manifest["artifact_kind"], _ARTIFACT_KIND)
            self.assertEqual(manifest["schema_version"], _SCHEMA_VERSION)
            self.assertEqual(manifest["selected_point"]["diagnostic_only"], True)
            self.assertEqual(manifest["selected_point"]["deployable"], False)
            self.assertEqual(manifest["coverage"]["n_rows_total"], len(rows))
            self.assertTrue(manifest["coverage"]["once_only"])
            for fold in manifest["fold_metadata"]:
                self.assertTrue(fold["group_disjoint"])
                self.assertEqual(
                    set(fold["train_groups"]) & set(fold["test_groups"]),
                    set(),
                )
            parity = manifest["official_parity"]
            for key in ("custom", "official", "deltas", "parity"):
                self.assertIn(key, parity)
            self.assertIn("false_reject_rate", parity["official"])
            self.assertIn("false_accept_rate", parity["official"])
            summary = json.loads(files["summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary["artifact_kind"], _ARTIFACT_KIND)
            self.assertEqual(summary["schema_version"], _SCHEMA_VERSION)
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_scores_roundtrip_without_label_or_reference_text(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            lines = files["scores"].read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), len(rows))
            for line in lines:
                row = json.loads(line)
                self.assertEqual(set(row), {"id", "group", "fold", *result["scores_by_model"]})
                for model_name in result["scores_by_model"]:
                    score = row[model_name]
                    self.assertIsInstance(score, float)
                    self.assertTrue(0.0 <= score <= 1.0)
                    self.assertTrue(math.isfinite(score))
                self.assertNotIn("label", row)
                self.assertNotIn("recognition_text", row)
                self.assertNotIn("target_present", row)
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_frontier_rows_contain_metrics_model_and_threshold_only(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            lines = files["frontier"].read_text(encoding="utf-8").strip().split("\n")
            self.assertGreater(len(lines), 0)
            for line in lines:
                row = json.loads(line)
                self.assertIn("model", row)
                self.assertIn("threshold", row)
                for key in ("cer", "rr", "overall", "accepted_positives", "accepted_negatives"):
                    self.assertIn(key, row)
                for forbidden in ("label", "reference", "text", "chosen_action"):
                    self.assertNotIn(forbidden, row)
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_reject_all_threshold_uses_json_safe_marker(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        result["selected_point"]["threshold"] = math.inf
        result["frontier"].append(
            {
                "model": result["selected_point"]["model"],
                "threshold": math.inf,
                "cer": 1.0,
                "rr": 1.0,
                "overall": 0.5,
                "accepted_positives": 0.0,
                "accepted_negatives": 0.0,
            }
        )
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_point"]["threshold"], _REJECT_ALL_THRESHOLD_MARKER)
            lines = files["frontier"].read_text(encoding="utf-8").strip().split("\n")
            thresholds = {json.loads(line)["threshold"] for line in lines}
            self.assertIn(_REJECT_ALL_THRESHOLD_MARKER, thresholds)
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_digest_and_config_hash_are_deterministic(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)

        def run(tmp: Path):
            paths = self._make_paths(tmp)
            out_root = tmp / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            return manifest["config_hash"], manifest["source_digest"]

        tmp_a = Path(tempfile.mkdtemp())
        tmp_b = Path(tempfile.mkdtemp())
        try:
            hash_a, digest_a = run(tmp_a)
            hash_b, digest_b = run(tmp_b)
            self.assertEqual(hash_a, hash_b)
            self.assertEqual(digest_a["aggregate"], digest_b["aggregate"])
            self.assertEqual(digest_a["per_source"], digest_b["per_source"])
        finally:
            import shutil
            shutil.rmtree(tmp_a)
            shutil.rmtree(tmp_b)

    def test_source_content_digest_per_source_and_aggregate(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            digest = manifest["source_digest"]
            self.assertIn("aggregate", digest)
            self.assertIn("per_source", digest)
            for name in (
                "candidate_fusion",
                "tse_asr",
                "audio_map",
                "r3_predictions",
                "group_manifest",
                "dataset_pos",
                "dataset_neg",
            ):
                self.assertIn(name, digest["per_source"])
                self.assertEqual(len(digest["per_source"][name]), 64)
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_source_content_digest_is_sensitive_to_each_raw_input(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)

        def base_digest():
            tmp_path = Path(tempfile.mkdtemp())
            try:
                paths = self._make_paths(tmp_path)
                out_root = tmp_path / "out"
                files = write_e0_artifacts(result, rows, groups, paths, out_root)
                manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
                return manifest["source_digest"], paths, tmp_path
            except Exception:
                shutil.rmtree(tmp_path)
                raise

        original, _, _ = base_digest()
        for name in (
            "candidate_fusion",
            "tse_asr",
            "audio_map",
            "r3_predictions",
            "group_manifest",
        ):
            tmp_path = Path(tempfile.mkdtemp())
            try:
                paths = self._make_paths(tmp_path)
                out_root = tmp_path / "out"
                path = paths[name]
                original_bytes = path.read_bytes()
                path.write_bytes(original_bytes + b"\n")
                files = write_e0_artifacts(result, rows, groups, paths, out_root)
                manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
                mutated = manifest["source_digest"]
                self.assertNotEqual(
                    mutated["per_source"][name],
                    original["per_source"][name],
                    f"{name} digest did not change after mutation",
                )
                self.assertNotEqual(
                    mutated["aggregate"], original["aggregate"], f"{name} mutation did not change aggregate"
                )
                for other in original["per_source"]:
                    if other != name:
                        self.assertEqual(
                            mutated["per_source"][other],
                            original["per_source"][other],
                            f"{name} mutation changed {other} digest",
                        )
            finally:
                import shutil
                shutil.rmtree(tmp_path)

        for split in ("pos", "neg"):
            tmp_path = Path(tempfile.mkdtemp())
            try:
                paths = self._make_paths(tmp_path)
                out_root = tmp_path / "out"
                split_path = paths["dataset_root"] / f"{split}.jsonl"
                split_path.write_bytes(split_path.read_bytes() + b"\n")
                files = write_e0_artifacts(result, rows, groups, paths, out_root)
                manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
                mutated = manifest["source_digest"]
                key = f"dataset_{split}"
                self.assertNotEqual(mutated["per_source"][key], original["per_source"][key])
                self.assertNotEqual(mutated["aggregate"], original["aggregate"])
            finally:
                import shutil
                shutil.rmtree(tmp_path)

    def test_report_contains_decision_and_next_branch(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            report = files["report"].read_text(encoding="utf-8")
            self.assertIn(f"Decision: {result['decision']}", report)
            self.assertIn(_next_branch(result["decision"]), report)
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_official_parity_for_selected_point(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        selected = result["selected_point"]
        threshold = float(selected["threshold"])
        model_scores = result["scores_by_model"][selected["model"]]
        contributions = build_oracle_contributions(rows, labels)
        accepted = model_scores >= threshold
        predictions = []
        for row, accept, action in zip(rows, accepted, contributions.chosen_actions):
            if labels[row.id] is None:
                text = _SENTINEL_ACCEPTED_NEGATIVE if accept else ""
            else:
                text = row.texts.get(action, "") if accept else ""
            predictions.append({"id": row.id, "recognition_text": text})
        samples = [
            Sample(
                id=row.id,
                split=row.split,
                wakeup_audio=Path("."),
                wakeup_text="",
                command_audio=row.original_command_audio or Path("."),
                label=labels[row.id],
            )
            for row in rows
        ]
        official = evaluate_rows(samples, predictions, missing_policy="empty").metrics
        self.assertAlmostEqual(selected["cer"], official["avg_cer"], places=9)
        self.assertAlmostEqual(selected["rr"], official["avg_rr"], places=9)
        expected_overall = ((1.0 - official["avg_cer"]) + official["avg_rr"]) / 2.0
        self.assertAlmostEqual(selected["overall"], expected_overall, places=9)

    def test_feature_state_digest_is_sensitive_to_joined_state(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            original = manifest["feature_state_digest"]

            mutated = list(rows)
            mutated[0] = replace(
                mutated[0],
                audio_features={**mutated[0].audio_features, "presence_score": 9.9},
            )
            out_root2 = tmp_path / "out2"
            files2 = write_e0_artifacts(result, mutated, groups, paths, out_root2)
            manifest2 = json.loads(files2["manifest"].read_text(encoding="utf-8"))
            self.assertNotEqual(original, manifest2["feature_state_digest"])
            self.assertEqual(
                manifest["source_digest"], manifest2["source_digest"]
            )
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_source_id_sets_record_exact_counts(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            for key in ("candidate_fusion", "tse_asr", "audio_map", "r3_predictions", "group_manifest"):
                self.assertEqual(manifest["source_id_sets"][key]["count"], len(rows))
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_coverage_matches_independent_recomputation(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            coverage = manifest["coverage"]
            assignments = np.asarray(result["fold_assignments"], dtype=np.int64)
            self.assertEqual(coverage["n_rows_total"], len(rows))
            self.assertEqual(coverage["n_rows_covered"], len(rows))
            self.assertTrue(coverage["once_only"])
            self.assertEqual(coverage["n_folds"], int(assignments.max()) + 1)
            ids = [row.id for row in rows]
            self.assertEqual(coverage["sample_ids"], ids)
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_non_finite_score_leaves_no_output(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        model_name = next(iter(result["scores_by_model"]))
        result["scores_by_model"][model_name][0] = float("nan")
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertFalse(out_root.exists())
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_malformed_frontier_leaves_no_output(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        result["frontier"].append({"model": "bad", "threshold": 0.5})
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertFalse(out_root.exists())
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_infinite_non_threshold_metric_raises(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        result["frontier"].append(
            {
                "model": "bad",
                "threshold": 0.5,
                "cer": float("inf"),
                "rr": 1.0,
                "overall": 1.0,
                "accepted_positives": 1.0,
                "accepted_negatives": 0.0,
            }
        )
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertFalse(out_root.exists())
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def test_next_branch_text_is_exact_for_all_decisions(self):
        expected = {
            "continue_cached": "E1 — complete cached negative control: compare positive-only expected-CER regression and LambdaMART; do not tune cached-gate hyperparameters further.",
            "falsified_cached": "Stop cached-gate tuning; go directly to E2 — FireRedChat-pVAD zero-shot and fused gate-oracle.",
            "proceed_pvad": "Begin E2 — FireRedChat-pVAD zero-shot and fused gate-oracle.",
        }
        for decision, text in expected.items():
            self.assertEqual(_next_branch(decision), text)

    def test_missing_official_metrics_raises_before_output(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        del result["official_metrics"]
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertFalse(out_root.exists())
        finally:
            shutil.rmtree(tmp_path)

    def test_official_parity_in_manifest_separates_values(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            parity = manifest["official_parity"]
            for key in ("custom", "official", "deltas", "parity"):
                self.assertIn(key, parity)
            for key in ("avg_cer", "avg_rr", "overall"):
                self.assertAlmostEqual(parity["deltas"][key], 0.0, places=9)
                self.assertTrue(parity["parity"][key])
            self.assertIn("false_reject_rate", parity["official"])
            self.assertIn("false_accept_rate", parity["official"])
        finally:
            shutil.rmtree(tmp_path)

    def test_check_evaluator_parity_records_false_reject_for_empty_accepted_positive(self):
        rows = [
            _row("p", "甲", r3="", primary="", energy="", tse=""),
            _row("n", None, r3="", primary="", energy="", tse=""),
        ]
        labels = {"p": "甲", "n": None}
        groups = ["g0", "g1"]
        result = {
            "selected_point": {
                "model": "m",
                "threshold": 0.5,
                "cer": 1.0,
                "rr": 1.0,
                "overall": 0.5,
                "accepted_positives": 1.0,
                "accepted_negatives": 0.0,
                "diagnostic_only": True,
                "deployable": False,
            },
            "scores_by_model": {"m": np.array([0.9, 0.1])},
        }
        official = _check_evaluator_parity(rows, labels, result)
        self.assertEqual(official["false_reject_rate"], 1.0)
        self.assertEqual(official["false_accept_rate"], 0.0)

    def test_metric_disagreement_fails_closed(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        result["selected_point"]["cer"] = 0.999
        result["selected_point"]["overall"] = (
            (1.0 - result["selected_point"]["cer"]) + result["selected_point"]["rr"]
        ) / 2.0
        result["official_metrics"] = {
            "avg_cer": 0.0,
            "avg_rr": 1.0,
            "overall": 1.0,
            "false_reject_rate": 1.0,
            "false_accept_rate": 0.0,
        }
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertFalse(out_root.exists())
            self.assertFalse(any(p.name.startswith("out.staging") for p in tmp_path.iterdir()))
        finally:
            shutil.rmtree(tmp_path)

    def test_non_finite_official_metric_fails_closed(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        result["official_metrics"]["avg_cer"] = float("nan")
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertFalse(out_root.exists())
        finally:
            shutil.rmtree(tmp_path)

    def test_invalid_official_rate_fails_closed_before_output(self):
        rows, labels, groups = self._synthetic_rows()
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            for key, value in (
                ("avg_cer", -0.01),
                ("avg_rr", 1.01),
                ("false_reject_rate", -0.01),
                ("false_accept_rate", 1.01),
            ):
                with self.subTest(key=key, value=value):
                    result = self._evaluate(rows, labels, groups)
                    result["official_metrics"][key] = value
                    out_root = tmp_path / f"out-{key}"
                    with self.assertRaises(ValueError):
                        write_e0_artifacts(result, rows, groups, paths, out_root)
                    self.assertFalse(out_root.exists())
                    self.assertFalse((tmp_path / f"out-{key}.publish.lock").exists())
        finally:
            shutil.rmtree(tmp_path)

    def test_all_parity_flags_true_in_published_package(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            files = write_e0_artifacts(result, rows, groups, paths, out_root)
            manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
            self.assertTrue(all(manifest["official_parity"]["parity"].values()))
        finally:
            shutil.rmtree(tmp_path)

    def _valid_result_with_folds(self, rows, labels, groups):
        result = self._evaluate(rows, labels, groups)
        result["fold_metadata"] = (
            {
                "fold_index": 0,
                "train_indices": np.array([2, 3], dtype=np.int64),
                "test_indices": np.array([0, 1], dtype=np.int64),
                "train_groups": ["g1"],
                "test_groups": ["g0"],
            },
            {
                "fold_index": 1,
                "train_indices": np.array([0, 1], dtype=np.int64),
                "test_indices": np.array([2, 3], dtype=np.int64),
                "train_groups": ["g0"],
                "test_groups": ["g1"],
            },
        )
        result["fold_assignments"] = np.array([0, 0, 1, 1], dtype=np.int64)
        return result

    def test_fold_validation_rejects_duplicate_test_index(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._valid_result_with_folds(rows, labels, groups)
        result["fold_metadata"] = (
            {
                "fold_index": 0,
                "train_indices": np.array([2, 3], dtype=np.int64),
                "test_indices": np.array([0, 1], dtype=np.int64),
                "train_groups": ["g1"],
                "test_groups": ["g0"],
            },
            {
                "fold_index": 1,
                "train_indices": np.array([0], dtype=np.int64),
                "test_indices": np.array([1, 2, 3], dtype=np.int64),
                "train_groups": ["g0"],
                "test_groups": ["g0", "g1"],
            },
        )
        result["fold_assignments"] = np.array([0, 0, 1, 1], dtype=np.int64)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertFalse(out_root.exists())
            self.assertFalse(any(p.name.startswith("out.staging") for p in tmp_path.iterdir()))
        finally:
            shutil.rmtree(tmp_path)

    def test_fold_validation_rejects_missing_test_index(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._valid_result_with_folds(rows, labels, groups)
        result["fold_metadata"] = (
            {
                "fold_index": 0,
                "train_indices": np.array([1, 2, 3], dtype=np.int64),
                "test_indices": np.array([0], dtype=np.int64),
                "train_groups": ["g0", "g1"],
                "test_groups": ["g0"],
            },
            {
                "fold_index": 1,
                "train_indices": np.array([0], dtype=np.int64),
                "test_indices": np.array([1], dtype=np.int64),
                "train_groups": ["g0"],
                "test_groups": ["g0"],
            },
        )
        result["fold_assignments"] = np.array([0, 1, -1, -1], dtype=np.int64)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertFalse(out_root.exists())
        finally:
            shutil.rmtree(tmp_path)

    def test_fold_validation_rejects_assignment_disagreement(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._valid_result_with_folds(rows, labels, groups)
        result["fold_assignments"] = np.array([1, 1, 0, 0], dtype=np.int64)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertFalse(out_root.exists())
        finally:
            shutil.rmtree(tmp_path)

    def test_fold_validation_rejects_train_test_group_intersection(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._valid_result_with_folds(rows, labels, groups)
        meta = list(result["fold_metadata"])
        meta[0] = {
            "fold_index": 0,
            "train_indices": np.array([2, 3], dtype=np.int64),
            "test_indices": np.array([0, 1], dtype=np.int64),
            "train_groups": ["g0", "g1"],
            "test_groups": ["g0"],
        }
        result["fold_metadata"] = tuple(meta)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertFalse(out_root.exists())
        finally:
            shutil.rmtree(tmp_path)

    def test_fold_validation_rejects_metadata_groups_not_derived_from_rows(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._valid_result_with_folds(rows, labels, groups)
        meta = list(result["fold_metadata"])
        meta[0] = {
            "fold_index": 0,
            "train_indices": np.array([2, 3], dtype=np.int64),
            "test_indices": np.array([0, 1], dtype=np.int64),
            "train_groups": ["g1"],
            "test_groups": ["wrong_group"],
        }
        result["fold_metadata"] = tuple(meta)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertFalse(out_root.exists())
        finally:
            shutil.rmtree(tmp_path)

    def test_unique_staging_path_is_not_fixed_sibling(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            stale = out_root.with_name(out_root.name + ".staging")
            stale.mkdir()
            captured = {}

            def fake_rename(src, dst):
                captured["src"] = str(src)
                captured["dst"] = str(dst)
                os_rename(src, dst)

            from xh202615 import artifact_publish
            os_rename = artifact_publish.os.rename
            with patch.object(artifact_publish.os, "rename", side_effect=fake_rename):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertTrue(out_root.is_dir())
            self.assertIn("staging", captured["src"])
            self.assertNotEqual(Path(captured["src"]), stale)
            self.assertTrue(stale.exists())
        finally:
            shutil.rmtree(tmp_path)

    def test_existing_unrecognized_output_root_fails_closed(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            out_root.mkdir()
            (out_root / "stray_file.txt").write_text("not evidence", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertTrue((out_root / "stray_file.txt").exists())
            self.assertFalse(any(p.name.startswith("out.staging") for p in tmp_path.iterdir()))
        finally:
            shutil.rmtree(tmp_path)

    def test_existing_recognized_output_root_is_replaced(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            write_e0_artifacts(result, rows, groups, paths, out_root)
            old_manifest = json.loads((out_root / "e0_manifest.json").read_text(encoding="utf-8"))

            mutated = list(rows)
            mutated[0] = replace(mutated[0], audio_features={**mutated[0].audio_features, "presence_score": 9.9})
            result2 = self._evaluate(mutated, labels, groups)
            write_e0_artifacts(result2, mutated, groups, paths, out_root)
            new_manifest = json.loads((out_root / "e0_manifest.json").read_text(encoding="utf-8"))
            self.assertNotEqual(
                old_manifest["feature_state_digest"]["aggregate"],
                new_manifest["feature_state_digest"]["aggregate"],
            )
            self.assertFalse(any(p.name.startswith("out.backup") for p in tmp_path.iterdir()))
        finally:
            shutil.rmtree(tmp_path)

    def test_forced_rename_failure_restores_backup(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            write_e0_artifacts(result, rows, groups, paths, out_root)
            old_manifest_bytes = (out_root / "e0_manifest.json").read_bytes()

            mutated = list(rows)
            mutated[0] = replace(mutated[0], audio_features={**mutated[0].audio_features, "presence_score": 9.9})
            result2 = self._evaluate(mutated, labels, groups)

            from xh202615 import artifact_publish
            original_rename = artifact_publish.os.rename
            calls = {"count": 0}

            def failing_rename(src, dst):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("forced rename failure")
                return original_rename(src, dst)

            with patch.object(artifact_publish.os, "rename", side_effect=failing_rename):
                with self.assertRaises(OSError):
                    write_e0_artifacts(result2, mutated, groups, paths, out_root)

            self.assertTrue(out_root.is_dir())
            restored_bytes = (out_root / "e0_manifest.json").read_bytes()
            self.assertEqual(restored_bytes, old_manifest_bytes)
            self.assertFalse(any(p.name.startswith("out.staging") for p in tmp_path.iterdir()))
        finally:
            shutil.rmtree(tmp_path)

    def test_double_rename_failure_retains_recovery_backup(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            write_e0_artifacts(result, rows, groups, paths, out_root)
            old_manifest_bytes = (out_root / "e0_manifest.json").read_bytes()

            mutated = list(rows)
            mutated[0] = replace(mutated[0], audio_features={**mutated[0].audio_features, "presence_score": 9.9})
            result2 = self._evaluate(mutated, labels, groups)

            from xh202615 import artifact_publish
            original_rename = artifact_publish.os.rename
            calls = {"count": 0}

            def failing_rename(src, dst):
                calls["count"] += 1
                if calls["count"] == 1:
                    return original_rename(src, dst)
                raise OSError("forced rename failure")

            with patch.object(artifact_publish.os, "rename", side_effect=failing_rename):
                with self.assertRaises(RuntimeError) as ctx:
                    write_e0_artifacts(result2, mutated, groups, paths, out_root)
                self.assertIn("recovery backup", str(ctx.exception).lower())

            backups = [p for p in tmp_path.iterdir() if p.name.startswith("out.backup")]
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / "e0_manifest.json").read_bytes(), old_manifest_bytes)
        finally:
            shutil.rmtree(tmp_path)

    def test_existing_publication_lock_fails_closed_without_cleanup(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            lock = tmp_path / "out.publish.lock"
            lock.mkdir()
            owner = lock / "owner.json"
            owner.write_text('{"writer":"other"}', encoding="utf-8")

            with self.assertRaises(RuntimeError) as ctx:
                write_e0_artifacts(result, rows, groups, paths, out_root)

            self.assertIn("publication lock", str(ctx.exception).lower())
            self.assertEqual(owner.read_text(encoding="utf-8"), '{"writer":"other"}')
            self.assertFalse(out_root.exists())
            self.assertFalse(any(p.name.startswith("out.staging") for p in tmp_path.iterdir()))
        finally:
            shutil.rmtree(tmp_path)

    def test_competing_output_after_publish_failure_is_preserved_with_backup(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            write_e0_artifacts(result, rows, groups, paths, out_root)
            old_manifest_bytes = (out_root / "e0_manifest.json").read_bytes()

            mutated = list(rows)
            mutated[0] = replace(mutated[0], audio_features={**mutated[0].audio_features, "presence_score": 9.9})
            result2 = self._evaluate(mutated, labels, groups)

            from xh202615 import artifact_publish
            original_rename = artifact_publish.os.rename
            calls = {"count": 0}

            def competing_writer_rename(src, dst):
                calls["count"] += 1
                if calls["count"] == 2:
                    backup = next(p for p in tmp_path.iterdir() if p.name.startswith("out.backup"))
                    shutil.copytree(backup, out_root)
                    raise OSError("simulated competing publication")
                return original_rename(src, dst)

            with patch.object(artifact_publish.os, "rename", side_effect=competing_writer_rename):
                with self.assertRaises(RuntimeError) as ctx:
                    write_e0_artifacts(result2, mutated, groups, paths, out_root)

            self.assertIn(str(out_root), str(ctx.exception))
            backups = [p for p in tmp_path.iterdir() if p.name.startswith("out.backup")]
            self.assertEqual(len(backups), 1)
            self.assertEqual((out_root / "e0_manifest.json").read_bytes(), old_manifest_bytes)
            self.assertEqual((backups[0] / "e0_manifest.json").read_bytes(), old_manifest_bytes)
        finally:
            shutil.rmtree(tmp_path)

    def test_existing_output_with_directory_named_artifacts_is_preserved(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            out_root.mkdir()
            for name in ("e0_manifest.json", "e0_summary.json", "e0_oof_scores.jsonl",
                         "e0_frontier.jsonl", "e0_report.md"):
                (out_root / name).mkdir()
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertTrue(out_root.is_dir())
            self.assertTrue((out_root / "e0_manifest.json").is_dir())
        finally:
            shutil.rmtree(tmp_path)

    def test_existing_output_with_malformed_manifest_is_preserved(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            out_root.mkdir()
            for name in ("e0_manifest.json", "e0_summary.json", "e0_oof_scores.jsonl",
                         "e0_frontier.jsonl", "e0_report.md"):
                (out_root / name).write_text("not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertTrue(out_root.is_dir())
            self.assertEqual((out_root / "e0_manifest.json").read_text(encoding="utf-8"), "not json")
        finally:
            shutil.rmtree(tmp_path)

    def test_existing_output_with_wrong_identity_is_preserved(self):
        rows, labels, groups = self._synthetic_rows()
        result = self._evaluate(rows, labels, groups)
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._make_paths(tmp_path)
            out_root = tmp_path / "out"
            out_root.mkdir()
            for name in ("e0_manifest.json", "e0_summary.json", "e0_oof_scores.jsonl",
                         "e0_frontier.jsonl", "e0_report.md"):
                (out_root / name).write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_e0_artifacts(result, rows, groups, paths, out_root)
            self.assertTrue(out_root.is_dir())
            self.assertEqual((out_root / "e0_manifest.json").read_text(encoding="utf-8"), "{}")
        finally:
            shutil.rmtree(tmp_path)


class GateOracleCLITests(unittest.TestCase):
    def test_help_exits_cleanly(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_missing_input_file_fails_before_writing_output(self):
        tmp_path = Path(tempfile.mkdtemp())
        try:
            out_root = tmp_path / "out"
            with self.assertRaises((FileNotFoundError, ValueError)):
                main([
                    "--dataset-root", str(tmp_path / "missing"),
                    "--candidate-fusion", str(tmp_path / "missing.jsonl"),
                    "--tse-asr", str(tmp_path / "missing.jsonl"),
                    "--audio-map", str(tmp_path / "missing.jsonl"),
                    "--r3-predictions", str(tmp_path / "missing.jsonl"),
                    "--group-manifest", str(tmp_path / "missing.jsonl"),
                    "--output-root", str(out_root),
                    "--n-outer", "2",
                    "--n-boot", "4",
                ])
            self.assertFalse(out_root.exists())
        finally:
            import shutil
            shutil.rmtree(tmp_path)

    def _write_synthetic_bundle(self, tmp_path: Path, pos: list[dict], neg: list[dict]):
        dataset_root = tmp_path / "datasetA"
        dataset_root.mkdir()
        (dataset_root / "pos.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pos), encoding="utf-8"
        )
        (dataset_root / "neg.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in neg), encoding="utf-8"
        )

        def write_jsonl(path: Path, records: list[dict]):
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                encoding="utf-8",
            )

        all_ids = sorted({str(r["id"]) for r in pos + neg}, key=lambda x: int(x) if x.isdigit() else x)
        labels = {"0": "甲", "1": None, "2": "乙", "3": None}
        fusion = [
            {"id": sid, "recognition_text": labels[sid] or "", "candidate_texts": {"primary": labels[sid] or "", "energy": ""}}
            for sid in all_ids
        ]
        tse = [{"id": rec["id"], "text": rec["recognition_text"]} for rec in fusion]
        audio = [
            {"id": sid, "presence_score": 0.9 if labels[sid] else 0.1, "enhanced_cosine": 0.8,
             "mixture_cosine": 0.7, "max_cosine": 0.8, "latency_ms": 100.0}
            for sid in all_ids
        ]
        r3 = [{"id": rec["id"], "recognition_text": rec["recognition_text"]} for rec in fusion]
        manifest = {
            "rows": [
                {"id": "0", "split": "pos", "label": "甲", "wake_component": "g0"},
                {"id": "1", "split": "neg", "label": None, "wake_component": "g0"},
                {"id": "2", "split": "pos", "label": "乙", "wake_component": "g1"},
                {"id": "3", "split": "neg", "label": None, "wake_component": "g1"},
            ]
        }
        candidate_fusion = tmp_path / "fusion.jsonl"
        tse_asr = tmp_path / "tse.jsonl"
        audio_map = tmp_path / "audio.jsonl"
        r3_predictions = tmp_path / "r3.jsonl"
        group_manifest = tmp_path / "manifest.json"
        write_jsonl(candidate_fusion, fusion)
        write_jsonl(tse_asr, tse)
        write_jsonl(audio_map, audio)
        write_jsonl(r3_predictions, r3)
        group_manifest.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return {
            "dataset_root": dataset_root,
            "candidate_fusion": candidate_fusion,
            "tse_asr": tse_asr,
            "audio_map": audio_map,
            "r3_predictions": r3_predictions,
            "group_manifest": group_manifest,
        }

    def test_end_to_end_runs_on_synthetic_inputs(self):
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._write_synthetic_bundle(
                tmp_path,
                pos=[{"id": "0", "wakeup_audio": "a.wav", "command_audio": "b.wav", "label": "甲"},
                     {"id": "2", "wakeup_audio": "a.wav", "command_audio": "b.wav", "label": "乙"}],
                neg=[{"id": "1", "wakeup_audio": "a.wav", "command_audio": "b.wav"},
                     {"id": "3", "wakeup_audio": "a.wav", "command_audio": "b.wav"}],
            )
            out_root = tmp_path / "out"
            rc = main([
                "--dataset-root", str(paths["dataset_root"]),
                "--candidate-fusion", str(paths["candidate_fusion"]),
                "--tse-asr", str(paths["tse_asr"]),
                "--audio-map", str(paths["audio_map"]),
                "--r3-predictions", str(paths["r3_predictions"]),
                "--group-manifest", str(paths["group_manifest"]),
                "--output-root", str(out_root),
                "--n-outer", "2",
                "--n-boot", "4",
                "--seed", "1",
                "--rr-floor", "0.5",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out_root.is_dir())
            self.assertEqual(
                {p.name for p in out_root.iterdir()},
                {"e0_manifest.json", "e0_summary.json", "e0_oof_scores.jsonl",
                 "e0_frontier.jsonl", "e0_report.md"},
            )
        finally:
            import shutil
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_duplicate_within_split_rejects_and_leaves_no_output(self):
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._write_synthetic_bundle(
                tmp_path,
                pos=[{"id": "0", "wakeup_audio": "a.wav", "command_audio": "b.wav", "label": "甲"},
                     {"id": "0", "wakeup_audio": "a.wav", "command_audio": "b.wav", "label": "甲"}],
                neg=[{"id": "1", "wakeup_audio": "a.wav", "command_audio": "b.wav"}],
            )
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                main([
                    "--dataset-root", str(paths["dataset_root"]),
                    "--candidate-fusion", str(paths["candidate_fusion"]),
                    "--tse-asr", str(paths["tse_asr"]),
                    "--audio-map", str(paths["audio_map"]),
                    "--r3-predictions", str(paths["r3_predictions"]),
                    "--group-manifest", str(paths["group_manifest"]),
                    "--output-root", str(out_root),
                    "--n-outer", "2",
                    "--n-boot", "4",
                ])
            self.assertFalse(out_root.exists())
        finally:
            import shutil
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_duplicate_across_splits_rejects_and_leaves_no_output(self):
        tmp_path = Path(tempfile.mkdtemp())
        try:
            paths = self._write_synthetic_bundle(
                tmp_path,
                pos=[{"id": "0", "wakeup_audio": "a.wav", "command_audio": "b.wav", "label": "甲"},
                     {"id": "1", "wakeup_audio": "a.wav", "command_audio": "b.wav", "label": "甲"}],
                neg=[{"id": "1", "wakeup_audio": "a.wav", "command_audio": "b.wav"}],
            )
            out_root = tmp_path / "out"
            with self.assertRaises(ValueError):
                main([
                    "--dataset-root", str(paths["dataset_root"]),
                    "--candidate-fusion", str(paths["candidate_fusion"]),
                    "--tse-asr", str(paths["tse_asr"]),
                    "--audio-map", str(paths["audio_map"]),
                    "--r3-predictions", str(paths["r3_predictions"]),
                    "--group-manifest", str(paths["group_manifest"]),
                    "--output-root", str(out_root),
                    "--n-outer", "2",
                    "--n-boot", "4",
                ])
            self.assertFalse(out_root.exists())
        finally:
            import shutil
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

