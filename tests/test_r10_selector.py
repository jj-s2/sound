"""Tests for R10 multi-candidate grouped OOF selector.

Task 1: candidate join, oracle audit, and evaluator equivalence.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xh202615.data import Sample
from xh202615.evaluation import evaluate_rows
from xh202615.text import normalize_text
from xh202615.r10_selector import (
    ACTION_ORDER,
    CandidateRow,
    FEATURE_SCHEMA,
    FoldArtifacts,
    _action_to_index,
    _actions_to_predictions,
    _build_folds,
    _fit_bigram_index,
    _group_to_folds,
    _vocab_distance_bigrams_fast,
    _optimal_action,
    _predict_actions_from_stacked,
    _row_feature_vectors,
    _train_model,
    bootstrap_grouped_ci,
    build_all_action_features,
    build_inference_features,
    compute_oracle_metrics,
    fit_fold_artifacts,
    load_candidate_bundle,
    run_grouped_nested_oof,
    score_action_policy,
)


def _samples_from_labels(labels: dict[str, str | None]) -> list[Sample]:
    return [
        Sample(
            id=sid,
            split="pos" if label is not None else "neg",
            wakeup_audio=Path("."),
            wakeup_text="",
            command_audio=Path("."),
            label=label,
        )
        for sid, label in labels.items()
    ]


class CandidateBundleTests(unittest.TestCase):
    def _write(self, handle, records: list[dict]) -> None:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        handle.flush()

    def _bundle(
        self,
        fusion: list[dict],
        tse: list[dict],
        audio_map: list[dict],
        manifest: dict,
        r3_pred: list[dict] | None = None,
    ) -> tuple[dict[str, CandidateRow], dict[str, str], dict[str, str | None]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fusion_path = tmp / "fusion.jsonl"
            tse_path = tmp / "tse.jsonl"
            audio_path = tmp / "audio.jsonl"
            manifest_path = tmp / "manifest.jsonl"
            for path, recs in [(fusion_path, fusion), (tse_path, tse), (audio_path, audio_map)]:
                with path.open("w", encoding="utf-8") as handle:
                    self._write(handle, recs)
            with manifest_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            r3_path = None
            if r3_pred is not None:
                r3_path = tmp / "r3.jsonl"
                with r3_path.open("w", encoding="utf-8") as handle:
                    self._write(handle, r3_pred)
            return load_candidate_bundle(
                fusion_path,
                tse_path,
                audio_path,
                manifest_path,
                r3_predictions_path=r3_path,
            )

    def test_loads_all_candidates(self):
        labels = {"0": "打开空调", "1": None}
        manifest = {
            "rows": [
                {"id": "0", "split": "pos", "label": "打开空调", "wake_component": "g0"},
                {"id": "1", "split": "neg", "label": None, "wake_component": "g1"},
            ]
        }
        fusion = [
            {
                "id": "0",
                "split": "pos",
                "recognition_text": "打开空调",
                "candidate_texts": {"primary": "打开空调", "energy": "开空调"},
            },
            {
                "id": "1",
                "split": "neg",
                "recognition_text": "",
                "candidate_texts": {"primary": "", "energy": ""},
            },
        ]
        tse = [
            {"id": "0", "text": "打开空调了"},
            {"id": "1", "text": ""},
        ]
        audio_map = [
            {"id": "0", "presence_score": 0.5, "enhanced_cosine": 0.6, "latency_ms": 100.0},
            {"id": "1", "presence_score": 0.1, "enhanced_cosine": 0.2, "latency_ms": 100.0},
        ]
        rows, groups, loaded_labels = self._bundle(fusion, tse, audio_map, manifest)
        self.assertEqual(sorted(rows), ["0", "1"])
        self.assertEqual(rows["0"].r3_text, "打开空调")
        self.assertEqual(rows["0"].primary_text, "打开空调")
        self.assertEqual(rows["0"].energy_text, "开空调")
        self.assertEqual(rows["0"].tse_text, "打开空调了")
        self.assertEqual(loaded_labels, labels)

    def test_duplicate_ids_raise(self):
        manifest = {"rows": [{"id": "0", "split": "pos", "wake_component": "g0"}]}
        fusion = [
            {"id": "0", "recognition_text": "a", "candidate_texts": {"primary": "a"}},
            {"id": "0", "recognition_text": "b", "candidate_texts": {"primary": "b"}},
        ]
        tse = [{"id": "0", "text": "a"}]
        audio_map = [{"id": "0", "presence_score": 0.5}]
        with self.assertRaises(ValueError):
            self._bundle(fusion, tse, audio_map, manifest)

    def test_missing_ids_raise(self):
        manifest = {"rows": [{"id": "0", "split": "pos", "wake_component": "g0"}]}
        fusion = [{"id": "0", "recognition_text": "a", "candidate_texts": {"primary": "a"}}]
        tse = []  # missing TSE for id 0
        audio_map = [{"id": "0", "presence_score": 0.5}]
        with self.assertRaises(ValueError):
            self._bundle(fusion, tse, audio_map, manifest)

    def test_deduplication_retains_source_identity(self):
        manifest = {"rows": [{"id": "0", "split": "pos", "wake_component": "g0"}]}
        fusion = [
            {
                "id": "0",
                "recognition_text": "打开空调",
                "candidate_texts": {"primary": "打开空调", "energy": "打开空调"},
            }
        ]
        tse = [{"id": "0", "text": "打开空调"}]
        audio_map = [{"id": "0", "presence_score": 0.5}]
        rows, _, _ = self._bundle(fusion, tse, audio_map, manifest)
        diag = rows["0"].dedup_sources
        all_sources = [src for sources in diag.values() for src in sources]
        self.assertIn("primary", all_sources)
        self.assertIn("energy", all_sources)

    def test_r3_gated_predictions_override(self):
        manifest = {"rows": [{"id": "0", "split": "neg", "wake_component": "g0"}]}
        fusion = [
            {
                "id": "0",
                "recognition_text": "噪音",
                "candidate_texts": {"primary": "噪音", "energy": ""},
            }
        ]
        r3_pred = [{"id": "0", "recognition_text": ""}]
        tse = [{"id": "0", "text": "噪音"}]
        audio_map = [{"id": "0", "presence_score": 0.1}]
        rows, _, _ = self._bundle(fusion, tse, audio_map, manifest, r3_pred=r3_pred)
        self.assertEqual(rows["0"].r3_text, "")


class FeatureLeakageTests(unittest.TestCase):
    def _row(self, label: str | None = None) -> CandidateRow:
        return CandidateRow(
            id="x",
            split="pos" if label is not None else "neg",
            label=label,
            r3_text="打开空调",
            primary_text="开空调",
            energy_text="",
            tse_text="打开空调",
            audio_features={
                "presence_score": 0.5,
                "enhanced_cosine": 0.6,
                "mixture_cosine": 0.55,
                "max_cosine": 0.6,
                "latency_ms": 100.0,
            },
            original_command_audio=None,
            source_digest="",
            dedup_sources={},
        )

    def test_schema_excludes_label_derived_names(self):
        for name in FEATURE_SCHEMA:
            lower = name.lower()
            self.assertNotIn("label", lower, name)
            self.assertNotIn("ref_text", lower, name)
            self.assertNotIn("split", lower, name)
            self.assertNotIn("target", lower, name)

    def test_features_finite_after_imputation(self):
        rows = [self._row("打开空调")]
        artifacts = fit_fold_artifacts(["x"], {r.id: r for r in rows})
        X = build_inference_features(rows, artifacts, "r3")
        self.assertTrue(np.isfinite(X).all())
        self.assertEqual(X.shape[0], 1)
        self.assertEqual(X.shape[1], len(FEATURE_SCHEMA))

    def _make_row(self, sid: str, label: str | None, r3: str, primary: str, energy: str, tse: str) -> CandidateRow:
        return CandidateRow(
            id=sid,
            split="pos" if label is not None else "neg",
            label=label,
            r3_text=r3,
            primary_text=primary,
            energy_text=energy,
            tse_text=tse,
            audio_features={
                "presence_score": 0.5,
                "enhanced_cosine": 0.6,
                "mixture_cosine": 0.55,
                "max_cosine": 0.6,
                "latency_ms": 100.0,
            },
            original_command_audio=None,
            source_digest="",
            dedup_sources={},
        )

    def test_fold_vocab_excludes_outer_test_labels(self):
        train_rows = {
            "a": self._make_row("a", "打开空调", "打开空调", "开空调", "", "打开空调"),
            "b": self._make_row("b", "关闭电视", "关闭电视", "关电视", "", "关闭电视"),
        }
        artifacts = fit_fold_artifacts(["a", "b"], train_rows)
        self.assertNotIn("播放音乐", artifacts.vocab)
        self.assertIn("打开空调", artifacts.vocab)

    def test_feature_vector_independent_of_test_label(self):
        # A row's feature vector must be identical even if its label changes,
        # because labels are not inference features.
        row_pos = self._make_row("c", "打开空调", "打开空调", "开空调", "", "打开空调")
        row_neg = self._make_row("c", None, "打开空调", "开空调", "", "打开空调")
        train_rows = {"a": self._make_row("a", "dummy", "dummy", "dummy", "", "dummy")}
        artifacts = fit_fold_artifacts(["a"], train_rows)
        X_pos = build_inference_features([row_pos], artifacts, "r3")
        X_neg = build_inference_features([row_neg], artifacts, "r3")
        np.testing.assert_array_equal(X_pos, X_neg)

    def test_vectorizer_fitted_from_train_vocab_only(self):
        # The fold-local bigram index must be fitted only on training labels.
        # Adding test-only vocabulary must not alter training or validation
        # feature vectors, and unseen test bigrams must not change the index.
        train_row = self._make_row("a", "打开空调", "打开空调", "开空调", "", "打开空调")
        val_row = self._make_row("b", "关闭电视", "关闭电视", "关电视", "", "关闭电视")
        test_only_row = self._make_row("c", None, "播放音乐", "播放音乐", "", "播放音乐")

        rows = {"a": train_row, "b": val_row}
        artifacts = fit_fold_artifacts(["a"], rows)

        X_train_before = build_inference_features([train_row], artifacts, "r3")
        X_val_before = build_inference_features([val_row], artifacts, "r3")

        # Re-fit with a test-only row present in the mapping but not in train_ids.
        rows_with_test = {"a": train_row, "b": val_row, "c": test_only_row}
        artifacts_after = fit_fold_artifacts(["a"], rows_with_test)

        X_train_after = build_inference_features([train_row], artifacts_after, "r3")
        X_val_after = build_inference_features([val_row], artifacts_after, "r3")

        np.testing.assert_array_equal(X_train_before, X_train_after)
        np.testing.assert_array_equal(X_val_before, X_val_after)

        # The vectorizer vocabulary should contain only bigrams from the train label.
        train_bigrams = set()
        norm_train = normalize_text(train_row.label)
        for i in range(len(norm_train) - 1):
            train_bigrams.add(norm_train[i : i + 2])
        self.assertTrue(set(artifacts.vectorizer.vocabulary_).issubset(train_bigrams))

    def test_empty_vocab_returns_null_index_and_max_distance(self):
        vectorizer, matrix, norms = _fit_bigram_index([])
        self.assertIsNone(vectorizer)
        self.assertIsNone(matrix)
        self.assertEqual(norms.size, 0)

        # With a null index the raw distance output is deterministic.
        min_d, margin = _vocab_distance_bigrams_fast(
            ["abc", "ab", "a"], None, None, np.zeros(0, dtype=np.int64)
        )
        np.testing.assert_array_equal(min_d, np.ones(3))
        np.testing.assert_array_equal(margin, np.zeros(3))

        row = self._make_row("x", None, "abc", "ab", "", "a")
        artifacts = fit_fold_artifacts(["x"], {"x": row})
        self.assertIsNone(artifacts.vectorizer)
        # Standardized value is not informative here; just assert finite and no crash.
        X = build_inference_features([row], artifacts, "r3")
        self.assertTrue(np.isfinite(X).all())

    def test_single_char_vocab_returns_null_index_and_max_distance(self):
        vectorizer, matrix, norms = _fit_bigram_index(["a", "b", "c"])
        self.assertIsNone(vectorizer)
        self.assertIsNone(matrix)
        self.assertEqual(norms.size, 0)

        min_d, margin = _vocab_distance_bigrams_fast(
            ["abc", "ab", "a"], None, None, np.zeros(0, dtype=np.int64)
        )
        np.testing.assert_array_equal(min_d, np.ones(3))
        np.testing.assert_array_equal(margin, np.zeros(3))

        row = self._make_row("x", "a", "abc", "ab", "", "a")
        artifacts = fit_fold_artifacts(["x"], {"x": row})
        self.assertIsNone(artifacts.vectorizer)
        X = build_inference_features([row], artifacts, "r3")
        self.assertTrue(np.isfinite(X).all())


class GroupedOOFTests(unittest.TestCase):
    def _make_row(self, sid: str, label: str | None, r3: str, primary: str, energy: str, tse: str, **audio) -> CandidateRow:
        defaults = {
            "presence_score": 0.5,
            "enhanced_cosine": 0.5,
            "mixture_cosine": 0.5,
            "max_cosine": 0.5,
            "latency_ms": 100.0,
        }
        defaults.update(audio)
        return CandidateRow(
            id=sid,
            split="pos" if label is not None else "neg",
            label=label,
            r3_text=r3,
            primary_text=primary,
            energy_text=energy,
            tse_text=tse,
            audio_features=defaults,
            original_command_audio=None,
            source_digest="",
            dedup_sources={},
        )

    def test_group_folds_keep_components_intact(self):
        groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g3", "e": "g3", "f": "g4"}
        folds = _group_to_folds(list(groups), groups, n_folds=3, seed=1)
        all_assigned = [sid for fold in folds for sid in fold]
        self.assertEqual(sorted(all_assigned), sorted(groups))
        for members in [["a", "b"], ["c"], ["d", "e"], ["f"]]:
            count = sum(sum(1 for sid in fold if sid in members) for fold in folds)
            self.assertEqual(count, len(members))

    def test_nested_folds_disjoint(self):
        groups = {str(i): f"g{i // 2}" for i in range(20)}
        folds = _build_folds(list(groups), groups, n_outer=5, n_inner=2, seed=7)
        self.assertEqual(len(folds), 5)
        for fold in folds:
            self.assertEqual(len(set(fold["test"]) & set(fold["train"])), 0)
            inner_all = [sid for inner in fold["inner_folds"] for sid in inner]
            self.assertEqual(sorted(inner_all), sorted(fold["train"]))

    def test_optimal_action_prefers_reject_for_negatives(self):
        row = self._make_row("x", None, "打开空调", "开空调", "", "打开空调")
        self.assertEqual(_optimal_action(row), "reject")

    def test_optimal_action_prefers_lowest_cer_positive(self):
        row = self._make_row("x", "打开空调", "开空调", "打开空调", "", "关空调")
        self.assertEqual(_optimal_action(row), "primary")

    def test_run_grouped_nested_oof_coverage_and_equivalence(self):
        sample_ids = [str(i) for i in range(30)]
        labels = {sid: ("打开空调" if int(sid) % 3 == 0 else "关闭电视") if int(sid) < 25 else None for sid in sample_ids}
        rows = {
            sid: self._make_row(
                sid,
                labels[sid],
                r3=labels[sid] or "",
                primary=labels[sid] or "",
                energy="",
                tse=labels[sid] or "",
                presence_score=0.7 if labels[sid] is not None else 0.3,
            )
            for sid in sample_ids
        }
        groups = {sid: f"g{int(sid) % 6}" for sid in sample_ids}
        result = run_grouped_nested_oof(
            rows,
            labels,
            groups,
            n_outer=3,
            n_inner=2,
            C_values=(1.0,),
            tau_values=(0.5,),
            seed=42,
        )
        oof_ids = [p["id"] for p in result["oof_predictions"]]
        self.assertEqual(sorted(oof_ids), sorted(sample_ids))
        # Pooled metrics agree with official evaluator.
        from xh202615.evaluation import evaluate_rows
        samples = [Sample(id=r.id, split=r.split, wakeup_audio=Path("."), wakeup_text="", command_audio=Path("."), label=r.label) for r in rows.values()]
        preds = [{"id": p["id"], "recognition_text": p["recognition_text"]} for p in result["oof_predictions"]]
        official = evaluate_rows(samples, preds, missing_policy="empty").metrics
        pooled = result["pooled_metrics"]
        for key in ("avg_cer", "avg_rr", "substitutions", "insertions", "deletions", "ref_chars", "false_reject_rate", "false_accept_rate"):
            self.assertAlmostEqual(pooled[key], official[key], places=9, msg=key)

    def test_r3_fallback_on_empty_predicted_candidate(self):
        row = self._make_row("x", "打开空调", "打开空调", "", "", "")
        # Force primary (empty) action; fallback should yield r3 text.
        preds = _actions_to_predictions([row], [_action_to_index("primary")], fallback_to_r3=True)
        self.assertEqual(preds[0]["recognition_text"], "打开空调")

    def test_infeasible_inner_cv_falls_back_to_exact_r3(self):
        # All-positive data: no negatives exist, so no learned policy can meet
        # the RR >= 0.95 floor. Every outer fold must fall back to exact R3.
        sample_ids = [str(i) for i in range(30)]
        labels = {sid: "打开空调" for sid in sample_ids}
        rows = {
            sid: self._make_row(sid, labels[sid], r3="打开空调", primary="打开空调", energy="打开空调", tse="打开空调")
            for sid in sample_ids
        }
        groups = {sid: f"g{int(sid) % 6}" for sid in sample_ids}
        result = run_grouped_nested_oof(
            rows,
            labels,
            groups,
            n_outer=3,
            n_inner=2,
            C_values=(1.0,),
            tau_values=(0.5,),
            seed=42,
        )
        self.assertEqual(result["n_infeasible_folds"], 3)
        for f in result["fold_reports"]:
            self.assertEqual(f["fallback"], "r3_no_feasible_inner_policy")
            self.assertIsNone(f["selected_C"])
            self.assertIsNone(f["selected_tau"])
            self.assertTrue(f["group_disjoint"])
        for p in result["oof_predictions"]:
            self.assertEqual(p["recognition_text"], "打开空调")
        self.assertEqual(result["pooled_metrics"]["avg_cer"], 0.0)
        self.assertEqual(result["pooled_metrics"]["avg_rr"], 0.0)

    def test_group_bootstrap_samples_groups_with_replacement(self):
        # Singleton groups with heterogeneous contributions. Row-level
        # resampling would keep the distribution nearly fixed; group-level
        # resampling with replacement must produce a non-degenerate CI.
        labels = {str(i): ("打开空调" if i % 2 == 0 else None) for i in range(10)}
        preds = ["打", "", "打开", "", "打开空", "", "打开空调", "", "空调打开", ""]
        oof = [
            {"id": str(i), "recognition_text": preds[i], "outer_fold": 0}
            for i in range(10)
        ]
        groups = {str(i): f"g{i}" for i in range(10)}
        ci = bootstrap_grouped_ci(oof, labels, groups, n_boot=500, seed=1)
        self.assertEqual(ci["n_groups"], 10)
        self.assertLess(ci["ci_low"], ci["ci_high"])
        self.assertGreater(ci["ci_high"] - ci["ci_low"], 1e-6)


class OracleAndScorerTests(unittest.TestCase):
    def test_oracle_rejects_all_negatives(self):
        labels = {"p0": "打开空调", "n0": None}
        rows = [
            CandidateRow(
                id="p0",
                split="pos",
                label="打开空调",
                r3_text="开空调",
                primary_text="打开空调",
                energy_text="",
                tse_text="关空调",
                audio_features={},
                original_command_audio=None,
                source_digest="",
                dedup_sources={},
            ),
            CandidateRow(
                id="n0",
                split="neg",
                label=None,
                r3_text="",
                primary_text="",
                energy_text="",
                tse_text="",
                audio_features={},
                original_command_audio=None,
                source_digest="",
                dedup_sources={},
            ),
        ]
        metrics = compute_oracle_metrics(rows, labels, ["r3", "primary", "energy", "tse"])
        self.assertEqual(metrics["avg_rr"], 1.0)
        self.assertEqual(metrics["avg_cer"], 0.0)
        self.assertEqual(metrics["pos_count"], 1)

    def test_score_action_policy_matches_official_evaluator(self):
        labels = {"p0": "打开空调", "p1": "关闭电视", "n0": None}
        rows = [
            CandidateRow(
                id="p0",
                split="pos",
                label="打开空调",
                r3_text="打开空调",
                primary_text="开空调",
                energy_text="",
                tse_text="",
                audio_features={},
                original_command_audio=None,
                source_digest="",
                dedup_sources={},
            ),
            CandidateRow(
                id="p1",
                split="pos",
                label="关闭电视",
                r3_text="关电视",
                primary_text="关闭电视",
                energy_text="",
                tse_text="",
                audio_features={},
                original_command_audio=None,
                source_digest="",
                dedup_sources={},
            ),
            CandidateRow(
                id="n0",
                split="neg",
                label=None,
                r3_text="",
                primary_text="",
                energy_text="",
                tse_text="",
                audio_features={},
                original_command_audio=None,
                source_digest="",
                dedup_sources={},
            ),
        ]
        metrics = score_action_policy(rows, labels, lambda row: "r3")
        samples = _samples_from_labels(labels)
        preds = [{"id": row.id, "recognition_text": row.r3_text} for row in rows]
        official = evaluate_rows(samples, preds, missing_policy="empty").metrics
        for key in ("avg_cer", "avg_rr", "substitutions", "insertions", "deletions", "ref_chars", "false_reject_rate", "false_accept_rate"):
            self.assertAlmostEqual(metrics[key], official[key], places=9, msg=key)


if __name__ == "__main__":
    unittest.main()
