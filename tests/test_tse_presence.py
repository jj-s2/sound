"""Tests for the presence-gated ASR composition evaluator (R6).

Pure-Python tests (no WeSpeaker, no Dataset-A) cover presence loading
fail-closed behaviour, gating semantics, agreement with the official
``evaluate_rows`` scorer, threshold calibration, and the public-manifest sample
builder. The final class writes a minimal torch checkpoint to regression-test
``--threshold-from-checkpoint`` (a binary ``.pt`` file, not JSON).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from xh202615.data import Sample
from xh202615.evaluation import evaluate_rows
from scripts.evaluate_tse_presence import _resolve_threshold, _resolve_score_field, parse_args, main
from xh202615.tse_presence import (
    REJECT_TEXT,
    SCORE_FIELDS,
    calibrate_threshold_overall,
    gate_predictions,
    load_all_score_fields,
    load_asr_text,
    load_presence,
    load_scores,
    overall_at_threshold,
    overall_from_metrics,
    samples_from_manifest,
)


def _sample(sid: str, label: str | None, split: str = "val") -> Sample:
    return Sample(
        id=sid,
        split=split,
        wakeup_audio=Path("."),
        wakeup_text="",
        command_audio=Path("."),
        label=label,
    )


def _perfect_split() -> tuple[list[Sample], dict[str, str], dict[str, float]]:
    """Positives with correct ASR at high presence; negatives at low presence."""
    samples = [
        _sample("p0", "你好世界"),
        _sample("p1", "打开空调"),
        _sample("n0", None),
        _sample("n1", None),
    ]
    asr = {"p0": "你好世界", "p1": "打开空调", "n0": "干扰说话", "n1": "噪音"}
    presence = {"p0": 0.9, "p1": 0.85, "n0": 0.1, "n1": 0.15}
    return samples, asr, presence


class LoadPresenceTests(unittest.TestCase):
    def _write(self, tmp: Path, rows: list[dict]) -> Path:
        path = tmp / "pres.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_loads_id_to_score(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), [{"id": "a", "presence_score": 0.3},
                                          {"id": "b", "presence_score": 0.7}])
            self.assertEqual(load_presence(path), {"a": 0.3, "b": 0.7})

    def test_missing_score_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), [{"id": "a"}])
            with self.assertRaisesRegex(ValueError, "presence_score"):
                load_presence(path)

    def test_none_score_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), [{"id": "a", "presence_score": None}])
            with self.assertRaisesRegex(ValueError, "presence_score"):
                load_presence(path)

    def test_non_finite_score_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), [{"id": "a", "presence_score": float("nan")}])
            with self.assertRaisesRegex(ValueError, "finite"):
                load_presence(path)

    def test_duplicate_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), [{"id": "a", "presence_score": 0.1},
                                          {"id": "a", "presence_score": 0.2}])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_presence(path)

    def test_empty_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no records"):
                load_presence(path)


class LoadAsrTextTests(unittest.TestCase):
    def test_text_alias_and_missing_become_empty(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "asr.jsonl"
            path.write_text(
                "\n".join([
                    json.dumps({"id": "a", "text": "你好"}),
                    json.dumps({"id": "b", "recognition_text": "世界"}),
                    json.dumps({"id": "c"}),
                ])
                + "\n",
                encoding="utf-8",
            )
            asr = load_asr_text(path)
            self.assertEqual(asr, {"a": "你好", "b": "世界", "c": ""})


class GatePredictionsTests(unittest.TestCase):
    def test_accept_keeps_text_reject_empties(self):
        asr = {"p0": "你好", "n0": "噪音"}
        presence = {"p0": 0.9, "n0": 0.1}
        preds = {r["id"]: r["recognition_text"] for r in gate_predictions(asr, presence, 0.5)}
        self.assertEqual(preds, {"p0": "你好", "n0": REJECT_TEXT})

    def test_missing_presence_raises(self):
        with self.assertRaisesRegex(ValueError, "missing presence"):
            gate_predictions({"p0": "x"}, {}, 0.5)

    def test_boundary_threshold_accepts_equal(self):
        # score == threshold is accepted (>=).
        preds = {r["id"]: r["recognition_text"]
                 for r in gate_predictions({"a": "x"}, {"a": 0.5}, 0.5)}
        self.assertEqual(preds, {"a": "x"})


class OverallAtThresholdTests(unittest.TestCase):
    def test_perfect_separation_yields_perfect_overall(self):
        samples, asr, presence = _perfect_split()
        m = overall_at_threshold(samples, asr, presence, 0.5)
        self.assertAlmostEqual(m["avg_cer"], 0.0, places=6)
        self.assertEqual(m["avg_rr"], 1.0)
        self.assertAlmostEqual(m["overall"], 1.0, places=6)
        self.assertEqual(m["pos_count"], 2)
        self.assertEqual(m["neg_count"], 2)
        self.assertEqual(m["false_accept_rate"], 0.0)
        self.assertEqual(m["false_reject_rate"], 0.0)

    def test_reject_all_collapses_cer_keeps_rr(self):
        samples, asr, presence = _perfect_split()
        m = overall_at_threshold(samples, asr, presence, 1.5)
        # All positives rejected -> full deletion -> CER 1.0; all negatives rejected -> RR 1.0.
        self.assertAlmostEqual(m["avg_cer"], 1.0, places=6)
        self.assertEqual(m["avg_rr"], 1.0)
        self.assertAlmostEqual(m["overall"], 0.5, places=6)
        self.assertEqual(m["false_reject_rate"], 1.0)

    def test_accept_all_rr_zero_when_all_negatives_transcribed(self):
        samples, asr, presence = _perfect_split()
        m = overall_at_threshold(samples, asr, presence, 0.0)
        # No negatives rejected by the gate (their ASR text is non-empty) -> RR 0.
        self.assertEqual(m["avg_rr"], 0.0)
        self.assertEqual(m["false_accept_rate"], 1.0)
        self.assertAlmostEqual(m["avg_cer"], 0.0, places=6)

    def test_agrees_with_official_evaluator(self):
        samples, asr, presence = _perfect_split()
        for thr in (0.0, 0.2, 0.5, 0.8, 1.5):
            m = overall_at_threshold(samples, asr, presence, thr)
            preds = gate_predictions(asr, presence, thr)
            report = evaluate_rows(samples, preds, missing_policy="empty")
            official = dict(report.metrics)
            for key in ("avg_cer", "avg_rr", "pos_count", "neg_count",
                        "false_reject_rate", "false_accept_rate"):
                self.assertAlmostEqual(m[key], official[key], places=6,
                                       msg=f"mismatch at thr={thr} key={key}")
            self.assertAlmostEqual(m["overall"], ((1 - official["avg_cer"]) + official["avg_rr"]) / 2,
                                   places=6)


class CalibrateThresholdTests(unittest.TestCase):
    def test_picks_threshold_maximising_overall(self):
        samples, asr, presence = _perfect_split()
        cal = calibrate_threshold_overall(samples, asr, presence)
        self.assertEqual(cal["threshold_source"], "public_val_max_overall")
        # Perfect separation -> a threshold in (0.15, 0.85] gives Overall 1.0.
        self.assertAlmostEqual(cal["metrics"]["overall"], 1.0, places=6)
        self.assertGreater(cal["threshold"], 0.15)
        self.assertLessEqual(cal["threshold"], 0.85)

    def test_single_class_fails_closed(self):
        samples = [_sample("p0", "你好"), _sample("p1", "世界")]
        asr = {"p0": "你好", "p1": "世界"}
        presence = {"p0": 0.9, "p1": 0.8}
        with self.assertRaisesRegex(ValueError, "both classes"):
            calibrate_threshold_overall(samples, asr, presence)

    def test_missing_presence_fails_closed(self):
        samples, asr, _ = _perfect_split()
        with self.assertRaisesRegex(ValueError, "missing presence"):
            calibrate_threshold_overall(samples, asr, {"p0": 0.9})

    def test_tie_break_prefers_higher_rr(self):
        # Two thresholds give identical Overall but one rejects more negatives.
        samples = [_sample("p0", "你好"), _sample("n0", None)]
        asr = {"p0": "错", "n0": "噪音"}  # positive always wrong (CER 1.0)
        presence = {"p0": 0.9, "n0": 0.2}
        cal = calibrate_threshold_overall(samples, asr, presence)
        # Reject-all (thr>0.9): CER=1.0, RR=1.0 -> Overall=0.5.
        # Accept-pos (thr in (0.2,0.9]): CER=1.0, RR=1.0 -> Overall=0.5 (same).
        # Accept-all (thr<=0.2): CER=1.0, RR=0.0 -> Overall=0.0.
        # Tie at 0.5 broken toward higher RR -> reject-all or accept-pos (both RR=1).
        self.assertEqual(cal["metrics"]["avg_rr"], 1.0)


class SamplesFromManifestTests(unittest.TestCase):
    def test_builds_pos_and_neg_samples(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            base = tmp / "audio"
            for split, present, text in [("val", True, "你好"), ("val", False, None)]:
                rid = f"{split}-{'pos' if present else 'neg'}"
                for fld in ("enrollment", "mixture", "target"):
                    p = base / fld / f"{rid}.wav"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b"")
                row = {
                    "row_id": rid, "split": split, "source": "t",
                    "enrollment_audio": str(base / "enrollment" / f"{rid}.wav"),
                    "target_audio": str(base / "target" / f"{rid}.wav"),
                    "mixture_audio": str(base / "mixture" / f"{rid}.wav"),
                    "target_speaker_id": "S1", "interferer_speaker_id": "I1" if present else None,
                    "target_present": present, "overlap_ratio": 0.5 if present else 0.0,
                    "snr_db": 5.0 if present else None, "sir_db": 0.0 if present else None,
                    "text": text, "seed": 1,
                }
                with (tmp / "manifest.jsonl").open("a", encoding="utf-8") as h:
                    h.write(json.dumps(row, ensure_ascii=False) + "\n")
            samples = samples_from_manifest(tmp / "manifest.jsonl", "val")
            self.assertEqual(len(samples), 2)
            labels = {s.id: s.label for s in samples}
            self.assertEqual(labels["val-pos"], "你好")
            self.assertIsNone(labels["val-neg"])


class OverallFromMetricsTests(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(overall_from_metrics({"avg_cer": 0.5, "avg_rr": 0.9}), 0.7)
        self.assertAlmostEqual(overall_from_metrics({"avg_cer": 0.0, "avg_rr": 1.0}), 1.0)
        self.assertAlmostEqual(overall_from_metrics({"avg_cer": 1.0, "avg_rr": 0.0}), 0.0)


def _ns(**kwargs) -> "argparse.Namespace":  # type: ignore[name-defined]
    import argparse

    base = {"threshold": None, "threshold_from_checkpoint": None}
    base.update(kwargs)
    return argparse.Namespace(**base)


class CheckpointThresholdResolutionTests(unittest.TestCase):
    """Regression: --threshold-from-checkpoint reads a binary .pt checkpoint,
    not JSON (the documented blind-eval command passes best.pt)."""

    def _write_ckpt(self, path: Path, **meta) -> None:
        payload = {"model_state_dict": {}, "model_config": {"with_presence": True}}
        payload.update(meta)
        torch.save(payload, path)

    def test_reads_threshold_and_source_from_pt_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "best.pt"
            self._write_ckpt(
                ckpt,
                presence_threshold=0.421,
                presence_threshold_source="public_val_youden_j",
                presence_auc=0.81,
            )
            threshold, source = _resolve_threshold(_ns(threshold_from_checkpoint=str(ckpt)))
            self.assertAlmostEqual(threshold, 0.421, places=6)
            self.assertEqual(source, "public_val_youden_j")

    def test_cli_parser_to_resolution_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "best.pt"
            self._write_ckpt(ckpt, presence_threshold=0.37,
                             presence_threshold_source="public_val_youden_j")
            args = parse_args([
                "--asr-predictions", "x.jsonl",
                "--audio-map", "map.jsonl",
                "--threshold-from-checkpoint", str(ckpt),
            ])
            threshold, source = _resolve_threshold(args)
            self.assertAlmostEqual(threshold, 0.37, places=6)
            self.assertEqual(source, "public_val_youden_j")

    def test_explicit_threshold_overrides_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "best.pt"
            self._write_ckpt(ckpt, presence_threshold=0.421)
            threshold, source = _resolve_threshold(
                _ns(threshold=0.9, threshold_from_checkpoint=str(ckpt))
            )
            self.assertAlmostEqual(threshold, 0.9, places=6)
            self.assertEqual(source, "explicit_override")

    def test_checkpoint_without_presence_threshold_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "best.pt"
            self._write_ckpt(ckpt)  # no presence_threshold
            with self.assertRaises(SystemExit):
                _resolve_threshold(_ns(threshold_from_checkpoint=str(ckpt)))

    def test_non_torch_file_fails_closed(self):
        # The original bug: a text/JSON file must not be parsed as a checkpoint.
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "not_a_ckpt.pt"
            bad.write_text('{"presence_threshold": 0.5}', encoding="utf-8")
            with self.assertRaises(SystemExit):
                _resolve_threshold(_ns(threshold_from_checkpoint=str(bad)))

    def test_missing_checkpoint_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                _resolve_threshold(_ns(threshold_from_checkpoint=str(Path(td) / "ghost.pt")))

    def test_no_threshold_source_fails_closed(self):
        with self.assertRaises(SystemExit):
            _resolve_threshold(_ns())


class LoadScoresTests(unittest.TestCase):
    def _write(self, tmp: Path, rows: list[dict], name: str = "scores.jsonl") -> Path:
        path = tmp / name
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_loads_named_field(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), [
                {"id": "a", "enhanced_cosine": 0.6, "mixture_cosine": 0.2},
                {"id": "b", "enhanced_cosine": 0.1, "mixture_cosine": 0.8},
            ])
            self.assertEqual(load_scores(path, score_field="enhanced_cosine"),
                             {"a": 0.6, "b": 0.1})

    def test_missing_named_field_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), [{"id": "a", "mixture_cosine": 0.2}])
            with self.assertRaisesRegex(ValueError, "enhanced_cosine"):
                load_scores(path, score_field="enhanced_cosine")


class LoadAllScoreFieldsTests(unittest.TestCase):
    def _write(self, tmp: Path, rows: list[dict]) -> Path:
        path = tmp / "amap.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_loads_all_present_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), [
                {"id": "a", "enhanced_cosine": 0.6, "mixture_cosine": 0.2,
                 "max_cosine": 0.6, "presence_score": 0.9},
                {"id": "b", "enhanced_cosine": 0.1, "mixture_cosine": 0.8,
                 "max_cosine": 0.8, "presence_score": 0.1},
            ])
            fields = load_all_score_fields(path)
            self.assertEqual(set(fields), set(SCORE_FIELDS))
            self.assertEqual(fields["enhanced_cosine"], {"a": 0.6, "b": 0.1})

    def test_half_populated_field_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), [
                {"id": "a", "enhanced_cosine": 0.6, "mixture_cosine": 0.2},
                {"id": "b", "mixture_cosine": 0.8},  # missing enhanced_cosine
            ])
            with self.assertRaisesRegex(ValueError, "enhanced_cosine"):
                load_all_score_fields(path)

    def test_absent_fields_omitted(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), [
                {"id": "a", "presence_score": 0.9},
                {"id": "b", "presence_score": 0.1},
            ])
            fields = load_all_score_fields(path)
            self.assertEqual(set(fields), {"presence_score"})

    def test_non_finite_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(Path(td), [
                {"id": "a", "enhanced_cosine": float("nan")},
                {"id": "b", "enhanced_cosine": 0.1},
            ])
            with self.assertRaisesRegex(ValueError, "finite"):
                load_all_score_fields(path)


class SpeakerCheckpointResolutionTests(unittest.TestCase):
    def _write_ckpt(self, path: Path, **meta) -> None:
        payload = {"model_state_dict": {}, "model_config": {"with_presence": True}}
        payload.update(meta)
        torch.save(payload, path)

    def test_resolves_speaker_threshold_and_field_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "best.pt"
            self._write_ckpt(
                ckpt,
                with_speaker_score=True,
                speaker_score_type="enhanced_cosine",
                speaker_threshold=0.55,
                speaker_threshold_source="public_val_max_overall",
                speaker_auc=0.93,
                presence_threshold=0.42,
                presence_threshold_source="public_val_youden_j",
            )
            args = parse_args([
                "--asr-predictions", "x.jsonl", "--audio-map", "m.jsonl",
                "--threshold-from-checkpoint", str(ckpt),
            ])
            threshold, source = _resolve_threshold(args)
            self.assertAlmostEqual(threshold, 0.55, places=6)
            self.assertEqual(source, "public_val_max_overall")
            self.assertEqual(_resolve_score_field(args), "enhanced_cosine")

    def test_explicit_score_field_overrides_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "best.pt"
            self._write_ckpt(ckpt, speaker_score_type="enhanced_cosine",
                             speaker_threshold=0.55)
            args = parse_args([
                "--asr-predictions", "x.jsonl", "--audio-map", "m.jsonl",
                "--threshold-from-checkpoint", str(ckpt),
                "--score-field", "mixture_cosine", "--threshold", "0.4",
            ])
            self.assertEqual(_resolve_score_field(args), "mixture_cosine")
            threshold, _ = _resolve_threshold(args)
            self.assertAlmostEqual(threshold, 0.4, places=6)

    def test_presence_only_checkpoint_resolves_presence_field(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "best.pt"
            self._write_ckpt(ckpt, presence_threshold=0.42)
            args = parse_args([
                "--asr-predictions", "x.jsonl", "--audio-map", "m.jsonl",
                "--threshold-from-checkpoint", str(ckpt),
            ])
            self.assertEqual(_resolve_score_field(args), "presence_score")

    def test_unknown_score_field_fails_closed(self):
        args = parse_args([
            "--asr-predictions", "x.jsonl", "--audio-map", "m.jsonl",
            "--score-field", "bogus",
        ])
        with self.assertRaises(SystemExit):
            _resolve_score_field(args)


def _write_public_manifest(tmp: Path) -> Path:
    """2 present + 2 absent training rows (dummy audio paths; never read)."""
    rows = []
    for rid, present, text in [("p0", True, "你好"), ("p1", True, "世界"),
                               ("n0", False, None), ("n1", False, None)]:
        rows.append({
            "row_id": rid, "split": "val", "source": "t",
            "enrollment_audio": f"/dummy/{rid}-e.wav",
            "target_audio": f"/dummy/{rid}-t.wav",
            "mixture_audio": f"/dummy/{rid}-m.wav",
            "target_speaker_id": "S1", "interferer_speaker_id": "I1",
            "target_present": present, "overlap_ratio": 0.5,
            "snr_db": 5.0, "sir_db": 0.0, "text": text, "seed": 1,
        })
    path = tmp / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    return path


def _write_amap_asr(tmp: Path) -> tuple[Path, Path]:
    amap_rows = [
        {"id": "p0", "enhanced_cosine": 0.60, "mixture_cosine": 0.20,
         "max_cosine": 0.60, "presence_score": 0.90},
        {"id": "p1", "enhanced_cosine": 0.55, "mixture_cosine": 0.30,
         "max_cosine": 0.55, "presence_score": 0.85},
        {"id": "n0", "enhanced_cosine": 0.10, "mixture_cosine": 0.80,
         "max_cosine": 0.80, "presence_score": 0.10},
        {"id": "n1", "enhanced_cosine": 0.15, "mixture_cosine": 0.70,
         "max_cosine": 0.70, "presence_score": 0.15},
    ]
    asr_rows = [
        {"id": "p0", "recognition_text": "你好"},
        {"id": "p1", "recognition_text": "世界"},
        {"id": "n0", "recognition_text": "干扰"},
        {"id": "n1", "recognition_text": "噪音"},
    ]
    amap = tmp / "amap.jsonl"
    asr = tmp / "asr.jsonl"
    amap.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in amap_rows) + "\n",
                    encoding="utf-8")
    asr.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in asr_rows) + "\n",
                   encoding="utf-8")
    return amap, asr


class SpeakerEvaluatorEndToEndTests(unittest.TestCase):
    def test_calibrate_selects_best_variant_and_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            manifest = _write_public_manifest(tmp)
            amap, asr = _write_amap_asr(tmp)
            out = tmp / "calib.json"
            result = main([
                "--calibrate", "--asr-predictions", str(asr), "--audio-map", str(amap),
                "--public-manifest", str(manifest), "--public-split", "val",
                "--output", str(out),
            ])
            self.assertEqual(result["score_type"], "enhanced_cosine")
            self.assertEqual(result["threshold_source"], "public_val_max_overall")
            self.assertAlmostEqual(result["metrics"]["overall"], 1.0, places=6)
            self.assertEqual(set(result["per_variant"]),
                             {"enhanced_cosine", "mixture_cosine", "max_cosine"})
            self.assertTrue(out.is_file())

    def test_evaluate_gates_on_selected_speaker_field(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            manifest = _write_public_manifest(tmp)
            amap, asr = _write_amap_asr(tmp)
            out = tmp / "eval.json"
            result = main([
                "--asr-predictions", str(asr), "--audio-map", str(amap),
                "--public-manifest", str(manifest), "--public-split", "val",
                "--threshold", "0.3", "--score-field", "enhanced_cosine",
                "--gated-predictions", str(tmp / "gated.jsonl"),
                "--output", str(out),
            ])
            self.assertEqual(result["score_field"], "enhanced_cosine")
            # At 0.3: pos accepted (CER 0), neg rejected (RR 1) -> Overall 1.0.
            self.assertAlmostEqual(result["metrics"]["overall"], 1.0, places=6)
            self.assertEqual(result["metrics"]["avg_rr"], 1.0)
            self.assertIn("official_evaluator_crosscheck", result)

    def test_evaluate_falls_back_to_presence_for_r6_map(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            manifest = _write_public_manifest(tmp)
            # R6-style audio map: presence_score only, no cosines.
            amap = tmp / "amap.jsonl"
            amap.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
                {"id": "p0", "presence_score": 0.90},
                {"id": "p1", "presence_score": 0.85},
                {"id": "n0", "presence_score": 0.10},
                {"id": "n1", "presence_score": 0.15},
            ]) + "\n", encoding="utf-8")
            # _write_amap_asr returns (amap, asr); keep the asr path, overwrite amap.
            _, asr = _write_amap_asr(tmp)
            out = tmp / "eval.json"
            result = main([
                "--asr-predictions", str(asr), "--audio-map", str(amap),
                "--public-manifest", str(manifest), "--public-split", "val",
                "--threshold", "0.3", "--score-field", "presence_score",
                "--output", str(out),
            ])
            self.assertEqual(result["score_field"], "presence_score")
            self.assertAlmostEqual(result["metrics"]["overall"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
