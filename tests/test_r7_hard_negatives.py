"""Tests for R7 impostor hard-negative construction.

Synthetic R3 counterfactual pairs (with real tiny WAV files) exercise the
impostor builder: balanced 1:1 present/absent, impostor enrollment from a
*different* speaker over the *same* mixture, speaker-disjointness, no duplicate
IDs, deterministic digest, and fail-closed behaviour for bad fractions and
single-speaker splits.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from xh202615.r3_data import R3MixtureRow
from xh202615.r7_hard_negatives import (
    IMPOSTOR_SOURCE,
    PUBLIC_SOURCE,
    build_r7_training_rows,
    prepare_r7_manifest,
    read_aishell_transcripts,
    speaker_of_utterance,
)
from xh202615.tse_presence import samples_from_manifest


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 1 s of low-amplitude noise; content is irrelevant to the manifest logic.
    rng = np.random.default_rng(abs(hash(str(path))) % (2 ** 32))
    sf.write(str(path), rng.standard_normal(16000).astype(np.float32) * 0.01, 16000)


def _make_pair(tmp: Path, pair_id: str, split: str, target_utt: str,
               interferer_utts: tuple[str, ...]) -> tuple[R3MixtureRow, R3MixtureRow]:
    """Create a counterfactual pair with real (tiny) WAV files."""
    enroll = tmp / "enrollment" / split / f"{pair_id}.wav"
    pos_mix = tmp / "mixture" / split / f"{pair_id}-pos.wav"
    neg_mix = tmp / "mixture" / split / f"{pair_id}-neg.wav"
    pos_tgt = tmp / "clean_target" / split / f"{pair_id}-pos.wav"
    neg_tgt = tmp / "clean_target" / split / f"{pair_id}-neg.wav"
    for p in (enroll, pos_mix, neg_mix, pos_tgt, neg_tgt):
        _write_wav(p)
    common = dict(
        pair_id=pair_id, split=split,
        enrollment_audio=enroll, target_source_id=target_utt,
        interferer_source_ids=interferer_utts, noise_source_id="noise-1",
        target_rir_id="rir-1", interferer_rir_ids=("rir-1",),
        renderer_family="r3-test", snr_db=5.0, sir_db=0.0, overlap_ratio=0.5,
        codec="pcm16", clip_threshold=1.0, nuisance_fingerprint="fp-" + pair_id,
    )
    pos = R3MixtureRow(row_id=f"{pair_id}-pos", target_present=True,
                       mixture_audio=pos_mix, clean_target_audio=pos_tgt, **common)
    neg = R3MixtureRow(row_id=f"{pair_id}-neg", target_present=False,
                       mixture_audio=neg_mix, clean_target_audio=neg_tgt, **common)
    return pos, neg


def _four_pairs(tmp: Path) -> list[R3MixtureRow]:
    """4 pairs across 4 distinct speakers in the train split."""
    rows: list[R3MixtureRow] = []
    for i, spk in enumerate(("S0002", "S0003", "S0004", "S0005")):
        target = f"BAC009{spk}W0001"
        interferer = (f"BAC009S0099W0001",)
        pos, neg = _make_pair(tmp, f"r3-train-{i + 1:04d}", "train", target, interferer)
        rows.extend([pos, neg])
    return rows


class SpeakerOfUtteranceTests(unittest.TestCase):
    def test_extracts_speaker_prefix(self):
        self.assertEqual(speaker_of_utterance("BAC009S0002W0421"), "S0002")
        self.assertEqual(speaker_of_utterance("BAC009S0156W0126"), "S0156")

    def test_non_aishell_falls_back_to_whole(self):
        self.assertEqual(speaker_of_utterance("some_other_id"), "some_other_id")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            speaker_of_utterance("")


class BuildR7RowsTests(unittest.TestCase):
    def test_all_impostor_balanced_and_disjoint(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rows = _four_pairs(tmp)
            training_rows, neg_counts = build_r7_training_rows(
                rows, dataset_a_root=tmp / "datasetA",
                impostor_fraction=1.0, seed=20260806, check_audio=True,
            )
            # 4 positives + 4 negatives, balanced 1:1.
            self.assertEqual(len(training_rows), 8)
            present = [r for r in training_rows if r.target_present]
            absent = [r for r in training_rows if not r.target_present]
            self.assertEqual(len(present), 4)
            self.assertEqual(len(absent), 4)
            # All negatives are impostors at fraction 1.0.
            self.assertTrue(all(r.source == IMPOSTOR_SOURCE for r in absent))
            self.assertEqual(neg_counts["impostor_negatives"]["train"], 4)
            self.assertEqual(neg_counts["counterfactual_negatives"]["train"], 0)

            # No duplicate row ids.
            ids = [r.row_id for r in training_rows]
            self.assertEqual(len(ids), len(set(ids)))

            # Impostor structure: same mixture, different-speaker enrollment.
            by_pair = {}
            for r in training_rows:
                by_pair.setdefault(r.row_id.rsplit("-", 1)[0], {})[
                    "pos" if r.target_present else "imp"
                ] = r
            for pair_id, pair in by_pair.items():
                pos, imp = pair["pos"], pair["imp"]
                self.assertEqual(imp.mixture_audio, pos.mixture_audio)  # A's mixture
                self.assertNotEqual(imp.enrollment_audio, pos.enrollment_audio)  # B != A
                # Enrolled speaker (B) differs from the impostor (A) actually present.
                self.assertNotEqual(
                    speaker_of_utterance(imp.target_speaker_id),
                    speaker_of_utterance(imp.interferer_speaker_id),
                )
                # The impostor actually in the mixture is A (the pair's target).
                self.assertEqual(imp.interferer_speaker_id, pos.target_speaker_id)
                self.assertIs(imp.target_present, False)
                # Both speakers stay in the train split (no cross-split leakage).
                self.assertEqual(imp.split, "train")

    def test_all_counterfactual_when_fraction_zero(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rows = _four_pairs(tmp)
            training_rows, neg_counts = build_r7_training_rows(
                rows, dataset_a_root=tmp / "datasetA",
                impostor_fraction=0.0, seed=20260806, check_audio=True,
            )
            absent = [r for r in training_rows if not r.target_present]
            self.assertTrue(all(r.source == PUBLIC_SOURCE for r in absent))
            self.assertEqual(neg_counts["impostor_negatives"]["train"], 0)
            self.assertEqual(neg_counts["counterfactual_negatives"]["train"], 4)

    def test_bad_fraction_raises(self):
        with tempfile.TemporaryDirectory() as td:
            rows = _four_pairs(Path(td))
            for bad in (-0.1, 1.5):
                with self.assertRaisesRegex(ValueError, "impostor_fraction"):
                    build_r7_training_rows(rows, dataset_a_root="datasetA",
                                           impostor_fraction=bad, seed=1, check_audio=False)

    def test_single_speaker_split_with_impostor_raises(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # One pair, one target speaker -> no different-speaker partner.
            pos, neg = _make_pair(tmp, "r3-train-0001", "train",
                                  "BAC009S0002W0001", ("BAC009S0099W0001",))
            with self.assertRaisesRegex(ValueError, "no different-speaker partner"):
                build_r7_training_rows([pos, neg], dataset_a_root=tmp / "datasetA",
                                       impostor_fraction=1.0, seed=1, check_audio=True)

    def test_missing_audio_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rows = _four_pairs(tmp)
            # Delete an enrollment file the impostor would reference.
            enroll_files = sorted((tmp / "enrollment" / "train").glob("*.wav"))
            enroll_files[0].unlink()
            with self.assertRaisesRegex(ValueError, "missing audio"):
                build_r7_training_rows(rows, dataset_a_root=tmp / "datasetA",
                                       impostor_fraction=1.0, seed=20260806, check_audio=True)


class PrepareR7ManifestTests(unittest.TestCase):
    def test_writes_manifest_with_digest_and_balance(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            r3_manifest = tmp / "r3_manifest.jsonl"
            from xh202615.r3_data import write_r3_manifest
            write_r3_manifest(r3_manifest, _four_pairs(tmp))
            out = tmp / "r7" / "manifest.jsonl"
            summary = prepare_r7_manifest(
                r3_manifest, out, dataset_a_root=tmp / "datasetA",
                impostor_fraction=0.5, seed=20260806, check_audio=True,
            )
            self.assertEqual(summary["row_count"], 8)
            self.assertEqual(summary["split_rows"]["train"]["present"], 4)
            self.assertEqual(summary["split_rows"]["train"]["absent"], 4)
            self.assertEqual(len(summary["manifest_digest"]), 64)
            self.assertIn(IMPOSTOR_SOURCE, summary["sources"])
            self.assertIn(PUBLIC_SOURCE, summary["sources"])
            # The written file parses back into valid training rows.
            from xh202615.training_data import read_training_manifest
            rows = read_training_manifest(out)
            self.assertEqual(len(rows), 8)

    def test_deterministic_digest(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            r3_manifest = tmp / "r3_manifest.jsonl"
            from xh202615.r3_data import write_r3_manifest
            write_r3_manifest(r3_manifest, _four_pairs(tmp))
            out1 = tmp / "r7a" / "manifest.jsonl"
            out2 = tmp / "r7b" / "manifest.jsonl"
            s1 = prepare_r7_manifest(r3_manifest, out1, dataset_a_root=tmp / "datasetA",
                                     impostor_fraction=0.5, seed=20260806)
            s2 = prepare_r7_manifest(r3_manifest, out2, dataset_a_root=tmp / "datasetA",
                                     impostor_fraction=0.5, seed=20260806)
            self.assertEqual(s1["manifest_digest"], s2["manifest_digest"])


def _write_transcript(tmp: Path, mapping: dict[str, str]) -> Path:
    """Write a minimal AISHELL-style transcript (UTT_ID token token ...)."""
    path = tmp / "transcript.txt"
    path.write_text(
        "\n".join(f"{utt} {text}" for utt, text in mapping.items()) + "\n",
        encoding="utf-8",
    )
    return path


def _four_pair_target_utts() -> tuple[str, ...]:
    return tuple(f"BAC009{spk}W0001" for spk in ("S0002", "S0003", "S0004", "S0005"))


class ReadAishellTranscriptsTests(unittest.TestCase):
    def test_parses_utt_id_to_text(self):
        with tempfile.TemporaryDirectory() as td:
            path = _write_transcript(Path(td), {
                "BAC009S0002W0001": "一 二 三",
                "BAC009S0003W0001": "打开 空调",
            })
            trans = read_aishell_transcripts(path)
            self.assertEqual(trans["BAC009S0002W0001"], "一 二 三")
            self.assertEqual(trans["BAC009S0003W0001"], "打开 空调")

    def test_duplicate_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.txt"
            path.write_text("BAC009S0002W0001 一\nBAC009S0002W0001 二\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate utterance id"):
                read_aishell_transcripts(path)

    def test_malformed_line_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.txt"
            path.write_text("BAC009S0002W0001 一\nlonelyid\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed AISHELL transcript"):
                read_aishell_transcripts(path)

    def test_empty_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.txt"
            path.write_text("\n  \n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no transcript entries"):
                read_aishell_transcripts(path)


class TranscriptPopulationTests(unittest.TestCase):
    def test_with_transcript_populates_positive_text_only(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rows = _four_pairs(tmp)
            trans = _write_transcript(tmp, {u: f"textfor {u}" for u in _four_pair_target_utts()})
            training_rows, neg_counts = build_r7_training_rows(
                rows, dataset_a_root=tmp / "datasetA", impostor_fraction=0.0,
                seed=20260806, check_audio=True, transcript_path=trans,
            )
            present = [r for r in training_rows if r.target_present]
            absent = [r for r in training_rows if not r.target_present]
            self.assertTrue(all(r.text is not None and r.text for r in present))
            self.assertTrue(all(r.text is None for r in absent))
            self.assertEqual(neg_counts["positive_text_coverage"], "4/4")
            self.assertEqual(neg_counts["transcript"], str(trans.resolve()))

    def test_without_transcript_leaves_positive_text_none(self):
        # The original bug: without a transcript, samples_from_manifest sees no
        # labeled positives, so --public-manifest calibration cannot run.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rows = _four_pairs(tmp)
            training_rows, neg_counts = build_r7_training_rows(
                rows, dataset_a_root=tmp / "datasetA", impostor_fraction=0.0,
                seed=20260806, check_audio=True, transcript_path=None,
            )
            present = [r for r in training_rows if r.target_present]
            self.assertTrue(all(r.text is None for r in present))
            self.assertEqual(neg_counts["positive_text_coverage"], "disabled (no transcript)")

    def test_missing_target_id_in_transcript_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rows = _four_pairs(tmp)
            # Transcript covers only 3 of the 4 target utterances.
            utts = _four_pair_target_utts()
            trans = _write_transcript(tmp, {u: f"text {u}" for u in utts[:3]})
            with self.assertRaisesRegex(ValueError, "missing from the transcript"):
                build_r7_training_rows(
                    rows, dataset_a_root=tmp / "datasetA", impostor_fraction=0.0,
                    seed=20260806, check_audio=False, transcript_path=trans,
                )


class PublicManifestCalibrationFixtureTests(unittest.TestCase):
    """The documented `--public-manifest` calibration path must see both classes.

    Regression for the audit finding: without positive text, samples_from_manifest
    labels every row as absent and calibration fails. With the transcript path,
    calibration runs end-to-end via the CLI.
    """

    def _build_manifest_with_text(self, tmp: Path) -> Path:
        from xh202615.r3_data import write_r3_manifest
        r3_manifest = tmp / "r3_manifest.jsonl"
        write_r3_manifest(r3_manifest, _four_pairs(tmp))
        utts = _four_pair_target_utts()
        trans = _write_transcript(tmp, {u: f"文本 {u}" for u in utts})
        out = tmp / "r7" / "manifest.jsonl"
        prepare_r7_manifest(
            r3_manifest, out, dataset_a_root=tmp / "datasetA",
            impostor_fraction=0.5, seed=20260806, transcript_path=trans,
        )
        return out

    def test_samples_from_manifest_sees_both_classes_with_transcript(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = self._build_manifest_with_text(tmp)
            samples = samples_from_manifest(out, "train")
            pos = [s for s in samples if s.label is not None]
            neg = [s for s in samples if s.label is None]
            self.assertEqual(len(samples), 8)
            self.assertEqual(len(pos), 4)
            self.assertEqual(len(neg), 4)
            self.assertTrue(all(s.label for s in pos))

    def test_cli_calibrate_succeeds_via_public_manifest(self):
        from scripts.evaluate_tse_presence import main as eval_main
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            out = self._build_manifest_with_text(tmp)
            rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
            present_ids = {r["row_id"] for r in rows if r["target_present"]}
            all_ids = [r["row_id"] for r in rows]
            # audio_map: present rows score high on every variant; absent low.
            amap = tmp / "amap.jsonl"
            amap.write_text("\n".join(json.dumps({
                "id": rid,
                "enhanced_cosine": 0.60 if rid in present_ids else 0.10,
                "mixture_cosine": 0.55 if rid in present_ids else 0.12,
                "max_cosine": 0.60 if rid in present_ids else 0.12,
                "presence_score": 0.90 if rid in present_ids else 0.10,
            }, ensure_ascii=False) for rid in all_ids) + "\n", encoding="utf-8")
            # ASR: perfect text for present (matches the manifest text), noise for absent.
            asr = tmp / "asr.jsonl"
            asr.write_text("\n".join(json.dumps({
                "id": r["row_id"],
                "recognition_text": r["text"] if r["target_present"] else "干扰语音",
            }, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            calib_out = tmp / "calib.json"
            result = eval_main([
                "--calibrate", "--asr-predictions", str(asr), "--audio-map", str(amap),
                "--public-manifest", str(out), "--public-split", "train",
                "--output", str(calib_out),
            ])
            self.assertIn(result["score_type"],
                          ("enhanced_cosine", "mixture_cosine", "max_cosine"))
            self.assertEqual(result["threshold_source"], "public_val_max_overall")
            # Perfect ASR + perfect gating -> Overall 1.0.
            self.assertAlmostEqual(result["metrics"]["overall"], 1.0, places=6)
            self.assertTrue(calib_out.is_file())


if __name__ == "__main__":
    unittest.main()
