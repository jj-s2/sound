"""Tests for the R3 domain-matched TSE manifest contract and leakage validation."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from xh202615.r3_data import (
    R3MixtureRow,
    assert_r3_manifest_safe,
    read_r3_manifest,
    validate_r3_manifest,
    write_r3_manifest,
)


class R3ManifestContractTests(unittest.TestCase):
    def make_row(
        self,
        row_id="r1",
        target_present=True,
        *,
        fingerprint="fp-default",
        mixture_audio=None,
        **changes,
    ):
        values = {
            "row_id": row_id,
            "pair_id": "pair-1",
            "split": "train",
            "target_present": target_present,
            "enrollment_audio": Path("audio/enroll.wav"),
            "mixture_audio": Path("audio/mix.wav"),
            "clean_target_audio": Path("audio/clean.wav"),
            "target_source_id": "src-target",
            "interferer_source_ids": ("src-interferer",),
            "noise_source_id": "noise-1",
            "target_rir_id": "rir-1",
            "interferer_rir_ids": ("rir-2",),
            "renderer_family": "renderer-A",
            "snr_db": 10.0,
            "sir_db": 5.0,
            "overlap_ratio": 0.5,
            "codec": "pcm16",
            "clip_threshold": 1.0,
            "nuisance_fingerprint": fingerprint,
        }
        if mixture_audio is not None:
            values["mixture_audio"] = mixture_audio
        values.update(changes)
        return R3MixtureRow(**values)

    def make_pair(self, pair_id="pair-1", *, split="train", **shared):
        pos = self.make_row(
            f"{pair_id}-p",
            True,
            pair_id=pair_id,
            split=split,
            mixture_audio=Path(f"audio/mix_{pair_id}_p.wav"),
            **shared,
        )
        neg = self.make_row(
            f"{pair_id}-n",
            False,
            pair_id=pair_id,
            split=split,
            mixture_audio=Path(f"audio/mix_{pair_id}_n.wav"),
            **shared,
        )
        return (pos, neg)

    def codes(self, rows):
        return {issue.code for issue in validate_r3_manifest(rows)}

    # --- pair invariants ---

    def test_pair_validation_requires_one_positive_and_one_negative_with_same_nuisance(self):
        rows = (
            self.make_row("p", True, fingerprint="same"),
            self.make_row("n", False, fingerprint="same"),
        )
        self.assertEqual(validate_r3_manifest(rows), ())

    def test_pair_validation_rejects_nuisance_mismatch(self):
        rows = (
            self.make_row("p", True, fingerprint="a"),
            self.make_row("n", False, fingerprint="b"),
        )
        self.assertIn("counterfactual_nuisance_mismatch", self.codes(rows))

    def test_pair_rejects_two_positives(self):
        rows = (self.make_row("p", True), self.make_row("n", True))
        self.assertIn("pair_polarity_imbalance", self.codes(rows))

    def test_pair_rejects_oversized_group(self):
        rows = (
            self.make_row("p", True, fingerprint="same"),
            self.make_row("n", False, fingerprint="same"),
            self.make_row("x", True, fingerprint="same"),
        )
        self.assertIn("pair_size", self.codes(rows))

    def test_singleton_pair_rejected(self):
        self.assertIn("pair_size", self.codes((self.make_row("solo", True),)))

    def test_pair_split_mismatch_rejected(self):
        rows = (
            self.make_row("p", True, fingerprint="same", split="train"),
            self.make_row("n", False, fingerprint="same", split="val"),
        )
        self.assertIn("pair_split_mismatch", self.codes(rows))

    def test_pair_field_mismatch_enrollment_rejected(self):
        rows = (
            self.make_row("p", True, fingerprint="same", enrollment_audio=Path("audio/enroll_a.wav")),
            self.make_row("n", False, fingerprint="same", enrollment_audio=Path("audio/enroll_b.wav")),
        )
        self.assertIn("counterfactual_nuisance_mismatch", self.codes(rows))

    def test_pair_field_mismatch_snr_rejected(self):
        rows = (
            self.make_row("p", True, fingerprint="same", snr_db=10.0),
            self.make_row("n", False, fingerprint="same", snr_db=20.0),
        )
        self.assertIn("counterfactual_nuisance_mismatch", self.codes(rows))

    # --- Dataset-A containment guard ---

    def test_safe_guard_rejects_dataset_a_before_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_a_root = Path(temp_dir)
            row = self.make_row("p", True, mixture_audio=dataset_a_root / "pos" / "x.wav")
            with self.assertRaisesRegex(ValueError, "Dataset-A"):
                assert_r3_manifest_safe((row,), dataset_a_root)

    def test_safe_guard_rejects_enrollment_under_dataset_a(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_a_root = Path(temp_dir)
            row = self.make_row("p", True, enrollment_audio=dataset_a_root / "enroll.wav")
            with self.assertRaisesRegex(ValueError, "Dataset-A"):
                assert_r3_manifest_safe((row,), dataset_a_root)

    def test_safe_guard_rejects_clean_target_under_dataset_a(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_a_root = Path(temp_dir)
            row = self.make_row("p", True, clean_target_audio=dataset_a_root / "clean.wav")
            with self.assertRaisesRegex(ValueError, "Dataset-A"):
                assert_r3_manifest_safe((row,), dataset_a_root)

    def test_safe_guard_passes_for_safe_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_a_root = Path(temp_dir) / "datasetA"
            public = Path(temp_dir) / "public"
            common = dict(
                enrollment_audio=public / "enroll.wav",
                clean_target_audio=public / "clean.wav",
                fingerprint="same",
            )
            rows = (
                self.make_row("p", True, mixture_audio=public / "mix_p.wav", **common),
                self.make_row("n", False, mixture_audio=public / "mix_n.wav", **common),
            )
            self.assertEqual(assert_r3_manifest_safe(rows, dataset_a_root), rows)

    def test_safe_guard_also_raises_on_validation_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_a_root = Path(temp_dir) / "datasetA"
            public = Path(temp_dir) / "public"
            common = dict(
                enrollment_audio=public / "enroll.wav",
                clean_target_audio=public / "clean.wav",
                fingerprint="same",
                overlap_ratio=2.0,
            )
            rows = (
                self.make_row("p", True, mixture_audio=public / "mix_p.wav", **common),
                self.make_row("n", False, mixture_audio=public / "mix_n.wav", **common),
            )
            with self.assertRaises(ValueError):
                assert_r3_manifest_safe(rows, dataset_a_root)

    def test_relative_path_under_sibling_dataset_a_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_a_root = root / "datasetA"
            configs = root / "configs"
            configs.mkdir()
            manifest = configs / "manifest.jsonl"
            rows = (self.make_row("p", True, mixture_audio=Path("../datasetA/pos/x.wav")),)
            write_r3_manifest(manifest, rows)
            restored = read_r3_manifest(manifest)
            with self.assertRaisesRegex(ValueError, "Dataset-A"):
                assert_r3_manifest_safe(restored, dataset_a_root)

    # --- finite / range validation ---

    def test_invalid_overlap_ratio(self):
        for ratio in (-0.01, 1.01, math.inf, math.nan):
            with self.subTest(ratio=ratio):
                rows = (
                    self.make_row("p", True, overlap_ratio=ratio),
                    self.make_row("n", False, overlap_ratio=ratio),
                )
                self.assertIn("invalid_overlap_ratio", self.codes(rows))

    def test_invalid_snr_and_sir(self):
        rows = (
            self.make_row("p", True, snr_db=math.inf),
            self.make_row("n", False, snr_db=math.inf),
        )
        self.assertIn("invalid_snr_db", self.codes(rows))
        rows = (
            self.make_row("p", True, sir_db=math.nan),
            self.make_row("n", False, sir_db=math.nan),
        )
        self.assertIn("invalid_sir_db", self.codes(rows))

    def test_clip_threshold_must_be_in_open_zero_closed_one(self):
        for value in (0.0, -0.5, 1.5, math.nan, math.inf):
            with self.subTest(value=value):
                rows = (
                    self.make_row("p", True, clip_threshold=value),
                    self.make_row("n", False, clip_threshold=value),
                )
                self.assertIn("invalid_clip_threshold", self.codes(rows))
        valid = (
            self.make_row("p", True, clip_threshold=1.0),
            self.make_row("n", False, clip_threshold=1.0),
        )
        self.assertEqual(self.codes(valid), set())

    # --- unique ids and split values ---

    def test_duplicate_row_ids_rejected(self):
        rows = (
            self.make_row("dup", True),
            self.make_row("dup", False, fingerprint="same"),
        )
        self.assertIn("duplicate_row_id", self.codes(rows))

    def test_invalid_split_rejected(self):
        rows = (
            self.make_row("p", True, split="bogus"),
            self.make_row("n", False, split="bogus"),
        )
        self.assertIn("invalid_split", self.codes(rows))

    def test_empty_interferer_source_id_rejected(self):
        rows = (
            self.make_row("p", True, interferer_source_ids=("",)),
            self.make_row("n", False, interferer_source_ids=("",)),
        )
        self.assertIn("empty_interferer_source_id", self.codes(rows))

    def test_interferer_rir_count_must_match_or_be_empty(self):
        mismatch = (
            self.make_row("p", True, interferer_source_ids=("a", "b"), interferer_rir_ids=("r1",)),
            self.make_row("n", False, interferer_source_ids=("a", "b"), interferer_rir_ids=("r1",)),
        )
        self.assertIn("interferer_rir_count_mismatch", self.codes(mismatch))
        valid = (
            self.make_row("p", True, interferer_source_ids=("a",), interferer_rir_ids=()),
            self.make_row("n", False, interferer_source_ids=("a",), interferer_rir_ids=()),
        )
        self.assertEqual(self.codes(valid), set())

    # --- cross-split entity leakage (role-independent) ---

    def test_cross_split_target_source_leakage_rejected(self):
        train = self.make_pair(
            "pt",
            split="train",
            target_source_id="spk-shared",
            interferer_source_ids=("spk-t-int",),
            noise_source_id="noise-t",
            target_rir_id="rir-t",
            interferer_rir_ids=("rir-t-int",),
            renderer_family="renderer-t",
        )
        val = self.make_pair(
            "pv",
            split="val",
            target_source_id="spk-shared",
            interferer_source_ids=("spk-v-int",),
            noise_source_id="noise-v",
            target_rir_id="rir-v",
            interferer_rir_ids=("rir-v-int",),
            renderer_family="renderer-v",
        )
        self.assertIn("entity_split_leakage", self.codes(train + val))

    def test_cross_role_speaker_leakage_rejected(self):
        train = self.make_pair(
            "pt",
            split="train",
            target_source_id="spk-X",
            interferer_source_ids=("spk-t-int",),
            noise_source_id="noise-t",
            target_rir_id="rir-t",
            interferer_rir_ids=("rir-t-int",),
            renderer_family="renderer-t",
        )
        val = self.make_pair(
            "pv",
            split="val",
            target_source_id="spk-v",
            interferer_source_ids=("spk-X",),
            noise_source_id="noise-v",
            target_rir_id="rir-v",
            interferer_rir_ids=("rir-v-int",),
            renderer_family="renderer-v",
        )
        self.assertIn("entity_split_leakage", self.codes(train + val))

    def test_cross_role_rir_leakage_rejected(self):
        train = self.make_pair(
            "pt",
            split="train",
            target_source_id="spk-t",
            interferer_source_ids=("spk-t-int",),
            noise_source_id="noise-t",
            target_rir_id="rir-X",
            interferer_rir_ids=("rir-t-int",),
            renderer_family="renderer-t",
        )
        val = self.make_pair(
            "pv",
            split="val",
            target_source_id="spk-v",
            interferer_source_ids=("spk-v-int",),
            noise_source_id="noise-v",
            target_rir_id="rir-v",
            interferer_rir_ids=("rir-X",),
            renderer_family="renderer-v",
        )
        self.assertIn("entity_split_leakage", self.codes(train + val))

    def test_no_leakage_when_entities_disjoint(self):
        train = self.make_pair(
            "pt",
            split="train",
            target_source_id="spk-t",
            interferer_source_ids=("spk-t-int",),
            noise_source_id="noise-t",
            target_rir_id="rir-t",
            interferer_rir_ids=("rir-t-int",),
            renderer_family="renderer-t",
        )
        val = self.make_pair(
            "pv",
            split="val",
            target_source_id="spk-v",
            interferer_source_ids=("spk-v-int",),
            noise_source_id="noise-v",
            target_rir_id="rir-v",
            interferer_rir_ids=("rir-v-int",),
            renderer_family="renderer-v",
        )
        self.assertEqual(validate_r3_manifest(train + val), ())

    # --- deterministic JSONL round-trip ---

    def test_to_dict_is_json_compatible(self):
        row = self.make_row("r1", True)
        payload = row.to_dict()
        self.assertEqual(json.loads(json.dumps(payload, sort_keys=True)), payload)

    def test_optional_fields_round_trip(self):
        row = self.make_row(
            "r1",
            True,
            noise_source_id=None,
            target_rir_id=None,
            snr_db=None,
            interferer_source_ids=(),
            interferer_rir_ids=(),
        )
        restored = R3MixtureRow.from_dict(row.to_dict())
        self.assertEqual(restored, row)

    def test_from_dict_resolves_relative_paths_against_base(self):
        row = self.make_row("r1", True)
        payload = row.to_dict()
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            restored = R3MixtureRow.from_dict(payload, base_dir=base_dir)
            self.assertEqual(
                restored.enrollment_audio,
                base_dir.resolve(strict=False) / row.enrollment_audio,
            )

    def test_jsonl_round_trip_is_deterministic_and_preserves_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            rows = (
                self.make_row(
                    "p",
                    True,
                    fingerprint="same",
                    enrollment_audio=base / "enroll.wav",
                    mixture_audio=base / "mix_p.wav",
                    clean_target_audio=base / "clean_p.wav",
                ),
                self.make_row(
                    "n",
                    False,
                    fingerprint="same",
                    enrollment_audio=base / "enroll.wav",
                    mixture_audio=base / "mix_n.wav",
                    clean_target_audio=base / "clean_n.wav",
                ),
            )
            path = base / "manifest.jsonl"
            write_r3_manifest(path, rows)
            first_text = path.read_text(encoding="utf-8")
            write_r3_manifest(path, rows)
            self.assertEqual(path.read_text(encoding="utf-8"), first_text)
            self.assertEqual(read_r3_manifest(path), rows)

    def test_read_manifest_reports_malformed_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.jsonl"
            path.write_text("{not-json}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"line 1.*malformed JSON"):
                read_r3_manifest(path)

    def test_missing_required_field_raises_with_field_name(self):
        with self.assertRaisesRegex(ValueError, r"pair_id.*is missing"):
            R3MixtureRow.from_dict({"row_id": "r1"})


if __name__ == "__main__":
    unittest.main()
