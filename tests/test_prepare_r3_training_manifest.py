"""Tests for the R3 -> training-manifest conversion handoff.

The converter must map the public R3 counterfactual manifest onto the existing
``TrainingManifestRow`` JSONL contract without ever reading Dataset-A. These
tests cover field mapping, fail-closed rejection (Dataset-A containment, missing
audio, malformed/unknown rows), speaker split isolation, the deterministic seed,
and row-count preservation. Audio fixtures are tiny public WAVs written under a
temp directory; the Dataset-A root is a separate temp directory used only as a
forbidden containment root.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.prepare_r3_training_manifest import (
    PUBLIC_SOURCE,
    build_training_rows,
    convert_r3_row,
    deterministic_seed,
    prepare_training_manifest,
)
from xh202615.r3_data import R3MixtureRow, write_r3_manifest
from xh202615.training_data import (
    read_training_manifest,
    validate_training_manifest,
)


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(3200, dtype="float32"), 16000)


def _pair_dict(
    *,
    pair_id: str,
    split: str,
    target_source_id: str,
    interferer_source_ids: list[str],
    base: Path,
    mixture_pos: Path | None = None,
    mixture_neg: Path | None = None,
    overlap_ratio: float = 0.5,
    snr_db: float | None = 20.0,
    sir_db: float | None = -5.0,
) -> tuple[dict, dict]:
    """Build a structurally valid R3 counterfactual pair (pos + neg) as dicts."""
    enrollment = base / "enrollment" / split / f"{pair_id}.wav"
    clean_pos = base / "clean_target" / split / f"{pair_id}-pos.wav"
    clean_neg = base / "clean_target" / split / f"{pair_id}-neg.wav"
    mix_pos = mixture_pos if mixture_pos is not None else base / "mixture" / split / f"{pair_id}-pos.wav"
    mix_neg = mixture_neg if mixture_neg is not None else base / "mixture" / split / f"{pair_id}-neg.wav"
    for path in (enrollment, clean_pos, clean_neg, mix_pos, mix_neg):
        _write_wav(path)
    common = {
        "pair_id": pair_id,
        "split": split,
        "enrollment_audio": str(enrollment),
        "target_source_id": target_source_id,
        "interferer_source_ids": list(interferer_source_ids),
        # R3 partitions *every* nuisance entity (noise, RIR, renderer) across
        # splits; mirror that here so the manifest is leakage-clean.
        "noise_source_id": f"noise-{split}-0001",
        "target_rir_id": f"Room-{split}-00001",
        "interferer_rir_ids": [f"Room-{split}-00002", f"Room-{split}-00003"],
        "renderer_family": f"r3-{split}",
        "snr_db": snr_db,
        "sir_db": sir_db,
        "overlap_ratio": overlap_ratio,
        "codec": "pcm16",
        "clip_threshold": 1.0,
        "nuisance_fingerprint": f"fingerprint-{pair_id}",
    }
    pos = dict(common)
    pos.update(
        row_id=f"{pair_id}-pos",
        target_present=True,
        mixture_audio=str(mix_pos),
        clean_target_audio=str(clean_pos),
    )
    neg = dict(common)
    neg.update(
        row_id=f"{pair_id}-neg",
        target_present=False,
        mixture_audio=str(mix_neg),
        clean_target_audio=str(clean_neg),
    )
    return pos, neg


def _write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _make_r3_row(**changes) -> R3MixtureRow:
    base = {
        "row_id": "r3-train-0001-pos",
        "pair_id": "r3-train-0001",
        "split": "train",
        "target_present": True,
        "enrollment_audio": "audio/enrollment/r3-train-0001.wav",
        "mixture_audio": "audio/mixture/r3-train-0001-pos.wav",
        "clean_target_audio": "audio/clean/r3-train-0001-pos.wav",
        "target_source_id": "BAC009S0002W0421",
        "interferer_source_ids": ["BAC009S0055W0121", "BAC009S0046W0122"],
        "noise_source_id": "noise-free-sound-0051",
        "target_rir_id": "Room001-00057",
        "interferer_rir_ids": ["Room001-00025", "Room001-00069"],
        "renderer_family": "r3-train",
        "snr_db": 20.0,
        "sir_db": -10.0,
        "overlap_ratio": 0.5,
        "codec": "pcm16",
        "clip_threshold": 1.0,
        "nuisance_fingerprint": "9afceec3fd97415ef31eebf9000a721db9bad4d5b7fc50c903841d6648a8dad8",
    }
    base.update(changes)
    return R3MixtureRow.from_dict(base)


class ConvertR3RowTests(unittest.TestCase):
    def test_field_mapping_matches_contract(self):
        row = _make_r3_row()
        converted = convert_r3_row(row)

        self.assertEqual(converted.row_id, row.row_id)
        self.assertEqual(converted.split, row.split)
        self.assertEqual(converted.target_present, row.target_present)
        self.assertEqual(converted.overlap_ratio, row.overlap_ratio)
        self.assertEqual(converted.snr_db, row.snr_db)
        self.assertEqual(converted.sir_db, row.sir_db)

        # Audio remapping: clean_target -> target_audio, mixture and enrollment unchanged.
        self.assertEqual(converted.target_audio, row.clean_target_audio)
        self.assertEqual(converted.mixture_audio, row.mixture_audio)
        self.assertEqual(converted.enrollment_audio, row.enrollment_audio)

        # Speaker mapping: target_source_id -> target_speaker_id, first interferer.
        self.assertEqual(converted.target_speaker_id, row.target_source_id)
        self.assertEqual(converted.interferer_speaker_id, row.interferer_source_ids[0])

        # R3 carries no command text; source is a nonempty public string.
        self.assertIsNone(converted.text)
        self.assertEqual(converted.source, PUBLIC_SOURCE)
        self.assertTrue(converted.source.strip())

    def test_empty_interferers_map_to_null_interferer_speaker_id(self):
        row = _make_r3_row(
            interferer_source_ids=[],
            interferer_rir_ids=[],
            overlap_ratio=0.0,
            snr_db=None,
            sir_db=None,
        )
        converted = convert_r3_row(row)
        self.assertIsNone(converted.interferer_speaker_id)

    def test_seed_is_deterministic_int_derived_from_row_identity(self):
        row = _make_r3_row()
        converted = convert_r3_row(row)
        self.assertIsInstance(converted.seed, int)
        self.assertGreaterEqual(converted.seed, 0)
        self.assertLess(converted.seed, 2**31)
        self.assertEqual(converted.seed, deterministic_seed(row.row_id))
        self.assertEqual(deterministic_seed(row.row_id), deterministic_seed(row.row_id))


class DeterministicSeedTests(unittest.TestCase):
    def test_same_row_id_is_reproducible_and_siblings_differ(self):
        pos = "r3-train-0001-pos"
        neg = "r3-train-0001-neg"
        self.assertEqual(deterministic_seed(pos), deterministic_seed(pos))
        self.assertNotEqual(deterministic_seed(pos), deterministic_seed(neg))

    def test_seed_is_a_stable_non_negative_int(self):
        for row_id in ("r3-train-0001-pos", "r3-val-0042-neg", "r3-test-0999-pos"):
            seed = deterministic_seed(row_id)
            self.assertIsInstance(seed, int)
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 2**31)

    def test_empty_row_id_is_rejected(self):
        with self.assertRaises(ValueError):
            deterministic_seed("")


class BuildTrainingRowsTests(unittest.TestCase):
    def _fixture(self, tmp: Path) -> tuple[list[dict], Path]:
        dataset_a = tmp / "datasetA_root"
        dataset_a.mkdir()
        base = tmp / "public"
        pairs = [
            _pair_dict(
                pair_id="r3-train-0001",
                split="train",
                target_source_id="BAC009S0001W0001",
                interferer_source_ids=["BAC009S0100W0002", "BAC009S0101W0003"],
                base=base,
            ),
            _pair_dict(
                pair_id="r3-val-0001",
                split="val",
                target_source_id="BAC009S0002W0001",
                interferer_source_ids=["BAC009S0200W0002", "BAC009S0201W0003"],
                base=base,
            ),
            _pair_dict(
                pair_id="r3-test-0001",
                split="test",
                target_source_id="BAC009S0003W0001",
                interferer_source_ids=["BAC009S0300W0002", "BAC009S0301W0003"],
                base=base,
            ),
        ]
        rows = [row for pair in pairs for row in pair]
        return rows, dataset_a

    def test_valid_manifest_converts_and_passes_training_validation(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            rows, dataset_a = self._fixture(tmp)
            training_rows = build_training_rows(
                (R3MixtureRow.from_dict(r) for r in rows),
                dataset_a_root=dataset_a,
            )
            self.assertEqual(len(training_rows), len(rows))
            issues = validate_training_manifest(training_rows, forbidden_roots=(dataset_a,))
            self.assertEqual(issues, ())

    def test_dataset_a_containment_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            rows, dataset_a = self._fixture(tmp)
            # Move one mixture path under the Dataset-A root.
            inside = dataset_a / "leaked.wav"
            _write_wav(inside)
            rows[0] = dict(rows[0])
            rows[0]["mixture_audio"] = str(inside)
            with self.assertRaisesRegex(ValueError, r"Dataset-A containment"):
                build_training_rows(
                    (R3MixtureRow.from_dict(r) for r in rows),
                    dataset_a_root=dataset_a,
                )

    def test_missing_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            rows, dataset_a = self._fixture(tmp)
            rows[0] = dict(rows[0])
            rows[0]["mixture_audio"] = str(tmp / "public" / "missing.wav")
            with self.assertRaisesRegex(ValueError, r"missing audio"):
                build_training_rows(
                    (R3MixtureRow.from_dict(r) for r in rows),
                    dataset_a_root=dataset_a,
                )

    def test_speaker_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            rows, dataset_a = self._fixture(tmp)
            # Reuse the train target speaker in the val pair -> cross-split leak.
            # Both siblings of the val pair are changed so the pair nuisance
            # field still matches and the only error is the speaker leak.
            for index in (2, 3):
                rows[index] = dict(rows[index])
                rows[index]["target_source_id"] = "BAC009S0001W0001"
            with self.assertRaisesRegex(ValueError, r"validation failed|leakage"):
                build_training_rows(
                    (R3MixtureRow.from_dict(r) for r in rows),
                    dataset_a_root=dataset_a,
                )

    def test_split_isolation_is_preserved_in_output(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            rows, dataset_a = self._fixture(tmp)
            training_rows = build_training_rows(
                (R3MixtureRow.from_dict(r) for r in rows),
                dataset_a_root=dataset_a,
            )
            issues = validate_training_manifest(training_rows, forbidden_roots=(dataset_a,))
            self.assertNotIn(
                "speaker_split_leakage", {issue.code for issue in issues}
            )


class PrepareTrainingManifestTests(unittest.TestCase):
    def _write_fixture(self, tmp: Path) -> tuple[Path, Path]:
        dataset_a = tmp / "datasetA_root"
        dataset_a.mkdir()
        base = tmp / "public"
        pairs = [
            _pair_dict(
                pair_id="r3-train-0001",
                split="train",
                target_source_id="BAC009S0001W0001",
                interferer_source_ids=["BAC009S0100W0002", "BAC009S0101W0003"],
                base=base,
            ),
            _pair_dict(
                pair_id="r3-val-0001",
                split="val",
                target_source_id="BAC009S0002W0001",
                interferer_source_ids=["BAC009S0200W0002", "BAC009S0201W0003"],
                base=base,
            ),
            _pair_dict(
                pair_id="r3-test-0001",
                split="test",
                target_source_id="BAC009S0003W0001",
                interferer_source_ids=["BAC009S0300W0002", "BAC009S0301W0003"],
                base=base,
            ),
        ]
        rows = [row for pair in pairs for row in pair]
        manifest = tmp / "r3_manifest.jsonl"
        _write_manifest(rows, manifest)
        return manifest, dataset_a

    def test_row_counts_and_splits_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, dataset_a = self._write_fixture(tmp)
            output = tmp / "out" / "training.jsonl"
            summary = prepare_training_manifest(manifest, output, dataset_a)

            self.assertEqual(summary["row_count"], 6)
            self.assertEqual(summary["split_rows"], {"train": 2, "val": 2, "test": 2})
            self.assertEqual(summary["dataset_a_used_for_training"], False)
            self.assertEqual(summary["source"], PUBLIC_SOURCE)
            self.assertTrue(output.is_file())

            lines = output.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 6)

    def test_output_round_trips_and_contains_only_public_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, dataset_a = self._write_fixture(tmp)
            output = tmp / "out" / "training.jsonl"
            prepare_training_manifest(manifest, output, dataset_a)

            rows = read_training_manifest(output)
            self.assertEqual(len(rows), 6)
            for row in rows:
                for field in ("enrollment_audio", "target_audio", "mixture_audio"):
                    path = Path(getattr(row, field))
                    self.assertTrue(path.is_absolute(), f"{field} should be absolute")
                    self.assertNotIn(
                        "dataseta",
                        str(path).lower(),
                        "no Dataset-A path may appear in the training manifest",
                    )
                self.assertIsNone(row.text)
                self.assertEqual(row.source, PUBLIC_SOURCE)

    def test_manifest_digest_is_sha256_hex(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, dataset_a = self._write_fixture(tmp)
            output = tmp / "out" / "training.jsonl"
            summary = prepare_training_manifest(manifest, output, dataset_a)
            digest = summary["manifest_digest"]
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_malformed_r3_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, dataset_a = self._write_fixture(tmp)
            # Append a row with an unknown field.
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"row_id": "bad", "unknown_r3_field": "x"}) + "\n"
                )
            output = tmp / "out" / "training.jsonl"
            with self.assertRaises(ValueError):
                prepare_training_manifest(manifest, output, dataset_a)


if __name__ == "__main__":
    unittest.main()
