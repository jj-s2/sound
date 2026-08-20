"""Tests for the public AISHELL synthetic overlap builder."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.prepare_phase2_synthetic import (
    build_synthetic_manifest,
    discover_aishell_utterances,
    split_speakers,
)
from xh202615.training_data import (
    assert_valid_training_manifest,
    read_training_manifest,
)


class Phase2SyntheticBuilderTests(unittest.TestCase):
    def make_fake_aishell(self, root: Path, speaker_count: int = 3) -> Path:
        corpus = root / "aishell"
        transcript = corpus / "transcript" / "aishell_transcript_v0.8.txt"
        transcript.parent.mkdir(parents=True)
        transcript_lines = []
        for speaker_index in range(1, speaker_count + 1):
            speaker_id = f"S{speaker_index:04d}"
            speaker_dir = corpus / "wav" / "train" / speaker_id
            speaker_dir.mkdir(parents=True)
            for utterance_index in range(1, 3):
                utterance_id = f"{speaker_id}{utterance_index:04d}"
                sample_rate = 8_000 if speaker_index == 1 else 16_000
                duration = 0.08 + utterance_index * 0.01
                times = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
                tone = 0.1 * np.sin(
                    2.0 * np.pi * (180 + speaker_index * 30 + utterance_index) * times
                )
                if speaker_index == 2:
                    tone = np.column_stack((tone, tone * 0.8))
                sf.write(str(speaker_dir / f"{utterance_id}.wav"), tone, sample_rate)
                transcript_lines.append(f"{utterance_id}\t测试文本 {utterance_id}")

        missing_transcript = corpus / "misc" / "S9999" / "SKIP0001.wav"
        missing_transcript.parent.mkdir(parents=True)
        sf.write(str(missing_transcript), np.zeros(320, dtype=np.float32), 16_000)
        transcript.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
        return corpus

    def make_augmentation_assets(self, root: Path) -> tuple[Path, Path]:
        rir_root = root / "rir_assets"
        noise_root = root / "noise_assets"
        rir_root.mkdir(parents=True)
        noise_root.mkdir(parents=True)
        impulse = np.zeros(160, dtype=np.float32)
        impulse[3] = 0.8
        impulse[40] = 0.3
        for index in range(6):
            sf.write(str(rir_root / f"rir_{index:02d}.wav"), impulse, 16_000)
            times = np.arange(800 + index * 10, dtype=np.float64) / 16_000
            noise = 0.02 * np.sin(2.0 * np.pi * (80 + index * 10) * times)
            sf.write(str(noise_root / f"noise_{index:02d}.wav"), noise, 16_000)
        return rir_root, noise_root

    def test_discovery_supports_layout_and_skips_untranscribed_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = self.make_fake_aishell(Path(temp_dir))
            utterances = discover_aishell_utterances(corpus)

        self.assertEqual(len(utterances), 6)
        self.assertEqual(
            {utterance["speaker_id"] for utterance in utterances},
            {"S0001", "S0002", "S0003"},
        )
        self.assertTrue(all(utterance["text"].startswith("测试文本") for utterance in utterances))
        self.assertTrue(all(utterance["split"] == "train" for utterance in utterances))

    def test_speaker_split_is_deterministic_and_disjoint(self):
        speakers = [f"S{index:04d}" for index in range(20)]
        first = split_speakers(speakers, seed=1234)
        second = split_speakers(reversed(speakers), seed=1234)
        changed = split_speakers(speakers, seed=4321)

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(set(first), set(speakers))
        self.assertEqual(list(first.values()).count("val"), 2)
        self.assertEqual(list(first.values()).count("test"), 2)
        self.assertEqual(list(first.values()).count("train"), 16)

        small = split_speakers(["S1", "S2", "S3"], seed=1234)
        self.assertEqual(list(small.values()).count("val"), 1)
        self.assertEqual(list(small.values()).count("test"), 1)
        self.assertEqual(list(small.values()).count("train"), 1)

    def test_build_writes_16khz_mono_files_and_valid_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = self.make_fake_aishell(root)
            output = root / "generated"
            rows = build_synthetic_manifest(
                corpus,
                output,
                max_speakers_per_split=3,
                utterances_per_speaker=2,
                seed=99,
                snr_db=5.0,
                sir_db=0.0,
            )

            self.assertEqual(len(rows), 12)
            self.assertTrue((output / "manifest.jsonl").is_file())
            self.assertTrue((output / "metadata.json").is_file())
            self.assertEqual(
                assert_valid_training_manifest(
                    rows, manifest_path=output / "manifest.jsonl"
                ),
                rows,
            )
            restored = read_training_manifest(output / "manifest.jsonl")
            self.assertEqual(
                [row.to_dict() for row in restored],
                [row.to_dict() for row in rows],
            )

            for row in rows:
                for path in (
                    row.enrollment_audio,
                    row.target_audio,
                    row.mixture_audio,
                ):
                    self.assertIsNotNone(path)
                    info = sf.info(str(path))
                    self.assertEqual(info.samplerate, 16_000)
                    self.assertEqual(info.channels, 1)
                self.assertTrue(math.isfinite(row.overlap_ratio))
                self.assertTrue(math.isfinite(row.snr_db))
                if row.sir_db is not None:
                    self.assertTrue(math.isfinite(row.sir_db))

            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["seed"], 99)
            self.assertEqual(metadata["source_root"], str(corpus.resolve()))
            self.assertEqual(metadata["row_count"], len(rows))
            self.assertEqual(metadata["speaker_split_counts"]["train"], 3)

    def test_build_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = self.make_fake_aishell(root)
            first = build_synthetic_manifest(corpus, root / "first", seed=17)
            second = build_synthetic_manifest(corpus, root / "second", seed=17)

            first_payload = [
                {
                    **row.to_dict(),
                    "enrollment_audio": row.enrollment_audio.name,
                    "target_audio": row.target_audio.name,
                    "mixture_audio": row.mixture_audio.name,
                }
                for row in first
            ]
            second_payload = [
                {
                    **row.to_dict(),
                    "enrollment_audio": row.enrollment_audio.name,
                    "target_audio": row.target_audio.name,
                    "mixture_audio": row.mixture_audio.name,
                }
                for row in second
            ]
            self.assertEqual(first_payload, second_payload)
            for first_row, second_row in zip(first, second):
                first_audio, _ = sf.read(str(first_row.mixture_audio), dtype="float32")
                second_audio, _ = sf.read(str(second_row.mixture_audio), dtype="float32")
                np.testing.assert_array_equal(first_audio, second_audio)

    def test_forbidden_input_root_is_rejected_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            forbidden = root / "evaluation_only"
            corpus = self.make_fake_aishell(forbidden)
            output = root / "generated"

            with self.assertRaisesRegex(ValueError, "under forbidden root"):
                build_synthetic_manifest(
                    corpus,
                    output,
                    forbidden_roots=(forbidden,),
                )
            self.assertFalse(output.exists())

    def test_forbidden_output_root_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = self.make_fake_aishell(root)
            forbidden = root / "evaluation_only"
            output = forbidden / "generated"
            with self.assertRaisesRegex(ValueError, "output root .* forbidden root"):
                build_synthetic_manifest(
                    corpus,
                    output,
                    forbidden_roots=(forbidden,),
                )
            self.assertFalse(output.exists())

    def test_target_absent_rows_have_null_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = self.make_fake_aishell(root, speaker_count=2)
            rows = build_synthetic_manifest(corpus, root / "generated")
            negative_rows = [row for row in rows if not row.target_present]
            positive_rows = [row for row in rows if row.target_present]
            self.assertTrue(negative_rows)
            self.assertTrue(positive_rows)
            self.assertTrue(all(row.text is None for row in negative_rows))
            self.assertTrue(all(row.text for row in positive_rows))
            for row in negative_rows:
                audio, _ = sf.read(str(row.target_audio), dtype="float32")
                self.assertTrue(np.all(audio == 0.0))

    def test_requested_partial_overlap_is_realized_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = self.make_fake_aishell(root)
            first = build_synthetic_manifest(
                corpus,
                root / "first",
                overlap_values=(0.25,),
                snr_values=(None,),
                sir_values=(0.0,),
                seed=73,
            )
            second = build_synthetic_manifest(
                corpus,
                root / "second",
                overlap_values=(0.25,),
                snr_values=(None,),
                sir_values=(0.0,),
                seed=73,
            )

            first_positive = [row for row in first if row.target_present]
            second_positive = [row for row in second if row.target_present]
            self.assertTrue(first_positive)
            self.assertTrue(all(row.overlap_ratio == 0.25 for row in first_positive))
            for first_row, second_row in zip(first_positive, second_positive):
                first_audio, _ = sf.read(str(first_row.mixture_audio), dtype="float32")
                second_audio, _ = sf.read(str(second_row.mixture_audio), dtype="float32")
                np.testing.assert_array_equal(first_audio, second_audio)

    def test_assets_are_split_disjoint_and_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = self.make_fake_aishell(root, speaker_count=12)
            rir_root, noise_root = self.make_augmentation_assets(root)
            output = root / "generated"
            rows = build_synthetic_manifest(
                corpus,
                output,
                max_speakers_per_split=4,
                utterances_per_speaker=2,
                snr_values=(0.0,),
                sir_values=(0.0,),
                overlap_values=(0.5,),
                rir_root=rir_root,
                noise_root=noise_root,
                reverb_probability=1.0,
                seed=99,
            )

            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            asset_ids = metadata["asset_split_ids"]
            self.assertFalse(set(asset_ids["train"]) & set(asset_ids["val"]))
            self.assertFalse(set(asset_ids["train"]) & set(asset_ids["test"]))
            self.assertFalse(set(asset_ids["val"]) & set(asset_ids["test"]))
            self.assertTrue(metadata["asset_split_counts"]["train"]["rir"])
            self.assertTrue(metadata["asset_split_counts"]["train"]["noise"])
            self.assertEqual(read_training_manifest(output / "manifest.jsonl"), rows)

    def test_same_asset_root_stays_disjoint_across_rir_and_noise_roles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = self.make_fake_aishell(root, speaker_count=12)
            asset_root = root / "combined_assets"
            asset_root.mkdir()
            impulse = np.zeros(80, dtype=np.float32)
            impulse[2] = 1.0
            for index in range(12):
                sf.write(str(asset_root / f"rir_noise_{index:02d}.wav"), impulse, 16_000)

            output = root / "generated"
            build_synthetic_manifest(
                corpus,
                output,
                max_speakers_per_split=4,
                rir_root=asset_root,
                noise_root=asset_root,
                reverb_probability=1.0,
                seed=101,
            )
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            paths_by_split = {
                split: {item.split(":", maxsplit=1)[1] for item in values}
                for split, values in metadata["asset_split_ids"].items()
            }
            self.assertFalse(paths_by_split["train"] & paths_by_split["val"])
            self.assertFalse(paths_by_split["train"] & paths_by_split["test"])
            self.assertFalse(paths_by_split["val"] & paths_by_split["test"])


if __name__ == "__main__":
    unittest.main()
