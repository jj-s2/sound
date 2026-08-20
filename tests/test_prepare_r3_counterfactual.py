"""Tests for the R3 counterfactual preparation CLI."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.prepare_r3_counterfactual import _plan_pairs, build_r3_manifest, parse_args
from scripts.prepare_phase2_synthetic import discover_aishell_utterances, split_speakers
from xh202615.r3_data import (
    assert_r3_manifest_safe,
    read_r3_manifest,
    validate_r3_manifest,
)


class PrepareR3CounterfactualTests(unittest.TestCase):
    SR = 16_000

    def make_fake_aishell(self, root: Path, speaker_count: int = 12) -> Path:
        corpus = root / "aishell"
        transcript = corpus / "transcript" / "aishell_transcript_v0.8.txt"
        transcript.parent.mkdir(parents=True)
        lines = []
        for speaker_index in range(1, speaker_count + 1):
            speaker_id = f"S{speaker_index:04d}"
            speaker_dir = corpus / "wav" / "train" / speaker_id
            speaker_dir.mkdir(parents=True)
            for utterance_index in range(1, 3):
                utterance_id = f"{speaker_id}{utterance_index:04d}"
                n = int(self.SR * (0.08 + utterance_index * 0.01))
                t = np.arange(n) / self.SR
                tone = 0.1 * np.sin(
                    2.0 * np.pi * (180 + speaker_index * 30 + utterance_index) * t
                )
                sf.write(str(speaker_dir / f"{utterance_id}.wav"), tone, self.SR)
                lines.append(f"{utterance_id}\t测试文本 {utterance_id}")
        transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return corpus

    def make_assets(self, root: Path, count: int = 6) -> tuple[Path, Path]:
        rir_root = root / "rir_assets"
        noise_root = root / "noise_assets"
        rir_root.mkdir(parents=True)
        noise_root.mkdir(parents=True)
        impulse = np.zeros(160, dtype=np.float32)
        impulse[3] = 0.8
        impulse[40] = 0.3
        for index in range(count):
            sf.write(str(rir_root / f"rir_{index:02d}.wav"), impulse, self.SR)
            n = 800 + index * 10
            t = np.arange(n) / self.SR
            noise = 0.02 * np.sin(2.0 * np.pi * (80 + index * 10) * t)
            sf.write(str(noise_root / f"noise_{index:02d}.wav"), noise, self.SR)
        return rir_root, noise_root

    def test_parse_args_defaults_and_required(self):
        args = parse_args(
            [
                "--aishell-root", "corpus",
                "--output-root", "out",
                "--dataset-a-root", "datasetA",
            ]
        )
        self.assertEqual(args.aishell_root, "corpus")
        self.assertEqual(args.output_root, "out")
        self.assertEqual(args.dataset_a_root, "datasetA")
        self.assertEqual(args.pairs, 2)
        self.assertFalse(args.dry_run)

    def test_dry_run_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_fake_aishell(root)
            output = root / "out"
            plan = build_r3_manifest(
                corpus,
                output,
                dataset_a_root=root / "datasetA",
                pairs=2,
                seed=1,
                dry_run=True,
            )
            self.assertFalse(output.exists())
            self.assertGreater(plan["pair_count"], 0)
            self.assertTrue(plan["dry_run"])

    def test_fail_closed_on_dataset_a_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_a = root / "datasetA"
            corpus = self.make_fake_aishell(dataset_a)
            output = root / "out"
            with self.assertRaisesRegex(ValueError, "Dataset-A"):
                build_r3_manifest(
                    corpus, output, dataset_a_root=dataset_a, pairs=2, seed=1
                )
            self.assertFalse(output.exists())

    def test_fail_closed_on_dataset_a_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_fake_aishell(root)
            dataset_a = root / "datasetA"
            output = dataset_a / "out"
            with self.assertRaisesRegex(ValueError, "Dataset-A"):
                build_r3_manifest(
                    corpus, output, dataset_a_root=dataset_a, pairs=2, seed=1
                )
            self.assertFalse(output.exists())

    def test_renders_valid_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_fake_aishell(root)
            rir_root, noise_root = self.make_assets(root)
            output = root / "out"
            dataset_a_root = root / "datasetA"
            result = build_r3_manifest(
                corpus,
                output,
                dataset_a_root=dataset_a_root,
                pairs=2,
                seed=3,
                rir_root=rir_root,
                noise_root=noise_root,
            )
            rows = result["rows"]
            self.assertTrue((output / "manifest.jsonl").is_file())
            self.assertTrue((output / "metadata.json").is_file())
            self.assertGreater(len(rows), 0)
            # Every pair is one positive and one negative and Task-1 safe.
            assert_r3_manifest_safe(rows, dataset_a_root=dataset_a_root)
            pair_ids = {row.pair_id for row in rows}
            for pair_id in pair_ids:
                members = [row for row in rows if row.pair_id == pair_id]
                self.assertEqual(len(members), 2)
                self.assertEqual(
                    {m.target_present for m in members}, {True, False}
                )
            for row in rows:
                for path in (row.enrollment_audio, row.mixture_audio, row.clean_target_audio):
                    info = sf.info(str(path))
                    self.assertEqual(info.samplerate, 16_000)
                    self.assertEqual(info.channels, 1)

    def test_cli_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_fake_aishell(root)
            first = build_r3_manifest(corpus, root / "first", dataset_a_root=root / "da", pairs=2, seed=17)
            second = build_r3_manifest(corpus, root / "second", dataset_a_root=root / "da", pairs=2, seed=17)

            def normalized(rows):
                payload = []
                for row in rows:
                    d = row.to_dict()
                    for key in ("enrollment_audio", "mixture_audio", "clean_target_audio"):
                        d[key] = Path(d[key]).name
                    payload.append(d)
                return payload

            self.assertEqual(normalized(first["rows"]), normalized(second["rows"]))
            for r1, r2 in zip(first["rows"], second["rows"]):
                a1, _ = sf.read(str(r1.mixture_audio), dtype="float64")
                a2, _ = sf.read(str(r2.mixture_audio), dtype="float64")
                np.testing.assert_array_equal(a1, a2)

    def test_partitioning_is_disjoint_across_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_fake_aishell(root)
            rir_root, noise_root = self.make_assets(root)
            output = root / "out"
            build_r3_manifest(
                corpus, output, dataset_a_root=root / "da",
                pairs=2, seed=5, rir_root=rir_root, noise_root=noise_root,
            )
            rows = read_r3_manifest(output / "manifest.jsonl")
            by_split = {"train": set(), "val": set(), "test": set()}
            for row in rows:
                entities = {row.target_source_id, row.noise_source_id, row.target_rir_id, row.renderer_family}
                entities.update(row.interferer_source_ids)
                entities.update(row.interferer_rir_ids)
                entities.discard(None)
                by_split[row.split].update(entities)
            self.assertFalse(by_split["train"] & by_split["val"])
            self.assertFalse(by_split["train"] & by_split["test"])
            self.assertFalse(by_split["val"] & by_split["test"])

    def test_plan_pairs_distinct_interferer_speakers(self):
        """When interferer_count > 1, the N interferers must be N DISTINCT
        speakers from the same split (excluding the target), one utterance each.

        The plan (interferer count 1..3) and final_decision.md ('One to three
        interfering speakers') intend different interfering speakers per
        mixture, not multiple utterances of a single speaker.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_fake_aishell(root, speaker_count=20)
            utterances = discover_aishell_utterances(corpus)
            speaker_splits = split_speakers(
                (u["speaker_id"] for u in utterances),
                seed=20260805,
                val_fraction=0.2,
                test_fraction=0.2,
            )
            plan = _plan_pairs(utterances, speaker_splits, pairs=15, seed=20260805)

            self.assertGreater(len(plan), 0)
            found_multi_interferer = False
            for spec in plan:
                infos = spec["interferer_infos"]
                target_spk = spec["target_info"]["speaker_id"]
                interferer_speakers = [ii["speaker_id"] for ii in infos]
                # Target must never be an interferer.
                self.assertNotIn(
                    target_spk, interferer_speakers,
                    f"Pair {spec['pair_id']}: target speaker is also an interferer",
                )
                if len(infos) > 1:
                    found_multi_interferer = True
                    self.assertEqual(
                        len(set(interferer_speakers)), len(infos),
                        f"Pair {spec['pair_id']}: interferer speakers are not "
                        f"distinct: {interferer_speakers}",
                    )
            self.assertTrue(
                found_multi_interferer,
                "Expected at least one pair with >1 interferer to verify distinctness",
            )

    def test_manifest_distinct_interferer_speakers(self):
        """The rendered manifest must carry interferer utterances from distinct
        speakers whenever a pair has more than one interferer.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_fake_aishell(root, speaker_count=20)
            rir_root, noise_root = self.make_assets(root)
            output = root / "out"
            build_r3_manifest(
                corpus, output, dataset_a_root=root / "da",
                pairs=12, seed=20260805, rir_root=rir_root, noise_root=noise_root,
            )
            rows = read_r3_manifest(output / "manifest.jsonl")
            # Build utterance_id -> speaker_id map from the discovered corpus.
            utt_to_speaker = {
                u["utterance_id"]: u["speaker_id"]
                for u in discover_aishell_utterances(corpus)
            }
            found_multi = False
            for row in rows:
                int_spk_ids = [
                    utt_to_speaker[iid] for iid in row.interferer_source_ids
                ]
                # Target speaker must never appear among interferers.
                target_spk = utt_to_speaker.get(row.target_source_id)
                self.assertNotIn(
                    target_spk, int_spk_ids,
                    f"Row {row.row_id}: target speaker appears among interferers",
                )
                if len(int_spk_ids) > 1:
                    found_multi = True
                    self.assertEqual(
                        len(set(int_spk_ids)), len(int_spk_ids),
                        f"Row {row.row_id}: interferer speakers are not distinct: "
                        f"{int_spk_ids}",
                    )
            self.assertTrue(
                found_multi,
                "Expected at least one row with >1 interferer to verify distinctness",
            )


if __name__ == "__main__":
    unittest.main()
