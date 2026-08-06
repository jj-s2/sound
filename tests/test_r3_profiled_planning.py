"""Tests for profile-aware public R3 pair planning (Task 3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.prepare_r3_counterfactual import (
    _discover_durations,
    _plan_pairs,
    build_r3_manifest,
    parse_args,
)
from scripts.prepare_phase2_synthetic import discover_aishell_utterances, split_speakers
from xh202615.acoustic_profile import profile_audio_paths, write_aggregate_profile
from xh202615.r3_data import assert_r3_manifest_safe, read_r3_manifest

DURATIONS = (0.2, 0.5, 1.0, 2.0)


class R3ProfiledPlanningTests(unittest.TestCase):
    SR = 16_000

    def make_varied_aishell(self, root: Path, speaker_count: int = 6) -> Path:
        corpus = root / "aishell"
        transcript = corpus / "transcript" / "aishell_transcript_v0.8.txt"
        transcript.parent.mkdir(parents=True)
        lines = []
        for speaker_index in range(1, speaker_count + 1):
            speaker_id = f"S{speaker_index:04d}"
            speaker_dir = corpus / "wav" / "train" / speaker_id
            speaker_dir.mkdir(parents=True)
            for utterance_index, dur in enumerate(DURATIONS, start=1):
                utterance_id = f"{speaker_id}{utterance_index:04d}"
                n = int(self.SR * dur)
                t = np.arange(n) / self.SR
                tone = 0.1 * np.sin(
                    2.0 * np.pi * (180 + speaker_index * 30 + utterance_index) * t
                )
                sf.write(str(speaker_dir / f"{utterance_id}.wav"), tone, self.SR)
                lines.append(f"{utterance_id}\t文本 {utterance_id}")
        transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return corpus

    def make_assets(self, root: Path, count: int = 6) -> tuple[Path, Path]:
        rir_root = root / "rir_assets"
        noise_root = root / "noise_assets"
        rir_root.mkdir(parents=True)
        noise_root.mkdir(parents=True)
        impulse = np.zeros(160, dtype=np.float32)
        impulse[3] = 0.8
        for index in range(count):
            sf.write(str(rir_root / f"rir_{index:02d}.wav"), impulse, self.SR)
            n = 800 + index * 10
            t = np.arange(n) / self.SR
            sf.write(str(noise_root / f"noise_{index:02d}.wav"),
                     0.02 * np.sin(2.0 * np.pi * (80 + index * 10) * t), self.SR)
        return rir_root, noise_root

    def _write_target_profile(self, root: Path, target_duration: float, n: int = 4) -> Path:
        wavs = []
        for i in range(n):
            wav = root / f"p_{i}.wav"
            length = int(self.SR * target_duration)
            t = np.arange(length) / self.SR
            sf.write(str(wav), (0.1 * np.sin(2 * np.pi * (300 + i * 50) * t)).astype(np.float32), self.SR)
            wavs.append(wav)
        path = root / "profile.json"
        write_aggregate_profile(path, profile_audio_paths(wavs))
        return path

    def _utt_durations(self, corpus: Path) -> dict[str, float]:
        out = {}
        for u in discover_aishell_utterances(corpus):
            info = sf.info(str(u["audio_path"]))
            out[u["utterance_id"]] = info.frames / info.samplerate
        return out

    # ----------------------------------------------------------- CLI parsing

    def test_parse_args_acoustic_profile_optional(self):
        args = parse_args(
            ["--aishell-root", "c", "--output-root", "o", "--dataset-a-root", "d"]
        )
        self.assertIsNone(args.acoustic_profile)
        args2 = parse_args(
            ["--aishell-root", "c", "--output-root", "o", "--dataset-a-root", "d",
             "--acoustic-profile", "profile.json"]
        )
        self.assertEqual(args2.acoustic_profile, "profile.json")

    # --------------------------------------------------- metadata + no-change

    def test_profile_metadata_fields_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_varied_aishell(root)
            rir_root, noise_root = self.make_assets(root)
            profile = self._write_target_profile(root, target_duration=2.0)
            output = root / "out"
            build_r3_manifest(
                corpus, output, dataset_a_root=root / "da",
                pairs=3, seed=7, rir_root=rir_root, noise_root=noise_root,
                acoustic_profile=profile,
            )
            meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertIn("acoustic_profile_hash", meta)
            self.assertEqual(len(meta["acoustic_profile_hash"]), 64)
            self.assertIn("duration_target_seconds", meta)
            self.assertGreater(meta["duration_target_seconds"], 0.0)
            self.assertIn("profiled_source_duration_seconds", meta)
            self.assertGreater(meta["profiled_source_duration_seconds"], 0.0)

    def test_no_profile_metadata_unchanged_and_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_varied_aishell(root)
            rir_root, noise_root = self.make_assets(root)
            output = root / "out"
            build_r3_manifest(
                corpus, output, dataset_a_root=root / "da",
                pairs=3, seed=7, rir_root=rir_root, noise_root=noise_root,
            )
            meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertNotIn("acoustic_profile_hash", meta)
            self.assertNotIn("duration_target_seconds", meta)
            self.assertNotIn("profiled_source_duration_seconds", meta)

    def test_default_equals_explicit_none_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_varied_aishell(root)
            rir_root, noise_root = self.make_assets(root)
            a = build_r3_manifest(corpus, root / "a", dataset_a_root=root / "da",
                                  pairs=4, seed=11, rir_root=rir_root, noise_root=noise_root)
            b = build_r3_manifest(corpus, root / "b", dataset_a_root=root / "da",
                                  pairs=4, seed=11, rir_root=rir_root, noise_root=noise_root,
                                  acoustic_profile=None)

            def normalized(rows):
                payload = []
                for row in rows:
                    d = row.to_dict()
                    for key in ("enrollment_audio", "mixture_audio", "clean_target_audio"):
                        d[key] = Path(d[key]).name
                    payload.append(d)
                return payload

            self.assertEqual(normalized(a["rows"]), normalized(b["rows"]))

    # ----------------------------------------------- duration-matched planning

    def test_plan_picks_longest_utterances_for_long_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_varied_aishell(root, speaker_count=8)
            utterances = discover_aishell_utterances(corpus)
            speaker_splits = split_speakers(
                (u["speaker_id"] for u in utterances), seed=20260805,
                val_fraction=0.2, test_fraction=0.2,
            )
            # Build a real profile object targeting ~2.0s.
            wavs = []
            for i in range(4):
                w = root / f"tgt_{i}.wav"
                length = int(self.SR * 2.0)
                t = np.arange(length) / self.SR
                sf.write(str(w), (0.1 * np.sin(2 * np.pi * (300 + i * 40) * t)).astype(np.float32), self.SR)
                wavs.append(w)
            profile = profile_audio_paths(wavs)
            durations = {
                str(u["audio_path"]): sf.info(str(u["audio_path"])).frames / self.SR
                for u in utterances
            }
            plan = _plan_pairs(
                utterances, speaker_splits, pairs=6, seed=20260805,
                acoustic_profile=profile, durations=durations,
            )
            self.assertGreater(len(plan), 0)
            utt_dur = {u["utterance_id"]: durations[str(u["audio_path"])] for u in utterances}
            for spec in plan:
                chosen = utt_dur[spec["target_info"]["utterance_id"]]
                # Closest available duration to a 2.0s target is the 2.0s utterance.
                self.assertAlmostEqual(chosen, 2.0, places=2,
                                       msg=f"{spec['pair_id']} chose {chosen}s")

    def test_plan_picks_shortest_utterances_for_short_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_varied_aishell(root, speaker_count=8)
            utterances = discover_aishell_utterances(corpus)
            speaker_splits = split_speakers(
                (u["speaker_id"] for u in utterances), seed=20260805,
                val_fraction=0.2, test_fraction=0.2,
            )
            wavs = []
            for i in range(4):
                w = root / f"tgt_{i}.wav"
                length = int(self.SR * 0.2)
                t = np.arange(length) / self.SR
                sf.write(str(w), (0.1 * np.sin(2 * np.pi * (300 + i * 40) * t)).astype(np.float32), self.SR)
                wavs.append(w)
            profile = profile_audio_paths(wavs)
            durations = {
                str(u["audio_path"]): sf.info(str(u["audio_path"])).frames / self.SR
                for u in utterances
            }
            plan = _plan_pairs(
                utterances, speaker_splits, pairs=6, seed=20260805,
                acoustic_profile=profile, durations=durations,
            )
            utt_dur = {u["utterance_id"]: durations[str(u["audio_path"])] for u in utterances}
            for spec in plan:
                self.assertAlmostEqual(
                    utt_dur[spec["target_info"]["utterance_id"]], 0.2, places=2
                )

    # ----------------------------------------------- invariants + provenance

    def test_profiled_manifest_is_valid_and_public_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_varied_aishell(root, speaker_count=10)
            rir_root, noise_root = self.make_assets(root)
            profile = self._write_target_profile(root, target_duration=1.0)
            output = root / "out"
            dataset_a_root = root / "datasetA"
            build_r3_manifest(
                corpus, output, dataset_a_root=dataset_a_root,
                pairs=8, seed=20260805, rir_root=rir_root, noise_root=noise_root,
                acoustic_profile=profile,
            )
            rows = read_r3_manifest(output / "manifest.jsonl")
            assert_r3_manifest_safe(rows, dataset_a_root=dataset_a_root)
            # Every audio path is public (under the public corpus or assets), never Dataset-A.
            public_roots = {corpus.resolve(), rir_root.resolve(), noise_root.resolve(), output.resolve()}
            da = dataset_a_root.resolve()
            for row in rows:
                for field in ("enrollment_audio", "mixture_audio", "clean_target_audio"):
                    p = getattr(row, field).resolve()
                    self.assertFalse(
                        p == da or da in p.parents,
                        f"{field} for {row.row_id} resolves under Dataset-A",
                    )
                    self.assertTrue(
                        any(p == pr or pr in p.parents for pr in public_roots),
                        f"{field} for {row.row_id} is not under a public root",
                    )
            # Pairs are well-formed.
            pair_ids = {r.pair_id for r in rows}
            for pid in pair_ids:
                members = [r for r in rows if r.pair_id == pid]
                self.assertEqual(len(members), 2)
                self.assertEqual({m.target_present for m in members}, {True, False})

    def test_profiled_manifest_disjoint_across_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_varied_aishell(root, speaker_count=12)
            rir_root, noise_root = self.make_assets(root)
            profile = self._write_target_profile(root, target_duration=1.0)
            output = root / "out"
            build_r3_manifest(
                corpus, output, dataset_a_root=root / "da",
                pairs=10, seed=5, rir_root=rir_root, noise_root=noise_root,
                acoustic_profile=profile,
            )
            rows = read_r3_manifest(output / "manifest.jsonl")
            by_split = {"train": set(), "val": set(), "test": set()}
            for row in rows:
                entities = {row.target_source_id, row.noise_source_id,
                            row.target_rir_id, row.renderer_family}
                entities.update(row.interferer_source_ids)
                entities.update(row.interferer_rir_ids)
                entities.discard(None)
                by_split[row.split].update(entities)
            self.assertFalse(by_split["train"] & by_split["val"])
            self.assertFalse(by_split["train"] & by_split["test"])
            self.assertFalse(by_split["val"] & by_split["test"])

    def test_profiled_planning_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_varied_aishell(root, speaker_count=8)
            rir_root, noise_root = self.make_assets(root)
            profile = self._write_target_profile(root, target_duration=1.0)
            first = build_r3_manifest(corpus, root / "a", dataset_a_root=root / "da",
                                      pairs=5, seed=33, rir_root=rir_root, noise_root=noise_root,
                                      acoustic_profile=profile)
            second = build_r3_manifest(corpus, root / "b", dataset_a_root=root / "da",
                                       pairs=5, seed=33, rir_root=rir_root, noise_root=noise_root,
                                       acoustic_profile=profile)

            def normalized(rows):
                payload = []
                for row in rows:
                    d = row.to_dict()
                    for key in ("enrollment_audio", "mixture_audio", "clean_target_audio"):
                        d[key] = Path(d[key]).name
                    payload.append(d)
                return payload

            self.assertEqual(normalized(first["rows"]), normalized(second["rows"]))

    # ----------------------------------------------------- duration discovery

    def test_discover_durations_pcm16_and_float(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pcm16 = root / "pcm.wav"
            flt = root / "flt.wav"
            n = int(self.SR * 0.625)
            t = np.arange(n) / self.SR
            tone = 0.1 * np.sin(2 * np.pi * 440.0 * t)
            # PCM_16 subtype exercises the fast wave-header path.
            sf.write(str(pcm16), tone, self.SR, subtype="PCM_16")
            # Default float subtype exercises the soundfile.info fallback path.
            sf.write(str(flt), tone.astype(np.float32), self.SR)
            utterances = [
                {"utterance_id": "pcm", "audio_path": pcm16},
                {"utterance_id": "flt", "audio_path": flt},
            ]
            durations = _discover_durations(utterances)
            self.assertAlmostEqual(durations[str(pcm16)], 0.625, places=3)
            self.assertAlmostEqual(durations[str(flt)], 0.625, places=3)

    # ----------------------------------------------------- profile validation

    def _valid_profile_payload(self, root: Path) -> dict:
        profile = self._write_target_profile(root, target_duration=1.0)
        return json.loads(profile.read_text(encoding="utf-8"))

    def test_rejects_profile_with_paths_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_varied_aishell(root)
            payload = self._valid_profile_payload(root)
            payload["paths"] = ["/secret/datasetA/file.wav"]
            bad = root / "bad.json"
            bad.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prohibited|non-aggregate"):
                build_r3_manifest(corpus, root / "out", dataset_a_root=root / "da",
                                  pairs=2, seed=1, acoustic_profile=bad)
            self.assertFalse((root / "out").exists())

    def test_rejects_profile_with_label_in_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_varied_aishell(root)
            payload = self._valid_profile_payload(root)
            payload["config"]["label"] = "secret"
            bad = root / "bad.json"
            bad.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prohibited|non-aggregate"):
                build_r3_manifest(corpus, root / "out", dataset_a_root=root / "da",
                                  pairs=2, seed=1, acoustic_profile=bad)

    def test_rejects_profile_with_sample_level_metric_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = self.make_varied_aishell(root)
            payload = self._valid_profile_payload(root)
            payload["metrics"]["duration"]["values"] = [0.1, 0.2, 0.3]
            bad = root / "bad.json"
            bad.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-aggregate"):
                build_r3_manifest(corpus, root / "out", dataset_a_root=root / "da",
                                  pairs=2, seed=1, acoustic_profile=bad)


if __name__ == "__main__":
    unittest.main()
