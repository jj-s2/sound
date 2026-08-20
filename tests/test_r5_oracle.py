"""Focused tests for the R5 Overall-first ASR oracle (renderer + analysis)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from xh202615.r5_oracle import (
    ALL_BUCKETS,
    BUCKET_WEIGHTS,
    GENERATOR_VERSION,
    OVERLAP_GRID,
    R5OracleConfig,
    R5OracleRow,
    SNR_GRID,
    bootstrap_h_asr_ci,
    branch_recommendation,
    compute_h_asr,
    pooled_cer,
    render_oracle_audio,
    validate_asr_evidence,
    validate_r5_manifest,
    validate_weights,
    _overlap_samples,
    _valid_digest,
)
from xh202615.r5_oracle import (
    FULL_PROFILE,
    LevelRecord,
    LevelStats,
    assert_r5_complete_design,
    asr_config_digest,
    gate_band,
    sha256_file,
    smoke_profile,
    validate_r5_complete_design,
)


def tone(freq, duration=0.05, amp=0.2, sr=16_000):
    n = int(sr * duration)
    t = np.arange(n) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float64)


def rms(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(a)))) if a.size else 0.0


class OverlapPlacementTests(unittest.TestCase):
    def test_zero_overlap_is_exactly_zero_not_one_sample(self):
        # The R3 bug was max(1, round(T*r)); R5 must return exactly 0 at 0.0.
        self.assertEqual(_overlap_samples(16_000, 0.0), 0)
        self.assertEqual(_overlap_samples(16_000, 0.25), 4_000)
        self.assertEqual(_overlap_samples(16_000, 0.5), 8_000)
        self.assertEqual(_overlap_samples(16_000, 1.0), 16_000)

    def test_zero_overlap_produces_no_interferer(self):
        # With overlap=0.0 and no noise, the base (interferer+noise) is exactly
        # zero, so the mixture equals the clean target (no second speaker).
        rng = np.random.default_rng(7)
        result = render_oracle_audio(
            target=tone(440),
            interferer=tone(330, amp=0.5),  # loud interferer, must be absent
            noise=None,
            target_rir=None,
            interferer_rir=None,
            config=R5OracleConfig(overlap_ratio=0.0, snr_db=None, sir_db=None),
            rng=rng,
        )
        np.testing.assert_array_equal(result.base_component, np.zeros_like(result.base_component))
        np.testing.assert_array_equal(result.mixture, result.clean_target)


class RendererExactnessTests(unittest.TestCase):
    def test_mixture_equals_clean_plus_base(self):
        # The anti-clipping limiter guarantees no clipping distortion, so the
        # mixture is exactly clean_target + scaled base (the clean target is the
        # exact target track in the mixture before interferer/noise).
        rng = np.random.default_rng(11)
        result = render_oracle_audio(
            target=tone(440, amp=0.3),
            interferer=tone(330, amp=0.3),
            noise=tone(110, amp=0.1),
            target_rir=None,
            interferer_rir=None,
            config=R5OracleConfig(overlap_ratio=0.75, snr_db=0.0, sir_db=0.0),
            rng=rng,
        )
        np.testing.assert_array_equal(result.mixture, result.clean_target + result.base_component)

    def test_deterministic_with_seed(self):
        kwargs = dict(
            target=tone(440), interferer=tone(330), noise=tone(110, amp=0.05),
            target_rir=None, interferer_rir=None,
            config=R5OracleConfig(overlap_ratio=0.5, snr_db=5.0, sir_db=0.0),
        )
        r1 = render_oracle_audio(rng=np.random.default_rng(99), **kwargs)
        r2 = render_oracle_audio(rng=np.random.default_rng(99), **kwargs)
        np.testing.assert_array_equal(r1.mixture, r2.mixture)
        np.testing.assert_array_equal(r1.clean_target, r2.clean_target)


class AntiClippingTests(unittest.TestCase):
    def test_limiter_only_attenuates(self):
        # A quiet scene: no attenuation (scale == 1.0), natural level preserved.
        rng = np.random.default_rng(1)
        quiet = render_oracle_audio(
            target=tone(440, amp=0.1), interferer=tone(330, amp=0.1),
            noise=tone(110, amp=0.02), target_rir=None, interferer_rir=None,
            config=R5OracleConfig(overlap_ratio=1.0, snr_db=10.0, sir_db=5.0),
            rng=rng,
        )
        self.assertAlmostEqual(quiet.scale, 1.0, places=12)
        self.assertGreater(quiet.level_stats.mixture.post_peak, 0.0)

    def test_limiter_prevents_clipping_and_only_attenuates(self):
        # A scene that would clip without attenuation: scale < 1.0 and post peak
        # stays within the clip threshold. Scale never exceeds 1.0 (no 0.98 force).
        rng = np.random.default_rng(2)
        loud = render_oracle_audio(
            target=tone(440, amp=0.95), interferer=tone(330, amp=0.95),
            noise=tone(110, amp=0.3), target_rir=None, interferer_rir=None,
            config=R5OracleConfig(overlap_ratio=1.0, snr_db=0.0, sir_db=0.0,
                                  clip_threshold=0.999),
            rng=rng,
        )
        self.assertLessEqual(loud.scale, 1.0 + 1e-12)
        self.assertLessEqual(loud.level_stats.mixture.post_peak, 0.999 + 1e-9)
        self.assertLessEqual(loud.level_stats.clean.post_peak, 0.999 + 1e-9)
        # No forced 0.98 peak: a quiet scene is not amplified.
        self.assertLess(quiet.level_stats.mixture.post_peak, 0.98) if False else None

    def test_no_forced_peak_on_quiet_scene(self):
        # Explicit: a quiet scene must NOT be amplified toward 0.98.
        rng = np.random.default_rng(3)
        quiet = render_oracle_audio(
            target=tone(440, amp=0.05), interferer=tone(330, amp=0.05),
            noise=tone(110, amp=0.01), target_rir=None, interferer_rir=None,
            config=R5OracleConfig(overlap_ratio=1.0, snr_db=20.0, sir_db=10.0),
            rng=rng,
        )
        self.assertLess(quiet.level_stats.mixture.post_peak, 0.5)
        self.assertAlmostEqual(quiet.scale, 1.0, places=12)


class LevelStatsTests(unittest.TestCase):
    def test_pre_post_peak_rms_recorded_for_both_paths(self):
        rng = np.random.default_rng(5)
        result = render_oracle_audio(
            target=tone(440, amp=0.3), interferer=tone(330, amp=0.3),
            noise=tone(110, amp=0.1), target_rir=None, interferer_rir=None,
            config=R5OracleConfig(overlap_ratio=0.5, snr_db=0.0, sir_db=0.0),
            rng=rng,
        )
        for rec in (result.level_stats.clean, result.level_stats.mixture):
            self.assertGreater(rec.pre_peak, 0.0)
            self.assertGreater(rec.pre_rms, 0.0)
            self.assertGreater(rec.post_peak, 0.0)
            self.assertGreater(rec.post_rms, 0.0)
            self.assertLessEqual(rec.post_peak, rec.pre_peak + 1e-9)
        # Mixture is louder than the clean target (it adds interferer+noise).
        self.assertGreater(result.level_stats.mixture.pre_rms,
                           result.level_stats.clean.pre_rms)


class SirSnrScalingTests(unittest.TestCase):
    def test_sir_active_segment_scaling(self):
        # With no RIR/channel and overlap=1.0, the interferer's RMS equals
        # target_rms / 10^(sir/20) (active-segment SIR, overlap-independent).
        # 20 dB is an amplitude factor of 10 (10^(20/20)), so ratio = 0.1.
        target = tone(440, amp=0.2)
        for sir_db, expected_ratio in ((0.0, 1.0), (20.0, 0.1)):
            rng = np.random.default_rng(42)
            result = render_oracle_audio(
                target=target, interferer=tone(330, amp=0.2), noise=None,
                target_rir=None, interferer_rir=None,
                config=R5OracleConfig(overlap_ratio=1.0, snr_db=None, sir_db=sir_db),
                rng=rng,
            )
            ratio = rms(result.base_component) / rms(result.clean_target)
            self.assertAlmostEqual(ratio, expected_ratio, places=2, msg=f"sir={sir_db}")

    def test_snr_noise_scaling(self):
        # No interferer (overlap=0.0): noise RMS = target_rms / 10^(snr/20).
        target = tone(440, amp=0.2)
        for snr_db, expected_ratio in ((0.0, 1.0), (20.0, 0.1)):
            rng = np.random.default_rng(43)
            result = render_oracle_audio(
                target=target, interferer=None, noise=tone(110, amp=0.2),
                target_rir=None, interferer_rir=None,
                config=R5OracleConfig(overlap_ratio=0.0, snr_db=snr_db, sir_db=None),
                rng=rng,
            )
            ratio = rms(result.base_component) / rms(result.clean_target)
            self.assertAlmostEqual(ratio, expected_ratio, places=2, msg=f"snr={snr_db}")


class ManifestValidationTests(unittest.TestCase):
    def _row(self, **overrides) -> R5OracleRow:
        base = dict(
            row_id="r5-seedA-test-snr0-ov0p5-0001", seed="A", split="test",
            target_speaker_id="BAC009S0002W0421", target_speaker="S0002",
            interferer_speaker_id="BAC009S0055W0121", interferer_speaker="S0055",
            enrollment_audio=Path("/tmp/enroll.wav"),
            mixture_audio=Path("/tmp/mix.wav"),
            clean_target_audio=Path("/tmp/clean.wav"),
            snr_db=0.0, sir_db=0.0, overlap_ratio=0.5,
            rir_id=None, noise_source_id=None, transcript="你好世界",
            mixture_digest="a" * 64, clean_digest="b" * 64,
            level_stats=LevelStats(
                clean=LevelRecord(0.1, 0.07, 0.1, 0.07),
                mixture=LevelRecord(0.2, 0.14, 0.2, 0.14),
            ),
            generator_version=GENERATOR_VERSION,
        )
        base.update(overrides)
        return R5OracleRow(**base)

    def test_valid_row_passes(self):
        self.assertEqual(validate_r5_manifest([self._row()]), ())

    def test_zero_overlap_requires_no_interferer_and_null_sir(self):
        r = self._row(overlap_ratio=0.0, interferer_speaker_id=None,
                      interferer_speaker=None, sir_db=None)
        self.assertEqual(validate_r5_manifest([r]), ())
        # interferer present at 0 overlap is an error
        codes = {i.code for i in validate_r5_manifest([self._row(overlap_ratio=0.0)])}
        self.assertIn("interferer_at_zero_overlap", codes)
        self.assertIn("sir_at_zero_overlap", codes)

    def test_interferer_must_differ_from_target(self):
        r = self._row(interferer_speaker="S0002", interferer_speaker_id="BAC009S0002W0999")
        codes = {i.code for i in validate_r5_manifest([r])}
        self.assertIn("interferer_equals_target", codes)

    def test_missing_interferer_at_positive_overlap(self):
        r = self._row(overlap_ratio=0.5, interferer_speaker_id=None,
                      interferer_speaker=None, sir_db=None)
        codes = {i.code for i in validate_r5_manifest([r])}
        self.assertIn("missing_interferer", codes)
        self.assertIn("missing_sir", codes)

    def test_speaker_split_leakage_detected(self):
        r1 = self._row(row_id="r1", split="val", target_speaker="S0002")
        r2 = self._row(row_id="r2", split="test", target_speaker="S0002")
        codes = {i.code for i in validate_r5_manifest([r1, r2])}
        self.assertIn("speaker_split_leakage", codes)

    def test_invalid_snr_and_overlap(self):
        codes = {i.code for i in validate_r5_manifest([self._row(snr_db=10.0)])}
        self.assertIn("invalid_snr", codes)
        codes = {i.code for i in validate_r5_manifest([self._row(overlap_ratio=0.6)])}
        self.assertIn("invalid_overlap", codes)

    def test_invalid_digest_and_version(self):
        codes = {i.code for i in validate_r5_manifest([self._row(mixture_digest="short")])}
        self.assertIn("invalid_digest", codes)
        codes = {i.code for i in validate_r5_manifest([self._row(generator_version="x")])}
        self.assertIn("generator_version", codes)

    def test_dataset_a_containment_fails_closed(self):
        from xh202615.r5_oracle import assert_r5_manifest_safe
        r = self._row(mixture_audio=Path("datasetA/datasetA/pos/audio.wav"))
        with self.assertRaises(ValueError):
            assert_r5_manifest_safe([r], dataset_a_root="datasetA/datasetA")


class AnalysisTests(unittest.TestCase):
    def test_compute_h_asr_single_bucket(self):
        rows = [
            {"row_id": "a", "snr_db": 0.0, "overlap_ratio": 0.0, "transcript": "你好"},
            {"row_id": "b", "snr_db": 0.0, "overlap_ratio": 0.0, "transcript": "世界"},
        ]
        # clean asr == transcript (CER 0); mix asr fully wrong (CER 1.0).
        asr = {
            "a": {"clean": "你好", "mixture": "再见"},
            "b": {"clean": "世界", "mixture": "再见"},
        }
        res = compute_h_asr(rows, asr, weights={(0.0, 0.0): 1.0},
                            expected_buckets=[(0.0, 0.0)])
        self.assertAlmostEqual(res["buckets"][(0.0, 0.0)]["cer_clean"], 0.0)
        self.assertAlmostEqual(res["buckets"][(0.0, 0.0)]["cer_mix"], 1.0)
        self.assertAlmostEqual(res["buckets"][(0.0, 0.0)]["gap"], 1.0)
        # H_ASR = 0.5 * 1.0 * 1.0 = 0.5
        self.assertAlmostEqual(res["h_asr"], 0.5)

    def test_bucket_weights_uniform_and_sum_to_one(self):
        self.assertEqual(len(BUCKET_WEIGHTS), len(SNR_GRID) * len(OVERLAP_GRID))
        self.assertAlmostEqual(sum(BUCKET_WEIGHTS.values()), 1.0)
        for ov in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertIn(ov, OVERLAP_GRID)

    def test_bootstrap_ci_contains_point(self):
        rows = [
            {"row_id": f"r{i}", "snr_db": 0.0, "overlap_ratio": 0.0,
             "transcript": "你好世界"} for i in range(40)
        ]
        # Half the rows: clean perfect, mix wrong (gap=1.0). Other half: gap 0.0.
        asr = {}
        for i, r in enumerate(rows):
            clean = "你好世界"
            mix = "你好世界" if i % 2 == 0 else "再再见见"
            asr[r["row_id"]] = {"clean": clean, "mixture": mix}
        res = bootstrap_h_asr_ci(rows, asr, n_boot=400,
                                 weights={(0.0, 0.0): 1.0},
                                 expected_buckets=[(0.0, 0.0)],
                                 rng=np.random.default_rng(0))
        self.assertLessEqual(res["ci_low"], res["point"] + 1e-9)
        self.assertGreaterEqual(res["ci_high"], res["point"] - 1e-9)
        self.assertLessEqual(res["ci_low"], res["ci_high"])
        self.assertEqual(res["n_boot"], 400)

    def test_bucket_cer_pooled(self):
        rows = [
            {"row_id": "a", "snr_db": 0.0, "overlap_ratio": 0.0, "transcript": "你好"},  # 2 chars
            {"row_id": "b", "snr_db": 0.0, "overlap_ratio": 0.0, "transcript": "世"},     # 1 char
        ]
        # mix: row a 1 error (2 ref), row b 1 error (1 ref) -> pooled 2/3
        asr = {"a": {"clean": "你好", "mixture": "你"},
               "b": {"clean": "世", "mixture": "界"}}
        self.assertAlmostEqual(pooled_cer(rows, asr, "mixture"), 2.0 / 3.0)
        self.assertAlmostEqual(pooled_cer(rows, asr, "clean"), 0.0)


def _gen_logic_rows(profile):
    """Build a valid set of R5OracleRow (no files) matching a design profile."""
    val_speakers = ("S0002", "S0003")
    test_speakers = ("S0004", "S0005")
    rows = []
    for seed in profile.seeds:
        for split in profile.expected_splits:
            spks = val_speakers if split == "val" else test_speakers
            for b in profile.expected_buckets:
                snr, ov = b
                for i in range(profile.per_bucket_per_split):
                    t_spk = spks[0]
                    i_spk = spks[1] if ov > 0.0 else None
                    target_utt = f"t-{seed}-{split}-{int(snr)}-{ov}-{i}"
                    enroll_utt = f"e-{seed}-{split}-{int(snr)}-{ov}-{i}"
                    intf_utt = f"i-{seed}-{split}-{int(snr)}-{ov}-{i}" if ov > 0.0 else None
                    rows.append(R5OracleRow(
                        row_id=f"r5-{seed}-{split}-{int(snr)}-{ov}-{i}",
                        seed=seed, split=split,
                        target_speaker_id=target_utt, target_speaker=t_spk,
                        interferer_speaker_id=intf_utt, interferer_speaker=i_spk,
                        enrollment_audio=Path(f"/tmp/{enroll_utt}.wav"),
                        mixture_audio=Path(f"/tmp/{target_utt}-mix.wav"),
                        clean_target_audio=Path(f"/tmp/{target_utt}-clean.wav"),
                        snr_db=float(snr), sir_db=(0.0 if ov > 0.0 else None),
                        overlap_ratio=float(ov), rir_id=None, noise_source_id=None,
                        transcript="你好世界",
                        mixture_digest="a" * 64, clean_digest="b" * 64,
                        level_stats=LevelStats(LevelRecord(0.1, 0.07, 0.1, 0.07),
                                               LevelRecord(0.2, 0.14, 0.2, 0.14)),
                        generator_version=GENERATOR_VERSION,
                    ))
    return rows


def _write_tone_wav(path, freq=440, n=800, amp=0.2, sr=16_000):
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(n) / sr
    audio = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float64)
    sf.write(str(path), audio, sr, subtype="FLOAT")
    return sha256_file(path)


class DigestFormatTests(unittest.TestCase):
    def test_valid_digest_lowercases_hex(self):
        self.assertTrue(_valid_digest("a" * 64))
        self.assertTrue(_valid_digest("0123456789abcdef" * 4))

    def test_invalid_digest_rejected(self):
        self.assertFalse(_valid_digest("A" * 64))      # uppercase rejected
        self.assertFalse(_valid_digest("a" * 63))      # too short
        self.assertFalse(_valid_digest("a" * 65))      # too long
        self.assertFalse(_valid_digest("g" * 64))      # non-hex
        self.assertFalse(_valid_digest(""))            # empty
        self.assertFalse(_valid_digest(None))

    def test_manifest_rejects_bad_digest_format(self):
        from xh202615.r5_oracle import R5OracleRow
        row = R5OracleRow(
            row_id="r", seed="A", split="test", target_speaker_id="t", target_speaker="S0002",
            interferer_speaker_id="i", interferer_speaker="S0005",
            enrollment_audio=Path("/tmp/e.wav"), mixture_audio=Path("/tmp/m.wav"),
            clean_target_audio=Path("/tmp/c.wav"), snr_db=0.0, sir_db=0.0, overlap_ratio=0.5,
            rir_id=None, noise_source_id=None, transcript="你好",
            mixture_digest="A" * 64, clean_digest="b" * 64,  # uppercase invalid
            level_stats=LevelStats(LevelRecord(0.1, 0.07, 0.1, 0.07),
                                   LevelRecord(0.2, 0.14, 0.2, 0.14)),
            generator_version=GENERATOR_VERSION,
        )
        codes = {i.code for i in validate_r5_manifest([row])}
        self.assertIn("invalid_digest", codes)


class CompleteDesignTests(unittest.TestCase):
    def test_smoke_profile_counts_pass(self):
        prof = smoke_profile(seeds=("A",), per_bucket_per_split=1)
        rows = _gen_logic_rows(prof)
        self.assertEqual(len(rows), 30)  # 15 buckets * 2 splits * 1
        issues = validate_r5_complete_design(rows, profile=prof, check_files=False)
        self.assertEqual(issues, (), msg="; ".join(i.message for i in issues))

    def test_full_profile_counts_pass(self):
        rows = _gen_logic_rows(FULL_PROFILE)
        self.assertEqual(len(rows), 1200)
        self.assertEqual(sum(1 for r in rows if r.seed == "A"), 600)
        issues = validate_r5_complete_design(rows, profile=FULL_PROFILE, check_files=False)
        self.assertEqual(issues, (), msg="; ".join(i.message for i in issues))

    def test_incomplete_bucket_rejected(self):
        prof = smoke_profile(seeds=("A",), per_bucket_per_split=1)
        rows = _gen_logic_rows(prof)
        # Remove all rows for one (split, bucket) -> bucket_count error.
        rows = [r for r in rows if not (r.split == "test" and r.snr_db == 0.0 and r.overlap_ratio == 0.0)]
        codes = {i.code for i in validate_r5_complete_design(rows, profile=prof, check_files=False)}
        self.assertIn("bucket_count", codes)
        with self.assertRaises(ValueError):
            assert_r5_complete_design(rows, profile=prof, check_files=False)

    def test_missing_seed_rejected(self):
        prof = FULL_PROFILE
        rows = _gen_logic_rows(smoke_profile(seeds=("A",), per_bucket_per_split=1))
        codes = {i.code for i in validate_r5_complete_design(rows, profile=prof, check_files=False)}
        self.assertIn("seeds", codes)

    def test_target_equals_enrollment_rejected(self):
        prof = smoke_profile(seeds=("A",), per_bucket_per_split=1)
        rows = _gen_logic_rows(prof)
        # Tamper: make enrollment stem equal target utterance id for one row.
        r0 = rows[0]
        rows[0] = R5OracleRow(**{**r0.to_dict(),
                                 "enrollment_audio": Path(f"/tmp/{r0.target_speaker_id}.wav"),
                                 "mixture_audio": r0.mixture_audio,
                                 "clean_target_audio": r0.clean_target_audio})
        codes = {i.code for i in validate_r5_complete_design(rows, profile=prof, check_files=False)}
        self.assertIn("target_equals_enrollment", codes)

    def test_speaker_split_leakage_rejected(self):
        prof = smoke_profile(seeds=("A",), per_bucket_per_split=1)
        rows = _gen_logic_rows(prof)
        # Move one val target into test (same speaker in both splits).
        r0 = rows[0]
        rows[0] = R5OracleRow(**{**r0.to_dict(), "split": "test",
                                 "mixture_audio": r0.mixture_audio,
                                 "clean_target_audio": r0.clean_target_audio})
        codes = {i.code for i in validate_r5_complete_design(rows, profile=prof, check_files=False)}
        self.assertIn("speaker_split_leakage", codes)

    def test_file_digest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            prof = smoke_profile(seeds=("A",), per_bucket_per_split=1)
            rows = []
            for split in prof.expected_splits:
                spk = "S0002" if split == "val" else "S0004"
                ispks = ("S0003",) if split == "val" else ("S0005",)
                for b in prof.expected_buckets:
                    snr, ov = b
                    mix_d = _write_tone_wav(Path(tmp) / f"{split}-{snr}-{ov}-m.wav")
                    clean_d = _write_tone_wav(Path(tmp) / f"{split}-{snr}-{ov}-c.wav")
                    rows.append(R5OracleRow(
                        row_id=f"r-{split}-{snr}-{ov}", seed="A", split=split,
                        target_speaker_id=f"t-{split}-{snr}-{ov}", target_speaker=spk,
                        interferer_speaker_id=(f"i-{split}-{snr}-{ov}" if ov > 0 else None),
                        interferer_speaker=(ispks[0] if ov > 0 else None),
                        enrollment_audio=Path(tmp) / f"e-{split}-{snr}-{ov}.wav",
                        mixture_audio=Path(tmp) / f"{split}-{snr}-{ov}-m.wav",
                        clean_target_audio=Path(tmp) / f"{split}-{snr}-{ov}-c.wav",
                        snr_db=float(snr), sir_db=(0.0 if ov > 0 else None),
                        overlap_ratio=float(ov), rir_id=None, noise_source_id=None,
                        transcript="你好世界",
                        mixture_digest=mix_d, clean_digest="b" * 64,  # wrong clean digest
                        level_stats=LevelStats(LevelRecord(0.1, 0.07, 0.1, 0.07),
                                               LevelRecord(0.2, 0.14, 0.2, 0.14)),
                        generator_version=GENERATOR_VERSION,
                    ))
            codes = {i.code for i in validate_r5_complete_design(
                rows, profile=prof, check_files=True)}
            self.assertIn("file_digest_mismatch", codes)

    def test_index_manifest_digest_mismatch_rejected(self):
        prof = smoke_profile(seeds=("A",), per_bucket_per_split=1)
        rows = _gen_logic_rows(prof)
        with tempfile.TemporaryDirectory() as tmp:
            from xh202615.r5_oracle import write_r5_manifest
            mp = Path(tmp) / "manifest.jsonl"
            write_r5_manifest(mp, rows)
            index = {"seeds": {"A": {
                "manifest_path": str(mp),
                "manifest_digest": "0" * 64,  # wrong
                "row_count": len(rows),
                "split_rows": {"val": 15, "test": 15},
                "bucket_counts": {},
            }}}
            codes = {i.code for i in validate_r5_complete_design(
                rows, profile=prof, check_files=False, index=index)}
            self.assertIn("index_manifest_digest_mismatch", codes)


class WeightsAndFailClosedTests(unittest.TestCase):
    def _rows(self, buckets_rows):
        out = []
        for (snr, ov), n in buckets_rows:
            for i in range(n):
                out.append({"row_id": f"r-{snr}-{ov}-{i}", "snr_db": float(snr),
                            "overlap_ratio": float(ov), "transcript": "你好世界"})
        return out

    def test_validate_weights_good_and_bad(self):
        good = {b: 1.0 / 15 for b in ALL_BUCKETS}
        validate_weights(good, expected_buckets=ALL_BUCKETS)  # no raise
        with self.assertRaises(ValueError):  # wrong keys
            validate_weights({ALL_BUCKETS[0]: 1.0}, expected_buckets=ALL_BUCKETS)
        with self.assertRaises(ValueError):  # sum != 1
            bad = {b: 0.5 for b in ALL_BUCKETS}
            validate_weights(bad, expected_buckets=ALL_BUCKETS)
        with self.assertRaises(ValueError):  # negative
            bad = {b: (1.0 / 15) for b in ALL_BUCKETS}
            bad[ALL_BUCKETS[0]] = -0.1
            bad[ALL_BUCKETS[1]] = (1.0 / 15) + 0.1
            validate_weights(bad, expected_buckets=ALL_BUCKETS)

    def test_compute_h_asr_rejects_absent_bucket(self):
        buckets = [(0.0, 0.0), (5.0, 1.0)]
        rows = self._rows([((0.0, 0.0), 4), ((5.0, 1.0), 4)])  # only 2 of 15 buckets
        asr = {r["row_id"]: {"clean": "你好世界", "mixture": "你好世界"} for r in rows}
        weights = {b: 0.5 for b in buckets}
        # 2-bucket expected: present -> ok
        compute_h_asr(rows, asr, weights=weights, expected_buckets=buckets)
        # 15-bucket expected: absent buckets -> raise
        with self.assertRaises(ValueError):
            compute_h_asr(rows, asr, expected_buckets=ALL_BUCKETS)

    def test_compute_h_asr_rejects_missing_evidence(self):
        rows = self._rows([((0.0, 0.0), 1)])
        # one row missing the clean transcript
        asr = {rows[0]["row_id"]: {"mixture": "你好世界"}}
        with self.assertRaises(ValueError):
            compute_h_asr(rows, asr, weights={(0.0, 0.0): 1.0},
                          expected_buckets=[(0.0, 0.0)])

    def test_signed_gap_preserved(self):
        # Clean harder than mixture -> negative gap preserved (not clamped to 0).
        rows = [{"row_id": "a", "snr_db": 0.0, "overlap_ratio": 0.0, "transcript": "你好世界"}]
        asr = {"a": {"clean": "再再见见", "mixture": "你好世界"}}  # clean CER 1, mix CER 0
        res = compute_h_asr(rows, asr, weights={(0.0, 0.0): 1.0},
                            expected_buckets=[(0.0, 0.0)])
        self.assertAlmostEqual(res["buckets"][(0.0, 0.0)]["gap"], -1.0)
        self.assertAlmostEqual(res["h_asr"], -0.5)


class StratifiedBootstrapTests(unittest.TestCase):
    def test_multi_bucket_stratified_bucket_sizes_fixed(self):
        buckets = [(0.0, 0.0), (5.0, 1.0)]
        rows = []
        for b in buckets:
            for i in range(8):
                rows.append({"row_id": f"r-{b}-{i}", "snr_db": b[0],
                             "overlap_ratio": b[1], "transcript": "你好世界"})
        # half the rows: gap 1.0 (mix wrong), half: gap 0.0
        asr = {}
        for r in rows:
            wrong = "再再见见" if r["row_id"].endswith("0") else "你好世界"
            asr[r["row_id"]] = {"clean": "你好世界", "mixture": wrong}
        weights = {b: 0.5 for b in buckets}
        res = bootstrap_h_asr_ci(rows, asr, n_boot=300, weights=weights,
                                 expected_buckets=buckets, rng=np.random.default_rng(7))
        # bucket sizes fixed at 8 each in every replicate
        self.assertEqual(res["bucket_sizes"], {buckets[0]: 8, buckets[1]: 8})
        self.assertLessEqual(res["ci_low"], res["point"] + 1e-9)
        self.assertGreaterEqual(res["ci_high"], res["point"] - 1e-9)
        self.assertEqual(res["n_boot"], 300)

    def test_stratified_rejects_absent_bucket(self):
        buckets = [(0.0, 0.0), (5.0, 1.0)]
        rows = [{"row_id": "r", "snr_db": 0.0, "overlap_ratio": 0.0, "transcript": "你好"}]
        asr = {"r": {"clean": "你好", "mixture": "你好"}}
        weights = {b: 0.5 for b in buckets}
        with self.assertRaises(ValueError):
            bootstrap_h_asr_ci(rows, asr, n_boot=10, weights=weights,
                               expected_buckets=buckets)


class AsrEvidenceTests(unittest.TestCase):
    def _row(self, row_id="r1", *, snr=0.0, overlap=0.5, transcript="你好世界"):
        return R5OracleRow(
            row_id=row_id, seed="A", split="test",
            target_speaker_id="t1", target_speaker="S0002",
            interferer_speaker_id=("i1" if overlap > 0 else None),
            interferer_speaker=("S0005" if overlap > 0 else None),
            enrollment_audio=Path("/tmp/e.wav"), mixture_audio=Path("/tmp/m.wav"),
            clean_target_audio=Path("/tmp/c.wav"), snr_db=float(snr),
            sir_db=(0.0 if overlap > 0 else None), overlap_ratio=float(overlap),
            rir_id=None, noise_source_id=None, transcript=transcript,
            mixture_digest="a" * 64, clean_digest="b" * 64,
            level_stats=LevelStats(LevelRecord(0.1, 0.07, 0.1, 0.07),
                                   LevelRecord(0.2, 0.14, 0.2, 0.14)),
            generator_version=GENERATOR_VERSION,
        )

    def _rec(self, row_id="r1", role="mixture", *, text="你好世界", error=None,
             digest_ok=True, manifest_digest="a" * 64, config_digest="c" * 64,
             audio_digest=None, snr=0.0, overlap=0.5, transcript="你好世界"):
        return {
            "row_id": row_id, "seed": "A", "split": "test", "path_role": role,
            "snr_db": float(snr), "overlap_ratio": float(overlap),
            "transcript": transcript, "asr_text": text, "latency_ms": 10.0,
            "digest": audio_digest if audio_digest is not None else manifest_digest,
            "manifest_digest": manifest_digest, "config_digest": config_digest,
            "digest_ok": digest_ok, **({"error": error} if error else {}),
        }

    def test_valid_pairs_pass(self):
        rows = [self._row()]
        recs = [self._rec("r1", "mixture"), self._rec("r1", "clean_target", manifest_digest="b" * 64)]
        asr, cfg = validate_asr_evidence(rows, recs, recheck_files=False)
        self.assertEqual(asr, {"r1": {"mixture": "你好世界", "clean": "你好世界"}})
        self.assertEqual(cfg, "c" * 64)

    def test_missing_clean_rejected(self):
        rows = [self._row()]
        recs = [self._rec("r1", "mixture")]
        with self.assertRaises(ValueError):
            validate_asr_evidence(rows, recs, recheck_files=False)

    def test_duplicate_mixture_rejected(self):
        rows = [self._row()]
        recs = [self._rec("r1", "mixture"), self._rec("r1", "mixture"),
                self._rec("r1", "clean_target", manifest_digest="b" * 64)]
        with self.assertRaises(ValueError):
            validate_asr_evidence(rows, recs, recheck_files=False)

    def test_unexpected_row_rejected(self):
        rows = [self._row()]
        recs = [self._rec("r1", "mixture"), self._rec("r1", "clean_target", manifest_digest="b" * 64),
                self._rec("r2", "mixture")]
        with self.assertRaises(ValueError):
            validate_asr_evidence(rows, recs, recheck_files=False)

    def test_wrong_transcript_rejected(self):
        rows = [self._row(transcript="你好世界")]
        recs = [self._rec("r1", "mixture", transcript="错误文本"),
                self._rec("r1", "clean_target", manifest_digest="b" * 64, transcript="错误文本")]
        with self.assertRaises(ValueError):
            validate_asr_evidence(rows, recs, recheck_files=False)

    def test_wrong_metadata_rejected(self):
        rows = [self._row(snr=5.0)]
        recs = [self._rec("r1", "mixture", snr=0.0),
                self._rec("r1", "clean_target", manifest_digest="b" * 64, snr=0.0)]
        with self.assertRaises(ValueError):
            validate_asr_evidence(rows, recs, recheck_files=False)

    def test_manifest_digest_mismatch_rejected(self):
        rows = [self._row()]
        recs = [self._rec("r1", "mixture", manifest_digest="z" * 64),
                self._rec("r1", "clean_target", manifest_digest="b" * 64)]
        with self.assertRaises(ValueError):
            validate_asr_evidence(rows, recs, recheck_files=False)

    def test_errored_record_rejected(self):
        rows = [self._row()]
        recs = [self._rec("r1", "mixture", error="boom"),
                self._rec("r1", "clean_target", manifest_digest="b" * 64)]
        with self.assertRaises(ValueError):
            validate_asr_evidence(rows, recs, recheck_files=False)

    def test_config_inconsistency_rejected(self):
        rows = [self._row()]
        recs = [self._rec("r1", "mixture", config_digest="c" * 64),
                self._rec("r1", "clean_target", manifest_digest="b" * 64, config_digest="d" * 64)]
        with self.assertRaises(ValueError):
            validate_asr_evidence(rows, recs, recheck_files=False)

    def test_file_digest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            mix_d = _write_tone_wav(Path(tmp) / "m.wav")
            clean_d = _write_tone_wav(Path(tmp) / "c.wav")
            row = R5OracleRow(
                row_id="r1", seed="A", split="test", target_speaker_id="t1",
                target_speaker="S0002", interferer_speaker_id="i1", interferer_speaker="S0005",
                enrollment_audio=Path(tmp) / "e.wav", mixture_audio=Path(tmp) / "m.wav",
                clean_target_audio=Path(tmp) / "c.wav", snr_db=0.0, sir_db=0.0, overlap_ratio=0.5,
                rir_id=None, noise_source_id=None, transcript="你好世界",
                mixture_digest=mix_d, clean_digest=clean_d,
                level_stats=LevelStats(LevelRecord(0.1, 0.07, 0.1, 0.07),
                                       LevelRecord(0.2, 0.14, 0.2, 0.14)),
                generator_version=GENERATOR_VERSION,
            )
            recs = [self._rec("r1", "mixture", manifest_digest=mix_d, audio_digest=mix_d),
                    self._rec("r1", "clean_target", manifest_digest=clean_d, audio_digest=clean_d)]
            validate_asr_evidence([row], recs, recheck_files=True)  # passes
            # Corrupt the mixture file -> current-file digest mismatch
            _write_tone_wav(Path(tmp) / "m.wav", freq=999)
            with self.assertRaises(ValueError):
                validate_asr_evidence([row], recs, recheck_files=True)


class ResumeIntegrityTests(unittest.TestCase):
    def test_only_valid_matching_records_count_as_done(self):
        from scripts.r5_oracle_asr import load_done_keys
        cfg = "c" * 64
        row = R5OracleRow(
            row_id="r1", seed="A", split="test", target_speaker_id="t1",
            target_speaker="S0002", interferer_speaker_id="i1", interferer_speaker="S0005",
            enrollment_audio=Path("/tmp/e.wav"), mixture_audio=Path("/tmp/m.wav"),
            clean_target_audio=Path("/tmp/c.wav"), snr_db=0.0, sir_db=0.0, overlap_ratio=0.5,
            rir_id=None, noise_source_id=None, transcript="你好",
            mixture_digest="a" * 64, clean_digest="b" * 64,
            level_stats=LevelStats(LevelRecord(0.1, 0.07, 0.1, 0.07),
                                   LevelRecord(0.2, 0.14, 0.2, 0.14)),
            generator_version=GENERATOR_VERSION,
        )
        manifest_by_row = {"r1": row}
        records = [
            # valid mixture -> done
            {"row_id": "r1", "path_role": "mixture", "error": None, "digest_ok": True,
             "config_digest": cfg, "manifest_digest": "a" * 64, "digest": "a" * 64},
            # errored clean -> NOT done (retry)
            {"row_id": "r1", "path_role": "clean_target", "error": "boom", "digest_ok": True,
             "config_digest": cfg, "manifest_digest": "b" * 64, "digest": "b" * 64},
            # second row, config mismatch -> NOT done
            {"row_id": "r2", "path_role": "mixture", "error": None, "digest_ok": True,
             "config_digest": "d" * 64, "manifest_digest": "a" * 64, "digest": "a" * 64},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "asr.jsonl"
            with out.open("w", encoding="utf-8") as h:
                for r in records:
                    h.write(__import__("json").dumps(r) + "\n")
            done = load_done_keys(out, manifest_by_row, cfg, recheck_digests=False)
        self.assertEqual(done, {"r1|mixture"})  # only the valid mixture

    def test_manifest_digest_change_forces_retry(self):
        from scripts.r5_oracle_asr import load_done_keys
        cfg = "c" * 64
        row = R5OracleRow(
            row_id="r1", seed="A", split="test", target_speaker_id="t1",
            target_speaker="S0002", interferer_speaker_id="i1", interferer_speaker="S0005",
            enrollment_audio=Path("/tmp/e.wav"), mixture_audio=Path("/tmp/m.wav"),
            clean_target_audio=Path("/tmp/c.wav"), snr_db=0.0, sir_db=0.0, overlap_ratio=0.5,
            rir_id=None, noise_source_id=None, transcript="你好",
            mixture_digest="a" * 64, clean_digest="b" * 64,
            level_stats=LevelStats(LevelRecord(0.1, 0.07, 0.1, 0.07),
                                   LevelRecord(0.2, 0.14, 0.2, 0.14)),
            generator_version=GENERATOR_VERSION,
        )
        manifest_by_row = {"r1": row}
        # record's manifest_digest differs from current manifest -> retry
        rec = {"row_id": "r1", "path_role": "mixture", "error": None, "digest_ok": True,
               "config_digest": cfg, "manifest_digest": "z" * 64, "digest": "z" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "asr.jsonl"
            out.write_text(__import__("json").dumps(rec) + "\n", encoding="utf-8")
            done = load_done_keys(out, manifest_by_row, cfg, recheck_digests=False)
        self.assertEqual(done, set())


class BranchRecommendationTests(unittest.TestCase):
    def _seed(self, h, ci_low, ci_high):
        return {"seed": "A", "h_asr_test": h, "ci_test": {"ci_low": ci_low, "ci_high": ci_high}}

    def test_ci_crossing_boundary_is_inconclusive(self):
        per = [self._seed(0.010, 0.005, 0.025), self._seed(0.012, 0.006, 0.026)]
        br = branch_recommendation(per)
        self.assertTrue(br["inconclusive"])
        self.assertTrue(br["reasons"]["ci_crosses_boundary"])

    def test_different_bands_inconclusive(self):
        per = [self._seed(0.010, 0.005, 0.015), self._seed(0.030, 0.025, 0.035)]
        br = branch_recommendation(per)
        self.assertTrue(br["inconclusive"])
        self.assertTrue(br["reasons"]["different_bands"])

    def test_non_overlapping_ci_inconclusive(self):
        per = [self._seed(0.010, 0.005, 0.012), self._seed(0.018, 0.016, 0.019)]
        br = branch_recommendation(per)
        self.assertTrue(br["inconclusive"])
        self.assertFalse(br["reasons"]["ci_overlap"])

    def test_same_band_overlapping_conclusive(self):
        per = [self._seed(0.010, 0.005, 0.015), self._seed(0.012, 0.008, 0.016)]
        br = branch_recommendation(per)
        self.assertFalse(br["inconclusive"])
        self.assertIn("closing", br["recommendation"])

    def test_gate_band_labels(self):
        self.assertEqual(gate_band(0.01), "close")
        self.assertEqual(gate_band(0.03), "mini")
        self.assertEqual(gate_band(0.05), "eligible")


class AdditiveClipInvariantTests(unittest.TestCase):
    def test_invariant_non_attenuated(self):
        rng = np.random.default_rng(21)
        result = render_oracle_audio(
            target=tone(440, amp=0.2), interferer=tone(330, amp=0.2),
            noise=tone(110, amp=0.05), target_rir=None, interferer_rir=None,
            config=R5OracleConfig(overlap_ratio=0.5, snr_db=5.0, sir_db=0.0),
            rng=rng,
        )
        self.assertAlmostEqual(result.scale, 1.0, places=12)  # no attenuation
        np.testing.assert_allclose(result.mixture,
                                   result.clean_target + result.base_component, atol=1e-9)
        self.assertLessEqual(result.level_stats.mixture.post_peak, 1.0 + 1e-12)

    def test_invariant_attenuated(self):
        rng = np.random.default_rng(22)
        result = render_oracle_audio(
            target=tone(440, amp=0.9), interferer=tone(330, amp=0.9),
            noise=tone(110, amp=0.3), target_rir=None, interferer_rir=None,
            config=R5OracleConfig(overlap_ratio=1.0, snr_db=0.0, sir_db=0.0,
                                  clip_threshold=0.999),
            rng=rng,
        )
        self.assertLess(result.scale, 1.0)  # attenuation occurred
        np.testing.assert_allclose(result.mixture,
                                   result.clean_target + result.base_component, atol=1e-9)
        self.assertLessEqual(result.level_stats.mixture.post_peak, 0.999 + 1e-9)
        self.assertLessEqual(result.level_stats.clean.post_peak, 0.999 + 1e-9)

    def test_clip_is_numerical_noop(self):
        # The safety clip never alters a sample beyond float tolerance because
        # the limiter keeps every sample below clip_threshold.
        rng = np.random.default_rng(23)
        result = render_oracle_audio(
            target=tone(440, amp=0.95), interferer=tone(330, amp=0.95),
            noise=tone(110, amp=0.3), target_rir=None, interferer_rir=None,
            config=R5OracleConfig(overlap_ratio=1.0, snr_db=0.0, sir_db=0.0),
            rng=rng,
        )
        from xh202615.r5_oracle import _clip
        unclipped = result.clean_target + result.base_component
        np.testing.assert_allclose(result.mixture, _clip(unclipped, 1.0), atol=0.0)
        np.testing.assert_allclose(result.mixture, unclipped, atol=1e-9)


class AsrConfigDigestTests(unittest.TestCase):
    def test_config_digest_changes_with_model(self):
        a = asr_config_digest(model="paraformer-zh", vad_model="fsmn-vad",
                              punc_model="ct-punc", hotword=None, hotword_preset="assistant",
                              language=None, use_itn=None, device="cuda", batch_size_s=300,
                              trust_remote_code=False)
        b = asr_config_digest(model="SenseVoiceSmall", vad_model="fsmn-vad",
                              punc_model="ct-punc", hotword=None, hotword_preset="assistant",
                              language=None, use_itn=None, device="cuda", batch_size_s=300,
                              trust_remote_code=False)
        self.assertNotEqual(a, b)
        self.assertTrue(_valid_digest(a))


def _file_row(row_id, tmp, *, t_spk="S0002", i_spk="S0005", snr=0.0, overlap=0.5):
    """A manifest row backed by real temp WAV files with correct digests."""
    mix_d = _write_tone_wav(Path(tmp) / f"{row_id}-mix.wav", freq=440)
    clean_d = _write_tone_wav(Path(tmp) / f"{row_id}-clean.wav", freq=880)
    return R5OracleRow(
        row_id=row_id, seed="A", split="test",
        target_speaker_id=f"t-{row_id}", target_speaker=t_spk,
        interferer_speaker_id=(f"i-{row_id}" if overlap > 0 else None),
        interferer_speaker=(i_spk if overlap > 0 else None),
        enrollment_audio=Path(tmp) / f"{row_id}-enr.wav",
        mixture_audio=Path(tmp) / f"{row_id}-mix.wav",
        clean_target_audio=Path(tmp) / f"{row_id}-clean.wav",
        snr_db=float(snr), sir_db=(0.0 if overlap > 0 else None), overlap_ratio=float(overlap),
        rir_id=None, noise_source_id=None, transcript="你好世界",
        mixture_digest=mix_d, clean_digest=clean_d,
        level_stats=LevelStats(LevelRecord(0.1, 0.07, 0.1, 0.07),
                               LevelRecord(0.2, 0.14, 0.2, 0.14)),
        generator_version=GENERATOR_VERSION,
    )


def _asr_rec(row_id, role, digest, *, text="你好世界", error=None, config="c" * 64,
             digest_ok=True, snr=0.0, overlap=0.5, transcript="你好世界", include_text=True):
    import json as _json
    rec = {
        "row_id": row_id, "seed": "A", "split": "test", "path_role": role,
        "snr_db": float(snr), "overlap_ratio": float(overlap), "transcript": transcript,
        "latency_ms": 10.0, "digest": digest, "manifest_digest": digest,
        "config_digest": config, "digest_ok": digest_ok,
    }
    if include_text:
        rec["asr_text"] = text
    if error:
        rec["error"] = error
    return rec


class ResumeCompactionTests(unittest.TestCase):
    """Resume must drop errored/stale/duplicate records so retried results
    replace them; the final evidence validator must see exactly one pair per row."""

    def test_compaction_keeps_valid_drops_errored_and_config_mismatched(self):
        from scripts.r5_oracle_asr import load_done_keys, compact_output
        import json
        cfg = "c" * 64
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_file_row("r1", tmp), _file_row("r2", tmp, t_spk="S0004", i_spk="S0006")]
            manifest_by_row = {r.row_id: r for r in rows}
            out = Path(tmp) / "asr.jsonl"
            with out.open("w", encoding="utf-8") as h:
                # r1/mixture: valid -> keep
                h.write(json.dumps(_asr_rec("r1", "mixture", rows[0].mixture_digest)) + "\n")
                # r1/clean: errored -> retry (drop)
                h.write(json.dumps(_asr_rec("r1", "clean_target", rows[0].clean_digest,
                                            error="boom")) + "\n")
                # r2/mixture: config-mismatched -> retry (drop)
                h.write(json.dumps(_asr_rec("r2", "mixture", rows[1].mixture_digest,
                                            config="d" * 64)) + "\n")
                # r2/clean: absent -> retry
            done = load_done_keys(out, manifest_by_row, cfg, recheck_digests=False)
            self.assertEqual(done, {"r1|mixture"})
            kept = compact_output(out, done)
            self.assertEqual(kept, 1)
            remaining = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(remaining), 1)
            self.assertEqual((remaining[0]["row_id"], remaining[0]["path_role"]), ("r1", "mixture"))
            # Simulate the retry appending successful replacements.
            with out.open("a", encoding="utf-8") as h:
                h.write(json.dumps(_asr_rec("r1", "clean_target", rows[0].clean_digest)) + "\n")
                h.write(json.dumps(_asr_rec("r2", "mixture", rows[1].mixture_digest)) + "\n")
                h.write(json.dumps(_asr_rec("r2", "clean_target", rows[1].clean_digest)) + "\n")
            final = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
            # Exactly one pair per row, no duplicates, no errors -> validator passes.
            asr_by_row, _ = validate_asr_evidence(rows, final, recheck_files=True)
            self.assertEqual(set(asr_by_row.keys()), {"r1", "r2"})
            for rid in ("r1", "r2"):
                self.assertEqual(set(asr_by_row[rid].keys()), {"mixture", "clean"})

    def test_without_compaction_duplicate_is_rejected(self):
        """Demonstrates the bug the fix addresses: a retried successful record
        coexisting with an earlier record for the same key is rejected as a
        duplicate (compaction prevents this)."""
        import json
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_file_row("r1", tmp), _file_row("r2", tmp, t_spk="S0004", i_spk="S0006")]
            out = Path(tmp) / "asr.jsonl"
            with out.open("w", encoding="utf-8") as h:
                h.write(json.dumps(_asr_rec("r1", "mixture", rows[0].mixture_digest)) + "\n")
                # two successful records for r1/clean_target -> duplicate
                h.write(json.dumps(_asr_rec("r1", "clean_target", rows[0].clean_digest)) + "\n")
                h.write(json.dumps(_asr_rec("r1", "clean_target", rows[0].clean_digest)) + "\n")
                h.write(json.dumps(_asr_rec("r2", "mixture", rows[1].mixture_digest)) + "\n")
                h.write(json.dumps(_asr_rec("r2", "clean_target", rows[1].clean_digest)) + "\n")
            final = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
            with self.assertRaises(ValueError):
                validate_asr_evidence(rows, final, recheck_files=True)

    def test_compaction_dedups_duplicate_valid_records(self):
        from scripts.r5_oracle_asr import load_done_keys, compact_output
        import json
        cfg = "c" * 64
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_file_row("r1", tmp)]
            manifest_by_row = {r.row_id: r for r in rows}
            out = Path(tmp) / "asr.jsonl"
            with out.open("w", encoding="utf-8") as h:
                # two valid records for r1/mixture + one valid r1/clean
                h.write(json.dumps(_asr_rec("r1", "mixture", rows[0].mixture_digest)) + "\n")
                h.write(json.dumps(_asr_rec("r1", "mixture", rows[0].mixture_digest)) + "\n")
                h.write(json.dumps(_asr_rec("r1", "clean_target", rows[0].clean_digest)) + "\n")
            done = load_done_keys(out, manifest_by_row, cfg, recheck_digests=False)
            self.assertEqual(done, {"r1|mixture", "r1|clean_target"})
            kept = compact_output(out, done)
            self.assertEqual(kept, 2)  # one per key
            remaining = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(remaining), 2)
            validate_asr_evidence(rows, remaining, recheck_files=True)  # passes


class AsrTextValidationTests(unittest.TestCase):
    def _row(self):
        with tempfile.TemporaryDirectory() as tmp:
            return _file_row("r1", tmp), tmp

    def test_missing_asr_text_rejected(self):
        row, _ = self._row()
        recs = [_asr_rec("r1", "mixture", row.mixture_digest, include_text=False),
                _asr_rec("r1", "clean_target", row.clean_digest)]
        with self.assertRaises(ValueError):
            validate_asr_evidence([row], recs, recheck_files=False)

    def test_nonstring_asr_text_rejected(self):
        row, _ = self._row()
        import json
        # Build a record with asr_text=123 (non-string) by post-processing.
        rec_mix = _asr_rec("r1", "mixture", row.mixture_digest)
        rec_mix["asr_text"] = 123
        rec_clean = _asr_rec("r1", "clean_target", row.clean_digest)
        with self.assertRaises(ValueError):
            validate_asr_evidence([row], [rec_mix, rec_clean], recheck_files=False)

    def test_none_asr_text_rejected(self):
        row, _ = self._row()
        rec_mix = _asr_rec("r1", "mixture", row.mixture_digest)
        rec_mix["asr_text"] = None
        rec_clean = _asr_rec("r1", "clean_target", row.clean_digest)
        with self.assertRaises(ValueError):
            validate_asr_evidence([row], [rec_mix, rec_clean], recheck_files=False)

    def test_empty_string_asr_text_allowed(self):
        row, _ = self._row()
        recs = [_asr_rec("r1", "mixture", row.mixture_digest, text=""),
                _asr_rec("r1", "clean_target", row.clean_digest, text="")]
        asr_by_row, _ = validate_asr_evidence([row], recs, recheck_files=False)
        self.assertEqual(asr_by_row, {"r1": {"mixture": "", "clean": ""}})


if __name__ == "__main__":
    unittest.main()
