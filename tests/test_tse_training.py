"""CPU tests for the public-only TSE pilot trainer.

No WeSpeaker and no network are required: a mock enrollment-embedding provider
and a tiny FiLMCRNExtractor exercise the manifest guard, deterministic cropping,
the composite loss/backward path, and checkpoint/summary metadata. All tests run
on CPU. No Dataset-A access.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.train_tse import (
    TSEDataset,
    TSEJointDataset,
    assert_class_balance,
    calibrate_presence_threshold,
    check_audio,
    composite_loss,
    crop_pair,
    evaluate,
    evaluate_joint,
    joint_loss,
    load_positive_rows,
    load_training_rows,
    manifest_digest,
    presence_bce_loss,
    run_training,
    train_step,
    train_step_joint,
)
from xh202615.target_extractor import FiLMCRNExtractor, stft_waveform
from xh202615.training_data import TrainingManifestRow

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_wav(path: Path, samples: int = 16000, sample_rate: int = SAMPLE_RATE, tone: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(samples, dtype=np.float32) / sample_rate
    x = (0.1 * np.sin(2 * np.pi * tone * t)).astype(np.float32)
    sf.write(str(path), x, sample_rate, subtype="PCM_16")


def _row_dict(
    row_id: str,
    split: str,
    target_present: bool,
    base: Path,
    speaker: str,
    interferer: str,
    *,
    overlap: float,
    snr: float | None,
    sir: float | None,
) -> dict:
    enrollment = base / split / f"{row_id}-enr.wav"
    mixture = base / split / f"{row_id}-mix.wav"
    target = base / split / f"{row_id}-tgt.wav"
    for path in (enrollment, mixture, target):
        _write_wav(path)
    return {
        "row_id": row_id,
        "split": split,
        "source": "test-public",
        "enrollment_audio": str(enrollment),
        "target_audio": str(target),
        "mixture_audio": str(mixture),
        "target_speaker_id": speaker,
        "interferer_speaker_id": interferer if target_present else None,
        "target_present": target_present,
        "overlap_ratio": overlap,
        "snr_db": snr,
        "sir_db": sir,
        "text": None,
        "seed": 20260805,
    }


_SPEAKERS = {"train": "aishell1:S0001", "val": "aishell1:S0002", "test": "aishell1:S0003"}
_INTERFERERS = {"train": "aishell1:I0001", "val": "aishell1:I0002", "test": "aishell1:I0003"}


def _write_fixture(
    tmp: Path,
    *,
    pos_per_split: int = 2,
    neg_per_split: int = 1,
) -> tuple[Path, list[dict]]:
    base = tmp / "public"
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        speaker = _SPEAKERS[split]
        interferer = _INTERFERERS[split]
        for k in range(pos_per_split):
            rows.append(
                _row_dict(
                    f"{split}-pos-{k:03d}",
                    split,
                    True,
                    base,
                    speaker,
                    interferer,
                    overlap=0.5,
                    snr=5.0,
                    sir=0.0,
                )
            )
        for k in range(neg_per_split):
            rows.append(
                _row_dict(
                    f"{split}-neg-{k:03d}",
                    split,
                    False,
                    base,
                    speaker,
                    interferer,
                    overlap=0.0,
                    snr=None,
                    sir=None,
                )
            )
    manifest = tmp / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest, rows


def _mock_provider(dim: int = 8):
    """Deterministic embedding provider (no WeSpeaker) returning a fixed-size vector."""

    def provider(path: Path) -> np.ndarray:
        digest = hashlib.blake2b(str(path).encode("utf-8"), digest_size=dim * 4).digest()
        vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        return (vec / 255.0 - 0.5)[:dim]

    return provider


class _MockSpeakerEncoder:
    """Content-dependent speaker encoder (no WeSpeaker) for R7 wiring tests.

    ``embed_tensor`` fingerprints a waveform by averaging ``dim`` equal partitions
    so the cosine is finite and content-dependent; ``embed_path`` reuses the
    path-hash provider so the enrollment cache is in a comparable shape. Tests
    only exercise plumbing (scores computed, metadata written, calibration
    runs) - real discrimination is validated by the CUDA smoke with WeSpeaker.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_tensor(self, waveform: torch.Tensor) -> torch.Tensor:
        wav = waveform.float()
        if wav.ndim != 2:
            raise ValueError("expected [B, samples]")
        b, n = wav.shape
        emb = torch.zeros(b, self.dim)
        idx = np.array_split(np.arange(n), self.dim)
        for i, part in enumerate(idx):
            if len(part):
                emb[:, i] = wav[:, part].mean(dim=1)
        return emb

    def embed_audio(self, audio: np.ndarray) -> np.ndarray:
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
        return self.embed_tensor(wav).reshape(-1).numpy()

    def embed_path(self, path) -> np.ndarray:
        return _mock_provider(self.dim)(Path(path))


def _mock_speaker_encoder(dim: int = 8) -> _MockSpeakerEncoder:
    return _MockSpeakerEncoder(dim=dim)


# ---------------------------------------------------------------------------
# Manifest filtering and Dataset-A guard
# ---------------------------------------------------------------------------

class ManifestFilterTests(unittest.TestCase):
    def _fixture(self, tmp: Path) -> tuple[Path, Path]:
        dataset_a = tmp / "datasetA_root"
        dataset_a.mkdir()
        manifest, _ = _write_fixture(tmp)
        return manifest, dataset_a

    def test_positive_rows_filtered_and_counts_correct(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, dataset_a = self._fixture(tmp)
            rows = load_positive_rows(manifest, dataset_a)
            # 2 positive per split, negatives dropped.
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row.target_present for row in rows))
            self.assertEqual(
                {row.split for row in rows}, {"train", "val", "test"}
            )

    def test_dataset_a_containment_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, dataset_a = self._fixture(tmp)
            # Rewrite the manifest so one mixture leaks under the Dataset-A root.
            lines = manifest.read_text(encoding="utf-8").splitlines()
            leaked = dataset_a / "leaked.wav"
            _write_wav(leaked)
            first = json.loads(lines[0])
            first["mixture_audio"] = str(leaked)
            lines[0] = json.dumps(first, ensure_ascii=False, sort_keys=True)
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden|validation failed"):
                load_positive_rows(manifest, dataset_a)

    def test_duplicate_row_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, dataset_a = self._fixture(tmp)
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        _row_dict(
                            "train-pos-000",
                            "train",
                            True,
                            tmp / "public",
                            _SPEAKERS["train"],
                            _INTERFERERS["train"],
                            overlap=0.5,
                            snr=5.0,
                            sir=0.0,
                        ),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            with self.assertRaisesRegex(ValueError, "duplicate_row_id|validation failed"):
                load_positive_rows(manifest, dataset_a)

    def test_speaker_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, dataset_a = self._fixture(tmp)
            # Reuse the train target speaker in a val row -> cross-split leak.
            lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in lines:
                if row["row_id"].startswith("val-pos"):
                    row["target_speaker_id"] = _SPEAKERS["train"]
            manifest.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in lines) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "speaker_split_leakage|validation failed"):
                load_positive_rows(manifest, dataset_a)


class CheckAudioTests(unittest.TestCase):
    def _rows(self, tmp: Path) -> tuple[list[TrainingManifestRow], Path]:
        manifest, _ = _write_fixture(tmp)
        dataset_a = tmp / "datasetA_root"
        dataset_a.mkdir()
        from xh202615.training_data import read_training_manifest

        rows = [r for r in read_training_manifest(manifest) if r.target_present]
        return rows, dataset_a

    def test_missing_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            rows, _ = self._rows(tmp)
            Path(rows[0].mixture_audio).unlink()
            with self.assertRaisesRegex(ValueError, "missing audio"):
                check_audio(rows)

    def test_non_16khz_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            rows, _ = self._rows(tmp)
            # Overwrite one mixture with an 8 kHz WAV.
            _write_wav(Path(rows[0].mixture_audio), samples=8000, sample_rate=8000)
            with self.assertRaisesRegex(ValueError, "non-16kHz"):
                check_audio(rows)

    def test_valid_audio_passes(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            rows, _ = self._rows(tmp)
            check_audio(rows)  # must not raise


class ManifestDigestTests(unittest.TestCase):
    def test_digest_is_sha256_hex(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            path = Path(tmp_str) / "m.jsonl"
            path.write_text('{"row_id":"x"}\n', encoding="utf-8")
            digest = manifest_digest(path)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Deterministic crop / pad
# ---------------------------------------------------------------------------

class CropPairTests(unittest.TestCase):
    def test_pad_short_clips_to_segment(self):
        mixture = np.ones(4000, dtype=np.float32)
        target = np.full(4000, 2.0, dtype=np.float32)
        mix_c, tgt_c = crop_pair(mixture, target, 8000, seed=1, epoch=1, index=0)
        self.assertEqual(mix_c.shape, (8000,))
        self.assertEqual(tgt_c.shape, (8000,))
        # First 4000 samples preserve content; tail is zero-padded.
        self.assertTrue(np.allclose(mix_c[:4000], 1.0))
        self.assertTrue(np.allclose(mix_c[4000:], 0.0))
        self.assertTrue(np.allclose(tgt_c[:4000], 2.0))

    def test_aligned_window_for_mixture_and_target(self):
        rng = np.random.default_rng(0)
        target = rng.standard_normal(16000).astype(np.float32)
        # Mixture shares the target plus an independent interferer -> aligned.
        mixture = target + rng.standard_normal(16000).astype(np.float32) * 0.1
        mix_c, tgt_c = crop_pair(mixture, target, 4000, seed=7, epoch=2, index=3)
        self.assertEqual(mix_c.shape, tgt_c.shape)
        # The cropped windows must come from the same offset: reconstruct the
        # offset by locating the target window that matches tgt_c.
        match = None
        for start in range(0, 16000 - 4000 + 1):
            if np.allclose(target[start : start + 4000], tgt_c, atol=1e-6):
                match = start
                break
        self.assertIsNotNone(match, "target crop is not a contiguous window")
        self.assertTrue(np.allclose(mixture[match : match + 4000], mix_c, atol=1e-6))

    def test_deterministic_same_args_identical(self):
        rng = np.random.default_rng(0)
        mixture = rng.standard_normal(16000).astype(np.float32)
        target = rng.standard_normal(16000).astype(np.float32)
        a = crop_pair(mixture, target, 4000, seed=3, epoch=1, index=5)
        b = crop_pair(mixture, target, 4000, seed=3, epoch=1, index=5)
        self.assertTrue(np.array_equal(a[0], b[0]))
        self.assertTrue(np.array_equal(a[1], b[1]))

    def test_different_epoch_changes_window(self):
        rng = np.random.default_rng(0)
        mixture = rng.standard_normal(32000).astype(np.float32)
        target = rng.standard_normal(32000).astype(np.float32)
        a = crop_pair(mixture, target, 4000, seed=3, epoch=1, index=0)
        b = crop_pair(mixture, target, 4000, seed=3, epoch=2, index=0)
        self.assertFalse(np.array_equal(a[0], b[0]), "crop should vary across epochs")

    def test_full_length_crop_is_exact(self):
        mixture = np.arange(8000, dtype=np.float32)
        target = np.arange(8000, dtype=np.float32) * 2
        mix_c, tgt_c = crop_pair(mixture, target, 8000, seed=1, epoch=1, index=0)
        self.assertTrue(np.array_equal(mix_c, mixture))
        self.assertTrue(np.array_equal(tgt_c, target))


# ---------------------------------------------------------------------------
# Composite loss and tiny-model backward (CPU, mocked embedding)
# ---------------------------------------------------------------------------

class CompositeLossTests(unittest.TestCase):
    def test_loss_is_finite_scalar_and_differentiable(self):
        enhanced = torch.randn(2, 8000, requires_grad=True)
        target = torch.randn(2, 8000)
        loss = composite_loss(enhanced, target, stft_weight=1.0, si_sdr_weight=1.0)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(enhanced.grad)
        self.assertTrue(torch.isfinite(enhanced.grad).all())


def _tiny_model() -> FiLMCRNExtractor:
    return FiLMCRNExtractor(
        embedding_dim=8, channels=(4, 8), n_fft=512, hop_length=128,
        win_length=None, gru_hidden=8, gru_layers=1,
    )


class TrainStepTests(unittest.TestCase):
    def test_cpu_step_finite_and_produces_gradients(self):
        torch.manual_seed(0)
        model = _tiny_model()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mixture = torch.randn(2, 8000)
        target = torch.randn(2, 8000)
        embedding = torch.randn(2, 8)
        loss = train_step(
            model, optimizer, mixture, target, embedding,
            use_amp=False, scaler=None, grad_clip=5.0,
            stft_weight=1.0, si_sdr_weight=1.0,
        )
        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        self.assertTrue(has_grad, "no parameter received a non-zero gradient")


class EvaluateTests(unittest.TestCase):
    def test_evaluate_returns_finite_metrics(self):
        torch.manual_seed(0)
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, _ = _write_fixture(tmp)
            from xh202615.training_data import read_training_manifest

            rows = tuple(r for r in read_training_manifest(manifest) if r.target_present and r.split == "train")
            embeddings = {str(r.enrollment_audio): np.zeros(8, dtype=np.float32) for r in rows}
            dataset = TSEDataset(rows, embeddings, 8000, seed=1, epoch=0)
            metrics = evaluate(
                model, dataset, device=torch.device("cpu"), use_amp=False,
                batch_size=2, stft_weight=1.0, si_sdr_weight=1.0,
            )
        for key in ("loss", "si_sdr", "stft"):
            self.assertIn(key, metrics)
            self.assertTrue(np.isfinite(metrics[key]), f"{key} not finite")
        self.assertEqual(metrics["n"], len(rows))


# ---------------------------------------------------------------------------
# Checkpoint + summary metadata (end-to-end on CPU with mocked embeddings)
# ---------------------------------------------------------------------------

class RunTrainingMetadataTests(unittest.TestCase):
    def test_writes_checkpoint_and_summary_with_required_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            dataset_a = tmp / "datasetA_root"
            dataset_a.mkdir()
            manifest, _ = _write_fixture(tmp, pos_per_split=2, neg_per_split=1)
            output_dir = tmp / "tse_out"
            summary = run_training(
                manifest=manifest,
                output_dir=output_dir,
                dataset_a_root=dataset_a,
                embedding_dim=8,
                channels=(4, 8),
                gru_hidden=8,
                gru_layers=1,
                epochs=1,
                batch_size=2,
                segment_seconds=0.5,
                seed=20260805,
                device="cpu",
                reuse_cache=False,
                embedding_provider=_mock_provider(dim=8),
            )

            # --- summary.json on disk ---
            summary_path = output_dir / "summary.json"
            self.assertTrue(summary_path.is_file())
            on_disk = json.loads(summary_path.read_text(encoding="utf-8"))
            for key in (
                "manifest_digest", "data_boundary", "model_config", "seed", "history",
            ):
                self.assertIn(key, on_disk, f"summary missing {key}")
            self.assertFalse(on_disk["dataset_a_used_for_training"])
            self.assertEqual(on_disk["seed"], 20260805)
            self.assertEqual(on_disk["model_config"]["embedding_dim"], 8)
            self.assertEqual(on_disk["model_config"]["channels"], [4, 8])
            self.assertEqual(on_disk["manifest_digest"], manifest_digest(manifest))
            self.assertEqual(len(on_disk["history"]), 1)
            self.assertIn("val", on_disk["history"][0])
            self.assertIn("test", on_disk["history"][0])
            # Test metrics are recorded but never drive the checkpoint.
            self.assertEqual(on_disk["positive_split_rows"], {"train": 2, "val": 2, "test": 2})

            # --- best.pt checkpoint ---
            best_path = output_dir / "best.pt"
            self.assertTrue(best_path.is_file())
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            for key in (
                "model_state_dict", "model_config", "manifest_digest",
                "dataset_a_root", "dataset_a_used_for_training", "data_boundary", "seed",
            ):
                self.assertIn(key, ckpt, f"checkpoint missing {key}")
            self.assertFalse(ckpt["dataset_a_used_for_training"])
            self.assertEqual(ckpt["manifest_digest"], manifest_digest(manifest))
            self.assertEqual(ckpt["seed"], 20260805)

            # The checkpoint must reconstruct an identical-architecture model.
            cfg = ckpt["model_config"]
            rebuilt = FiLMCRNExtractor(
                embedding_dim=cfg["embedding_dim"],
                channels=tuple(cfg["channels"]),
                n_fft=cfg["n_fft"],
                hop_length=cfg["hop_length"],
                win_length=cfg["win_length"],
                gru_hidden=cfg["gru_hidden"],
                gru_layers=cfg["gru_layers"],
            )
            rebuilt.load_state_dict(ckpt["model_state_dict"])
            rebuilt.eval()
            with torch.no_grad():
                spec = stft_waveform(
                    torch.randn(1, 8000),
                    n_fft=cfg["n_fft"],
                    hop_length=cfg["hop_length"],
                    win_length=cfg["win_length"],
                )
                out = rebuilt(spec, torch.randn(1, cfg["embedding_dim"]))
            self.assertTrue(out.is_complex())
            self.assertTrue(torch.isfinite(out).all())

    def test_empty_split_after_filtering_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            dataset_a = tmp / "datasetA_root"
            dataset_a.mkdir()
            base = tmp / "public"
            # Only train positive rows; val/test have no positives.
            rows = [
                _row_dict("train-pos-000", "train", True, base, _SPEAKERS["train"], _INTERFERERS["train"], overlap=0.5, snr=5.0, sir=0.0),
                _row_dict("val-neg-000", "val", False, base, _SPEAKERS["val"], _INTERFERERS["val"], overlap=0.0, snr=None, sir=None),
                _row_dict("test-neg-000", "test", False, base, _SPEAKERS["test"], _INTERFERERS["test"], overlap=0.0, snr=None, sir=None),
            ]
            manifest = tmp / "manifest.jsonl"
            manifest.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no target-present rows"):
                run_training(
                    manifest=manifest,
                    output_dir=tmp / "out",
                    dataset_a_root=dataset_a,
                    embedding_dim=8,
                    channels=(4, 8),
                    gru_hidden=8,
                    gru_layers=1,
                    epochs=1,
                    batch_size=1,
                    segment_seconds=0.5,
                    seed=20260805,
                    device="cpu",
                    reuse_cache=False,
                    embedding_provider=_mock_provider(dim=8),
                )


# ---------------------------------------------------------------------------
# R6 joint presence objective: negative rows, class balance, calibration,
# joint train/eval, checkpoint metadata, malformed rows, cache reuse.
# ---------------------------------------------------------------------------

class LoadTrainingRowsTests(unittest.TestCase):
    def test_returns_present_and_absent_rows(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            dataset_a = tmp / "datasetA_root"
            dataset_a.mkdir()
            manifest, _ = _write_fixture(tmp)  # 2 pos + 1 neg per split
            rows = load_training_rows(manifest, dataset_a)
            self.assertEqual(len(rows), 9)
            present = sum(1 for r in rows if r.target_present)
            absent = sum(1 for r in rows if not r.target_present)
            self.assertEqual(present, 6)
            self.assertEqual(absent, 3)

    def test_absent_row_with_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            dataset_a = tmp / "datasetA_root"
            dataset_a.mkdir()
            manifest, _ = _write_fixture(tmp)
            # Corrupt a negative row by giving it a text (forbidden: text must
            # be null when target_present is false).
            lines = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
            for row in lines:
                if not row["target_present"]:
                    row["text"] = "不应存在的文本"
            manifest.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in lines) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "target_absent_text|validation failed"):
                load_training_rows(manifest, dataset_a)


class ClassBalanceTests(unittest.TestCase):
    def test_balanced_passes(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, _ = _write_fixture(tmp)
            rows = load_training_rows(manifest, tmp / "da_root")
            counts = assert_class_balance(rows)
            for split in ("train", "val", "test"):
                self.assertGreater(counts[split]["present"], 0)
                self.assertGreater(counts[split]["absent"], 0)

    def test_imbalanced_split_fails(self):
        rows = [
            TrainingManifestRow(
                row_id="train-pos", split="train", source="t",
                enrollment_audio=Path("a"), target_audio=Path("b"), mixture_audio=Path("c"),
                target_speaker_id="S1", interferer_speaker_id="I1", target_present=True,
                overlap_ratio=0.5, snr_db=5.0, sir_db=0.0, text=None, seed=1,
            ),
            TrainingManifestRow(
                row_id="val-pos", split="val", source="t",
                enrollment_audio=Path("a"), target_audio=Path("b"), mixture_audio=Path("c"),
                target_speaker_id="S2", interferer_speaker_id="I2", target_present=True,
                overlap_ratio=0.5, snr_db=5.0, sir_db=0.0, text=None, seed=1,
            ),
        ]
        with self.assertRaisesRegex(ValueError, "class-imbalanced"):
            assert_class_balance(rows)


class CalibratePresenceThresholdTests(unittest.TestCase):
    def test_perfect_separation(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        cal = calibrate_presence_threshold(scores, labels)
        self.assertAlmostEqual(cal["auc"], 1.0, places=6)
        self.assertGreater(cal["threshold"], 0.2)
        self.assertLessEqual(cal["threshold"], 0.8)
        self.assertEqual(cal["rr_at_threshold"], 1.0)
        self.assertEqual(cal["false_accept_rate"], 0.0)

    def test_single_class_raises(self):
        with self.assertRaisesRegex(ValueError, "cannot calibrate"):
            calibrate_presence_threshold(np.array([0.1, 0.2]), np.array([1.0, 1.0]))

    def test_auc_random_is_half(self):
        # Scores uncorrelated with labels -> AUC ~ 0.5 (exact for this tie set).
        scores = np.array([0.1, 0.2, 0.1, 0.2])
        labels = np.array([0.0, 1.0, 1.0, 0.0])
        cal = calibrate_presence_threshold(scores, labels)
        self.assertAlmostEqual(cal["auc"], 0.5, places=6)


class PresenceBceLossTests(unittest.TestCase):
    def test_finite_and_differentiable(self):
        logit = torch.randn(4, 1, requires_grad=True)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = presence_bce_loss(logit, labels)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logit.grad)
        self.assertTrue(torch.isfinite(logit.grad).all())


class JointTrainStepTests(unittest.TestCase):
    def test_cpu_step_finite_and_produces_gradients(self):
        torch.manual_seed(0)
        model = FiLMCRNExtractor(
            embedding_dim=8, channels=(4, 8), n_fft=512, hop_length=128,
            win_length=None, gru_hidden=8, gru_layers=1, with_presence=True,
        )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mixture = torch.randn(4, 8000)
        target = torch.randn(4, 8000)
        embedding = torch.randn(4, 8)
        # 2 present, 2 absent.
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss_value, components = train_step_joint(
            model, optimizer, mixture, target, embedding, labels,
            use_amp=False, scaler=None, grad_clip=5.0,
            stft_weight=1.0, si_sdr_weight=1.0, presence_weight=1.0,
        )
        self.assertIsNotNone(loss_value)
        self.assertTrue(np.isfinite(loss_value))
        self.assertIn("recon", components)
        self.assertIn("presence", components)
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        self.assertTrue(has_grad, "no parameter received a non-zero gradient")


class JointEvaluateTests(unittest.TestCase):
    def test_returns_recon_and_presence_metrics(self):
        torch.manual_seed(0)
        model = FiLMCRNExtractor(
            embedding_dim=8, channels=(4, 8), n_fft=512, hop_length=128,
            win_length=None, gru_hidden=8, gru_layers=1, with_presence=True,
        )
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, _ = _write_fixture(tmp)
            from xh202615.training_data import read_training_manifest

            rows = tuple(r for r in read_training_manifest(manifest) if r.split == "train")
            embeddings = {str(r.enrollment_audio): np.zeros(8, dtype=np.float32) for r in rows}
            dataset = TSEJointDataset(rows, embeddings, 8000, seed=1, epoch=0)
            metrics = evaluate_joint(
                model, dataset, device=torch.device("cpu"), use_amp=False,
                batch_size=2, stft_weight=1.0, si_sdr_weight=1.0,
            )
        self.assertIn("recon", metrics)
        self.assertIn("presence", metrics)
        self.assertIn("auc", metrics["presence"])
        self.assertIn("threshold", metrics["presence"])
        self.assertEqual(metrics["n_rows"], len(rows))
        self.assertGreater(metrics["n_present"], 0)
        self.assertGreater(metrics["n_absent"], 0)


class RunTrainingJointMetadataTests(unittest.TestCase):
    def _run(self, tmp: Path, **overrides):
        dataset_a = tmp / "datasetA_root"
        dataset_a.mkdir()
        manifest, _ = _write_fixture(tmp, pos_per_split=2, neg_per_split=2)
        kwargs = dict(
            manifest=manifest,
            output_dir=tmp / "tse_out",
            dataset_a_root=dataset_a,
            embedding_dim=8,
            channels=(4, 8),
            gru_hidden=8,
            gru_layers=1,
            epochs=1,
            batch_size=2,
            segment_seconds=0.5,
            seed=20260805,
            device="cpu",
            reuse_cache=False,
            embedding_provider=_mock_provider(dim=8),
            with_presence=True,
        )
        kwargs.update(overrides)
        return run_training(**kwargs)

    def test_writes_checkpoint_and_summary_with_presence_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            summary = self._run(tmp)
            self.assertTrue(summary["with_presence"])
            self.assertIn("presence_threshold", summary)
            self.assertIn("presence_auc", summary)
            self.assertEqual(summary["presence_threshold_source"], "public_val_youden_j")
            self.assertIn("class_balance", summary)
            self.assertEqual(summary["model_config"]["with_presence"], True)
            self.assertIn("absent", summary["data_boundary"])
            # positive_split_rows counts only present rows.
            self.assertEqual(summary["positive_split_rows"], {"train": 2, "val": 2, "test": 2})

            ckpt = torch.load(tmp / "tse_out" / "best.pt", map_location="cpu", weights_only=False)
            self.assertTrue(ckpt["with_presence"])
            self.assertIn("presence_threshold", ckpt)
            self.assertIn("presence_auc", ckpt)
            self.assertEqual(ckpt["model_config"]["with_presence"], True)
            self.assertTrue(any("presence_head" in k for k in ckpt["model_state_dict"]))
            self.assertIn("absent", ckpt["data_boundary"])

            # Strict load into a with_presence model of matching config.
            cfg = ckpt["model_config"]
            rebuilt = FiLMCRNExtractor(
                embedding_dim=cfg["embedding_dim"], channels=tuple(cfg["channels"]),
                n_fft=cfg["n_fft"], hop_length=cfg["hop_length"], win_length=cfg["win_length"],
                gru_hidden=cfg["gru_hidden"], gru_layers=cfg["gru_layers"], with_presence=True,
            )
            rebuilt.load_state_dict(ckpt["model_state_dict"], strict=True)

    def test_old_checkpoint_still_loads_strict_without_presence(self):
        # The default (with_presence=False) path produces a checkpoint with no
        # presence head; an old-style model must still strict-load it.
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            dataset_a = tmp / "datasetA_root"
            dataset_a.mkdir()
            manifest, _ = _write_fixture(tmp, pos_per_split=2, neg_per_split=1)
            run_training(
                manifest=manifest, output_dir=tmp / "old_out", dataset_a_root=dataset_a,
                embedding_dim=8, channels=(4, 8), gru_hidden=8, gru_layers=1,
                epochs=1, batch_size=2, segment_seconds=0.5, seed=20260805,
                device="cpu", reuse_cache=False, embedding_provider=_mock_provider(dim=8),
            )
            ckpt = torch.load(tmp / "old_out" / "best.pt", map_location="cpu", weights_only=False)
            self.assertFalse(ckpt["model_config"].get("with_presence", False))
            self.assertNotIn("presence_threshold", ckpt)
            rebuilt = FiLMCRNExtractor(
                embedding_dim=8, channels=(4, 8), n_fft=512, hop_length=128,
                win_length=None, gru_hidden=8, gru_layers=1,
            )
            rebuilt.load_state_dict(ckpt["model_state_dict"], strict=True)


class JointCacheReuseTests(unittest.TestCase):
    def test_embedding_cache_reused_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            dataset_a = tmp / "datasetA_root"
            dataset_a.mkdir()
            manifest, _ = _write_fixture(tmp, pos_per_split=2, neg_per_split=2)
            kwargs = dict(
                manifest=manifest, output_dir=tmp / "out", dataset_a_root=dataset_a,
                embedding_dim=8, channels=(4, 8), gru_hidden=8, gru_layers=1,
                epochs=1, batch_size=2, segment_seconds=0.5, seed=20260805,
                device="cpu", embedding_provider=_mock_provider(dim=8), with_presence=True,
            )
            run_training(reuse_cache=False, **kwargs)
            cache_path = tmp / "out" / "enrollment_embeddings.pt"
            self.assertTrue(cache_path.is_file())
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            # Cache must include enrollment paths for absent rows too.
            self.assertGreater(len(payload["embeddings"]), 6)
            # Second run reuses the cache (same digest + model name).
            run_training(reuse_cache=True, **kwargs)
            payload2 = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.assertEqual(set(payload["embeddings"]), set(payload2["embeddings"]))
            for key, vec in payload["embeddings"].items():
                self.assertTrue(np.allclose(vec, payload2["embeddings"][key]))


class JointEvaluateSpeakerTests(unittest.TestCase):
    def test_returns_speaker_metrics_when_encoder_given(self):
        torch.manual_seed(0)
        model = FiLMCRNExtractor(
            embedding_dim=8, channels=(4, 8), n_fft=512, hop_length=128,
            win_length=None, gru_hidden=8, gru_layers=1, with_presence=True,
        )
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, _ = _write_fixture(tmp)
            from xh202615.training_data import read_training_manifest

            rows = tuple(r for r in read_training_manifest(manifest) if r.split == "train")
            embeddings = {str(r.enrollment_audio): np.zeros(8, dtype=np.float32) for r in rows}
            dataset = TSEJointDataset(rows, embeddings, 8000, seed=1, epoch=0)
            metrics = evaluate_joint(
                model, dataset, device=torch.device("cpu"), use_amp=False,
                batch_size=2, stft_weight=1.0, si_sdr_weight=1.0,
                speaker_encoder=_mock_speaker_encoder(dim=8),
            )
        self.assertIn("speaker", metrics)
        spk = metrics["speaker"]
        self.assertIn(spk["score_type"], ("enhanced_cosine", "mixture_cosine", "max_cosine"))
        self.assertEqual(spk["threshold_source"], "public_val_youden_j")
        self.assertIn("auc", spk)
        self.assertIn("threshold", spk)
        self.assertEqual(set(spk["per_variant"]),
                         {"enhanced_cosine", "mixture_cosine", "max_cosine"})

    def test_no_speaker_section_without_encoder(self):
        torch.manual_seed(0)
        model = FiLMCRNExtractor(
            embedding_dim=8, channels=(4, 8), n_fft=512, hop_length=128,
            win_length=None, gru_hidden=8, gru_layers=1, with_presence=True,
        )
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            manifest, _ = _write_fixture(tmp)
            from xh202615.training_data import read_training_manifest

            rows = tuple(r for r in read_training_manifest(manifest) if r.split == "train")
            embeddings = {str(r.enrollment_audio): np.zeros(8, dtype=np.float32) for r in rows}
            dataset = TSEJointDataset(rows, embeddings, 8000, seed=1, epoch=0)
            metrics = evaluate_joint(
                model, dataset, device=torch.device("cpu"), use_amp=False,
                batch_size=2, stft_weight=1.0, si_sdr_weight=1.0,
            )
        self.assertNotIn("speaker", metrics)


class RunTrainingSpeakerScoreTests(unittest.TestCase):
    def _run(self, tmp: Path, **overrides):
        dataset_a = tmp / "datasetA_root"
        dataset_a.mkdir()
        manifest, _ = _write_fixture(tmp, pos_per_split=2, neg_per_split=2)
        kwargs = dict(
            manifest=manifest, output_dir=tmp / "tse_out", dataset_a_root=dataset_a,
            embedding_dim=8, channels=(4, 8), gru_hidden=8, gru_layers=1,
            epochs=1, batch_size=2, segment_seconds=0.5, seed=20260805,
            device="cpu", reuse_cache=False,
            embedding_provider=_mock_provider(dim=8),
            with_presence=True, with_speaker_score=True,
            speaker_encoder=_mock_speaker_encoder(dim=8),
        )
        kwargs.update(overrides)
        return run_training(**kwargs)

    def test_writes_speaker_metadata_in_checkpoint_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            summary = self._run(tmp)
            self.assertTrue(summary["with_speaker_score"])
            self.assertIn(summary["speaker_score_type"],
                          ("enhanced_cosine", "mixture_cosine", "max_cosine"))
            self.assertEqual(summary["speaker_threshold_source"], "public_val_youden_j")
            self.assertIn("speaker_threshold", summary)
            self.assertIn("speaker_auc", summary)
            self.assertEqual(set(summary["speaker_val_per_variant"]),
                             {"enhanced_cosine", "mixture_cosine", "max_cosine"})

            ckpt = torch.load(tmp / "tse_out" / "best.pt", map_location="cpu", weights_only=False)
            self.assertTrue(ckpt["with_speaker_score"])
            self.assertIn("speaker_threshold", ckpt)
            self.assertIn("speaker_score_type", ckpt)
            self.assertEqual(ckpt["speaker_threshold_source"], "public_val_youden_j")
            self.assertEqual(set(ckpt["speaker_score_variants"]),
                             {"enhanced_cosine", "mixture_cosine", "max_cosine"})
            # Presence metadata still present (auxiliary head).
            self.assertTrue(ckpt["with_presence"])
            self.assertIn("presence_threshold", ckpt)

    def test_with_speaker_score_requires_with_presence(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            dataset_a = tmp / "datasetA_root"
            dataset_a.mkdir()
            manifest, _ = _write_fixture(tmp, pos_per_split=2, neg_per_split=2)
            with self.assertRaisesRegex(ValueError, "with_speaker_score"):
                run_training(
                    manifest=manifest, output_dir=tmp / "out", dataset_a_root=dataset_a,
                    embedding_dim=8, channels=(4, 8), gru_hidden=8, gru_layers=1,
                    epochs=1, batch_size=2, segment_seconds=0.5, seed=20260805,
                    device="cpu", reuse_cache=False, embedding_provider=_mock_provider(dim=8),
                    with_presence=False, with_speaker_score=True,
                    speaker_encoder=_mock_speaker_encoder(dim=8),
                )


if __name__ == "__main__":
    unittest.main()
