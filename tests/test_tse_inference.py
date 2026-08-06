"""CPU tests for the TSE inference adapter (no WeSpeaker or Dataset-A access)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.run_tse_inference import (
    InputRow,
    load_input_rows,
    load_checkpoint,
    read_input_jsonl,
    run_inference,
)
from xh202615.target_extractor import FiLMCRNExtractor


SR = 16_000


def _wav(path: Path, *, sr: int = SR, seconds: float = 0.5, freq: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, subtype="FLOAT")


def _provider(path: Path) -> np.ndarray:
    # Deterministic 256-D mock; no network/model download.
    value = (sum(path.name.encode("utf-8")) % 17) / 17.0
    return np.full(256, value, dtype=np.float32)


def _checkpoint(path: Path) -> None:
    model = FiLMCRNExtractor(
        embedding_dim=256,
        channels=(4, 8),
        n_fft=64,
        hop_length=16,
        win_length=64,
        gru_hidden=8,
        gru_layers=1,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "embedding_dim": 256,
                "channels": [4, 8],
                "n_fft": 64,
                "hop_length": 16,
                "win_length": 64,
                "gru_hidden": 8,
                "gru_layers": 1,
            },
        },
        path,
    )


def _checkpoint_with_presence(path: Path, threshold: float = 0.5) -> None:
    """A with-presence checkpoint carrying calibrated presence metadata."""
    model = FiLMCRNExtractor(
        embedding_dim=256,
        channels=(4, 8),
        n_fft=64,
        hop_length=16,
        win_length=64,
        gru_hidden=8,
        gru_layers=1,
        with_presence=True,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "embedding_dim": 256,
                "channels": [4, 8],
                "n_fft": 64,
                "hop_length": 16,
                "win_length": 64,
                "gru_hidden": 8,
                "gru_layers": 1,
                "with_presence": True,
            },
            "with_presence": True,
            "presence_threshold": threshold,
            "presence_threshold_source": "public_val_youden_j",
            "presence_auc": 0.81,
        },
        path,
    )


def _fixture(tmp: Path) -> tuple[Path, Path, Path, Path]:
    wake = tmp / "wake.wav"
    command = tmp / "command.wav"
    _wav(wake, freq=220.0)
    _wav(command, freq=440.0)
    manifest = tmp / "input.jsonl"
    # The label is intentionally present but must not be part of InputRow.
    manifest.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "split": "pos",
                "wakeup_audio": wake.name,
                "command_audio": command.name,
                "label": "秘密标签，不应被读取",
                "recognition_text": "同样不应被读取",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp / "model.pt"
    _checkpoint(checkpoint)
    return manifest, wake, command, checkpoint


class ManifestTests(unittest.TestCase):
    def test_parser_extracts_input_fields_only(self):
        with tempfile.TemporaryDirectory() as value:
            manifest, _, _, _ = _fixture(Path(value))
            rows = read_input_jsonl(manifest)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].sample_id, "sample-1")
            self.assertFalse(hasattr(rows[0], "label"))
            self.assertTrue(rows[0].command_audio.is_file())

    def test_duplicate_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as value:
            tmp = Path(value)
            manifest, wake, command, _ = _fixture(tmp)
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "sample-1", "wakeup_audio": str(wake), "command_audio": str(command)}) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_input_rows(input_jsonl=[manifest])

    def test_non_16k_audio_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as value:
            tmp = Path(value)
            manifest, wake, command, checkpoint = _fixture(tmp)
            _wav(command, sr=8000)
            row = read_input_jsonl(manifest)[0]
            with self.assertRaisesRegex(ValueError, "16 kHz"):
                run_inference(
                    [row], checkpoint=checkpoint, output_root=tmp / "out",
                    output_map=tmp / "map.jsonl", embedding_cache=tmp / "cache.pt",
                    device="cpu", embedding_provider=_provider,
                )

    def test_checkpoint_metadata_is_required(self):
        with tempfile.TemporaryDirectory() as value:
            tmp = Path(value)
            manifest, _, _, _ = _fixture(tmp)
            broken = tmp / "broken.pt"
            torch.save({}, broken)
            with self.assertRaisesRegex(ValueError, "model_config"):
                load_checkpoint(broken, torch.device("cpu"))


class InferenceTests(unittest.TestCase):
    def test_output_map_wav_and_resume(self):
        with tempfile.TemporaryDirectory() as value:
            tmp = Path(value)
            manifest, _, _, checkpoint = _fixture(tmp)
            row = read_input_jsonl(manifest)[0]
            kwargs = dict(
                checkpoint=checkpoint,
                output_root=tmp / "enhanced",
                output_map=tmp / "map.jsonl",
                embedding_cache=tmp / "cache.pt",
                device="cpu",
                model_name="mock",
                embedding_provider=_provider,
                manifest_digest="fixture-digest",
            )
            first = run_inference([row], **kwargs)
            self.assertEqual(first["rows"], 1)
            self.assertEqual(first["errors"], 0)
            record = json.loads((tmp / "map.jsonl").read_text(encoding="utf-8").splitlines()[0])
            output = Path(record["enhanced_command_audio"])
            audio, sr = sf.read(str(output), dtype="float32")
            self.assertEqual(sr, SR)
            self.assertTrue(np.isfinite(audio).all())
            self.assertNotEqual(output.resolve(), row.command_audio.resolve())

            second = run_inference([row], **kwargs, resume=True)
            self.assertEqual(second["rows"], 0)
            self.assertEqual(second["skipped"], 1)
            self.assertEqual(len((tmp / "map.jsonl").read_text(encoding="utf-8").splitlines()), 1)


class PresenceScoreTests(unittest.TestCase):
    def _fixture_with_presence(self, tmp: Path) -> tuple[Path, Path, Path, Path]:
        wake = tmp / "wake.wav"
        command = tmp / "command.wav"
        _wav(wake, freq=220.0)
        _wav(command, freq=440.0)
        manifest = tmp / "input.jsonl"
        manifest.write_text(
            json.dumps({"id": "sample-1", "split": "pos",
                        "wakeup_audio": wake.name, "command_audio": command.name})
            + "\n",
            encoding="utf-8",
        )
        checkpoint = tmp / "model.pt"
        _checkpoint_with_presence(checkpoint, threshold=0.42)
        return manifest, wake, command, checkpoint

    def test_presence_score_emitted_for_with_presence_checkpoint(self):
        with tempfile.TemporaryDirectory() as value:
            tmp = Path(value)
            manifest, _, _, checkpoint = self._fixture_with_presence(tmp)
            row = read_input_jsonl(manifest)[0]
            summary = run_inference(
                [row], checkpoint=checkpoint, output_root=tmp / "enhanced",
                output_map=tmp / "map.jsonl", embedding_cache=tmp / "cache.pt",
                device="cpu", model_name="mock", embedding_provider=_provider,
                manifest_digest="fixture-digest",
            )
            self.assertTrue(summary["with_presence"])
            self.assertAlmostEqual(summary["presence_threshold"], 0.42)
            self.assertEqual(summary["presence_threshold_source"], "public_val_youden_j")
            record = json.loads((tmp / "map.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("presence_score", record)
            self.assertIsInstance(record["presence_score"], float)
            self.assertTrue(np.isfinite(record["presence_score"]))
            self.assertFalse(record.get("error"))

    def test_old_checkpoint_emits_no_presence_score(self):
        with tempfile.TemporaryDirectory() as value:
            tmp = Path(value)
            manifest, _, _, _ = _fixture(tmp)  # old-style checkpoint, no presence
            row = read_input_jsonl(manifest)[0]
            summary = run_inference(
                [row], checkpoint=tmp / "model.pt", output_root=tmp / "enhanced",
                output_map=tmp / "map.jsonl", embedding_cache=tmp / "cache.pt",
                device="cpu", model_name="mock", embedding_provider=_provider,
                manifest_digest="fixture-digest",
            )
            self.assertNotIn("with_presence", summary)
            record = json.loads((tmp / "map.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("presence_score", record)

    def test_resume_preserves_presence_score(self):
        with tempfile.TemporaryDirectory() as value:
            tmp = Path(value)
            manifest, _, _, checkpoint = self._fixture_with_presence(tmp)
            row = read_input_jsonl(manifest)[0]
            kwargs = dict(
                checkpoint=checkpoint, output_root=tmp / "enhanced",
                output_map=tmp / "map.jsonl", embedding_cache=tmp / "cache.pt",
                device="cpu", model_name="mock", embedding_provider=_provider,
                manifest_digest="fixture-digest",
            )
            run_inference([row], **kwargs)
            second = run_inference([row], **kwargs, resume=True)
            self.assertEqual(second["rows"], 0)
            self.assertEqual(second["skipped"], 1)
            record = json.loads((tmp / "map.jsonl").read_text(encoding="utf-8").splitlines()[0])
            # Resumed (kept) record retains its presence score.
            self.assertIsInstance(record.get("presence_score"), float)


if __name__ == "__main__":
    unittest.main()
