from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_personal_vad_features_repeat_enrollment_for_each_frame() -> None:
    from xh202615.r12_personal_vad import build_personal_vad_features

    log_mel = np.arange(6, dtype=np.float32).reshape(3, 2)
    enrollment = np.asarray([0.2, -0.4], dtype=np.float32)
    cosine = np.asarray([0.1, 0.3, 0.5], dtype=np.float32)

    features = build_personal_vad_features(log_mel, enrollment, cosine)

    assert features.shape == (3, 5)
    assert np.array_equal(features[:, :2], log_mel)
    assert np.array_equal(features[:, 2:4], np.tile(enrollment, (3, 1)))
    assert np.array_equal(features[:, 4], cosine)


def test_personal_vad_features_reject_misaligned_or_nonfinite_inputs() -> None:
    from xh202615.r12_personal_vad import build_personal_vad_features

    with pytest.raises(ValueError, match="same number of frames"):
        build_personal_vad_features(np.zeros((2, 3)), np.zeros(4), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        build_personal_vad_features(
            np.asarray([[np.nan, 0.0]], dtype=np.float32), np.zeros(2), np.zeros(1)
        )


def test_personal_vad_network_returns_three_frame_classes() -> None:
    from xh202615.r12_personal_vad import PersonalVADConfig, PersonalVADNet

    config = PersonalVADConfig(mel_bins=2, embedding_dim=2, hidden_size=4, num_layers=2)
    model = PersonalVADNet(config)
    logits = model(torch.zeros((3, 7, config.input_dim), dtype=torch.float32))

    assert logits.shape == (3, 7, 3)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_personal_vad_loss_is_finite_for_all_three_classes() -> None:
    from xh202615.r12_personal_vad import PersonalVADConfig, PersonalVADNet, personal_vad_loss

    config = PersonalVADConfig(mel_bins=2, embedding_dim=2, hidden_size=4, num_layers=1)
    model = PersonalVADNet(config)
    logits = model(torch.zeros((1, 5, config.input_dim), dtype=torch.float32))
    targets = torch.tensor([[0, 1, 2, 1, 0]], dtype=torch.long)

    loss = personal_vad_loss(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_personal_vad_aggregation_reports_target_ratio_and_longest_run() -> None:
    from xh202615.r12_personal_vad import aggregate_personal_vad

    probabilities = np.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    metrics = aggregate_personal_vad(probabilities, frame_seconds=0.01)

    assert metrics["target_speech_ratio"] == pytest.approx(0.38)
    assert metrics["target_speech_max"] == pytest.approx(0.8)
    assert metrics["target_longest_run_frames"] == 2.0
    assert metrics["target_longest_run_seconds"] == pytest.approx(0.02)
    assert 0.0 <= metrics["overlap_probability"] <= 1.0


def test_personal_vad_mixture_lineage_rejects_internal_test_paths(tmp_path: Path) -> None:
    from xh202615.r12_personal_vad import write_personal_vad_lineage

    source = tmp_path / "sources.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "x",
                "parent_id": "x",
                "enrollment_audio": "train/wake.wav",
                "target_audio": "train/command.wav",
                "target_speaker_id": "spk1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "lineage.jsonl"
    result = write_personal_vad_lineage(source, output, seed=17)
    assert result == 1
    assert json.loads(output.read_text(encoding="utf-8"))["seed"] == 17

    source.write_text(
        json.dumps(
            {
                "id": "x",
                "parent_id": "x",
                "enrollment_audio": "internal_test/wake.wav",
                "target_audio": "train/command.wav",
                "target_speaker_id": "spk1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="internal-test"):
        write_personal_vad_lineage(source, tmp_path / "blocked.jsonl", seed=17)
