from __future__ import annotations

from pathlib import Path

import numpy as np


def test_overall085_docs_describe_new_stack_and_evaluation_boundary() -> None:
    root = Path(__file__).parents[1]
    text = "\n".join(
        [
            (root / "README.md").read_text(encoding="utf-8"),
            (root / "docs/r12/r12-train-and-publish.md").read_text(encoding="utf-8"),
        ]
    ).lower()

    for term in ("personal vad", "no-vad", "lora", "8gb", "overall 0.85"):
        assert term in text
    assert "internal-test" in text
    assert "盲测" in text


def test_overall085_smoke_imports_new_pure_helpers() -> None:
    from xh202615.r12_personal_vad import (
        PersonalVADConfig,
        PersonalVADNet,
        aggregate_personal_vad,
        build_personal_vad_features,
    )

    config = PersonalVADConfig(mel_bins=2, embedding_dim=2, hidden_size=4, num_layers=1)
    features = build_personal_vad_features(
        np.zeros((2, 2), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    metrics = aggregate_personal_vad(np.full((2, 3), 1 / 3, dtype=np.float32))

    assert features.shape == (2, config.input_dim)
    assert PersonalVADNet(config) is not None
    assert set(metrics) >= {"target_speech_ratio", "overlap_probability"}
