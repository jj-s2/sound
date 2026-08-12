"""Contract tests for the offline, label-free FireRed pVAD adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from xh202615.firered_model_assets import FireRedModelPaths
from xh202615.firered_pvad import (
    PVAD_GATE_FEATURE_SCHEMA,
    FireRedPvadRuntime,
    PvadRuntimeConfig,
)


class FakeEncoder:
    def __init__(self, embedding: object = (3.0, 4.0, *([0.0] * 190))) -> None:
        self.embedding = embedding
        self.inputs: list[np.ndarray] = []

    def __call__(self, audio: np.ndarray) -> object:
        self.inputs.append(audio.copy())
        return self.embedding


class FakeSession:
    def __init__(
        self,
        probabilities: object = (0.1, 0.4, 0.8, 0.8, 0.2),
        *,
        mel: object | None = None,
        gru: object | None = None,
        providers: object = ("CPUExecutionProvider",),
    ) -> None:
        self.probabilities = list(probabilities)
        self.mel = mel
        self.gru = gru
        self.providers = providers
        self.calls: list[dict[str, np.ndarray]] = []

    def get_providers(self) -> object:
        return self.providers

    def run(self, _outputs: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append({name: value.copy() for name, value in inputs.items()})
        index = len(self.calls) - 1
        probability = self.probabilities[index % len(self.probabilities)]
        mel = self.mel
        if mel is None:
            mel = np.full((1, 80, 15), index + 1, dtype=np.float32)
        gru = self.gru
        if gru is None:
            gru = np.full((2, 1, 256), index + 1, dtype=np.float32)
        return [np.zeros((1, 1), np.float32), np.asarray([[probability]]), mel, gru]


@pytest.fixture
def paths(tmp_path: Path) -> FireRedModelPaths:
    return FireRedModelPaths(tmp_path, tmp_path / "pvad.onnx", tmp_path / "ecapa", tmp_path / "manifest")


@pytest.fixture
def audio(monkeypatch: pytest.MonkeyPatch):
    from xh202615 import firered_pvad

    samples: dict[Path, tuple[np.ndarray, int]] = {}

    def read(path: Path, *, always_2d: bool):
        assert always_2d is True
        return samples[Path(path)]

    monkeypatch.setattr(firered_pvad.soundfile, "read", read)
    return samples


def runtime(paths: FireRedModelPaths, encoder: FakeEncoder, session: FakeSession, **kwargs: object) -> FireRedPvadRuntime:
    return FireRedPvadRuntime(
        paths,
        config=kwargs.pop("config", PvadRuntimeConfig(minimum_audio_seconds=0.01)),
        ecapa_encoder=encoder,
        onnx_session=session,
        **kwargs,
    )


def put(audio: dict[Path, tuple[np.ndarray, int]], root: Path, name: str, values: object, rate: int = 16000) -> Path:
    path = root / name
    audio[path] = (np.asarray(values), rate)
    return path


def test_mono_stereo_resampling_clipping_and_wake_cap(paths, audio, tmp_path):
    wake = put(audio, tmp_path, "wake.wav", np.linspace(-2, 2, 96000).reshape(-1, 1), 16000)
    command = put(audio, tmp_path, "command.wav", np.column_stack((np.ones(240), -np.ones(240))), 8000)
    encoder, session = FakeEncoder(), FakeSession()

    result = runtime(paths, encoder, session).extract("x", wake, command)

    assert encoder.inputs[0].shape == (80000,)
    assert encoder.inputs[0].dtype == np.float32
    assert encoder.inputs[0].flags.c_contiguous
    assert np.min(encoder.inputs[0]) >= -1.0 and np.max(encoder.inputs[0]) <= 1.0
    assert result.values["command_duration_sec"] == pytest.approx(0.03)
    assert result.values["frame_count"] == 3
    assert all(call["input_audio"].shape == (1, 160) for call in session.calls)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_audio_is_rejected(paths, audio, tmp_path, bad):
    wake = put(audio, tmp_path, "wake.wav", np.full((4000, 1), bad))
    command = put(audio, tmp_path, "command.wav", np.zeros((4000, 1)))
    with pytest.raises(ValueError, match="non-finite"):
        runtime(paths, FakeEncoder(), FakeSession(), config=PvadRuntimeConfig()).extract("x", wake, command)


def test_short_audio_and_no_complete_command_frame_are_rejected(paths, audio, tmp_path):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((3999, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((4000, 1)))
    with pytest.raises(ValueError, match="shorter"):
        runtime(paths, FakeEncoder(), FakeSession(), config=PvadRuntimeConfig()).extract("x", wake, command)
    wake = put(audio, tmp_path, "wake2.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command2.wav", np.zeros((159, 1)))
    with pytest.raises(ValueError, match="complete 160-sample frame"):
        runtime(paths, FakeEncoder(), FakeSession(), config=PvadRuntimeConfig(minimum_audio_seconds=0.001)).extract("x", wake, command)


def test_complete_frames_drop_tail_without_padding(paths, audio, tmp_path):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((481, 1)))
    result = runtime(paths, FakeEncoder(), FakeSession()).extract("x", wake, command)
    assert result.values["frame_count"] == 3
    assert result.values["dropped_tail_samples"] == 1
    assert result.audit["dropped_tail_samples"] == 1


def test_embedding_is_normalized_once_and_onnx_uses_official_inputs(paths, audio, tmp_path):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((320, 1)))
    encoder, session = FakeEncoder(), FakeSession()
    result = runtime(paths, encoder, session).extract("x", wake, command)
    assert len(encoder.inputs) == 1
    assert result.values["embedding_norm_before"] == pytest.approx(5.0)
    assert result.values["embedding_norm_after"] == pytest.approx(1.0)
    assert set(session.calls[0]) == {"input_audio", "spkemb", "mel_buffer", "gru_buffer"}
    assert session.calls[0]["spkemb"].shape == (1, 192)
    assert session.calls[0]["spkemb"].dtype == np.float32
    assert np.linalg.norm(session.calls[0]["spkemb"]) == pytest.approx(1.0)


def test_recurrent_state_carries_within_command_and_resets_per_utterance(paths, audio, tmp_path):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((320, 1)))
    session = FakeSession([0.1, 0.4])
    subject = runtime(paths, FakeEncoder(), session)
    first = subject.extract("a", wake, command)
    second = subject.extract("b", wake, command)
    assert first.values == second.values
    assert np.all(session.calls[0]["mel_buffer"] == 0) and np.all(session.calls[0]["gru_buffer"] == 0)
    assert np.all(session.calls[1]["mel_buffer"] == 1) and np.all(session.calls[1]["gru_buffer"] == 1)
    assert np.all(session.calls[2]["mel_buffer"] == 0) and np.all(session.calls[2]["gru_buffer"] == 0)


@pytest.mark.parametrize(
    "embedding,match",
    [([0.0] * 192, "nonzero"), ([math.nan] * 192, "finite"), ([1.0] * 191, "192")],
)
def test_invalid_embeddings_fail_closed(paths, audio, tmp_path, embedding, match):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((160, 1)))
    with pytest.raises(ValueError, match=match):
        runtime(paths, FakeEncoder(embedding), FakeSession()).extract("x", wake, command)


@pytest.mark.parametrize("probability", [math.nan, math.inf, -0.01, 1.01, [0.5, 0.5]])
def test_invalid_probabilities_fail_closed(paths, audio, tmp_path, probability):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((160, 1)))
    with pytest.raises(ValueError, match="probability"):
        runtime(paths, FakeEncoder(), FakeSession([probability])).extract("x", wake, command)


@pytest.mark.parametrize("field,shape", [("mel", (1, 80, 14)), ("gru", (1, 2, 256))])
def test_invalid_recurrent_states_fail_closed(paths, audio, tmp_path, field, shape):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((160, 1)))
    kwargs = {field: np.zeros(shape, dtype=np.float32)}
    with pytest.raises(ValueError, match="state"):
        runtime(paths, FakeEncoder(), FakeSession(**kwargs)).extract("x", wake, command)


def test_fixed_aggregate_literal_and_schema_order(paths, audio, tmp_path):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((800, 1)))
    result = runtime(paths, FakeEncoder(), FakeSession()).extract("x", wake, command)
    values = result.values
    ema = np.array([0.1, 0.16, 0.288, 0.3904, 0.35232])
    assert tuple(values) == PVAD_GATE_FEATURE_SCHEMA
    assert values["raw_mean"] == pytest.approx(0.46)
    assert values["ema_q50"] == pytest.approx(float(np.quantile(ema, 0.5)))
    assert values["raw_fraction_ge_0_5"] == pytest.approx(0.4)
    assert values["ema_fraction_ge_0_3"] == pytest.approx(0.4)
    assert values["ema_longest_run_ge_0_3_frames"] == 2
    assert values["ema_longest_run_ge_0_3_seconds"] == pytest.approx(0.02)
    assert values["ema_first_crossing_ge_0_3_frame"] == 4
    assert values["ema_last_crossing_ge_0_3_frame"] == 5
    assert values["ema_active_span_ge_0_3_frames"] == 2
    assert values["ema_transitions_ge_0_3"] == 1
    assert PVAD_GATE_FEATURE_SCHEMA[:4] == (
        "frame_count", "analyzed_duration_sec", "dropped_tail_samples", "command_duration_sec"
    )
    assert PVAD_GATE_FEATURE_SCHEMA[-3:] == (
        "enrollment_duration_sec", "embedding_norm_before", "embedding_norm_after"
    )


@pytest.mark.parametrize(
    "ema,expected",
    [
        ([0.9, 0.9, 0.9], 0),
        ([0.9, 0.9, 0.1], 1),
        ([0.9, 0.1, 0.9, 0.1], 3),
        ([0.9], 0),
    ],
    ids=("all-active", "active-to-inactive", "alternating", "one-frame"),
)
def test_threshold_transitions_count_adjacent_state_flips_only(paths, ema, expected):
    subject = runtime(paths, FakeEncoder(), FakeSession())
    values = subject._aggregate(
        np.asarray(ema), np.asarray(ema), len(ema) * 160, 0, 160, 1.0, 1.0
    )
    assert values["ema_transitions_ge_0_3"] == expected


def test_inactive_ema_thresholds_use_explicit_sentinels(paths, audio, tmp_path):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((320, 1)))
    values = runtime(paths, FakeEncoder(), FakeSession([0.1, 0.2])).extract("x", wake, command).values
    assert values["ema_first_crossing_ge_0_7_frame"] == -1
    assert values["ema_last_crossing_ge_0_7_frame"] == -1
    assert values["ema_active_span_ge_0_7_frames"] == 0
    assert values["ema_transitions_ge_0_7"] == 0


def test_audit_only_timing_memory_and_label_mutation_independence(paths, audio, tmp_path):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((160, 1)))
    metadata_a = {"label": "secret", "text": "private"}
    metadata_b = {"label": "different", "text": "changed"}
    ticks = iter((10.0, 10.5, 11.0, 11.5))
    subject = runtime(paths, FakeEncoder(), FakeSession([0.1]), clock=lambda: next(ticks), rss_bytes=lambda: 100, cuda_peak_bytes=lambda: 7)
    first = subject.extract("x", wake, command)
    metadata_a["label"] = metadata_b["label"]
    metadata_a["text"] = metadata_b["text"]
    second = subject.extract("x", wake, command)
    canonical_a = json.dumps(dict(first.values), separators=(",", ":"), sort_keys=False)
    canonical_b = json.dumps(dict(second.values), separators=(",", ":"), sort_keys=False)
    assert metadata_a == metadata_b and canonical_a == canonical_b
    assert {"elapsed_seconds", "audio_seconds", "rtf", "peak_rss_delta_bytes", "cuda_peak_bytes"} <= set(first.audit)
    assert not ({"elapsed_seconds", "audio_seconds", "rtf", "peak_rss_delta_bytes", "cuda_peak_bytes"} & set(PVAD_GATE_FEATURE_SCHEMA))
    forbidden = " ".join((*first.values, *first.audit)).lower()
    assert "label" not in forbidden and "text" not in forbidden and "raw_frame" not in forbidden


def test_audit_marks_successful_cold_then_warm_and_retains_transient_rss_peak(paths, audio, tmp_path):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((160, 1)))
    ticks = iter((10.0, 10.5, 11.0, 11.5))
    rss_values = iter((100, 130, 190, 140, 120, 100, 110, 120, 115, 105))
    subject = runtime(
        paths,
        FakeEncoder(),
        FakeSession([0.1]),
        clock=lambda: next(ticks),
        rss_bytes=lambda: next(rss_values),
    )

    cold = subject.extract("cold", wake, command)
    warm = subject.extract("warm", wake, command)

    assert cold.audit["extraction_phase"] == "cold"
    assert warm.audit["extraction_phase"] == "warm"
    assert cold.audit["peak_rss_delta_bytes"] == 90
    assert warm.audit["peak_rss_delta_bytes"] == 20
    assert all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for result in (cold, warm)
        for value in result.audit.values()
        if isinstance(value, (int, float))
    )


def test_failed_extraction_does_not_consume_cold_phase(paths, audio, tmp_path):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((160, 1)))
    subject = runtime(paths, FakeEncoder([0.0] * 192), FakeSession([0.1]))

    with pytest.raises(ValueError, match="nonzero"):
        subject.extract("failed", wake, command)

    subject._ecapa_encoder = FakeEncoder()
    assert subject.extract("successful", wake, command).audit["extraction_phase"] == "cold"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": True},
        {"sample_rate": 16000.0},
        {"frame_samples": False},
        {"frame_samples": 160.5},
        {"enrollment_cap_seconds": math.nan},
        {"minimum_audio_seconds": math.inf},
        {"minimum_audio_seconds": 0.0},
        {"ema_alpha": math.nan},
        {"ema_alpha": -0.1},
        {"ema_alpha": 1.1},
        {"onnx_provider": " "},
        {"onnx_provider": "CPU Provider"},
        {"ecapa_device": ""},
        {"ecapa_device": "cuda:-1"},
    ],
)
def test_runtime_config_rejects_nonfinite_type_confused_and_invalid_values(kwargs):
    with pytest.raises(ValueError):
        PvadRuntimeConfig(**kwargs)


def test_runtime_config_preserves_approved_defaults():
    assert PvadRuntimeConfig() == PvadRuntimeConfig(
        sample_rate=16000,
        frame_samples=160,
        enrollment_cap_seconds=5.0,
        minimum_audio_seconds=0.25,
        ema_alpha=0.8,
        onnx_provider="CPUExecutionProvider",
        ecapa_device="cpu",
    )


def test_session_provider_must_be_explicit_and_match_requested_provider(paths, audio, tmp_path):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((160, 1)))
    config = PvadRuntimeConfig(minimum_audio_seconds=0.01, onnx_provider="CUDAExecutionProvider")

    with pytest.raises(ValueError, match="requested ONNX provider"):
        runtime(paths, FakeEncoder(), FakeSession([0.1], providers=("CPUExecutionProvider",)), config=config).extract("x", wake, command)

    class MissingProviderContract:
        run = FakeSession().run

    with pytest.raises(TypeError, match="get_providers"):
        runtime(paths, FakeEncoder(), MissingProviderContract()).extract("x", wake, command)


def test_default_session_factory_is_verified_before_onnx_inference(paths, audio, tmp_path, monkeypatch):
    from xh202615 import firered_pvad

    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((160, 1)))
    config = PvadRuntimeConfig(minimum_audio_seconds=0.01, onnx_provider="CUDAExecutionProvider")
    monkeypatch.setattr(
        firered_pvad,
        "_default_onnx_session",
        lambda _paths, _config: FakeSession([0.1], providers=("CPUExecutionProvider",)),
    )

    with pytest.raises(ValueError, match="requested ONNX provider"):
        FireRedPvadRuntime(paths, config=config, ecapa_encoder=FakeEncoder()).extract("x", wake, command)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"clock": lambda: math.nan}, "clock"),
        ({"rss_bytes": lambda: -1}, "RSS"),
        ({"cuda_peak_bytes": lambda: True}, "RSS"),
    ],
)
def test_invalid_audit_callback_values_fail_closed(paths, audio, tmp_path, kwargs, match):
    wake = put(audio, tmp_path, "wake.wav", np.zeros((4000, 1)))
    command = put(audio, tmp_path, "command.wav", np.zeros((160, 1)))

    with pytest.raises(ValueError, match=match):
        runtime(paths, FakeEncoder(), FakeSession([0.1]), **kwargs).extract("x", wake, command)
