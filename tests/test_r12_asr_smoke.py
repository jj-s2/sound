"""Tests for the FunASR M0 configuration/load smoke probe (R12 M0)."""

from __future__ import annotations

import dataclasses
import json
import sys
from typing import Iterator

import pytest

from xh202615.r12_asr_smoke import SmokeConfig, build_loader_kwargs, run_smoke


class _FakeParameter:
    def __init__(self, requires_grad: bool = True) -> None:
        self.requires_grad = requires_grad


class _FakeModel:
    def __init__(self, params: list[tuple[str, _FakeParameter]]) -> None:
        self._params = params

    def named_parameters(self) -> Iterator[tuple[str, _FakeParameter]]:
        return iter(self._params)


class _FakeLoaded:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model


def _fake_loader(
    calls: list[dict[str, object]], model: _FakeModel | None = None,
):
    def loader(**kwargs: object) -> _FakeLoaded:
        calls.append(kwargs)
        return _FakeLoaded(model if model is not None else _FakeModel([]))

    return loader


def _lora_model() -> _FakeModel:
    return _FakeModel([
        ("encoder.0.weight", _FakeParameter(False)),
        ("decoder.weight", _FakeParameter(False)),
        ("decoder.lora_A", _FakeParameter(True)),
        ("decoder.lora_B", _FakeParameter(True)),
    ])


def test_lora_smoke_passes_decoder_lora_configuration_to_loader() -> None:
    calls: list[dict[str, object]] = []
    result = run_smoke(
        SmokeConfig(model="paraformer-zh", mode="lora", device="cpu", level="load"),
        loader=_fake_loader(calls, _lora_model()),
    )
    assert result.loaded is True
    assert calls[0]["decoder_conf"]["lora_list"] == ["q", "k", "v", "o"]


def test_freeze_encoder_smoke_marks_encoder_parameters_not_trainable() -> None:
    params = [
        ("encoder.weight", _FakeParameter(True)),
        ("decoder.weight", _FakeParameter(True)),
    ]
    result = run_smoke(
        SmokeConfig(model="x", mode="freeze_encoder", device="cpu", level="load"),
        loader=_fake_loader([], _FakeModel(params)),
    )
    assert result.trainable_parameter_names == ("decoder.weight",)
    assert params[0][1].requires_grad is False
    assert params[1][1].requires_grad is True


def test_config_level_does_not_call_loader() -> None:
    calls: list[dict[str, object]] = []
    result = run_smoke(
        SmokeConfig(model="x", mode="lora", device="cpu", level="config"),
        loader=_fake_loader(calls),
    )
    assert calls == []
    assert result.loaded is False
    assert result.total_parameter_count == 0
    assert result.trainable_parameter_count == 0
    assert result.trainable_parameter_names == ()
    assert result.lora_parameter_count == 0


def test_invalid_mode_fails_before_loader_invocation() -> None:
    calls: list[dict[str, object]] = []
    loader = _fake_loader(calls)
    with pytest.raises(ValueError, match="mode"):
        run_smoke(SmokeConfig(model="x", mode="full_finetune", device="cpu", level="load"), loader=loader)  # type: ignore[arg-type]
    assert calls == []


def test_invalid_level_fails_before_loader_invocation() -> None:
    calls: list[dict[str, object]] = []
    loader = _fake_loader(calls)
    with pytest.raises(ValueError, match="level"):
        run_smoke(SmokeConfig(model="x", mode="lora", device="cpu", level="train"), loader=loader)  # type: ignore[arg-type]
    assert calls == []


def test_invalid_device_fails_before_loader_invocation() -> None:
    calls: list[dict[str, object]] = []
    loader = _fake_loader(calls)
    with pytest.raises(ValueError, match="device"):
        run_smoke(SmokeConfig(model="x", mode="lora", device="tpu", level="config"), loader=loader)
    assert calls == []


def test_invalid_lora_component_fails_before_loader_invocation() -> None:
    calls: list[dict[str, object]] = []
    loader = _fake_loader(calls)
    with pytest.raises(ValueError, match="lora"):
        run_smoke(
            SmokeConfig(model="x", mode="lora", device="cpu", level="load", lora_list=("q", "")),
            loader=loader,
        )
    assert calls == []


def test_lora_mode_fails_closed_when_no_lora_parameter() -> None:
    model = _FakeModel([
        ("encoder.weight", _FakeParameter(False)),
        ("decoder.weight", _FakeParameter(False)),
    ])
    with pytest.raises(ValueError, match="lora"):
        run_smoke(
            SmokeConfig(model="x", mode="lora", device="cpu", level="load"),
            loader=_fake_loader([], model),
        )


def test_lora_loader_kwargs_omit_encoder_conf_and_keep_decoder_lora_list() -> None:
    # FunASR deep_update treats an empty dict as replacement, so encoder_conf={}
    # would wipe the pretrained Paraformer encoder config. It must be omitted.
    kwargs = build_loader_kwargs(
        SmokeConfig(model="paraformer-zh", mode="lora", device="cpu", level="load"),
    )
    assert "encoder_conf" not in kwargs
    assert kwargs["decoder_conf"] == {"lora_list": ["q", "k", "v", "o"]}


def test_lora_loader_kwargs_contain_only_asr_arguments() -> None:
    kwargs = build_loader_kwargs(
        SmokeConfig(model="paraformer-zh", mode="lora", device="cpu", level="load"),
    )
    assert set(kwargs) == {"model", "device", "disable_update", "decoder_conf", "lora_only"}
    assert kwargs["model"] == "paraformer-zh"
    assert kwargs["device"] == "cpu"
    assert kwargs["disable_update"] is True
    assert kwargs["lora_only"] is True
    assert kwargs["decoder_conf"] == {"lora_list": ["q", "k", "v", "o"]}
    assert "encoder_conf" not in kwargs
    assert "vad_model" not in kwargs
    assert "punc_model" not in kwargs


def test_freeze_encoder_loader_kwargs_are_plain_asr_only() -> None:
    kwargs = build_loader_kwargs(
        SmokeConfig(model="paraformer-zh", mode="freeze_encoder", device="cpu", level="load"),
    )
    assert set(kwargs) == {"model", "device"}
    assert "vad_model" not in kwargs
    assert "punc_model" not in kwargs
    assert "disable_update" not in kwargs
    assert "decoder_conf" not in kwargs


def test_result_is_json_safe() -> None:
    result = run_smoke(
        SmokeConfig(model="paraformer-zh", mode="lora", device="cpu", level="load"),
        loader=_fake_loader([], _lora_model()),
    )
    rendered = json.dumps(dataclasses.asdict(result), sort_keys=True)
    payload = json.loads(rendered)
    assert payload["loaded"] is True
    assert payload["total_parameter_count"] == 4
    assert payload["trainable_parameter_count"] == 2
    assert payload["lora_parameter_count"] == 2
    assert payload["config"]["mode"] == "lora"


def test_parameter_prefix_inventory_is_descriptive_and_sha_free() -> None:
    result = run_smoke(
        SmokeConfig(model="x", mode="freeze_encoder", device="cpu", level="load"),
        loader=_fake_loader([], _FakeModel([
            ("encoder.0.weight", _FakeParameter(True)),
            ("decoder.weight", _FakeParameter(True)),
        ])),
    )
    assert result.parameter_prefixes == ("decoder", "encoder")


def test_cli_smoke_config_level_is_offline(capsys: pytest.CaptureFixture[str]) -> None:
    sys.modules.pop("funasr", None)
    from scripts.r12_asr_train import main

    assert main(["smoke", "--model", "paraformer-zh", "--level", "config"]) == 0
    assert "funasr" not in sys.modules
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["loaded"] is False
    assert payload["config"]["mode"] == "lora"
    assert payload["config"]["device"] == "cpu"


def test_cli_rejects_train_and_validation_label_paths() -> None:
    from scripts.r12_asr_train import main

    with pytest.raises(SystemExit):
        main(["smoke", "--train-labels", "train.json"])
    with pytest.raises(SystemExit):
        main(["smoke", "--valid-labels", "valid.json"])
    with pytest.raises(SystemExit):
        main(["smoke", "--internal-test-labels", "internal.json"])
