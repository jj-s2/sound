"""FunASR M0 configuration/load smoke probe for R12 Paraformer preparation.

The probe validates a ``SmokeConfig`` and, at ``load`` level only, constructs
one ASR model behind an injectable loader and inspects its parameters.  It never
calls ``generate``, runs a backward pass, writes checkpoints, instantiates VAD
or punctuation models, or reads any audio, label, or dataset path.  CUDA is not
allocated by default; the safe default device is ``cpu``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

Mode = Literal["lora", "freeze_encoder"]
Level = Literal["config", "load"]

_VALID_MODES = frozenset({"lora", "freeze_encoder"})
_VALID_LEVELS = frozenset({"config", "load"})
_DEVICE_RE = re.compile(r"^(?:cpu|cuda(?::\d+)?|mps|npu)$")


@dataclass(frozen=True)
class SmokeConfig:
    model: str
    mode: Mode
    device: str
    level: Level
    lora_list: tuple[str, ...] = ("q", "k", "v", "o")


@dataclass(frozen=True)
class SmokeResult:
    config: SmokeConfig
    loaded: bool
    total_parameter_count: int
    trainable_parameter_count: int
    trainable_parameter_names: tuple[str, ...]
    lora_parameter_count: int
    parameter_prefixes: tuple[str, ...]


def _validate_config(config: SmokeConfig) -> None:
    if not isinstance(config.model, str) or not config.model.strip():
        raise ValueError("model must be a nonempty string")
    if config.mode not in _VALID_MODES:
        raise ValueError(f"unsupported mode {config.mode!r}")
    if config.level not in _VALID_LEVELS:
        raise ValueError(f"unsupported level {config.level!r}")
    if not isinstance(config.device, str) or not _DEVICE_RE.fullmatch(config.device):
        raise ValueError(f"unsupported device {config.device!r}")
    seen: set[str] = set()
    for component in config.lora_list:
        if not isinstance(component, str) or not component:
            raise ValueError(f"invalid lora component {component!r}")
        if component in seen:
            raise ValueError(f"duplicate lora component {component!r}")
        seen.add(component)


def build_loader_kwargs(config: SmokeConfig) -> dict[str, object]:
    """Return only ASR model arguments; never add VAD, punctuation, or a dataset path."""
    _validate_config(config)
    if config.mode == "lora":
        return {
            "model": config.model,
            "device": config.device,
            "disable_update": True,
            "encoder_conf": {},
            "decoder_conf": {"lora_list": list(config.lora_list)},
            "lora_only": True,
        }
    return {
        "model": config.model,
        "device": config.device,
    }


def _default_loader(**kwargs: object) -> object:
    from funasr import AutoModel  # imported only at load time

    return AutoModel(**kwargs)


def _parameter_inventory(model: object) -> tuple[int, int, tuple[str, ...], int, tuple[str, ...]]:
    entries = list(model.named_parameters())  # type: ignore[attr-defined]
    total = len(entries)
    lora_count = sum(1 for name, _ in entries if "lora_" in name)
    trainable = tuple(name for name, param in entries if param.requires_grad)
    prefixes = tuple(sorted({name.split(".", 1)[0] for name, _ in entries}))
    return total, len(trainable), trainable, lora_count, prefixes


def run_smoke(config: SmokeConfig, *, loader: Callable[..., object] | None = None) -> SmokeResult:
    """Validate configuration or load one ASR model without decoding, training, or checkpoint writes."""
    _validate_config(config)
    if config.level == "config":
        return SmokeResult(
            config=config,
            loaded=False,
            total_parameter_count=0,
            trainable_parameter_count=0,
            trainable_parameter_names=(),
            lora_parameter_count=0,
            parameter_prefixes=(),
        )

    kwargs = build_loader_kwargs(config)
    loaded = (loader or _default_loader)(**kwargs)
    model = loaded.model  # type: ignore[attr-defined]

    if config.mode == "freeze_encoder":
        for name, param in model.named_parameters():  # type: ignore[attr-defined]
            if name.startswith("encoder."):
                param.requires_grad = False

    total, trainable_count, trainable_names, lora_count, prefixes = _parameter_inventory(model)
    if config.mode == "lora" and lora_count == 0:
        raise ValueError("lora mode loaded a model with no lora_ parameters")

    return SmokeResult(
        config=config,
        loaded=True,
        total_parameter_count=total,
        trainable_parameter_count=trainable_count,
        trainable_parameter_names=trainable_names,
        lora_parameter_count=lora_count,
        parameter_prefixes=prefixes,
    )
