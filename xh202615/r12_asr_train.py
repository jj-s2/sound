"""Validated R12 Paraformer training command builder.

The module has no FunASR import at import time.  ``dry_run`` validates only
private manifests and returns the exact command; ``train`` is explicit and
delegates to FunASR only after all local safeguards pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

TrainMode = Literal["lora", "freeze_encoder"]
_PRIVATE_KEYS = frozenset({"key", "source", "target", "parent_id", "augmentation_id"})


@dataclass(frozen=True)
class TrainingConfig:
    train_manifest: Path
    valid_manifest: Path
    output_dir: Path
    model: str
    device: str
    mode: TrainMode
    seed: int = 20260814


@dataclass(frozen=True)
class TrainingResult:
    argv: tuple[str, ...]
    executed: bool
    return_code: int | None


def _reject_internal_test(path: Path) -> None:
    if "internal_test" in str(path).lower() or "held_out" in str(path).lower():
        raise ValueError("internal-test paths are forbidden for ASR training")


def _validate_manifest(path: Path) -> None:
    _reject_internal_test(path)
    if not path.is_file():
        raise ValueError(f"manifest does not exist: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"manifest is empty: {path}")
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict) or frozenset(row) != _PRIVATE_KEYS:
            raise ValueError(f"manifest row has invalid keys at {path}:{line_number}")
        if not all(isinstance(row[key], str) and row[key].strip() for key in _PRIVATE_KEYS):
            raise ValueError(f"manifest row has an empty field at {path}:{line_number}")
        _reject_internal_test(Path(row["source"]))


def _validate_config(config: TrainingConfig) -> None:
    if config.mode not in {"lora", "freeze_encoder"}:
        raise ValueError(f"unsupported training mode: {config.mode!r}")
    if not config.model.strip():
        raise ValueError("model must be nonempty")
    if not config.device.startswith("cuda:"):
        raise ValueError("actual training requires an explicit cuda:N device")
    _validate_manifest(config.train_manifest)
    _validate_manifest(config.valid_manifest)
    if config.output_dir.exists():
        raise ValueError(f"output directory already exists: {config.output_dir}")


def build_train_argv(config: TrainingConfig) -> tuple[str, ...]:
    """Build the minimal FunASR ``train_ds`` invocation for one R12 fold."""
    _validate_config(config)
    argv = [
        sys.executable,
        "-m",
        "funasr.bin.train_ds",
        f"model={config.model}",
        f"device={config.device}",
        f"output_dir={config.output_dir}",
        f"seed={config.seed}",
        f"dataset_conf.data_list={config.train_manifest}",
        f"dataset_conf.data_list_valid={config.valid_manifest}",
    ]
    if config.mode == "lora":
        argv.extend(("lora_only=true", "decoder_conf.lora_list=[q,k,v,o]"))
    else:
        argv.append("freeze_param=encoder")
    return tuple(argv)


def _subprocess_runner(argv: Sequence[str]) -> int:
    return subprocess.run(tuple(argv), check=False).returncode


def run_training(
    config: TrainingConfig,
    *,
    dry_run: bool = False,
    runner: Callable[[Sequence[str]], int] | None = None,
) -> TrainingResult:
    """Validate the recipe, or explicitly execute the FunASR training module."""
    argv = build_train_argv(config)
    if dry_run:
        return TrainingResult(argv=argv, executed=False, return_code=None)
    code = (runner or _subprocess_runner)(argv)
    return TrainingResult(argv=argv, executed=True, return_code=code)
