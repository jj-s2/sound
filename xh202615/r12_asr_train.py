"""Validated R12 Paraformer training command builder.

The module has no FunASR import at import time.  ``dry_run`` validates only
private manifests and returns the exact command; ``train`` is explicit and
delegates to FunASR only after all local safeguards pass.
"""

from __future__ import annotations

import json
import math
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
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    max_epoch: int = 30
    keep_nbest_models: int = 10
    avg_nbest_model: int = 5
    batch_size: int = 800
    accum_grad: int = 8
    num_workers: int = 2


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
    for name in (
        "lora_rank",
        "lora_alpha",
        "max_epoch",
        "keep_nbest_models",
        "avg_nbest_model",
        "batch_size",
        "accum_grad",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(config.num_workers, bool) or not isinstance(config.num_workers, int) or config.num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if not isinstance(config.lora_dropout, (int, float)) or not math.isfinite(config.lora_dropout):
        raise ValueError("lora_dropout must be finite")
    if not 0.0 <= float(config.lora_dropout) < 1.0:
        raise ValueError("lora_dropout must be in [0, 1)")
    if not isinstance(config.learning_rate, (int, float)) or not math.isfinite(config.learning_rate):
        raise ValueError("learning_rate must be finite")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.avg_nbest_model > config.keep_nbest_models:
        raise ValueError("avg_nbest_model cannot exceed keep_nbest_models")
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
        f"+model={config.model}",
        f"+device={config.device}",
        f"+output_dir={config.output_dir}",
        f"+seed={config.seed}",
        f"+train_data_set_list={config.train_manifest}",
        f"+valid_data_set_list={config.valid_manifest}",
    ]
    if config.mode == "lora":
        argv.extend(
            (
                "+lora_only=true",
                "+encoder_conf.lora_list=[q,k,v,o]",
                "+decoder_conf.lora_list=[q,k,v,o]",
                f"+encoder_conf.lora_rank={config.lora_rank}",
                f"+decoder_conf.lora_rank={config.lora_rank}",
                f"+encoder_conf.lora_alpha={config.lora_alpha}",
                f"+decoder_conf.lora_alpha={config.lora_alpha}",
                f"+encoder_conf.lora_dropout={config.lora_dropout}",
                f"+decoder_conf.lora_dropout={config.lora_dropout}",
                f"+optim_conf.lr={config.learning_rate}",
                f"+train_conf.max_epoch={config.max_epoch}",
                f"+train_conf.keep_nbest_models={config.keep_nbest_models}",
                f"+train_conf.avg_nbest_model={config.avg_nbest_model}",
                f"+dataset_conf.batch_size={config.batch_size}",
                f"+dataset_conf.num_workers={config.num_workers}",
                f"+train_conf.accum_grad={config.accum_grad}",
            )
        )
    else:
        argv.append("+freeze_param=encoder")
    return tuple(argv)


def _manifest_source_root(path: Path) -> Path | None:
    """Find the augmented dataset root for relative manifest audio paths."""
    resolved = Path(path).resolve(strict=False)
    for ancestor in resolved.parents:
        candidate = ancestor / "augmented"
        if candidate.is_dir():
            return candidate
    return None


def _subprocess_runner(argv: Sequence[str], *, cwd: Path | None = None) -> int:
    return subprocess.run(
        tuple(argv), check=False, cwd=str(cwd) if cwd is not None else None
    ).returncode


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
    if runner is None:
        code = _subprocess_runner(argv, cwd=_manifest_source_root(config.train_manifest))
    else:
        code = runner(argv)
    return TrainingResult(argv=argv, executed=True, return_code=code)
