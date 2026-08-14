from __future__ import annotations

import json
from pathlib import Path

import pytest


def _manifest(path: Path, *, key: str = "p", source: str = "private/audio.wav") -> Path:
    path.write_text(
        json.dumps(
            {
                "key": key,
                "source": source,
                "target": "开灯",
                "parent_id": "p",
                "augmentation_id": "original",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _config(tmp_path: Path, **overrides: object):
    from xh202615.r12_asr_train import TrainingConfig

    values: dict[str, object] = {
        "train_manifest": _manifest(tmp_path / "train.jsonl"),
        "valid_manifest": _manifest(tmp_path / "valid.jsonl", key="v"),
        "output_dir": tmp_path / "run",
        "model": "paraformer-zh",
        "device": "cuda:0",
        "mode": "lora",
    }
    values.update(overrides)
    return TrainingConfig(**values)


def test_dry_run_returns_train_ds_command_without_running_funasr(tmp_path: Path) -> None:
    from xh202615.r12_asr_train import run_training

    result = run_training(_config(tmp_path), dry_run=True)

    assert result.executed is False
    assert result.argv[1:3] == ("-m", "funasr.bin.train_ds")
    assert not (tmp_path / "run").exists()


def test_training_rejects_internal_test_path_before_runner(tmp_path: Path) -> None:
    from xh202615.r12_asr_train import run_training

    def fail_if_called(_: tuple[str, ...]) -> int:
        raise AssertionError("runner must not be called")

    with pytest.raises(ValueError, match="internal-test"):
        run_training(
            _config(tmp_path, train_manifest=_manifest(tmp_path / "internal_test.jsonl")),
            runner=fail_if_called,
        )


def test_training_rejects_internal_test_audio_source_before_runner(tmp_path: Path) -> None:
    from xh202615.r12_asr_train import run_training

    with pytest.raises(ValueError, match="internal-test"):
        run_training(
            _config(tmp_path, train_manifest=_manifest(tmp_path / "train.jsonl", source="private/internal_test/p.wav")),
            dry_run=True,
        )


def test_train_argv_is_lora_only_and_contains_no_vad_or_punctuation(tmp_path: Path) -> None:
    from xh202615.r12_asr_train import build_train_argv

    rendered = " ".join(build_train_argv(_config(tmp_path)))

    assert "lora_only=true" in rendered
    assert "decoder_conf.lora_list=[q,k,v,o]" in rendered
    assert "vad" not in rendered.lower()
    assert "punc" not in rendered.lower()


def test_freeze_mode_passes_encoder_freeze_param(tmp_path: Path) -> None:
    from xh202615.r12_asr_train import build_train_argv

    argv = build_train_argv(_config(tmp_path, mode="freeze_encoder"))

    assert "freeze_param=encoder" in argv
    assert not any(item == "lora_only=true" for item in argv)


def test_existing_output_directory_is_rejected_before_runner(tmp_path: Path) -> None:
    from xh202615.r12_asr_train import run_training

    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(ValueError, match="output directory already exists"):
        run_training(_config(tmp_path, output_dir=output), runner=lambda _: 0)
