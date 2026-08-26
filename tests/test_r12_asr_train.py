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

    config = _config(tmp_path)
    _manifest(config.train_manifest, source="private/internal_test/p.wav")
    with pytest.raises(ValueError, match="internal-test"):
        run_training(config, dry_run=True)


def test_train_argv_is_lora_only_and_contains_no_vad_or_punctuation(tmp_path: Path) -> None:
    from xh202615.r12_asr_train import build_train_argv

    rendered = " ".join(build_train_argv(_config(tmp_path)))

    assert "lora_only=true" in rendered
    assert "decoder_conf.lora_list=[q,k,v,o]" in rendered
    assert "vad" not in rendered.lower()
    assert "punc" not in rendered.lower()


def test_lora_argv_exposes_bounded_encoder_decoder_recipe(tmp_path: Path) -> None:
    from xh202615.r12_asr_train import build_train_argv

    rendered = " ".join(build_train_argv(_config(tmp_path)))

    assert "encoder_conf.lora_list=[q,k,v,o]" in rendered
    assert "decoder_conf.lora_list=[q,k,v,o]" in rendered
    assert "encoder_conf.lora_rank=8" in rendered
    assert "decoder_conf.lora_rank=8" in rendered
    assert "encoder_conf.lora_alpha=16" in rendered
    assert "decoder_conf.lora_alpha=16" in rendered
    assert "encoder_conf.lora_dropout=0.05" in rendered
    assert "decoder_conf.lora_dropout=0.05" in rendered
    assert "optim_conf.lr=0.0001" in rendered
    assert "train_conf.max_epoch=30" in rendered
    assert "train_conf.keep_nbest_models=10" in rendered
    assert "train_conf.avg_nbest_model=5" in rendered
    assert "dataset_conf.batch_size=800" in rendered
    assert "dataset_conf.num_workers=2" in rendered
    assert "accum_grad=8" in rendered


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lora_rank", 0, "lora_rank"),
        ("lora_alpha", 0, "lora_alpha"),
        ("lora_dropout", 1.0, "lora_dropout"),
        ("learning_rate", 0.0, "learning_rate"),
        ("batch_size", 0, "batch_size"),
        ("accum_grad", 0, "accum_grad"),
    ],
)
def test_lora_recipe_rejects_invalid_resource_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    from xh202615.r12_asr_train import build_train_argv

    with pytest.raises(ValueError, match=message):
        build_train_argv(_config(tmp_path, **{field: value}))


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
