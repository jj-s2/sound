from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf


def _config(tmp_path: Path):
    from xh202615.r12_training_bootstrap import BootstrapConfig

    root = tmp_path / "dataset"
    root.mkdir()
    labels: dict[str, str | None] = {}
    groups: dict[str, str] = {}
    for split in ("pos", "neg"):
        rows = []
        for index in range(20):
            sample_id = f"{split}-{index}"
            group = f"g-{index}"
            audio = Path(split) / f"{sample_id}.wav"
            (root / audio).parent.mkdir(exist_ok=True)
            sf.write(root / audio, np.full(160, 0.1, dtype=np.float32), 16000)
            rows.append({"id": sample_id, "wakeup_audio": str(audio), "command_audio": str(audio)})
            labels[sample_id] = f"命令{index}" if split == "pos" else None
            groups[sample_id] = group
        (root / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    label_path, group_path = tmp_path / "labels.json", tmp_path / "groups.json"
    label_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    group_path.write_text(json.dumps(groups), encoding="utf-8")
    return BootstrapConfig(root, label_path, group_path, tmp_path / "run")


def test_plan_selects_group_disjoint_inner_valid_from_train_only(tmp_path: Path) -> None:
    from xh202615.r12_training_bootstrap import plan_bootstrap

    plan = plan_bootstrap(_config(tmp_path))

    assert plan.inner_valid_parent_ids <= plan.train_parent_ids
    assert plan.inner_valid_groups.isdisjoint(plan.fit_groups)
    assert len(plan.inner_valid_groups) == 1


def test_materialize_excludes_internal_test_from_private_asr_rows(tmp_path: Path) -> None:
    from xh202615.r12_training_bootstrap import materialize_bootstrap

    result = materialize_bootstrap(_config(tmp_path))

    rendered = result.train_jsonl.read_text(encoding="utf-8")
    assert result.train_jsonl.is_file()
    assert result.inner_valid_jsonl.is_file()
    assert result.folds_path.is_file()
    assert result.hotword_summary.is_file()
    assert "internal_test" not in rendered


def test_dry_run_does_not_write_output_root(tmp_path: Path) -> None:
    from xh202615.r12_training_bootstrap import dry_run_bootstrap

    config = _config(tmp_path)

    dry_run_bootstrap(config)

    assert not config.output_root.exists()
