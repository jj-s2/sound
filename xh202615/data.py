"""Dataset-A style JSONL reading with robust field aliases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FIELD_ALIASES = {
    "id": ["id", "utt_id", "sample_id"],
    "wakeup_audio": ["wakeup_audio", "kws_path", "唤醒音频", "鍞ら啋闊抽"],
    "wakeup_text": ["wakeup_text", "kws_txt", "唤醒文本", "鍞ら啋鏂囨湰"],
    "command_audio": ["command_audio", "cmd_path", "识别音频", "璇嗗埆闊抽"],
    "label": ["recognition_text", "label", "识别文本", "璇嗗埆鏂囨湰"],
}


@dataclass(frozen=True)
class Sample:
    id: str
    split: str
    wakeup_audio: Path
    wakeup_text: str
    command_audio: Path
    label: str | None


def _first_present(row: dict, canonical: str, default=None):
    for key in FIELD_ALIASES[canonical]:
        if key in row:
            return row[key]
    return default


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def load_split(dataset_root: str | Path, split: str) -> list[Sample]:
    root = Path(dataset_root)
    jsonl_path = root / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing split file: {jsonl_path}")

    samples: list[Sample] = []
    for row in read_jsonl(jsonl_path):
        sample_id = str(_first_present(row, "id"))
        wake_rel = _first_present(row, "wakeup_audio")
        cmd_rel = _first_present(row, "command_audio")
        if not wake_rel or not cmd_rel:
            raise KeyError(f"Sample {sample_id} missing wakeup/command audio path")
        samples.append(
            Sample(
                id=sample_id,
                split=split,
                wakeup_audio=root / str(wake_rel),
                wakeup_text=str(_first_present(row, "wakeup_text", "")),
                command_audio=root / str(cmd_rel),
                label=_first_present(row, "label"),
            )
        )
    return samples


def load_dataset(dataset_root: str | Path, splits: Iterable[str] = ("pos", "neg")) -> list[Sample]:
    samples: list[Sample] = []
    for split in splits:
        samples.extend(load_split(dataset_root, split))
    return samples

