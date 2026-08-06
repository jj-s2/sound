"""Batch transcribe command_audio with FunASR and write an ASR map JSONL.

The output format is intentionally compatible with
`xh202615.backends.TranscriptMapAsrBackend`:

    {"id": "0", "text": "打开空调"}

Install dependencies first:

    pip install -U funasr modelscope huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import load_dataset, read_jsonl
from xh202615.text import clean_asr_text


MODEL_ALIASES = {
    "SenseVoiceSmall": "iic/SenseVoiceSmall",
    "sensevoice": "iic/SenseVoiceSmall",
}

ASSISTANT_HOTWORD = (
    "科慕 COLMO Hi COLMO 你好科慕 "
    "空调 灯光 灯 窗帘 纱帘 洗衣机 洗碗机 烟机 烤箱 冰箱 新风 热水器 电视 屏幕 "
    "打开 关闭 设置 调整 切换 播放 暂停 音乐 歌曲 专辑 电影 儿歌 新闻 故事 "
    "出门 回家 吃饭 睡觉 做饭 洗澡 什么 怎么 叫什么 推荐 查询 "
    "闹钟 提醒 定时 倒计时 预约 取消 天气 导航"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run FunASR on Dataset-A style command_audio files")
    parser.add_argument("--dataset-root", default="datasetA")
    parser.add_argument("--splits", default="pos,neg", help="Comma-separated splits, e.g. pos,neg")
    parser.add_argument("--output", default="output/asr/funasr_predictions.jsonl")
    parser.add_argument(
        "--command-audio-map",
        default=None,
        help="Optional CSV/JSONL mapping id to enhanced_command_audio/command_audio/audio path",
    )
    parser.add_argument("--ids-file", default=None, help="Optional CSV/JSONL/text file containing ids to transcribe")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-split-limit", type=int, default=None, help="Limit each split before combining")
    parser.add_argument("--resume", action="store_true", help="Append and skip ids already present in output")
    parser.add_argument("--device", default="cpu", help='FunASR device, e.g. "cpu" or "cuda:0"')
    parser.add_argument("--model", default="paraformer-zh")
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--punc-model", default="ct-punc")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to AutoModel")
    parser.add_argument("--batch-size-s", type=int, default=300)
    parser.add_argument("--hotword", default=None, help="Optional FunASR hotword string")
    parser.add_argument(
        "--hotword-preset",
        choices=("none", "assistant"),
        default="none",
        help="Optional broad smart-assistant hotword preset. Appends to --hotword when both are provided.",
    )
    parser.add_argument("--language", default=None, help='Optional generate language, e.g. "zh" or "auto"')
    parser.add_argument("--use-itn", choices=("true", "false"), default=None, help="Optional generate use_itn flag")
    parser.add_argument("--merge-vad", choices=("true", "false"), default=None, help="Optional generate merge_vad flag")
    parser.add_argument("--merge-length-s", type=int, default=None, help="Optional generate merge_length_s value")
    parser.add_argument(
        "--on-error",
        choices=("empty", "raise"),
        default="empty",
        help="Write empty text on per-sample errors, or stop immediately",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.lower() == "true"


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["id"]) for row in read_jsonl(path) if "id" in row}


def load_command_audio_map(path: str | Path | None) -> dict[str, Path]:
    if not path:
        return {}
    path = Path(path)
    values = {}
    if path.suffix.lower() == ".csv":
        import csv

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                sample_id = str(row.get("id", row.get("sample_id", "")))
                audio = row.get("enhanced_command_audio", row.get("command_audio", row.get("audio", "")))
                if sample_id and audio:
                    values[sample_id] = Path(audio)
        return values

    for row in read_jsonl(path):
        sample_id = str(row.get("id", row.get("sample_id", "")))
        audio = row.get("enhanced_command_audio", row.get("command_audio", row.get("audio", "")))
        if sample_id and audio:
            values[sample_id] = Path(audio)
    return values


def load_ids(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    path = Path(path)
    ids: set[str] = set()
    if path.suffix.lower() == ".csv":
        import csv

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("id"):
                    ids.add(str(row["id"]))
    elif path.suffix.lower() == ".jsonl":
        for row in read_jsonl(path):
            if "id" in row:
                ids.add(str(row["id"]))
    else:
        with path.open("r", encoding="utf-8-sig") as f:
            ids.update(line.strip() for line in f if line.strip())
    return ids


def make_model(args):
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise SystemExit(
            "FunASR is not installed. Install it with:\n"
            "  pip install -U funasr modelscope huggingface_hub"
        ) from exc

    kwargs = {
        "model": MODEL_ALIASES.get(args.model, args.model),
        "device": args.device,
        "disable_update": True,
    }
    if args.vad_model and args.vad_model.lower() not in {"none", "null", "off"}:
        kwargs["vad_model"] = args.vad_model
    if args.punc_model and args.punc_model.lower() not in {"none", "null", "off"}:
        kwargs["punc_model"] = args.punc_model
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    return AutoModel(**kwargs)


def extract_text(result: Any) -> str:
    """Extract text from FunASR results across common 1.x output shapes."""

    if result is None:
        return ""
    if isinstance(result, dict):
        return "" if result.get("text") is None else str(result.get("text"))
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict) and item.get("text") is not None:
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(result)


def transcribe_one(model, audio_path: Path, args) -> tuple[str, float]:
    start = time.perf_counter()
    kwargs = {"input": str(audio_path), "batch_size_s": args.batch_size_s}
    hotword = args.hotword or ""
    if args.hotword_preset == "assistant":
        hotword = (hotword + " " + ASSISTANT_HOTWORD).strip()
    if hotword:
        kwargs["hotword"] = hotword
    if args.language:
        kwargs["language"] = args.language
    if args.use_itn is not None:
        kwargs["use_itn"] = parse_bool(args.use_itn)
    if args.merge_vad is not None:
        kwargs["merge_vad"] = parse_bool(args.merge_vad)
    if args.merge_length_s is not None:
        kwargs["merge_length_s"] = args.merge_length_s
    result = model.generate(**kwargs)
    latency_ms = (time.perf_counter() - start) * 1000
    return clean_asr_text(extract_text(result)), latency_ms


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples = load_dataset(args.dataset_root, splits)
    if args.per_split_limit is not None:
        selected = []
        for split in splits:
            selected.extend([sample for sample in samples if sample.split == split][: args.per_split_limit])
        samples = selected
    ids = load_ids(args.ids_file)
    if ids is not None:
        samples = [sample for sample in samples if str(sample.id) in ids]
    if args.limit is not None:
        samples = samples[: args.limit]
    command_audio_map = load_command_audio_map(args.command_audio_map)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(out) if args.resume else set()
    mode = "a" if args.resume else "w"

    pending = [sample for sample in samples if str(sample.id) not in done_ids]
    print(f"Loaded {len(samples)} samples, pending {len(pending)}, output={out}")
    if not pending:
        return

    model = make_model(args)
    total = len(pending)
    with out.open(mode, encoding="utf-8", newline="\n") as f:
        for idx, sample in enumerate(pending, start=1):
            error = None
            text = ""
            latency_ms = 0.0
            try:
                audio_path = command_audio_map.get(str(sample.id), sample.command_audio)
                if not audio_path.exists():
                    raise FileNotFoundError(f"Missing command_audio: {audio_path}")
                text, latency_ms = transcribe_one(model, audio_path, args)
            except Exception as exc:
                if args.on_error == "raise":
                    raise
                error = str(exc)
                print(f"[{idx}/{total}] id={sample.id} ERROR: {error}", file=sys.stderr)

            row = {
                "id": str(sample.id),
                "text": text,
                "split": sample.split,
                "command_audio": str(command_audio_map.get(str(sample.id), sample.command_audio)),
                "original_command_audio": str(sample.command_audio),
                "asr_backend": "funasr",
                "latency_ms": latency_ms,
            }
            if error:
                row["error"] = error
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if idx == 1 or idx % 20 == 0 or idx == total:
                print(f"[{idx}/{total}] id={sample.id} text={text!r} latency_ms={latency_ms:.1f}")

    print(f"Wrote FunASR predictions to {out}")


if __name__ == "__main__":
    main()
