"""R4 public ASR bakeoff: transcribe public R3 mixtures with frozen FunASR.

This produces a fully-public CER reference by running the same FunASR family
used for the Dataset-A raw ASR (``stage_asr_only`` == ``funasr_full_hotword_safe``)
on the public R3 mixtures, then mapping each row's ``target_speaker_id`` to
its AISHELL transcript.  No Dataset-A file is read.

Output JSONL rows::

    {"row_id": ..., "split": ..., "target_present": bool,
     "target_speaker_id": "BAC009S0002W0421", "transcript": "<aishell>",
     "raw_asr_text": "<funasr>", "latency_ms": float, "mixture_audio": "..."}

The downstream ``r4_select_rescue_threshold`` script consumes this together
with the frozen temporal-head probabilities to select the public rescue
threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import read_jsonl
from xh202615.text import clean_asr_text

DEFAULT_TRANSCRIPT = "data_aishell (1)/data_aishell/transcript/aishell_transcript_v0.8.txt"


def load_aishell_transcripts(path: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]
    return transcripts


def load_manifest_rows(manifest: Path, splits: tuple[str, ...]) -> list[dict]:
    rows = [row for row in read_jsonl(manifest) if row.get("split") in splits]
    return rows


def make_model(args) -> "AutoModel":  # type: ignore[name-defined]
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise SystemExit(
            "FunASR is not installed. Install it with:\n"
            "  pip install -U funasr modelscope huggingface_hub"
        ) from exc

    kwargs = {
        "model": args.model,
        "device": args.device,
        "disable_update": True,
    }
    if args.vad_model:
        kwargs["vad_model"] = args.vad_model
    if args.punc_model:
        kwargs["punc_model"] = args.punc_model
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    return AutoModel(**kwargs)


def extract_text(result) -> str:
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


ASSISTANT_HOTWORD = (
    "科慕 COLMO Hi COLMO 你好科慕 "
    "空调 灯光 灯 窗帘 纱帘 洗衣机 洗碗机 烟机 烤箱 冰箱 新风 热水器 电视 屏幕 "
    "打开 关闭 设置 调整 切换 播放 暂停 音乐 歌曲 专辑 电影 儿歌 新闻 故事 "
    "出门 回家 吃饭 睡觉 做饭 洗澡 什么 怎么 叫什么 推荐 查询 "
    "闹钟 提醒 定时 倒计时 预约 取消 天气 导航"
)


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
        kwargs["use_itn"] = args.use_itn == "true"
    result = model.generate(**kwargs)
    latency_ms = (time.perf_counter() - start) * 1000
    return clean_asr_text(extract_text(result)), latency_ms


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["row_id"]) for row in read_jsonl(path) if "row_id" in row}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/r3_public_pilot_v1_training.jsonl")
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits to transcribe")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="paraformer-zh")
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--punc-model", default="ct-punc")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size-s", type=int, default=300)
    parser.add_argument("--hotword", default=None)
    parser.add_argument("--hotword-preset", choices=("none", "assistant"), default="assistant")
    parser.add_argument("--language", default=None)
    parser.add_argument("--use-itn", choices=("true", "false"), default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap rows (after split filter)")
    parser.add_argument("--resume", action="store_true", help="Skip row_ids already in output")
    parser.add_argument("--on-error", choices=("empty", "raise"), default="empty")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    rows = load_manifest_rows(Path(args.manifest), splits)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no rows matched the requested splits")

    transcripts = load_aishell_transcripts(Path(args.transcript))
    missing = sorted({r["target_speaker_id"] for r in rows if r["target_speaker_id"] not in transcripts})
    if missing:
        raise SystemExit(f"{len(missing)} target_speaker_id(s) missing from transcript file; e.g. {missing[:3]}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(out) if args.resume else set()
    pending = [r for r in rows if str(r["row_id"]) not in done_ids]
    print(f"manifest={args.manifest} splits={splits} rows={len(rows)} pending={len(pending)} output={out}", flush=True)
    if not pending:
        return

    model = make_model(args)
    total = len(pending)
    mode = "a" if (args.resume and done_ids) else "w"
    with out.open(mode, encoding="utf-8", newline="\n") as handle:
        for idx, row in enumerate(pending, start=1):
            error = None
            text = ""
            latency_ms = 0.0
            mixture = Path(row["mixture_audio"])
            try:
                if not mixture.exists():
                    raise FileNotFoundError(f"missing mixture: {mixture}")
                text, latency_ms = transcribe_one(model, mixture, args)
            except Exception as exc:
                if args.on_error == "raise":
                    raise
                error = str(exc)
                print(f"[{idx}/{total}] row={row['row_id']} ERROR: {error}", file=sys.stderr)
            record = {
                "row_id": str(row["row_id"]),
                "split": str(row["split"]),
                "target_present": bool(row["target_present"]),
                "target_speaker_id": str(row["target_speaker_id"]),
                "transcript": transcripts[row["target_speaker_id"]],
                "raw_asr_text": text,
                "latency_ms": latency_ms,
                "mixture_audio": str(mixture),
            }
            if error:
                record["error"] = error
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if idx == 1 or idx % 20 == 0 or idx == total:
                print(f"[{idx}/{total}] row={row['row_id']} present={row['target_present']} text={text!r}", flush=True)
    print(f"wrote {total} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
