"""Run the frozen FunASR configuration on R5 oracle mixture AND clean-target audio.

Uses the same FunASR family as the Dataset-A raw ASR (``paraformer-zh`` +
``fsmn-vad`` + ``ct-punc``, hotword preset ``assistant``) on CUDA, on both the
mixture and the clean-target path for every manifest row. Supports resume and
validates each audio file's SHA-256 against the manifest digest before accepting
its transcript (fail-closed on mismatch). Records ASR-segment batch-1 latency
(warm-up excluded, CUDA synchronization) - explicitly **not** full-pipeline
latency (no enrollment/TSE/routing overhead).

No Dataset-A file is read.
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

from xh202615.r5_oracle import asr_config_digest, read_r5_manifest, sha256_file

WARMUP = 5  # throwaway transcriptions before latency is recorded

ASSISTANT_HOTWORD = (
    "科慕 COLMO Hi COLMO 你好科慕 "
    "空调 灯光 灯 窗帘 纱帘 洗衣机 洗碗机 烟机 烤箱 冰箱 新风 热水器 电视 屏幕 "
    "打开 关闭 设置 调整 切换 播放 暂停 音乐 歌曲 专辑 电影 儿歌 新闻 故事 "
    "出门 回家 吃饭 睡觉 做饭 洗澡 什么 怎么 叫什么 推荐 查询 "
    "闹钟 提醒 定时 倒计时 预约 取消 天气 导航"
)


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


def make_model(args):
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise SystemExit(
            "FunASR is not installed. Install with: pip install -U funasr modelscope huggingface_hub"
        ) from exc
    kwargs = {"model": args.model, "device": args.device, "disable_update": True}
    if args.vad_model:
        kwargs["vad_model"] = args.vad_model
    if args.punc_model:
        kwargs["punc_model"] = args.punc_model
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    return AutoModel(**kwargs)


def transcribe_one(model, audio_path: Path, args, *, sync) -> tuple[str, float]:
    if sync:
        import torch
        torch.cuda.synchronize()
    start = time.perf_counter()
    kwargs = {"input": str(audio_path), "batch_size_s": args.batch_size_s}
    hotword = (args.hotword or "")
    if args.hotword_preset == "assistant":
        hotword = (hotword + " " + ASSISTANT_HOTWORD).strip()
    if hotword:
        kwargs["hotword"] = hotword
    if args.language:
        kwargs["language"] = args.language
    if args.use_itn is not None:
        kwargs["use_itn"] = args.use_itn == "true"
    result = model.generate(**kwargs)
    if sync:
        import torch
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    from xh202615.text import clean_asr_text
    return clean_asr_text(extract_text(result)), latency_ms


def load_manifests(args) -> list:
    rows = []
    if args.index:
        index = json.loads(Path(args.index).read_text(encoding="utf-8"))
        for seed, info in index["seeds"].items():
            rows.extend(read_r5_manifest(info["manifest_path"]))
    else:
        for m in args.manifest:
            rows.extend(read_r5_manifest(m))
    return rows


def load_done_keys(path: Path, manifest_by_row: dict, config_digest: str,
                   *, recheck_digests: bool = True) -> set[str]:
    """Return ``row_id|path_role`` keys that are integrity-safe to skip on resume.

    A saved record counts as done only if it is successful (no error), digest_ok,
    its stored ``config_digest`` matches the current configuration, its stored
    ``manifest_digest`` matches the current manifest digest for that row+path,
    and (unless ``recheck_digests`` is False) the current audio file's recomputed
    SHA-256 matches the manifest digest. Errored, digest-failed, config-mismatched,
    manifest-changed, or audio-changed records are NOT done and must be retried.
    """
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "row_id" not in rec or "path_role" not in rec:
                continue
            if rec.get("error"):
                continue
            if not rec.get("digest_ok"):
                continue
            if rec.get("config_digest") != config_digest:
                continue
            rid = rec["row_id"]
            role = rec["path_role"]
            row = manifest_by_row.get(rid)
            if row is None:
                continue  # row gone from manifest -> cannot validate -> retry
            manifest_digest = row.mixture_digest if role == "mixture" else row.clean_digest
            if rec.get("manifest_digest") != manifest_digest:
                continue  # manifest changed since this record -> retry
            if recheck_digests:
                audio_path = Path(row.mixture_audio if role == "mixture" else row.clean_target_audio)
                if not audio_path.is_file():
                    continue
                if sha256_file(audio_path) != manifest_digest:
                    continue  # audio file changed -> retry
            done.add(f"{rid}|{role}")
    return done


def compact_output(path: Path, keep_keys: set[str]) -> int:
    """Atomically rewrite the output keeping at most one current record per
    ``(row_id, path_role)``.

    Retains only records whose key is in ``keep_keys`` (the validated set from
    :func:`load_done_keys`), one record per key, dropping errored, stale,
    config/manifest/digest-mismatched, and duplicate records. This guarantees a
    retried result replaces an earlier failed/stale record instead of coexisting
    with it. Returns the number of records retained.
    """
    if not path.exists():
        return 0
    kept: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                continue  # corrupt line -> drop
            rid = rec.get("row_id")
            role = rec.get("path_role")
            if rid is None or role is None:
                continue  # unknown record -> drop
            key = f"{rid}|{role}"
            if key in keep_keys and key not in seen:
                seen.add(key)
                kept.append(stripped)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    tmp.replace(path)  # atomic rename
    return len(kept)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=None,
                        help="R5 manifest.jsonl (repeatable)")
    parser.add_argument("--index", default=None,
                        help="r5_oracle_v1/index.json (processes both seeds)")
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
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--on-error", choices=("empty", "raise"), default="empty")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.index and not args.manifest:
        raise SystemExit("provide --index or at least one --manifest")
    rows = load_manifests(args)
    if args.limit is not None:
        rows = rows[: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    config_digest = asr_config_digest(
        model=args.model, vad_model=args.vad_model, punc_model=args.punc_model,
        hotword=args.hotword, hotword_preset=args.hotword_preset,
        language=args.language, use_itn=args.use_itn, device=args.device,
        batch_size_s=args.batch_size_s, trust_remote_code=args.trust_remote_code,
    )
    manifest_by_row = {row.row_id: row for row in rows}
    if args.resume:
        done = load_done_keys(out, manifest_by_row, config_digest)
        # Compaction: drop errored/stale/duplicate records so retried results
        # replace them without coexistence. Keeps one validated record per key.
        kept = compact_output(out, done)
        print(f"resume: compacted output to {kept} validated records "
              f"(done={len(done)})", flush=True)
        mode = "a"
    else:
        done = set()
        mode = "w"
    # Build the work list: (row, path_role, audio_path, manifest_digest).
    work = []
    for row in rows:
        for path_role, audio_field, digest_field in (
            ("mixture", "mixture_audio", "mixture_digest"),
            ("clean_target", "clean_target_audio", "clean_digest"),
        ):
            key = f"{row.row_id}|{path_role}"
            if key in done:
                continue
            work.append((row, path_role, Path(getattr(row, audio_field)),
                         getattr(row, digest_field)))
    print(f"rows={len(rows)} pending_transcriptions={len(work)} "
          f"skipped(resume)={len(done)} output={out}", flush=True)
    print(f"config_digest={config_digest}", flush=True)
    if not work:
        return

    model = make_model(args)
    sync = args.device.startswith("cuda")
    # Warm-up: throwaway transcriptions (not recorded) so recorded latency is warm.
    warmup_audios = []
    for row, path_role, audio_path, _ in work:
        if len(warmup_audios) >= args.warmup:
            break
        if audio_path.exists():
            warmup_audios.append(audio_path)
    for wa in warmup_audios:
        try:
            transcribe_one(model, wa, args, sync=sync)
        except Exception as exc:  # noqa: BLE001 - warmup failures are non-fatal
            print(f"warmup transcribe error ({wa}): {exc}", file=sys.stderr)
    print(f"warmup done ({len(warmup_audios)} throwaway)", flush=True)

    total = len(work)
    with out.open(mode, encoding="utf-8", newline="\n") as handle:
        for idx, (row, path_role, audio_path, manifest_digest) in enumerate(work, start=1):
            error = None
            text = ""
            latency_ms = 0.0
            digest = ""
            digest_ok = False
            try:
                if not audio_path.exists():
                    raise FileNotFoundError(f"missing audio: {audio_path}")
                digest = sha256_file(audio_path)
                digest_ok = digest == manifest_digest
                if not digest_ok:
                    raise ValueError(
                        f"digest mismatch for {row.row_id}/{path_role}: "
                        f"file={digest} manifest={manifest_digest}"
                    )
                text, latency_ms = transcribe_one(model, audio_path, args, sync=sync)
            except Exception as exc:  # noqa: BLE001
                if args.on_error == "raise":
                    raise
                error = str(exc)
                print(f"[{idx}/{total}] {row.row_id}/{path_role} ERROR: {error}", file=sys.stderr)
            rec = {
                "row_id": row.row_id,
                "seed": row.seed,
                "split": row.split,
                "path_role": path_role,
                "snr_db": row.snr_db,
                "overlap_ratio": row.overlap_ratio,
                "transcript": row.transcript,
                "asr_text": text,
                "latency_ms": latency_ms,
                "digest": digest,
                "manifest_digest": manifest_digest,
                "config_digest": config_digest,
                "digest_ok": digest_ok,
            }
            if error:
                rec["error"] = error
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            if idx == 1 or idx % 25 == 0 or idx == total:
                print(f"[{idx}/{total}] {row.row_id}/{path_role} text={text!r} "
                      f"lat={latency_ms:.1f}ms", flush=True)
    print(f"wrote {total} transcriptions to {out}", flush=True)


if __name__ == "__main__":
    main()
