"""Build enhanced command audio for V4 ASR experiments.

This script keeps the original Dataset-A files unchanged. It writes enhanced
WAV files plus a manifest CSV that can be passed to `run_funasr_asr.py` via
`--command-audio-map`.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import load_dataset, read_jsonl


FIELDNAMES = [
    "id",
    "split",
    "original_command_audio",
    "enhanced_command_audio",
    "wakeup_audio",
    "method",
    "selected_intervals_sec",
    "duration_in_sec",
    "duration_out_sec",
    "window_count",
    "top_similarity",
    "latency_ms",
    "error",
]


@dataclass(frozen=True)
class WindowScore:
    start: int
    end: int
    rms: float
    speaker_similarity: float | None

    @property
    def duration(self) -> int:
        return self.end - self.start


def parse_args():
    parser = argparse.ArgumentParser(description="Enhance command audio by energy gate and optional target-speaker windows")
    parser.add_argument("--dataset-root", default="datasetA")
    parser.add_argument("--splits", default="pos,neg")
    parser.add_argument("--output-root", default="output/enhanced/v4_target_speaker")
    parser.add_argument("--manifest", default="output/enhanced/v4_target_speaker_manifest.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-split-limit", type=int, default=None)
    parser.add_argument("--ids-file", default=None, help="Optional CSV/JSONL/text file containing ids to enhance")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--method", choices=("energy", "speaker", "hybrid"), default="hybrid")
    parser.add_argument("--window-sec", type=float, default=1.2)
    parser.add_argument("--hop-sec", type=float, default=0.4)
    parser.add_argument("--energy-candidates", type=int, default=8)
    parser.add_argument("--keep-top-k", type=int, default=3)
    parser.add_argument("--max-output-sec", type=float, default=4.0)
    parser.add_argument("--min-output-sec", type=float, default=0.6)
    parser.add_argument("--padding-sec", type=float, default=0.12)
    parser.add_argument("--energy-weight", type=float, default=0.15)
    parser.add_argument("--model", default="chinese", help='WeSpeaker model name, e.g. "chinese"')
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--on-error", choices=("copy", "raise"), default="copy")
    return parser.parse_args()


def load_ids(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    path = Path(path)
    ids: set[str] = set()
    if path.suffix.lower() == ".csv":
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


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {str(row["id"]) for row in csv.DictReader(f) if row.get("id")}


def make_speaker_model(args):
    if args.method == "energy":
        return None
    try:
        import wespeaker
    except ImportError as exc:
        raise SystemExit(
            "WeSpeaker is required for --method speaker/hybrid. Install it first or use --method energy."
        ) from exc

    model = wespeaker.load_model(args.model)
    if args.gpu is not None and hasattr(model, "set_gpu"):
        model.set_gpu(args.gpu)
    return model


def to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)) and value:
        return to_float(value[0])
    return float(value)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def write_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.size:
        peak = float(np.max(np.abs(audio)))
        if peak > 0.98:
            audio = audio * (0.98 / peak)
    sf.write(str(path), audio, sr)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def candidate_windows(audio: np.ndarray, sr: int, args) -> list[WindowScore]:
    win = max(1, int(args.window_sec * sr))
    hop = max(1, int(args.hop_sec * sr))
    if audio.size <= win:
        return [WindowScore(0, audio.size, rms(audio), None)]

    windows = []
    for start in range(0, max(1, audio.size - win + 1), hop):
        end = min(audio.size, start + win)
        windows.append(WindowScore(start, end, rms(audio[start:end]), None))
    if windows and windows[-1].end < audio.size:
        start = max(0, audio.size - win)
        windows.append(WindowScore(start, audio.size, rms(audio[start:]), None))
    windows.sort(key=lambda w: w.rms, reverse=True)
    return windows[: max(1, args.energy_candidates)]


def score_speaker_windows(model, wakeup_audio: Path, command_audio: np.ndarray, sr: int, windows: list[WindowScore]) -> list[WindowScore]:
    if model is None:
        return windows
    scored = []
    with tempfile.TemporaryDirectory(prefix="xh202615_v4_") as tmp:
        tmp_dir = Path(tmp)
        for idx, window in enumerate(windows):
            tmp_wav = tmp_dir / f"window_{idx}.wav"
            write_audio(tmp_wav, command_audio[window.start : window.end], sr)
            sim = to_float(model.compute_similarity(str(wakeup_audio), str(tmp_wav)))
            scored.append(WindowScore(window.start, window.end, window.rms, sim))
    return scored


def select_windows(windows: list[WindowScore], args) -> list[WindowScore]:
    if not windows:
        return []
    max_rms = max((w.rms for w in windows), default=0.0) or 1.0

    def score(window: WindowScore) -> float:
        energy_score = window.rms / max_rms
        if args.method == "energy" or window.speaker_similarity is None:
            return energy_score
        if args.method == "speaker":
            return window.speaker_similarity
        return window.speaker_similarity + args.energy_weight * energy_score

    ranked = sorted(windows, key=score, reverse=True)
    selected = []
    total = 0
    max_samples = math.inf
    if ranked:
        # All windows share the same sample rate indirectly through duration.
        # The caller enforces max duration after padding/merge.
        max_samples = None
    for window in ranked[: max(1, args.keep_top_k)]:
        selected.append(window)
        total += window.duration
    return sorted(selected, key=lambda w: w.start)


def merge_intervals(windows: list[WindowScore], audio_len: int, sr: int, args) -> list[tuple[int, int]]:
    if not windows:
        return [(0, audio_len)]
    pad = int(args.padding_sec * sr)
    intervals = [(max(0, w.start - pad), min(audio_len, w.end + pad)) for w in windows]
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    max_samples = int(args.max_output_sec * sr)
    if max_samples > 0 and sum(end - start for start, end in merged) > max_samples:
        trimmed = []
        remaining = max_samples
        for start, end in merged:
            if remaining <= 0:
                break
            take = min(end - start, remaining)
            trimmed.append((start, start + take))
            remaining -= take
        merged = trimmed

    min_samples = int(args.min_output_sec * sr)
    if min_samples > 0 and sum(end - start for start, end in merged) < min_samples:
        center = (merged[0][0] + merged[-1][1]) // 2
        half = min_samples // 2
        start = max(0, center - half)
        end = min(audio_len, start + min_samples)
        start = max(0, end - min_samples)
        merged = [(start, end)]
    return merged


def concatenate_intervals(audio: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    parts = [audio[start:end] for start, end in intervals if end > start]
    if not parts:
        return audio
    return np.concatenate(parts).astype(np.float32)


def enhance_one(model, sample, args, output_root: Path) -> dict:
    start_time = time.perf_counter()
    audio, sr = read_audio(sample.command_audio)
    windows = candidate_windows(audio, sr, args)
    windows = score_speaker_windows(model, sample.wakeup_audio, audio, sr, windows)
    selected = select_windows(windows, args)
    intervals = merge_intervals(selected, len(audio), sr, args)
    enhanced = concatenate_intervals(audio, intervals)

    out_wav = output_root / sample.split / f"cmd_{sample.id}.wav"
    write_audio(out_wav, enhanced, sr)
    latency_ms = (time.perf_counter() - start_time) * 1000
    top_similarity = ""
    sims = [w.speaker_similarity for w in selected if w.speaker_similarity is not None]
    if sims:
        top_similarity = max(sims)
    return {
        "id": str(sample.id),
        "split": sample.split,
        "original_command_audio": str(sample.command_audio),
        "enhanced_command_audio": str(out_wav),
        "wakeup_audio": str(sample.wakeup_audio),
        "method": args.method,
        "selected_intervals_sec": ";".join(f"{s / sr:.3f}-{e / sr:.3f}" for s, e in intervals),
        "duration_in_sec": len(audio) / sr if sr else 0.0,
        "duration_out_sec": len(enhanced) / sr if sr else 0.0,
        "window_count": len(windows),
        "top_similarity": top_similarity,
        "latency_ms": latency_ms,
        "error": "",
    }


def copy_row(sample, output_root: Path, error: str) -> dict:
    return {
        "id": str(sample.id),
        "split": sample.split,
        "original_command_audio": str(sample.command_audio),
        "enhanced_command_audio": str(sample.command_audio),
        "wakeup_audio": str(sample.wakeup_audio),
        "method": "copy_on_error",
        "selected_intervals_sec": "",
        "duration_in_sec": "",
        "duration_out_sec": "",
        "window_count": 0,
        "top_similarity": "",
        "latency_ms": 0.0,
        "error": error,
    }


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

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(manifest) if args.resume else set()
    pending = [sample for sample in samples if str(sample.id) not in done_ids]
    mode = "a" if args.resume and manifest.exists() else "w"
    write_header = mode == "w"

    print(f"Loaded {len(samples)} samples, pending {len(pending)}, output_root={output_root}, manifest={manifest}")
    if not pending:
        return

    model = make_speaker_model(args)
    with manifest.open(mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for idx, sample in enumerate(pending, start=1):
            try:
                row = enhance_one(model, sample, args, output_root)
            except Exception as exc:
                if args.on_error == "raise":
                    raise
                row = copy_row(sample, output_root, str(exc))
                print(f"[{idx}/{len(pending)}] id={sample.id} ERROR: {exc}", file=sys.stderr)
            writer.writerow(row)
            f.flush()
            if idx == 1 or idx % 20 == 0 or idx == len(pending):
                print(
                    f"[{idx}/{len(pending)}] id={sample.id} out={row['duration_out_sec']} "
                    f"sim={row['top_similarity']} file={row['enhanced_command_audio']}"
                )

    print(f"Wrote enhanced manifest to {manifest}")


if __name__ == "__main__":
    main()
