"""Generate speaker score CSV for V1 routing with WeSpeaker.

Each Dataset-A style sample contains:

    wakeup_audio  -> owner reference voice
    command_audio -> audio to accept/reject

This script compares the two files and writes the CSV consumed by
`ScoreCsvSpeakerBackend`.

Install WeSpeaker first:

    pip install git+https://github.com/wenet-e2e/wespeaker.git
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import load_dataset


FIELDNAMES = [
    "id",
    "target_probability",
    "global_similarity",
    "topk_similarity",
    "target_frame_ratio",
    "noise_score",
    "overlap_probability",
    "audio_quality",
    "backend",
    "latency_ms",
    "wakeup_audio",
    "command_audio",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate V1 speaker score CSV with WeSpeaker")
    parser.add_argument("--dataset-root", default="datasetA")
    parser.add_argument("--splits", default="pos,neg")
    parser.add_argument("--output", default="output/speaker/wespeaker_scores.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-split-limit", type=int, default=None, help="Limit each split before combining")
    parser.add_argument("--resume", action="store_true", help="Append and skip ids already present in output")
    parser.add_argument("--model", default="chinese", help='WeSpeaker model name, e.g. "chinese"')
    parser.add_argument("--gpu", type=int, default=None, help="GPU id if supported by the installed WeSpeaker version")
    parser.add_argument("--prob-center", type=float, default=0.35)
    parser.add_argument("--prob-scale", type=float, default=0.08)
    parser.add_argument("--target-threshold", type=float, default=0.45)
    parser.add_argument("--on-error", choices=("empty", "raise"), default="empty")
    return parser.parse_args()


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {str(row["id"]) for row in csv.DictReader(f) if row.get("id")}


def make_model(args):
    try:
        import wespeaker
    except ImportError as exc:
        raise SystemExit(
            "WeSpeaker is not installed. Install it with:\n"
            "  pip install git+https://github.com/wenet-e2e/wespeaker.git"
        ) from exc

    model = wespeaker.load_model(args.model)
    if args.gpu is not None:
        if hasattr(model, "set_gpu"):
            model.set_gpu(args.gpu)
        else:
            print("Warning: installed WeSpeaker model object has no set_gpu(); using its default device", file=sys.stderr)
    return model


def to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)) and value:
        return to_float(value[0])
    return float(value)


def probability_from_similarity(similarity: float, center: float, scale: float) -> float:
    scale = max(scale, 1e-6)
    z = (similarity - center) / scale
    return 1.0 / (1.0 + math.exp(-z))


def score_one(model, sample, args) -> dict:
    start = time.perf_counter()
    similarity = to_float(model.compute_similarity(str(sample.wakeup_audio), str(sample.command_audio)))
    latency_ms = (time.perf_counter() - start) * 1000
    target_probability = probability_from_similarity(similarity, args.prob_center, args.prob_scale)

    # This first V1 scorer uses global similarity only. The columns are still
    # populated so the existing conservative router can run unchanged.
    target_frame_ratio = 1.0 if similarity >= args.target_threshold else 0.0
    return {
        "id": str(sample.id),
        "target_probability": target_probability,
        "global_similarity": similarity,
        "topk_similarity": similarity,
        "target_frame_ratio": target_frame_ratio,
        "noise_score": "",
        "overlap_probability": "",
        "audio_quality": 1.0,
        "backend": "wespeaker",
        "latency_ms": latency_ms,
        "wakeup_audio": str(sample.wakeup_audio),
        "command_audio": str(sample.command_audio),
        "error": "",
    }


def empty_score(sample, error: str) -> dict:
    return {
        "id": str(sample.id),
        "target_probability": "",
        "global_similarity": "",
        "topk_similarity": "",
        "target_frame_ratio": "",
        "noise_score": "",
        "overlap_probability": "",
        "audio_quality": "",
        "backend": "wespeaker",
        "latency_ms": 0.0,
        "wakeup_audio": str(sample.wakeup_audio),
        "command_audio": str(sample.command_audio),
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
    if args.limit is not None:
        samples = samples[: args.limit]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(out) if args.resume else set()
    pending = [sample for sample in samples if str(sample.id) not in done_ids]
    mode = "a" if args.resume and out.exists() else "w"
    write_header = mode == "w"

    print(f"Loaded {len(samples)} samples, pending {len(pending)}, output={out}")
    if not pending:
        return

    model = make_model(args)
    with out.open(mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for idx, sample in enumerate(pending, start=1):
            try:
                if not sample.wakeup_audio.exists():
                    raise FileNotFoundError(f"Missing wakeup_audio: {sample.wakeup_audio}")
                if not sample.command_audio.exists():
                    raise FileNotFoundError(f"Missing command_audio: {sample.command_audio}")
                row = score_one(model, sample, args)
            except Exception as exc:
                if args.on_error == "raise":
                    raise
                row = empty_score(sample, str(exc))
                print(f"[{idx}/{len(pending)}] id={sample.id} ERROR: {exc}", file=sys.stderr)

            writer.writerow(row)
            f.flush()
            if idx == 1 or idx % 20 == 0 or idx == len(pending):
                print(
                    f"[{idx}/{len(pending)}] id={sample.id} "
                    f"sim={row['global_similarity']} prob={row['target_probability']} "
                    f"latency_ms={row['latency_ms']}"
                )

    print(f"Wrote speaker scores to {out}")


if __name__ == "__main__":
    main()
