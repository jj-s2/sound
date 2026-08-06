"""Run V0/V1/V2/V3 baseline inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_dataset
from .pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="XH-202615 baseline inference")
    parser.add_argument("--dataset-root", default="datasetA")
    parser.add_argument("--splits", default="pos,neg", help="Comma-separated splits, e.g. pos,neg")
    parser.add_argument("--config", default="configs/v0_asr_only.json")
    parser.add_argument("--output", default="output/predictions/predictions.jsonl")
    parser.add_argument("--asr-map", default=None, help="Optional JSONL/CSV with id,text or id,recognition_text")
    parser.add_argument("--speaker-scores", default=None, help="Optional CSV with id and speaker/difficulty scores")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-split-limit", type=int, default=None, help="Limit each split before combining")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples = load_dataset(args.dataset_root, splits)
    if args.per_split_limit is not None:
        selected = []
        for split in splits:
            selected.extend([sample for sample in samples if sample.split == split][: args.per_split_limit])
        samples = selected
    if args.limit is not None:
        samples = samples[: args.limit]

    pipeline = Pipeline(config, asr_map=args.asr_map, speaker_scores=args.speaker_scores)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for sample in samples:
            pred = pipeline.infer(sample).to_json()
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} predictions to {out}")


if __name__ == "__main__":
    main()
