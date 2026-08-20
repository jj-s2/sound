"""Evaluate prediction JSONL against Dataset-A style labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_dataset, read_jsonl
from .evaluation import evaluate_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XH-202615 predictions")
    parser.add_argument("--dataset-root", default="datasetA")
    parser.add_argument("--splits", default="pos,neg")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="output/metrics/metrics.json")
    parser.add_argument("--missing-policy", choices=["empty", "skip"], default="empty")
    parser.add_argument("--per-split-limit", type=int, default=None, help="Limit each split before combining")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples = load_dataset(args.dataset_root, splits)
    if args.per_split_limit is not None:
        selected = []
        for split in splits:
            selected.extend([sample for sample in samples if sample.split == split][: args.per_split_limit])
        samples = selected
    report = evaluate_rows(
        samples,
        read_jsonl(Path(args.predictions)),
        missing_policy=args.missing_policy,
    )
    output_payload = report.to_dict()
    # Keep the historical top-level shape for existing consumers. Buckets are
    # available through the pure evaluator but are opt-in at the CLI layer.
    output_payload.pop("buckets", None)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
