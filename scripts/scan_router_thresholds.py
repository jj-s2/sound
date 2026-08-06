"""Sweep speaker-router thresholds and report CER/RR tradeoffs.

This is a diagnostic tool. It evaluates thresholds after inference, but the
pipeline decisions themselves still use only ASR text and speaker scores.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import load_dataset
from xh202615.metrics import cer_stats, is_rejection
from xh202615.pipeline import Pipeline


FIELDNAMES = [
    "threshold",
    "cer",
    "rr",
    "false_reject_rate",
    "false_accept_rate",
    "pos_count",
    "neg_count",
    "official_80_score",
    "eligible",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep V4 router thresholds")
    parser.add_argument("--dataset-root", default="datasetA")
    parser.add_argument("--splits", default="pos,neg")
    parser.add_argument("--config", default="configs/v4_balanced_058.json")
    parser.add_argument("--asr-map", required=True)
    parser.add_argument("--speaker-scores", required=True)
    parser.add_argument("--thresholds", default="0.58,0.59,0.60,0.61,0.62,0.63,0.64")
    parser.add_argument("--output", default="output/reports/router_threshold_scan.csv")
    parser.add_argument("--pred-dir", default=None, help="Optional directory to write predictions for every threshold")
    parser.add_argument("--max-false-reject", type=float, default=0.10)
    parser.add_argument("--per-split-limit", type=int, default=None)
    return parser.parse_args()


def patch_threshold(config: dict, threshold: float) -> dict:
    cfg = copy.deepcopy(config)
    reject = cfg.setdefault("router", {}).setdefault("reject", {})
    reject["global_similarity_max"] = threshold
    reject["topk_similarity_max"] = threshold
    text_reject = cfg.setdefault("text_router", {}).setdefault("reject", {})
    if "speaker_similarity_max" in text_reject:
        text_reject["speaker_similarity_max"] = threshold
    return cfg


def evaluate_rows(samples, rows: list[dict]) -> dict:
    preds = {str(row["id"]): row.get("recognition_text", row.get("text", "")) for row in rows}
    pos_count = neg_count = correct_reject = false_reject = false_accept = 0
    subs = ins = dels = ref_chars = 0
    for sample in samples:
        hyp = "" if preds.get(str(sample.id)) is None else str(preds.get(str(sample.id), ""))
        if sample.label is None:
            neg_count += 1
            rejected = is_rejection(hyp)
            correct_reject += int(rejected)
            false_accept += int(not rejected)
        else:
            pos_count += 1
            false_reject += int(is_rejection(hyp))
            stats = cer_stats(sample.label, hyp)
            subs += stats.substitutions
            ins += stats.insertions
            dels += stats.deletions
            ref_chars += stats.ref_chars
    cer = (subs + ins + dels) / ref_chars if ref_chars else 0.0
    rr = correct_reject / neg_count if neg_count else 0.0
    return {
        "cer": cer,
        "rr": rr,
        "false_reject_rate": false_reject / pos_count if pos_count else 0.0,
        "false_accept_rate": false_accept / neg_count if neg_count else 0.0,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "official_80_score": (1.0 - cer) * 40.0 + rr * 40.0,
    }


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def main() -> None:
    args = parse_args()
    base_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples = load_dataset(args.dataset_root, splits)
    if args.per_split_limit is not None:
        selected = []
        for split in splits:
            selected.extend([sample for sample in samples if sample.split == split][: args.per_split_limit])
        samples = selected

    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for threshold in thresholds:
        pipeline = Pipeline(patch_threshold(base_config, threshold), asr_map=args.asr_map, speaker_scores=args.speaker_scores)
        pred_rows = [pipeline.infer(sample).to_json() for sample in samples]
        metrics = evaluate_rows(samples, pred_rows)
        metrics["threshold"] = threshold
        metrics["eligible"] = metrics["false_reject_rate"] < args.max_false_reject
        summary_rows.append(metrics)
        if pred_dir:
            pred_path = pred_dir / f"threshold_{threshold:.2f}.jsonl"
            with pred_path.open("w", encoding="utf-8", newline="\n") as f:
                for row in pred_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in FIELDNAMES})

    eligible = [row for row in summary_rows if row["eligible"]]
    best_pool = eligible or summary_rows
    best = max(best_pool, key=lambda row: row["official_80_score"]) if best_pool else None
    print(f"Wrote threshold scan to {out}")
    if best:
        print(
            "Best threshold="
            f"{best['threshold']:.2f}, CER={best['cer']:.4f}, RR={best['rr']:.4f}, "
            f"FR={best['false_reject_rate']:.4f}, eligible={best['eligible']}"
        )


if __name__ == "__main__":
    main()
