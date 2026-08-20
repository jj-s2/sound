"""Analyze prediction JSONL against Dataset-A style labels.

This script writes a compact CSV that is easier to inspect than the full
metrics JSON. It is useful for finding high-CER positive samples and false
accepted negative samples after each V0/V1/V2 experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import load_dataset, read_jsonl
from xh202615.metrics import cer_stats, is_rejection


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze XH-202615 prediction errors")
    parser.add_argument("--dataset-root", default="datasetA")
    parser.add_argument("--splits", default="pos,neg")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="output/reports/error_report.csv")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--missing-policy", choices=("empty", "skip"), default="empty")
    parser.add_argument("--per-split-limit", type=int, default=None, help="Limit each split before combining")
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict]:
    preds = {}
    for row in read_jsonl(path):
        sample_id = str(row["id"])
        text = row.get("recognition_text", row.get("text", ""))
        row["recognition_text"] = "" if text is None else str(text)
        preds[sample_id] = row
    return preds


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples = load_dataset(args.dataset_root, splits)
    if args.per_split_limit is not None:
        selected = []
        for split in splits:
            selected.extend([sample for sample in samples if sample.split == split][: args.per_split_limit])
        samples = selected
    preds = load_predictions(Path(args.predictions))

    rows = []
    pos_count = neg_count = missing = 0
    false_reject = false_accept = correct_reject = 0
    total_errors = total_ref_chars = 0

    for sample in samples:
        sample_id = str(sample.id)
        pred = preds.get(sample_id)
        if pred is None:
            missing += 1
            if args.missing_policy == "skip":
                continue
        hyp = pred["recognition_text"] if pred else ""
        rejected = is_rejection(hyp)
        base = {
            "id": sample_id,
            "split": sample.split,
            "command_audio": str(sample.command_audio),
            "reference_text": "" if sample.label is None else sample.label,
            "recognition_text": hyp,
            "route": "" if pred is None else pred.get("route", ""),
            "route_reason": "" if pred is None else pred.get("route_reason", ""),
            "asr_backend": "" if pred is None else pred.get("asr_backend", ""),
            "speaker_backend": "" if pred is None else pred.get("speaker_backend", ""),
            "latency_ms": "" if pred is None else pred.get("latency_ms", ""),
        }
        if sample.label is None:
            neg_count += 1
            correct_reject += int(rejected)
            false_accept += int(not rejected)
            rows.append({**base, "cer": "", "errors": "", "issue": "" if rejected else "false_accept"})
        else:
            pos_count += 1
            false_reject += int(rejected)
            stats = cer_stats(sample.label, hyp)
            total_errors += stats.errors
            total_ref_chars += stats.ref_chars
            if stats.errors or rejected:
                rows.append(
                    {
                        **base,
                        "cer": stats.cer,
                        "errors": stats.errors,
                        "issue": "false_reject" if rejected else "cer_error",
                    }
                )

    issue_rank = {"false_accept": 3, "false_reject": 2, "cer_error": 1}
    rows.sort(
        key=lambda row: (
            issue_rank.get(row["issue"], 0),
            float(row["cer"] or 0),
            int(row["errors"] or 0),
        ),
        reverse=True,
    )
    if args.top_k > 0:
        rows = rows[: args.top_k]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "split",
        "issue",
        "cer",
        "errors",
        "reference_text",
        "recognition_text",
        "route",
        "route_reason",
        "asr_backend",
        "speaker_backend",
        "latency_ms",
        "command_audio",
    ]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "pos_count": pos_count,
        "neg_count": neg_count,
        "missing_predictions": missing,
        "avg_cer": total_errors / total_ref_chars if total_ref_chars else 0.0,
        "avg_rr": correct_reject / neg_count if neg_count else 0.0,
        "false_reject_rate": false_reject / pos_count if pos_count else 0.0,
        "false_accept_rate": false_accept / neg_count if neg_count else 0.0,
        "report_rows": len(rows),
    }
    summary_out = Path(args.summary_output) if args.summary_output else out.with_suffix(".summary.json")
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote report to {out}")


if __name__ == "__main__":
    main()
