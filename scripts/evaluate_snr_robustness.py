"""Evaluate CER grouped by SNR for generated robustness sets."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import load_dataset, read_jsonl
from xh202615.metrics import cer_stats


SNR_RE = re.compile(r"_snr_([pm])([0-9]+(?:d[0-9]+)?)$")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SNR robustness CER")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="output/metrics/snr_robustness_metrics.json")
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--metadata", default=None, help="Optional snr_metadata.jsonl")
    parser.add_argument("--missing-policy", choices=("empty", "skip"), default="empty")
    return parser.parse_args()


def parse_snr_from_id(sample_id: str) -> float:
    match = SNR_RE.search(sample_id)
    if not match:
        raise ValueError(f"Cannot parse SNR from id: {sample_id}")
    sign = 1 if match.group(1) == "p" else -1
    value = float(match.group(2).replace("d", "."))
    return sign * value


def load_predictions(path: Path) -> dict[str, str]:
    preds = {}
    for row in read_jsonl(path):
        text = row.get("recognition_text", row.get("text", ""))
        preds[str(row["id"])] = "" if text is None else str(text)
    return preds


def load_metadata(path: str | Path | None) -> dict[str, dict]:
    if not path:
        return {}
    return {str(row["id"]): row for row in read_jsonl(Path(path))}


def main() -> None:
    args = parse_args()
    samples = load_dataset(args.dataset_root, ["pos"])
    preds = load_predictions(Path(args.predictions))
    metadata = load_metadata(args.metadata)

    grouped = {}
    missing = 0
    per_sample = []
    for sample in samples:
        sample_id = str(sample.id)
        if sample_id not in preds:
            missing += 1
            if args.missing_policy == "skip":
                continue
        hyp = preds.get(sample_id, "")
        snr_db = float(metadata.get(sample_id, {}).get("snr_db", parse_snr_from_id(sample_id)))
        stats = cer_stats(sample.label, hyp)
        group = grouped.setdefault(
            str(snr_db),
            {"snr_db": snr_db, "samples": 0, "substitutions": 0, "insertions": 0, "deletions": 0, "ref_chars": 0},
        )
        group["samples"] += 1
        group["substitutions"] += stats.substitutions
        group["insertions"] += stats.insertions
        group["deletions"] += stats.deletions
        group["ref_chars"] += stats.ref_chars
        per_sample.append(
            {
                "id": sample_id,
                "snr_db": snr_db,
                "cer": stats.cer,
                "errors": stats.errors,
                "reference_text": sample.label,
                "recognition_text": hyp,
            }
        )

    by_snr = []
    for group in sorted(grouped.values(), key=lambda row: row["snr_db"], reverse=True):
        errors = group["substitutions"] + group["insertions"] + group["deletions"]
        group["cer"] = errors / group["ref_chars"] if group["ref_chars"] else 0.0
        by_snr.append(group)
    average_cer = sum(row["cer"] for row in by_snr) / len(by_snr) if by_snr else 0.0
    weighted_errors = sum(row["substitutions"] + row["insertions"] + row["deletions"] for row in by_snr)
    weighted_ref_chars = sum(row["ref_chars"] for row in by_snr)
    weighted_cer = weighted_errors / weighted_ref_chars if weighted_ref_chars else 0.0

    result = {
        "metrics": {
            "average_cer_across_snr": average_cer,
            "weighted_cer": weighted_cer,
            "snr_count": len(by_snr),
            "missing_predictions": missing,
        },
        "by_snr": by_snr,
        "per_sample": per_sample,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_out = Path(args.csv_output) if args.csv_output else out.with_suffix(".csv")
    with csv_out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_db", "samples", "cer", "substitutions", "insertions", "deletions", "ref_chars"])
        writer.writeheader()
        writer.writerows(by_snr)

    print(json.dumps({"metrics": result["metrics"], "by_snr": by_snr}, ensure_ascii=False, indent=2))
    print(f"Wrote SNR robustness metrics to {out}")


if __name__ == "__main__":
    main()
