"""Compare one or more ASR map files against Dataset-A positive labels."""

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
from xh202615.metrics import cer_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ASR maps by CER on labeled samples")
    parser.add_argument("--dataset-root", default="datasetA")
    parser.add_argument("--splits", default="pos", help="Use labeled splits; normally pos")
    parser.add_argument("--maps", nargs="+", required=True, help="Entries like name=path.jsonl")
    parser.add_argument("--per-split-limit", type=int, default=None)
    parser.add_argument("--output", default="output/reports/asr_map_compare.csv")
    parser.add_argument("--summary-output", default=None)
    return parser.parse_args()


def load_asr_map(path: Path) -> dict[str, str]:
    rows = {}
    for row in read_jsonl(path):
        text = row.get("text", row.get("recognition_text", ""))
        rows[str(row["id"])] = "" if text is None else str(text)
    return rows


def parse_map_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples = [s for s in load_dataset(args.dataset_root, splits) if s.label is not None]
    if args.per_split_limit is not None:
        selected = []
        for split in splits:
            selected.extend([sample for sample in samples if sample.split == split][: args.per_split_limit])
        samples = selected

    map_specs = [parse_map_arg(value) for value in args.maps]
    maps = [(name, path, load_asr_map(path)) for name, path in map_specs]

    summary = []
    rows = []
    for name, path, predictions in maps:
        subs = ins = dels = ref_chars = missing = exact = 0
        for sample in samples:
            sample_id = str(sample.id)
            if sample_id not in predictions:
                missing += 1
            hyp = predictions.get(sample_id, "")
            stats = cer_stats(sample.label, hyp)
            subs += stats.substitutions
            ins += stats.insertions
            dels += stats.deletions
            ref_chars += stats.ref_chars
            exact += int(stats.errors == 0)
            rows.append(
                {
                    "model": name,
                    "id": sample_id,
                    "cer": stats.cer,
                    "errors": stats.errors,
                    "reference_text": sample.label,
                    "recognition_text": hyp,
                }
            )
        errors = subs + ins + dels
        summary.append(
            {
                "model": name,
                "path": str(path),
                "samples": len(samples),
                "missing": missing,
                "avg_cer": errors / ref_chars if ref_chars else 0.0,
                "exact_rate": exact / len(samples) if samples else 0.0,
                "substitutions": subs,
                "insertions": ins,
                "deletions": dels,
                "ref_chars": ref_chars,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "id", "cer", "errors", "reference_text", "recognition_text"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_out = Path(args.summary_output) if args.summary_output else out.with_suffix(".summary.json")
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote per-sample comparison to {out}")


if __name__ == "__main__":
    main()
