"""Prepare a small raw/energy/BSS/fusion ablation table for hard samples."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import load_dataset, read_jsonl
from xh202615.metrics import cer_stats


FIELDNAMES = [
    "id",
    "split",
    "reference_text",
    "raw_text",
    "raw_cer",
    "energy_text",
    "energy_cer",
    "bss_text",
    "bss_cer",
    "fusion_text",
    "fusion_cer",
    "best_branch",
    "command_audio",
    "note",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build BSS/TSE hard-sample ablation CSV")
    parser.add_argument("--dataset-root", default="datasetA")
    parser.add_argument("--splits", default="pos")
    parser.add_argument("--ids-file", required=True, help="20-100 hard sample ids, one per line")
    parser.add_argument("--raw", required=True, help="Raw ASR map JSONL")
    parser.add_argument("--energy", default=None, help="Energy-enhanced ASR map JSONL")
    parser.add_argument("--bss", default=None, help="BSS/TSE ASR map JSONL")
    parser.add_argument("--fusion", default=None, help="Fused ASR map JSONL")
    parser.add_argument("--output", default="output/reports/bss_ablation.csv")
    return parser.parse_args()


def load_text_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    values = {}
    for row in read_jsonl(Path(path)):
        text = row.get("recognition_text", row.get("text", ""))
        values[str(row["id"])] = "" if text is None else str(text)
    return values


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def branch_cer(reference: str | None, text: str) -> str:
    if reference is None:
        return ""
    return f"{cer_stats(reference, text).cer:.6f}".rstrip("0").rstrip(".")


def main() -> None:
    args = parse_args()
    ids = load_ids(Path(args.ids_file))
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples = {str(sample.id): sample for sample in load_dataset(args.dataset_root, splits)}
    maps = {
        "raw": load_text_map(args.raw),
        "energy": load_text_map(args.energy),
        "bss": load_text_map(args.bss),
        "fusion": load_text_map(args.fusion),
    }

    rows = []
    for sample_id in ids:
        sample = samples.get(sample_id)
        reference = None if sample is None else sample.label
        branch_scores = {}
        branch_texts = {name: values.get(sample_id, "") for name, values in maps.items()}
        for name, text in branch_texts.items():
            if reference is not None and text != "":
                branch_scores[name] = cer_stats(reference, text).cer
        best_branch = min(branch_scores, key=branch_scores.get) if branch_scores else ""
        rows.append(
            {
                "id": sample_id,
                "split": "" if sample is None else sample.split,
                "reference_text": "" if reference is None else reference,
                "raw_text": branch_texts["raw"],
                "raw_cer": branch_cer(reference, branch_texts["raw"]),
                "energy_text": branch_texts["energy"],
                "energy_cer": branch_cer(reference, branch_texts["energy"]),
                "bss_text": branch_texts["bss"],
                "bss_cer": branch_cer(reference, branch_texts["bss"]),
                "fusion_text": branch_texts["fusion"],
                "fusion_cer": branch_cer(reference, branch_texts["fusion"]),
                "best_branch": best_branch,
                "command_audio": "" if sample is None else str(sample.command_audio),
                "note": "diagnostic_only_not_for_rule_fitting",
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} BSS ablation rows to {out}")


if __name__ == "__main__":
    main()
