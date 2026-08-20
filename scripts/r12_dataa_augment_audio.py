"""Materialize train-only deterministic Dataset-A audio augmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.r12_dataa_augmentation import build_augmented_dataset
from xh202615.r12_dataa_augmented_split import load_augmented_internal_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasetA" / "datasetA")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    ids = []
    for source_split in ("pos", "neg"):
        ids.extend(str(json.loads(line)["id"]) for line in (args.dataset_root / f"{source_split}.jsonl").read_text(encoding="utf-8-sig").splitlines())
    summary = build_augmented_dataset(args.dataset_root, load_augmented_internal_split(args.split_manifest, ids), args.output_root)
    print(json.dumps({"dataset_root": str(summary.dataset_root), "lineage": str(summary.lineage_path), "lineage_digest": summary.lineage_digest, "rows": summary.row_count, "exclusions": summary.exclusion_count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
