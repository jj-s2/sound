"""Build the label-free R12 M0 ASR parent-group outer fold manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.r12_asr_folds import build_asr_folds, write_asr_folds
from xh202615.r12_dataa_augmentation import load_lineage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args(argv)
    lineage = load_lineage(args.lineage)
    manifest = build_asr_folds(lineage, fold_count=args.fold_count, seed=args.seed)
    write_asr_folds(args.output, manifest)
    print(json.dumps({
        "output": str(args.output),
        "fold_count": manifest.fold_count,
        "seed": manifest.seed,
        "train_rows": len(manifest.fold_by_id),
        "manifest_sha256": manifest.manifest_sha256,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
