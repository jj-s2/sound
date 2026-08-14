"""Prepare the private Dataset-A-train ASR manifest for R12 M0 Paraformer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.r12_asr_manifest import prepare_asr_manifests


def _parse_parent_ids(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item for item in value.split(",") if item]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inner-valid-parent-ids", type=_parse_parent_ids, default=[])
    args = parser.parse_args(argv)
    summary = prepare_asr_manifests(
        args.lineage, args.train_labels, args.output_root,
        inner_valid_parent_ids=args.inner_valid_parent_ids,
    )
    print(json.dumps({
        "train_jsonl": str(summary.train_jsonl),
        "inner_valid_jsonl": str(summary.inner_valid_jsonl),
        "public_summary": str(summary.public_summary),
        "train_rows": summary.train_rows,
        "inner_valid_rows": summary.inner_valid_rows,
        "manifest_sha256": summary.manifest_sha256,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
