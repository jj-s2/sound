"""Prepare train-only subphrase hotword candidates for R12 ASR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.r12_asr_hotword import prepare_subphrase_candidates


def _parse_csv(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def _parse_capacities(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--parent-ids", type=_parse_csv, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capacities", type=_parse_capacities, required=True)
    args = parser.parse_args(argv)
    summary = prepare_subphrase_candidates(
        args.train_labels, args.parent_ids, args.output_root, args.capacities
    )
    print(json.dumps({
        "private_hotwords": str(summary.private_hotwords),
        "public_summary": str(summary.public_summary),
        "phrase_count": summary.phrase_count,
        "source_labels_sha256": summary.source_labels_sha256,
        "summary_sha256": summary.summary_sha256,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
