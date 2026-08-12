"""Validate regenerated R12 sources and build a digest-safe canonical JSONL.

Neural commands remain explicit external inputs: run pVAD/ASR/TSE/temporal
inference against the derived root, add each command-audio digest to its JSONL,
then invoke this command to fail closed and publish canonical input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.r12_dataa_canonical import build_augmented_canonical


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--candidate-fusion", type=Path, required=True)
    parser.add_argument("--tse-asr", type=Path, required=True)
    parser.add_argument("--audio-map", type=Path, required=True)
    parser.add_argument("--r3-predictions", type=Path, required=True)
    parser.add_argument("--pvad-manifest", type=Path, required=True)
    parser.add_argument("--canonical-output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_augmented_canonical(args.lineage, args.candidate_fusion, args.tse_asr, args.audio_map, args.r3_predictions, args.pvad_manifest, args.canonical_output)
    print(json.dumps({"canonical": str(summary.output), "rows": summary.row_count, "sha256": summary.output_sha256}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
