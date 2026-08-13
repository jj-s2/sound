"""Attest regenerated R12 sources and publish digest-safe canonical inputs.

Neural inference remains explicit: run CPU pVAD, ASR, TSE and temporal
inference against the derived Dataset-A root.  Use ``attest`` on each JSONL
candidate source, then use ``canonical`` to fail closed on complete coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.r12_dataa_canonical import (
    attest_source_command_audio,
    build_augmented_canonical,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    attest = subparsers.add_parser("attest", help="bind one source JSONL to lineage command bytes")
    attest.add_argument("--lineage", type=Path, required=True)
    attest.add_argument("--source", type=Path, required=True)
    attest.add_argument("--output", type=Path, required=True)

    canonical = subparsers.add_parser("canonical", help="join fully attested sources into canonical input")
    canonical.add_argument("--lineage", type=Path, required=True)
    canonical.add_argument("--candidate-fusion", type=Path, required=True)
    canonical.add_argument("--tse-asr", type=Path, required=True)
    canonical.add_argument("--audio-map", type=Path, required=True)
    canonical.add_argument("--r3-predictions", type=Path, required=True)
    canonical.add_argument("--pvad-manifest", type=Path, required=True)
    canonical.add_argument("--canonical-output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "attest":
        summary = attest_source_command_audio(args.lineage, args.source, args.output)
        print(json.dumps({"attested": str(summary.output), "rows": summary.row_count, "sha256": summary.output_sha256}, ensure_ascii=False))
        return 0

    summary = build_augmented_canonical(
        args.lineage, args.candidate_fusion, args.tse_asr, args.audio_map,
        args.r3_predictions, args.pvad_manifest, args.canonical_output,
    )
    print(json.dumps({"canonical": str(summary.output), "rows": summary.row_count, "sha256": summary.output_sha256}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
