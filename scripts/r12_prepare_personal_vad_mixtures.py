"""Validate and materialize train-only Personal VAD mixture lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xh202615.r12_personal_vad import write_personal_vad_lineage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args(argv)
    count = write_personal_vad_lineage(args.source_manifest, args.output, seed=args.seed)
    print(json.dumps({"output": str(args.output), "rows": count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
