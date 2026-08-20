"""Download and preflight the pinned FireRedChat pVAD model assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xh202615.firered_model_assets import FIRERED_REVISION, download_and_verify_model


DEFAULT_MODEL_ROOT = (
    Path(__file__).resolve().parents[1]
    / "output"
    / "models"
    / "FireRedChat-pvad"
    / FIRERED_REVISION
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = download_and_verify_model(args.model_root)
    print(paths.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
