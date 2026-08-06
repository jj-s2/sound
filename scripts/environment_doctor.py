"""Print a JSON report of local runtime capabilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xh202615.environment import collect_environment


DEFAULT_PACKAGES = ("torch", "funasr", "modelscope", "wespeaker", "psutil")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", dest="packages", metavar="NAME")
    parser.add_argument("--artifact", action="append", dest="artifacts", metavar="PATH")
    parser.add_argument("--output", metavar="PATH")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = collect_environment(
        args.packages if args.packages is not None else DEFAULT_PACKAGES,
        args.artifacts if args.artifacts is not None else [],
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
