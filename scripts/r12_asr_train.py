"""R12 M0 Paraformer smoke probe (config-only by default; never trains or evaluates)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.r12_asr_smoke import SmokeConfig, run_smoke


def _parse_csv(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke", help="validate config or load one ASR model")
    smoke_parser.add_argument("--model", default="paraformer-zh")
    smoke_parser.add_argument("--mode", choices=("lora", "freeze_encoder"), default="lora")
    smoke_parser.add_argument("--device", default="cpu")
    smoke_parser.add_argument("--level", choices=("config", "load"), default="config")
    smoke_parser.add_argument("--lora-list", type=_parse_csv, default=["q", "k", "v", "o"])
    args = parser.parse_args(argv)

    config = SmokeConfig(
        model=args.model,
        mode=args.mode,
        device=args.device,
        level=args.level,
        lora_list=tuple(args.lora_list),
    )
    result = run_smoke(config)
    print(json.dumps(dataclasses.asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
