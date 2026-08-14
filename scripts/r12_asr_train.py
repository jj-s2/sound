"""R12 Paraformer smoke and explicit train launcher."""

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
from xh202615.r12_asr_train import TrainingConfig, run_training


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
    train_parser = subparsers.add_parser("train", help="dry-run or explicitly launch one FunASR training job")
    train_parser.add_argument("--train-manifest", type=Path, required=True)
    train_parser.add_argument("--valid-manifest", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--model", default="paraformer-zh")
    train_parser.add_argument("--mode", choices=("lora", "freeze_encoder"), default="lora")
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--seed", type=int, default=20260814)
    train_parser.add_argument("--execute", action="store_true", help="run FunASR; without this flag only validate and print argv")
    args = parser.parse_args(argv)

    if args.command == "train":
        result = run_training(
            TrainingConfig(
                train_manifest=args.train_manifest,
                valid_manifest=args.valid_manifest,
                output_dir=args.output_dir,
                model=args.model,
                device=args.device,
                mode=args.mode,
                seed=args.seed,
            ),
            dry_run=not args.execute,
        )
        print(json.dumps(dataclasses.asdict(result), ensure_ascii=False, sort_keys=True))
        return 0 if result.return_code in (None, 0) else result.return_code
    config = SmokeConfig(model=args.model, mode=args.mode, device=args.device, level=args.level, lora_list=tuple(args.lora_list))
    result = run_smoke(config)
    print(json.dumps(dataclasses.asdict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
