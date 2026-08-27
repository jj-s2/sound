"""Run a saved Personal VAD checkpoint on an NPZ feature archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from xh202615.r12_personal_vad import PersonalVADConfig, PersonalVADNet, aggregate_personal_vad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True, help="NPZ with features [N,T,D]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = PersonalVADConfig(**checkpoint["config"])
    model = PersonalVADNet(config).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    features = torch.from_numpy(np.asarray(np.load(args.features)["features"], dtype=np.float32)).to(args.device)
    with torch.no_grad():
        probabilities = torch.softmax(model(features), dim=-1).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, sample in enumerate(probabilities):
            handle.write(json.dumps({"index": index, **aggregate_personal_vad(sample)}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
