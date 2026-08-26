"""Train the small Personal VAD on a prepared NumPy feature archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from xh202615.r12_personal_vad import PersonalVADConfig, PersonalVADNet, personal_vad_loss


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="NPZ with features [N,T,D] and targets [N,T]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.lr <= 0:
        raise ValueError("epochs and lr must be positive")
    archive = np.load(args.features)
    features = torch.from_numpy(np.asarray(archive["features"], dtype=np.float32))
    targets = torch.from_numpy(np.asarray(archive["targets"], dtype=np.int64))
    if features.ndim != 3 or targets.shape != features.shape[:2]:
        raise ValueError("features must be [N,T,D] and targets must be [N,T]")
    embedding_dim = max(1, int(features.shape[-1] - 80 - 1))
    config = PersonalVADConfig(mel_bins=80, embedding_dim=embedding_dim)
    if features.shape[-1] != config.input_dim:
        raise ValueError("feature dimension must equal mel_bins + embedding_dim + 1")
    model = PersonalVADNet(config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    features, targets = features.to(args.device), targets.to(args.device)
    model.train()
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = personal_vad_loss(model(features), targets)
        loss.backward()
        optimizer.step()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.__dict__, "state_dict": model.state_dict()}, args.output)
    print(json.dumps({"output": str(args.output), "epochs": args.epochs}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
