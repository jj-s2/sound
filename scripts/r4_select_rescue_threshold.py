"""Select the R4 rescue threshold on public validation, then validate on public test.

Combines the public ASR bakeoff (raw FunASR transcripts vs AISHELL references)
with the frozen R3 temporal-head presence probabilities, then calls
``xh202615.r4_rescue.select_rescue_threshold`` to pick the public threshold
maximizing S_public subject to the rejection policy.  No Dataset-A file is
read at any step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import read_jsonl
from xh202615.r4_rescue import (
    public_baselines,
    rescue_metrics,
    rescue_text,
    select_rescue_threshold,
)
from xh202615.temporal_head import TemporalSpeakerHead


def head_probabilities(checkpoint_path: Path, cache_path: Path, device: str) -> dict[str, float]:
    """Run the frozen R3 head over its cached features -> {row_id: probability}."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TemporalSpeakerHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        mode=str(checkpoint["mode"]),
    )
    model.load_state_dict(checkpoint["model"])
    dev = torch.device(device)
    model.to(dev).eval()
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    features = payload["features"]
    row_ids = tuple(payload["row_ids"])
    probabilities: dict[str, float] = {}
    with torch.no_grad():
        logits, _ = model(features.to(dev))
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    for row_id, prob in zip(row_ids, probs):
        probabilities[str(row_id)] = float(prob)
    return probabilities


def load_bakeoff(path: Path) -> list[dict]:
    rows = list(read_jsonl(path))
    if not rows:
        raise SystemExit(f"bakeoff output is empty: {path}")
    return rows


def build_columns(bakeoff_rows: list[dict], probabilities: dict[str, float], split: str) -> dict:
    selected = [r for r in bakeoff_rows if r["split"] == split]
    if not selected:
        raise SystemExit(f"no bakeoff rows for split={split}")
    present = []
    probs = []
    refs = []
    raw = []
    fall = []
    missing = []
    for row in selected:
        rid = str(row["row_id"])
        if rid not in probabilities:
            missing.append(rid)
            continue
        present.append(1 if row["target_present"] else 0)
        probs.append(probabilities[rid])
        # Reference text only matters for present rows; absent rows use "".
        refs.append(str(row["transcript"]) if row["target_present"] else "")
        raw.append(str(row.get("raw_asr_text") or ""))
        # Public simulation stand-in for the Dataset-A fusion fallback: empty.
        fall.append("")
    if missing:
        raise SystemExit(f"{len(missing)} bakeoff row(s) missing a head probability; e.g. {missing[:3]}")
    return {
        "split": split,
        "present": present,
        "probabilities": probs,
        "references": refs,
        "raw_texts": raw,
        "fallback_texts": fall,
        "row_ids": [str(r["row_id"]) for r in selected],
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bakeoff", required=True, help="r4_public_asr_bakeoff output JSONL")
    parser.add_argument("--checkpoint", default="output/training/r3_temporal_head/best.pt")
    parser.add_argument("--feature-cache", default="output/training/r3_temporal_head/frozen_features.pt")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--max-frr", type=float, default=0.10)
    parser.add_argument("--min-rr", type=float, default=0.85)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> dict:
    args = parse_args(argv)
    probabilities = head_probabilities(Path(args.checkpoint), Path(args.feature_cache), args.device)
    bakeoff_rows = load_bakeoff(Path(args.bakeoff))

    val = build_columns(bakeoff_rows, probabilities, args.val_split)
    test = build_columns(bakeoff_rows, probabilities, args.test_split)

    selection = select_rescue_threshold(
        val["present"],
        val["probabilities"],
        val["references"],
        val["raw_texts"],
        val["fallback_texts"],
        max_frr=args.max_frr,
        min_rr=args.min_rr,
    )
    threshold = selection["threshold"]

    # Validate on public test with the frozen public threshold.
    test_outputs = [
        rescue_text(float(p), r, f, threshold=threshold)[0]
        for p, r, f in zip(test["probabilities"], test["raw_texts"], test["fallback_texts"])
    ]
    test_metrics = rescue_metrics(test["present"], test["references"], test_outputs)
    val_baselines = public_baselines(val["present"], val["references"], val["raw_texts"], val["fallback_texts"])
    test_baselines = public_baselines(test["present"], test["references"], test["raw_texts"], test["fallback_texts"])

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve(strict=False)),
        "feature_cache": str(Path(args.feature_cache).resolve(strict=False)),
        "bakeoff": str(Path(args.bakeoff).resolve(strict=False)),
        "val_split": args.val_split,
        "test_split": args.test_split,
        "policy": {"max_false_reject_rate": args.max_frr, "min_reject_accuracy": args.min_rr},
        "threshold": threshold,
        "threshold_source": "public_validation",
        "val_selection": selection,
        "val_baselines": val_baselines,
        "test_metrics": test_metrics,
        "test_baselines": test_baselines,
        "dataset_a_used_for_training": False,
        "label_usage": "public_aishell_transcripts_only",
        "val_count": len(val["present"]),
        "test_count": len(test["present"]),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "threshold": threshold,
        "val_overall": selection["metrics"]["overall"],
        "val_cer": selection["metrics"]["avg_cer"],
        "val_rr": selection["metrics"]["avg_rr"],
        "val_frr": selection["metrics"]["false_reject_rate"],
        "test_overall": test_metrics["overall"],
        "test_cer": test_metrics["avg_cer"],
        "test_rr": test_metrics["avg_rr"],
        "test_raw_alone_overall": test_baselines["raw_alone"]["overall"],
        "test_fallback_only_overall": test_baselines["fallback_only"]["overall"],
    }, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
