"""Audit a trained temporal head on Dataset-A without tuning on its labels."""

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

from scripts.train_temporal_head import _feature_sequence, prepare_frozen_encoder
from xh202615.data import load_dataset, read_jsonl
from xh202615.evaluation import evaluate_rows
from xh202615.temporal_head import TemporalSpeakerHead


def gated_text(probability: float, asr_text: str, *, threshold: float) -> str:
    """Keep a frozen ASR result only when the head accepts the sample."""

    return asr_text if probability >= threshold else ""


def select_candidate_text(
    probability: float,
    raw_text: str,
    fusion_text: str,
    *,
    threshold: float,
) -> tuple[str, str]:
    """Choose frozen ASR candidates with a public-trained presence decision."""

    if probability >= threshold and raw_text:
        return raw_text, "raw"
    return fusion_text, "fusion" if raw_text else "fusion_empty_raw_fallback"


def _asr_map(path: Path) -> dict[str, str]:
    result = {}
    for row in read_jsonl(path):
        result[str(row["id"])] = str(row.get("recognition_text", row.get("text", "")) or "")
    return result


def evaluate(args: argparse.Namespace) -> dict:
    dataset_root = Path(args.dataset_a_root).expanduser().resolve(strict=False)
    samples = load_dataset(dataset_root, ("pos", "neg"))
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    fusion_asr = _asr_map(Path(args.asr_predictions))
    raw_asr = (
        _asr_map(Path(args.raw_asr_predictions))
        if args.raw_asr_predictions is not None
        else {}
    )
    if args.router_mode in ("rescue", "safe_rescue") and args.raw_asr_predictions is None:
        raise ValueError(f"--raw-asr-predictions is required for --router-mode {args.router_mode}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.threshold is None:
        threshold = float(checkpoint.get("presence_threshold", 0.5))
        threshold_source = str(checkpoint.get("threshold_source", "legacy_default_0.5"))
    else:
        threshold = float(args.threshold)
        threshold_source = "explicit_override"
    model = TemporalSpeakerHead(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        mode=str(checkpoint["mode"]),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    try:
        import wespeaker
    except ImportError as exc:
        raise RuntimeError("WeSpeaker is required for Dataset-A audit") from exc
    encoder = prepare_frozen_encoder(wespeaker.load_model(args.model), device=str(device))
    predictions = []
    metadata = {}
    window_count = int(checkpoint["window_count"])
    for index, sample in enumerate(samples, start=1):
        sequence = _feature_sequence(encoder, sample.wakeup_audio, sample.command_audio, window_count)
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(sequence).unsqueeze(0).to(device))
            probability = float(torch.sigmoid(logits).item())
        fusion_text = fusion_asr.get(str(sample.id), "")
        raw_text = raw_asr.get(str(sample.id), "")
        if args.router_mode == "rescue":
            text, route = select_candidate_text(
                probability,
                raw_text,
                fusion_text,
                threshold=threshold,
            )
        elif args.router_mode == "safe_rescue":
            # R3-P1: use raw ASR only when the head is confident AND fusion
            # already accepted.  Fusion-rejected samples stay rejected, so RR
            # cannot drop below the fusion baseline by construction.
            from xh202615.r4_rescue import safe_rescue_text

            text, route = safe_rescue_text(probability, raw_text, fusion_text, threshold=threshold)
        else:
            text = gated_text(probability, fusion_text, threshold=threshold)
            route = "accepted_fusion" if text else "rejected"
        predictions.append({"id": str(sample.id), "recognition_text": text, "route": route})
        metadata[str(sample.id)] = {
            "temporal_probability": probability,
            "accepted": bool(text),
            "route": route,
        }
        if index == 1 or index % 100 == 0 or index == len(samples):
            print(f"audit {index}/{len(samples)}", flush=True)

    report = evaluate_rows(samples, predictions, missing_policy="empty")
    metrics = dict(report.metrics)
    metrics["overall"] = ((1.0 - metrics["avg_cer"]) + metrics["avg_rr"]) / 2.0
    payload = {
        "dataset_a_root": str(dataset_root),
        "checkpoint": str(Path(args.checkpoint).resolve(strict=False)),
        "asr_predictions": str(Path(args.asr_predictions).resolve(strict=False)),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "router_mode": args.router_mode,
        "metrics": metrics,
        "dataset_a_used_for_training": False,
        "label_usage": "audit_only",
        "temporal_probabilities": metadata,
    }
    output = Path(args.output).expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.gated_predictions:
        prediction_path = Path(args.gated_predictions).expanduser().resolve(strict=False)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return payload


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--asr-predictions", required=True)
    parser.add_argument("--dataset-a-root", default="datasetA/datasetA")
    parser.add_argument("--model", default="chinese")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--router-mode", choices=("reject", "rescue", "safe_rescue"), default="reject")
    parser.add_argument("--raw-asr-predictions", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gated-predictions", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
