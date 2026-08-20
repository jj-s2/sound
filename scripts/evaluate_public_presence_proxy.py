"""Dataset-A audit adapter for the global public-presence proxy checkpoint.

The public-validation checkpoint is frozen for this audit. Dataset-A labels are
loaded only to score the final gated predictions via ``evaluate_rows``; they are
never used to pick a model, threshold, feature parameter, or route.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_public_presence_proxy import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    _resolve_device,
    _row_proxy_features,
    prepare_frozen_encoder,
)
from xh202615.data import load_dataset, read_jsonl  # noqa: E402
from xh202615.evaluation import evaluate_rows  # noqa: E402
from xh202615.public_proxy import GlobalPresenceCalibrator  # noqa: E402


def gated_text(probability, asr_text, *, threshold):
    """Return the ASR text when probability is at least threshold, else empty."""
    if float(probability) >= float(threshold):
        return asr_text
    return ""


def _load_asr_map(path):
    """Frozen ASR text map keyed by sample id."""
    asr_map: dict[str, str] = {}
    for row in read_jsonl(path):
        if "id" not in row:
            raise ValueError("ASR prediction row is missing field 'id'")
        sample_id = str(row["id"])
        text = row.get("recognition_text", row.get("text", ""))
        asr_map[sample_id] = "" if text is None else str(text)
    return asr_map


def _load_global_checkpoint(path):
    """Load and validate a global presence-proxy calibrator checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint is not suitable for the global presence proxy")
    input_dim = payload.get("input_dim")
    presence_threshold = payload.get("presence_threshold")
    state_dict = payload.get("model")
    if (
        not isinstance(input_dim, int)
        or isinstance(input_dim, bool)
        or input_dim <= 0
    ):
        raise ValueError(
            "checkpoint is not suitable for the global presence proxy: missing input_dim"
        )
    if presence_threshold is None:
        raise ValueError(
            "checkpoint is not suitable for the global presence proxy: missing presence_threshold"
        )
    if (
        not isinstance(state_dict, dict)
        or "linear.weight" not in state_dict
        or "linear.bias" not in state_dict
    ):
        raise ValueError(
            "checkpoint is not suitable for the global presence proxy calibrator"
        )
    return payload, int(input_dim), float(presence_threshold)


def audit(args):
    """Run the Dataset-A audit for the global public-presence proxy checkpoint."""
    checkpoint_path = Path(args.checkpoint).expanduser().resolve(strict=False)
    asr_path = Path(args.asr_predictions).expanduser().resolve(strict=False)
    output_path = Path(args.output).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_a_root = Path(args.dataset_a_root).expanduser().resolve(strict=False)
    device = _resolve_device(args.device)

    payload, input_dim, presence_threshold = _load_global_checkpoint(checkpoint_path)
    threshold = float(args.threshold) if args.threshold is not None else presence_threshold
    threshold_source = payload.get("threshold_source")

    model = GlobalPresenceCalibrator(input_dim=input_dim).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    asr_map = _load_asr_map(asr_path)
    samples = load_dataset(dataset_a_root, ("pos", "neg"))

    try:
        import wespeaker
    except ImportError as exc:  # pragma: no cover - exercised only with WeSpeaker
        raise RuntimeError(
            "WeSpeaker is required for the public presence proxy audit"
        ) from exc
    encoder = prepare_frozen_encoder(wespeaker.load_model(args.model), device=str(device))

    per_sample = []
    prediction_rows = []
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample.id)
        row_like = SimpleNamespace(
            enrollment_audio=Path(sample.wakeup_audio),
            command_audio=Path(sample.command_audio),
        )
        global_features, _ = _row_proxy_features(
            encoder,
            row_like,
            input_field="command_audio",
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
            sample_rate=TARGET_SAMPLE_RATE,
        )
        with torch.no_grad():
            logits = model(
                torch.from_numpy(np.asarray(global_features, dtype=np.float32))
                .unsqueeze(0)
                .to(device)
            )
            probability = float(torch.sigmoid(logits).cpu().numpy().reshape(-1)[0])
        asr_text = asr_map.get(sample_id, "")
        gated = gated_text(probability, asr_text, threshold=threshold)
        route = "accept" if probability >= threshold else "reject"
        per_sample.append(
            {
                "id": sample_id,
                "split": sample.split,
                "probability": probability,
                "route": route,
                "gated_text": gated,
            }
        )
        prediction_rows.append({"id": sample_id, "recognition_text": gated})
        if index == 1 or index % 25 == 0 or index == len(samples):
            print(f"audit {index}/{len(samples)}", flush=True)

    report = evaluate_rows(samples, prediction_rows, missing_policy="empty")
    metrics = dict(report.metrics)
    avg_cer = float(metrics.get("avg_cer", 0.0))
    avg_rr = float(metrics.get("avg_rr", 0.0))
    metrics["overall"] = ((1.0 - avg_cer) + avg_rr) / 2.0

    audit_json = {
        "checkpoint": str(checkpoint_path),
        "asr_predictions": str(asr_path),
        "dataset_a_root": str(dataset_a_root),
        "model": args.model,
        "device": str(device),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "window_seconds": float(args.window_seconds),
        "hop_seconds": float(args.hop_seconds),
        "dataset_a_used_for_training": False,
        "label_usage": "audit_only",
        "metrics": metrics,
        "per_sample": per_sample,
    }
    output_path.write_text(
        json.dumps(audit_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.gated_predictions:
        gated_path = Path(args.gated_predictions).expanduser().resolve(strict=False)
        gated_path.parent.mkdir(parents=True, exist_ok=True)
        with gated_path.open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return audit_json


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--asr-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-a-root", default="datasetA/datasetA")
    parser.add_argument("--model", default="chinese")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--hop-seconds", type=float, default=0.5)
    parser.add_argument("--gated-predictions", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    audit(parse_args())
