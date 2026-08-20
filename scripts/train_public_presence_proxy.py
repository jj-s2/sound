"""Train the public-only presence-proxy calibration heads.

Dataset-A is never used for training. Manifest audio paths that resolve under
the configured Dataset-A root are rejected before any WeSpeaker import, WAV
read, output creation, or GPU work. Thresholds are selected only from the
public validation split via ``select_public_validation_threshold``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_temporal_head import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    _embedding,
    _read_audio,
    prepare_frozen_encoder,
)
from xh202615.public_proxy import (  # noqa: E402
    GlobalPresenceCalibrator,
    build_public_proxy_features,
    presence_proxy_metrics,
    select_public_validation_threshold,
)
from xh202615.temporal_head import TemporalSpeakerHead  # noqa: E402
from xh202615.training_data import (  # noqa: E402
    assert_valid_training_manifest,
    read_training_manifest,
)

# 0.2 s at 16 kHz; the minimum embedding window length used elsewhere.
_MIN_WINDOW_SAMPLES = 3200
_METRIC_KEYS = (
    "false_reject_rate",
    "reject_accuracy",
    "false_accept_rate",
    "target_accept_rate",
    "presence_proxy_utility",
)


def cache_metadata(rows, *, model_name, audio_field, window_seconds, hop_seconds):
    """Deterministic feature-identity metadata for the public-proxy cache.

    ``manifest_id`` is a nonempty SHA-256 digest of the ordered row IDs. Invalid
    rows or non-finite/positive window/hop sizes are rejected.
    """
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("rows must be non-empty")
    row_ids = []
    for row in materialized:
        row_id = getattr(row, "row_id", None)
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError("every row must carry a non-empty string row_id")
        row_ids.append(row_id)
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(audio_field, str) or not audio_field.strip():
        raise ValueError("audio_field must be a non-empty string")
    window_seconds = _as_finite_positive(window_seconds, "window_seconds")
    hop_seconds = _as_finite_positive(hop_seconds, "hop_seconds")
    digest = hashlib.sha256("\n".join(row_ids).encode("utf-8")).hexdigest()
    return {
        "manifest_id": digest,
        "model_name": model_name,
        "audio_field": audio_field,
        "window_seconds": float(window_seconds),
        "hop_seconds": float(hop_seconds),
    }


def _as_finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg):
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _bucket_key(row):
    """Class-comparable SNR tier used as the threshold-selection bucket key.

    SIR/overlap are intentionally excluded: target-absent synthetic rows carry
    no SIR/overlap, so a combined key would form single-class buckets that
    cannot be fairly ranked by ``presence_proxy_utility``.
    """
    snr = getattr(row, "snr_db", None)
    if snr is None:
        return "snr_none"
    if snr < 0:
        return "snr_low"
    if snr < 15:
        return "snr_med"
    return "snr_high"


def _fixed_hop_windows(audio, sample_rate, window_seconds, hop_seconds):
    """Fixed-hop windows; the final short window is padded to >= 0.2 s."""
    window_len = max(int(round(window_seconds * sample_rate)), _MIN_WINDOW_SAMPLES)
    hop_len = max(int(round(hop_seconds * sample_rate)), 1)
    windows = []
    total = len(audio)
    if total <= 0:
        return windows
    start = 0
    while start < total:
        chunk = audio[start : start + window_len]
        if len(chunk) < _MIN_WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, _MIN_WINDOW_SAMPLES - len(chunk)))
        windows.append(chunk.astype(np.float32, copy=False))
        start += hop_len
    return windows


def _window_log_energy(windows):
    energy = np.asarray(
        [float(np.sqrt(np.mean(np.square(w, dtype=np.float64)))) for w in windows],
        dtype=np.float32,
    )
    return np.log1p(np.maximum(energy, 0.0)).astype(np.float32)


def _row_proxy_features(
    encoder, row, *, input_field, window_seconds, hop_seconds, sample_rate
):
    input_path = getattr(row, input_field)
    if input_path is None:
        raise ValueError(f"row {row.row_id} has no {input_field}")
    enrollment_audio, _ = _read_audio(Path(row.enrollment_audio))
    input_audio, _ = _read_audio(Path(input_path))
    enrollment_embedding = _embedding(encoder, enrollment_audio, sample_rate)
    mixture_embedding = _embedding(encoder, input_audio, sample_rate)
    windows = _fixed_hop_windows(input_audio, sample_rate, window_seconds, hop_seconds)
    if not windows:
        raise ValueError(f"row {row.row_id} produced no windows")
    window_embeddings = np.stack(
        [_embedding(encoder, window, sample_rate) for window in windows], axis=0
    )
    log_energy = _window_log_energy(windows)
    global_features, frame_features = build_public_proxy_features(
        enrollment_embedding, mixture_embedding, window_embeddings, log_energy
    )
    return global_features, frame_features


def _build_feature_cache(
    rows,
    *,
    metadata,
    cache_path,
    model_name,
    input_field,
    device,
    window_seconds,
    hop_seconds,
):
    try:
        import wespeaker
    except ImportError as exc:  # pragma: no cover - exercised only with WeSpeaker
        raise RuntimeError("WeSpeaker is required for public-proxy training") from exc
    encoder = prepare_frozen_encoder(
        wespeaker.load_model(model_name), device=str(device)
    )

    global_features = []
    frame_features = []
    targets = []
    row_ids = []
    for index, row in enumerate(rows, start=1):
        global_vec, frame_mat = _row_proxy_features(
            encoder,
            row,
            input_field=input_field,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            sample_rate=TARGET_SAMPLE_RATE,
        )
        global_features.append(global_vec)
        frame_features.append(frame_mat)
        targets.append(float(bool(row.target_present)))
        row_ids.append(str(row.row_id))
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"public-proxy features {index}/{len(rows)}", flush=True)

    payload = {
        "global_features": torch.from_numpy(np.stack(global_features, axis=0)).float(),
        "frame_features": [torch.from_numpy(f).float() for f in frame_features],
        "targets": torch.tensor(targets, dtype=torch.float32).unsqueeze(1),
        "row_ids": tuple(row_ids),
        "metadata": metadata,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return (
        payload["global_features"],
        payload["frame_features"],
        payload["targets"],
        payload["row_ids"],
    )


def _load_or_build_feature_cache(
    rows,
    *,
    metadata,
    cache_path,
    reuse_cache,
    model_name,
    input_field,
    device,
    window_seconds,
    hop_seconds,
):
    if reuse_cache and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("metadata") == metadata:
            return (
                payload["global_features"],
                payload["frame_features"],
                payload["targets"],
                tuple(payload["row_ids"]),
            )
    return _build_feature_cache(
        rows,
        metadata=metadata,
        cache_path=cache_path,
        model_name=model_name,
        input_field=input_field,
        device=device,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )


def _split_indices(rows, split):
    return [index for index, row in enumerate(rows) if row.split == split]


def _labels(targets, indices):
    return [int(value) for value in targets[indices].numpy().reshape(-1).tolist()]


def _global_probabilities(model, global_features, indices, device):
    model.eval()
    with torch.no_grad():
        logits = model(global_features[indices].to(device))
        return torch.sigmoid(logits).cpu().numpy().reshape(-1)


def _frame_probabilities(model, frame_features, indices, device, max_t):
    model.eval()
    batch = torch.zeros(len(indices), max_t, 2, dtype=torch.float32)
    for row, index in enumerate(indices):
        frame = frame_features[index]
        length = min(frame.shape[0], max_t)
        batch[row, :length] = frame[:length]
    with torch.no_grad():
        presence_logits, _ = model(batch.to(device))
        return torch.sigmoid(presence_logits).cpu().numpy().reshape(-1)


def _presence_metrics(targets, indices, probabilities, threshold):
    labels = _labels(targets, indices)
    probs = [float(value) for value in np.asarray(probabilities).reshape(-1).tolist()]
    return presence_proxy_metrics(labels, probs, float(threshold))


def _select_on_val(labels, probabilities, bucket_keys, args):
    return select_public_validation_threshold(
        labels,
        [float(value) for value in np.asarray(probabilities).reshape(-1).tolist()],
        list(bucket_keys),
        max_false_reject_rate=args.max_false_reject_rate,
        min_reject_accuracy=args.min_reject_accuracy,
    )


def _train_global(
    global_features, targets, *, train_idx, val_idx, test_idx, rows, args, device, output_dir
):
    model = GlobalPresenceCalibrator(input_dim=int(global_features.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_x = global_features[train_idx].to(device)
    train_y = targets[train_idx].to(device)
    val_labels = _labels(targets, val_idx)
    val_bucket_keys = [_bucket_key(rows[index]) for index in val_idx]
    best = None
    best_utility = -float("inf")
    best_path = output_dir / "best_global.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(train_x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        val_probs = _global_probabilities(model, global_features, val_idx, device)
        try:
            selection = _select_on_val(val_labels, val_probs, val_bucket_keys, args)
        except ValueError:
            print(
                json.dumps(
                    {"candidate": "global", "epoch": epoch, "status": "no_eligible_threshold"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        utility = float(selection["worst_bucket_utility"])
        if utility > best_utility or best is None:
            best_utility = utility
            threshold = float(selection["threshold"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "input_dim": int(global_features.shape[-1]),
                    "presence_threshold": threshold,
                    "threshold_source": "public_validation",
                    "best_epoch": epoch,
                },
                best_path,
            )
            best = {
                "candidate": "global",
                "best_epoch": epoch,
                "selected_threshold": threshold,
                "threshold_source": "public_validation",
                "best_val_presence_proxy_utility": utility,
                "train": _presence_metrics(
                    targets,
                    train_idx,
                    _global_probabilities(model, global_features, train_idx, device),
                    threshold,
                ),
                "val": selection["metrics"],
                "best_checkpoint": str(best_path),
            }
            print(
                json.dumps(
                    {
                        "candidate": "global",
                        "epoch": epoch,
                        "val_presence_proxy_utility": utility,
                        "threshold": threshold,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if best is None:
        raise ValueError("no eligible public validation threshold")
    best["test"] = _presence_metrics(
        targets,
        test_idx,
        _global_probabilities(model, global_features, test_idx, device),
        best["selected_threshold"],
    )
    return best


def _train_frame(
    frame_features, targets, *, train_idx, val_idx, test_idx, rows, args, device, output_dir
):
    max_t = max(frame.shape[0] for frame in frame_features)
    model = TemporalSpeakerHead(
        input_dim=2, hidden_dim=args.hidden_dim, mode=args.mode
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_batch = torch.zeros(len(train_idx), max_t, 2, dtype=torch.float32)
    for row, index in enumerate(train_idx):
        frame = frame_features[index]
        train_batch[row, : frame.shape[0]] = frame
    train_batch = train_batch.to(device)
    train_y = targets[train_idx].to(device)
    val_labels = _labels(targets, val_idx)
    val_bucket_keys = [_bucket_key(rows[index]) for index in val_idx]
    best = None
    best_utility = -float("inf")
    best_path = output_dir / "best_frame.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        presence_logits, _ = model(train_batch)
        loss = nn.functional.binary_cross_entropy_with_logits(presence_logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        val_probs = _frame_probabilities(model, frame_features, val_idx, device, max_t)
        try:
            selection = _select_on_val(val_labels, val_probs, val_bucket_keys, args)
        except ValueError:
            print(
                json.dumps(
                    {"candidate": "frame", "epoch": epoch, "status": "no_eligible_threshold"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        utility = float(selection["worst_bucket_utility"])
        if utility > best_utility or best is None:
            best_utility = utility
            threshold = float(selection["threshold"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "input_dim": 2,
                    "hidden_dim": args.hidden_dim,
                    "mode": args.mode,
                    "presence_threshold": threshold,
                    "threshold_source": "public_validation",
                    "best_epoch": epoch,
                },
                best_path,
            )
            best = {
                "candidate": "frame",
                "best_epoch": epoch,
                "selected_threshold": threshold,
                "threshold_source": "public_validation",
                "best_val_presence_proxy_utility": utility,
                "train": _presence_metrics(
                    targets,
                    train_idx,
                    _frame_probabilities(model, frame_features, train_idx, device, max_t),
                    threshold,
                ),
                "val": selection["metrics"],
                "best_checkpoint": str(best_path),
            }
            print(
                json.dumps(
                    {
                        "candidate": "frame",
                        "epoch": epoch,
                        "val_presence_proxy_utility": utility,
                        "threshold": threshold,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if best is None:
        raise ValueError("no eligible public validation threshold")
    best["test"] = _presence_metrics(
        targets,
        test_idx,
        _frame_probabilities(model, frame_features, test_idx, device, max_t),
        best["selected_threshold"],
    )
    return best


_NO_ELIGIBLE_THRESHOLD = "no eligible public validation threshold"


def _attempt_candidate(train_fn, *args, **kwargs):
    """Run one candidate, isolating the no-eligible-threshold policy failure.

    Returns ``(report, None)`` on success or ``(None, message)`` when the
    candidate raises the public-validation policy ``ValueError``. Any other
    error propagates unchanged.
    """
    try:
        return train_fn(*args, **kwargs), None
    except ValueError as exc:
        if _NO_ELIGIBLE_THRESHOLD not in str(exc):
            raise
        return None, str(exc)


def train(args):
    """Run public-only presence-proxy training and write a summary.

    Manifest validation (including the Dataset-A forbidden-path check) and the
    non-empty split requirement happen before any WeSpeaker, audio, output, or
    GPU work.
    """
    rows = read_training_manifest(args.manifest)
    rows = assert_valid_training_manifest(
        rows,
        manifest_path=args.manifest,
        forbidden_roots=(Path(args.dataset_a_root),),
    )
    train_idx = _split_indices(rows, "train")
    val_idx = _split_indices(rows, "val")
    test_idx = _split_indices(rows, "test")
    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            "manifest must contain non-empty public train, val, and test splits"
        )

    _seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    metadata = cache_metadata(
        rows,
        model_name=args.model,
        audio_field=args.input_field,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
    )
    cache_path = output_dir / "public_proxy_features.pt"
    global_features, frame_features, targets, row_ids = _load_or_build_feature_cache(
        rows,
        metadata=metadata,
        cache_path=cache_path,
        reuse_cache=args.reuse_cache,
        model_name=args.model,
        input_field=args.input_field,
        device=device,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
    )

    candidate = args.candidate
    candidate_reports = {}
    selected_thresholds = {}
    public_test_metrics = {}
    succeeded = []

    if candidate in ("global", "all"):
        report, error = _attempt_candidate(
            _train_global,
            global_features,
            targets,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            rows=rows,
            args=args,
            device=device,
            output_dir=output_dir,
        )
        if report is not None:
            candidate_reports["global"] = report
            selected_thresholds["global"] = report["selected_threshold"]
            public_test_metrics["global"] = report["test"]
            succeeded.append("global")
        else:
            candidate_reports["global"] = {
                "candidate": "global",
                "status": "no_eligible_public_validation_threshold",
                "message": error,
            }

    if candidate in ("frame", "all"):
        report, error = _attempt_candidate(
            _train_frame,
            frame_features,
            targets,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            rows=rows,
            args=args,
            device=device,
            output_dir=output_dir,
        )
        if report is not None:
            candidate_reports["frame"] = report
            selected_thresholds["frame"] = report["selected_threshold"]
            public_test_metrics["frame"] = report["test"]
            succeeded.append("frame")
        else:
            candidate_reports["frame"] = {
                "candidate": "frame",
                "status": "no_eligible_public_validation_threshold",
                "message": error,
            }

    if not succeeded:
        raise ValueError(_NO_ELIGIBLE_THRESHOLD)

    summary = {
        "manifest": str(Path(args.manifest).resolve(strict=False)),
        "dataset_a_root": str(Path(args.dataset_a_root).resolve(strict=False)),
        "dataset_a_used_for_training": False,
        "threshold_source": "public_validation",
        "cache_metadata": metadata,
        "selected_thresholds": selected_thresholds,
        "candidate_reports": candidate_reports,
        "public_test_metrics": public_test_metrics,
        "rows": len(rows),
        "split_rows": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "candidate": candidate,
        "device": str(device),
        "window_seconds": float(args.window_seconds),
        "hop_seconds": float(args.hop_seconds),
        "input_field": args.input_field,
        "model": args.model,
        "row_ids": list(row_ids),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-a-root", default="datasetA/datasetA")
    parser.add_argument("--model", default="chinese")
    parser.add_argument("--input-field", default="mixture_audio")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--hop-seconds", type=float, default=0.5)
    parser.add_argument(
        "--candidate", choices=("global", "frame", "all"), default="all"
    )
    parser.add_argument("--mode", choices=("gru", "mlp", "fused"), default="gru")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-false-reject-rate", type=float, default=0.10)
    parser.add_argument("--min-reject-accuracy", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default=None)
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
