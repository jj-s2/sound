"""Train the Phase-2 temporal speaker head on a public/synthetic manifest.

The WeSpeaker encoder is frozen and used only to produce cached window features.
Dataset-A is deliberately not accepted as a training manifest; it remains a
held-out audit set for a separate evaluation step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.temporal_head import TemporalSpeakerHead
from xh202615.temporal_training import (
    binary_metrics,
    build_pair_features,
    select_presence_threshold,
)
from xh202615.training_data import assert_valid_training_manifest, read_training_manifest


TARGET_SAMPLE_RATE = 16_000
EMBEDDING_DIM = 256


def prepare_frozen_encoder(model, *, device: str | None = None):
    """Put a frozen encoder on its requested inference device when supported."""

    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()
    inner_model = getattr(model, "model", None)
    inner_eval = getattr(inner_model, "eval", None)
    if callable(inner_eval):
        inner_eval()
    if device is not None:
        set_device = getattr(model, "set_device", None)
        if callable(set_device):
            set_device(device)
    return model


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _embedding(model, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    pcm = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
    with torch.no_grad():
        encoder_device = torch.device(getattr(model, "device", "cpu"))
        if (
            encoder_device.type != "cpu"
            and callable(getattr(model, "compute_features", None))
            and callable(getattr(model, "model", None))
        ):
            features = model.compute_features(pcm, sample_rate=sample_rate, cmn=True)
            outputs = model.model(features.to(encoder_device))
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
            value = outputs[0].detach().cpu()
        else:
            value = model.extract_embedding_from_pcm(pcm, sample_rate)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.size != EMBEDDING_DIM or not np.isfinite(value).all():
        raise ValueError(f"unexpected WeSpeaker embedding shape/value: {value.shape}")
    return value


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}: {path}")
    mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    if mono.size < 3200 or not np.isfinite(mono).all():
        raise ValueError(f"audio is too short or non-finite: {path}")
    return mono, int(sample_rate)


def _windows(audio: np.ndarray, count: int) -> tuple[np.ndarray, ...]:
    if count < 1:
        raise ValueError("window_count must be positive")
    chunks = []
    for index in range(count):
        start = int(round(index * len(audio) / count))
        end = int(round((index + 1) * len(audio) / count))
        chunk = audio[start:end]
        if len(chunk) < 3200:
            chunk = np.pad(chunk, (0, 3200 - len(chunk)))
        chunks.append(chunk.astype(np.float32, copy=False))
    return tuple(chunks)


def _feature_sequence(model, enrollment_path: Path, input_path: Path, window_count: int) -> np.ndarray:
    enrollment_audio, sample_rate = _read_audio(enrollment_path)
    input_audio, _ = _read_audio(input_path)
    enrollment_embedding = _embedding(model, enrollment_audio, sample_rate)
    windows = _windows(input_audio, window_count)
    window_embeddings = np.stack(
        [_embedding(model, window, sample_rate) for window in windows], axis=0
    )
    energy = np.asarray(
        [float(np.sqrt(np.mean(np.square(window, dtype=np.float64)))) for window in windows],
        dtype=np.float32,
    )
    energy = np.log1p(np.maximum(energy, 0.0))
    return build_pair_features(enrollment_embedding, window_embeddings, energy)


def _manifest_digest(rows: Iterable[object]) -> str:
    payload = "\n".join(str(getattr(row, "row_id")) for row in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_feature_cache(
    rows: tuple,
    *,
    model_name: str,
    window_count: int,
    input_field: str,
    cache_path: Path,
    encoder_device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    """Extract frozen features once and persist tensors for ablations."""

    try:
        import wespeaker
    except ImportError as exc:
        raise RuntimeError("WeSpeaker is required for temporal-head training") from exc
    model = prepare_frozen_encoder(wespeaker.load_model(model_name), device=encoder_device)

    sequences = []
    labels = []
    overlaps = []
    row_ids = []
    for index, row in enumerate(rows, start=1):
        input_path = getattr(row, input_field)
        if input_path is None:
            raise ValueError(f"row {row.row_id} has no {input_field}")
        sequence = _feature_sequence(model, row.enrollment_audio, input_path, window_count)
        sequences.append(sequence)
        labels.append(float(row.target_present))
        overlaps.append(float(row.overlap_ratio))
        row_ids.append(str(row.row_id))
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"features {index}/{len(rows)}", flush=True)

    features = torch.from_numpy(np.stack(sequences, axis=0)).float()
    target = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    overlap = torch.tensor(overlaps, dtype=torch.float32).unsqueeze(1)
    payload = {
        "features": features,
        "target": target,
        "overlap": overlap,
        "row_ids": row_ids,
        "manifest_digest": _manifest_digest(rows),
        "input_field": input_field,
        "window_count": window_count,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return features, target, overlap, tuple(row_ids)


def _load_or_build_cache(
    rows: tuple,
    *,
    model_name: str,
    window_count: int,
    input_field: str,
    cache_path: Path,
    reuse_cache: bool,
    encoder_device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    if reuse_cache and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("manifest_digest") == _manifest_digest(rows) and payload.get(
            "input_field"
        ) == input_field and payload.get("window_count") == window_count:
            return payload["features"], payload["target"], payload["overlap"], tuple(
                payload["row_ids"]
            )
    return build_feature_cache(
        rows,
        model_name=model_name,
        window_count=window_count,
        input_field=input_field,
        cache_path=cache_path,
        encoder_device=encoder_device,
    )


def _split_indices(rows: tuple, split: str) -> list[int]:
    return [index for index, row in enumerate(rows) if row.split == split]


def _evaluate(
    model: TemporalSpeakerHead,
    features: torch.Tensor,
    target: torch.Tensor,
    overlap: torch.Tensor,
    indices: list[int],
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    if not indices:
        return {"loss": float("nan"), "overlap_mse": float("nan")}
    model.eval()
    with torch.no_grad():
        presence_logits, overlap_logits = model(features[indices].to(device))
        target_device = target[indices].to(device)
        overlap_device = overlap[indices].to(device)
        presence_loss = nn.functional.binary_cross_entropy_with_logits(
            presence_logits, target_device
        )
        overlap_loss = nn.functional.mse_loss(torch.sigmoid(overlap_logits), overlap_device)
        probabilities = torch.sigmoid(presence_logits).cpu().numpy().reshape(-1)
    metrics = binary_metrics(target[indices].numpy(), probabilities, threshold=threshold)
    metrics.update(
        {
            "loss": float((presence_loss + 0.2 * overlap_loss).item()),
            "overlap_mse": float(overlap_loss.item()),
        }
    )
    return metrics


def _presence_probabilities(
    model: TemporalSpeakerHead,
    features: torch.Tensor,
    indices: list[int],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        presence_logits, _ = model(features[indices].to(device))
        return torch.sigmoid(presence_logits).cpu().numpy().reshape(-1)


def train(args: argparse.Namespace) -> dict:
    _seed_everything(args.seed)
    forbidden = (Path(args.dataset_a_root),)
    rows = read_training_manifest(args.manifest)
    rows = assert_valid_training_manifest(
        rows,
        manifest_path=args.manifest,
        forbidden_roots=forbidden,
    )
    if args.max_rows is not None:
        rows = tuple(rows[: args.max_rows])
    if not rows:
        raise ValueError("training manifest is empty")
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_path = output_dir / "frozen_features.pt"
    features, target, overlap, row_ids = _load_or_build_cache(
        rows,
        model_name=args.model,
        window_count=args.window_count,
        input_field=args.input_field,
        cache_path=cache_path,
        reuse_cache=args.reuse_cache,
        encoder_device=str(device),
    )
    train_indices = _split_indices(rows, "train")
    val_indices = _split_indices(rows, "val")
    test_indices = _split_indices(rows, "test")
    if not train_indices or not val_indices or not test_indices:
        raise ValueError("manifest must contain train, val, and test rows")

    model = TemporalSpeakerHead(
        input_dim=int(features.shape[-1]), hidden_dim=args.hidden_dim, mode=args.mode
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = TensorDataset(features[train_indices], target[train_indices], overlap[train_indices])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    history = []
    best_val = float("inf")
    best_threshold = 0.5
    best_path = output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_features, batch_target, batch_overlap in loader:
            batch_features = batch_features.to(device)
            batch_target = batch_target.to(device)
            batch_overlap = batch_overlap.to(device)
            presence_logits, overlap_logits = model(batch_features)
            loss = nn.functional.binary_cross_entropy_with_logits(
                presence_logits, batch_target
            ) + 0.2 * nn.functional.mse_loss(torch.sigmoid(overlap_logits), batch_overlap)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val_probabilities = _presence_probabilities(model, features, val_indices, device)
        selected_threshold = select_presence_threshold(
            target[val_indices].numpy(),
            val_probabilities,
            min_recall=args.min_val_recall,
        )
        train_metrics = _evaluate(
            model, features, target, overlap, train_indices, device, selected_threshold
        )
        val_metrics = _evaluate(
            model, features, target, overlap, val_indices, device, selected_threshold
        )
        test_metrics = _evaluate(
            model, features, target, overlap, test_indices, device, selected_threshold
        )
        record = {
            "epoch": epoch,
            "presence_threshold": selected_threshold,
            "train_loss": float(np.mean(losses)),
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_threshold = selected_threshold
            torch.save(
                {
                    "model": model.state_dict(),
                    "input_dim": int(features.shape[-1]),
                    "hidden_dim": args.hidden_dim,
                    "mode": args.mode,
                    "window_count": args.window_count,
                    "input_field": args.input_field,
                    "presence_threshold": selected_threshold,
                    "threshold_source": "public_validation",
                    "min_val_recall": args.min_val_recall,
                    "row_ids": row_ids,
                },
                best_path,
            )

    summary = {
        "manifest": str(Path(args.manifest).resolve(strict=False)),
        "dataset_a_root": str(Path(args.dataset_a_root).resolve(strict=False)),
        "rows": len(rows),
        "split_rows": {split: len(_split_indices(rows, split)) for split in ("train", "val", "test")},
        "mode": args.mode,
        "device": str(device),
        "window_count": args.window_count,
        "input_field": args.input_field,
        "min_val_recall": args.min_val_recall,
        "best_presence_threshold": best_threshold,
        "history": history,
        "best_checkpoint": str(best_path),
        "dataset_a_used_for_training": False,
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
    parser.add_argument("--mode", choices=("gru", "mlp", "fused"), default="gru")
    parser.add_argument("--input-field", choices=("mixture_audio", "target_audio"), default="mixture_audio")
    parser.add_argument("--window-count", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-val-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
