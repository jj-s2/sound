"""Deterministic GPU trainer for the enrollment-conditioned TSE extractor.

Trains :class:`xh202615.target_extractor.FiLMCRNExtractor` to reconstruct a target
speaker from a mixture, conditioned on a frozen WeSpeaker enrollment embedding.

Public-only data boundary
-------------------------
This trainer never reads Dataset-A. It accepts an existing public/synthetic
training manifest (``TrainingManifestRow`` JSONL) and fail-closes on Dataset-A
containment, duplicate IDs, speaker split leakage, missing/non-16 kHz audio, and
non-positive rows (only ``target_present == true`` rows feed the reconstruction
objective). WeSpeaker is frozen and used only to produce cached enrollment
embeddings; no gradient flows through it. Audio is streamed per crop, never
loaded into RAM as a corpus.

The default configuration is a short controlled pilot (2 epochs, small batch).
``--limit-per-split`` reduces the positive rows per split for a smoke run. Do not
start the full pilot until the code and a CUDA smoke have been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import wave
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.target_extractor import (
    FiLMCRNExtractor,
    enhance_waveform,
    multi_resolution_stft_loss,
    negative_si_sdr_loss,
)
from xh202615.training_data import (
    TrainingManifestRow,
    assert_valid_training_manifest,
    read_training_manifest,
)


TARGET_SAMPLE_RATE = 16_000
EMBEDDING_DIM = 256
DEFAULT_DATASET_A_ROOT = "datasetA/datasetA"
DEFAULT_MANIFEST = "data/synthetic/aishell1_phase2_v2/manifest.jsonl"

# An enrollment-embedding provider maps an enrollment audio path to a fixed-size
# float vector. The default provider uses frozen WeSpeaker; tests inject a mock.
EmbeddingProvider = Callable[[Path], np.ndarray]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def manifest_digest(path: str | Path) -> str:
    """SHA-256 hex digest of the manifest file bytes (cache/audit key)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Manifest loading, validation, and positive-row filtering
# ---------------------------------------------------------------------------

def load_positive_rows(
    manifest_path: str | Path, dataset_a_root: str | Path
) -> tuple[TrainingManifestRow, ...]:
    """Read and validate the manifest, then return only target-present rows.

    Fail-closed guards (via ``assert_valid_training_manifest``): Dataset-A
    containment, duplicate IDs, speaker split leakage, and field ranges. Rows
    with ``target_present == false`` never enter the reconstruction objective;
    they are dropped here so the trainer cannot accidentally learn from them.
    """
    rows = read_training_manifest(manifest_path)
    rows = assert_valid_training_manifest(
        rows,
        manifest_path=manifest_path,
        forbidden_roots=(Path(dataset_a_root),),
    )
    return tuple(row for row in rows if row.target_present)


def _wav_sample_rate(path: Path) -> int:
    """Fast sample-rate read via the stdlib ``wave`` header (PCM fallback)."""
    try:
        with wave.open(str(path), "rb") as handle:
            return int(handle.getframerate())
    except (wave.Error, EOFError):
        return int(sf.info(str(path)).samplerate)


def check_audio(rows: Iterable[TrainingManifestRow]) -> None:
    """Fail closed on missing or non-16 kHz audio for the rows used.

    Only the three audio fields of each positive row are checked (existence +
    16 kHz sample rate). The stdlib ``wave`` header read keeps this fast even for
    thousands of files; ``soundfile.info`` is the fallback for non-PCM WAVs.
    """
    missing: list[tuple[str, str, str]] = []
    bad_rate: list[tuple[str, str, int]] = []
    for row in rows:
        for field_name in ("enrollment_audio", "mixture_audio", "target_audio"):
            path = Path(getattr(row, field_name))
            if not path.is_file():
                missing.append((row.row_id, field_name, str(path)))
                continue
            sample_rate = _wav_sample_rate(path)
            if sample_rate != TARGET_SAMPLE_RATE:
                bad_rate.append((row.row_id, field_name, sample_rate))
    if missing:
        sample = ", ".join(f"{rid}:{fld}" for rid, fld, _ in missing[:8])
        raise ValueError(f"missing audio for {len(missing)} row(s)/field(s): {sample}")
    if bad_rate:
        sample = ", ".join(f"{rid}:{fld}@{sr}Hz" for rid, fld, sr in bad_rate[:8])
        raise ValueError(
            f"non-16kHz audio for {len(bad_rate)} row(s)/field(s): {sample}"
        )


def _apply_limit_per_split(
    rows: tuple[TrainingManifestRow, ...], limit_per_split: int | None
) -> tuple[TrainingManifestRow, ...]:
    """Deterministically cap the positive rows per split (smoke runs)."""
    if limit_per_split is None:
        return rows
    by_split: dict[str, list[TrainingManifestRow]] = {}
    for row in rows:
        by_split.setdefault(row.split, []).append(row)
    out: list[TrainingManifestRow] = []
    for split in ("train", "val", "test"):
        out.extend(by_split.get(split, [])[:limit_per_split])
    return tuple(out)


# ---------------------------------------------------------------------------
# Audio I/O and deterministic cropping
# ---------------------------------------------------------------------------

def read_audio_mono(path: str | Path) -> np.ndarray:
    """Read a 16 kHz mono float32 waveform (multi-channel is averaged)."""
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}: {path}")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"audio is empty or non-finite: {path}")
    return mono


def crop_pair(
    mixture: np.ndarray,
    target: np.ndarray,
    segment_samples: int,
    *,
    seed: int,
    epoch: int,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic aligned random crop of a mixture/target pair.

    The same time window is cropped from both the mixture and the clean target
    (they are time-aligned). The crop offset is drawn from a Generator seeded by
    ``(seed, epoch, index)`` so it is reproducible yet varies across epochs.
    Clips shorter than ``segment_samples`` are right-padded with zeros.
    """
    if segment_samples <= 0:
        raise ValueError("segment_samples must be positive")
    n = min(mixture.size, target.size)
    if n < segment_samples:
        mix = np.pad(mixture[:n], (0, segment_samples - n)).astype(np.float32, copy=False)
        tgt = np.pad(target[:n], (0, segment_samples - n)).astype(np.float32, copy=False)
        return np.ascontiguousarray(mix), np.ascontiguousarray(tgt)
    rng = np.random.default_rng([int(seed), int(epoch), int(index)])
    start = int(rng.integers(0, n - segment_samples + 1))
    mix = np.ascontiguousarray(mixture[start : start + segment_samples])
    tgt = np.ascontiguousarray(target[start : start + segment_samples])
    return mix, tgt


# ---------------------------------------------------------------------------
# Composite reconstruction loss
# ---------------------------------------------------------------------------

def composite_loss(
    enhanced: torch.Tensor,
    target: torch.Tensor,
    *,
    stft_weight: float = 1.0,
    si_sdr_weight: float = 1.0,
) -> torch.Tensor:
    """Multi-resolution STFT loss plus negative SI-SDR loss (to be minimised)."""
    stft = multi_resolution_stft_loss(enhanced, target)
    si_sdr = negative_si_sdr_loss(enhanced, target)
    return stft_weight * stft + si_sdr_weight * si_sdr


# ---------------------------------------------------------------------------
# Frozen WeSpeaker enrollment embeddings
# ---------------------------------------------------------------------------

def prepare_frozen_encoder(model, *, device: str | None = None):
    """Put a WeSpeaker encoder in eval mode on the requested device."""
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


def extract_embedding(model, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract a detached enrollment embedding (no gradient through WeSpeaker)."""
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
    return np.asarray(value, dtype=np.float32).reshape(-1)


def wespeaker_provider(model_name: str, device: torch.device) -> EmbeddingProvider:
    """Build a frozen WeSpeaker enrollment-embedding provider."""
    try:
        import wespeaker
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("WeSpeaker is required for TSE training") from exc
    model = prepare_frozen_encoder(
        wespeaker.load_model(model_name), device=str(device)
    )

    def provider(path: Path) -> np.ndarray:
        audio = read_audio_mono(path)  # already validated 16 kHz mono
        embedding = extract_embedding(model, audio, TARGET_SAMPLE_RATE)
        if embedding.size != EMBEDDING_DIM or not np.isfinite(embedding).all():
            raise ValueError(
                f"unexpected WeSpeaker embedding for {path}: shape {embedding.shape}"
            )
        return embedding

    return provider


def load_or_build_embedding_cache(
    rows: tuple[TrainingManifestRow, ...],
    *,
    provider: EmbeddingProvider,
    digest: str,
    model_name: str,
    cache_path: Path,
    reuse_cache: bool,
) -> dict[str, np.ndarray]:
    """Precompute/cache frozen enrollment embeddings keyed by audio path.

    The cache is invalidated unless both the manifest digest and the WeSpeaker
    model name match. Only distinct enrollment paths are encoded; the result is
    a lightweight ``{path: vector}`` mapping held in memory (not the corpus).
    """
    if reuse_cache and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            payload.get("manifest_digest") == digest
            and payload.get("model_name") == model_name
        ):
            return payload["embeddings"]

    embeddings: dict[str, np.ndarray] = {}
    paths = sorted({str(row.enrollment_audio) for row in rows})
    for index, path in enumerate(paths, start=1):
        embeddings[path] = provider(Path(path))
        if index == 1 or index % 25 == 0 or index == len(paths):
            print(f"embeddings {index}/{len(paths)}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "manifest_digest": digest,
            "model_name": model_name,
            "embeddings": embeddings,
        },
        cache_path,
    )
    return embeddings


# ---------------------------------------------------------------------------
# Streaming dataset
# ---------------------------------------------------------------------------

class TSEDataset(Dataset):
    """Stream deterministic crops of mixture/target pairs with cached embeddings.

    Audio is read from disk per item (the corpus is never held in RAM). The crop
    offset depends on ``(seed, epoch, index)``; call :meth:`set_epoch` at the
    start of each training epoch. Evaluation datasets keep ``epoch=0`` so metrics
    are stable and comparable across epochs.
    """

    def __init__(
        self,
        rows: tuple[TrainingManifestRow, ...],
        embeddings: dict[str, np.ndarray],
        segment_samples: int,
        seed: int,
        *,
        epoch: int = 0,
    ) -> None:
        self.rows = rows
        self.embeddings = embeddings
        self.segment_samples = segment_samples
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        mixture = read_audio_mono(row.mixture_audio)
        target = read_audio_mono(row.target_audio)
        mixture_crop, target_crop = crop_pair(
            mixture,
            target,
            self.segment_samples,
            seed=self.seed,
            epoch=self.epoch,
            index=index,
        )
        embedding = self.embeddings[str(row.enrollment_audio)]
        return (
            torch.from_numpy(mixture_crop),
            torch.from_numpy(target_crop),
            torch.from_numpy(np.asarray(embedding, dtype=np.float32)),
        )


# ---------------------------------------------------------------------------
# Train / evaluate steps
# ---------------------------------------------------------------------------

def train_step(
    model: FiLMCRNExtractor,
    optimizer: torch.optim.Optimizer,
    mixture: torch.Tensor,
    target: torch.Tensor,
    embedding: torch.Tensor,
    *,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
    stft_weight: float,
    si_sdr_weight: float,
) -> float | None:
    """One optimised training step. Returns the loss, or ``None`` if skipped.

    AMP (autocast + GradScaler) is used only on CUDA. A finite-loss check skips
    the step (and leaves grads zeroed) when the loss is non-finite; GradScaler
    additionally skips on gradient overflow and downscales.
    """
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        with torch.amp.autocast(device_type="cuda", enabled=True):
            enhanced = enhance_waveform(model, mixture, embedding)
            loss = composite_loss(
                enhanced, target, stft_weight=stft_weight, si_sdr_weight=si_sdr_weight
            )
        if not torch.isfinite(loss):
            return None
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        enhanced = enhance_waveform(model, mixture, embedding)
        loss = composite_loss(
            enhanced, target, stft_weight=stft_weight, si_sdr_weight=si_sdr_weight
        )
        if not torch.isfinite(loss):
            return None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: FiLMCRNExtractor,
    dataset: TSEDataset,
    *,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
    stft_weight: float,
    si_sdr_weight: float,
) -> dict[str, float]:
    """Evaluate positive rows with SI-SDR and multi-resolution STFT loss.

    ``si_sdr`` is the actual SI-SDR (higher is better); ``stft`` is the spectral
    loss (lower is better); ``loss`` is the weighted composite (lower is better).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_si_sdr = 0.0
    total_stft = 0.0
    count = 0
    for mixture, target, embedding in loader:
        mixture = mixture.to(device)
        target = target.to(device)
        embedding = embedding.to(device)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                enhanced = enhance_waveform(model, mixture, embedding)
                stft = multi_resolution_stft_loss(enhanced, target)
                si_sdr_loss = negative_si_sdr_loss(enhanced, target)
        else:
            enhanced = enhance_waveform(model, mixture, embedding)
            stft = multi_resolution_stft_loss(enhanced, target)
            si_sdr_loss = negative_si_sdr_loss(enhanced, target)
        loss = stft_weight * stft + si_sdr_weight * si_sdr_loss
        batch = mixture.shape[0]
        total_loss += float(loss.item()) * batch
        total_si_sdr += float((-si_sdr_loss).item()) * batch
        total_stft += float(stft.item()) * batch
        count += batch
    if count == 0:
        return {"loss": float("nan"), "si_sdr": float("nan"), "stft": float("nan"), "n": 0}
    return {
        "loss": total_loss / count,
        "si_sdr": total_si_sdr / count,
        "stft": total_stft / count,
        "n": count,
    }


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def run_training(
    *,
    manifest: str | Path,
    output_dir: str | Path,
    dataset_a_root: str | Path = DEFAULT_DATASET_A_ROOT,
    model_name: str = "chinese",
    embedding_dim: int = EMBEDDING_DIM,
    channels: Iterable[int] = (16, 32, 64),
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int | None = None,
    gru_hidden: int = 256,
    gru_layers: int = 2,
    epochs: int = 2,
    batch_size: int = 8,
    segment_seconds: float = 2.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    stft_weight: float = 1.0,
    si_sdr_weight: float = 1.0,
    seed: int = 20260805,
    limit_per_split: int | None = None,
    device: str | None = None,
    reuse_cache: bool = True,
    num_workers: int = 0,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict:
    """Run a deterministic TSE training pilot and write checkpoint + summary.

    When ``embedding_provider`` is ``None`` a frozen WeSpeaker provider is built.
    Tests inject a mock provider so no WeSpeaker or network is required.
    """
    _seed_everything(seed)
    manifest_path = Path(manifest).expanduser().resolve(strict=False)
    output_path = Path(output_dir).expanduser().resolve(strict=False)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_a_resolved = Path(dataset_a_root).resolve(strict=False)

    digest = manifest_digest(manifest_path)
    rows = load_positive_rows(manifest_path, dataset_a_resolved)
    rows = tuple(sorted(rows, key=lambda row: row.row_id))
    rows = _apply_limit_per_split(rows, limit_per_split)
    check_audio(rows)

    split_rows = {
        split: tuple(row for row in rows if row.split == split)
        for split in ("train", "val", "test")
    }
    for split in ("train", "val", "test"):
        if not split_rows[split]:
            raise ValueError(
                f"no target-present rows in split {split!r} after filtering"
            )

    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = torch_device.type == "cuda"

    if embedding_provider is None:
        embedding_provider = wespeaker_provider(model_name, torch_device)
    cache_path = output_path / "enrollment_embeddings.pt"
    embeddings = load_or_build_embedding_cache(
        rows,
        provider=embedding_provider,
        digest=digest,
        model_name=model_name,
        cache_path=cache_path,
        reuse_cache=reuse_cache,
    )

    channels_tuple = tuple(channels)
    model = FiLMCRNExtractor(
        embedding_dim=embedding_dim,
        channels=channels_tuple,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
    ).to(torch_device)
    model_config = {
        "embedding_dim": embedding_dim,
        "channels": list(channels_tuple),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": model.win_length,
        "gru_hidden": gru_hidden,
        "gru_layers": gru_layers,
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    segment_samples = int(round(segment_seconds * TARGET_SAMPLE_RATE))

    train_dataset = TSEDataset(split_rows["train"], embeddings, segment_samples, seed)
    val_dataset = TSEDataset(
        split_rows["val"], embeddings, segment_samples, seed, epoch=0
    )
    test_dataset = TSEDataset(
        split_rows["test"], embeddings, segment_samples, seed, epoch=0
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
    )

    best_val_loss = float("inf")
    best_epoch = None
    best_path = output_path / "best.pt"
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        step_losses: list[float] = []
        for mixture, target, embedding in train_loader:
            mixture = mixture.to(torch_device)
            target = target.to(torch_device)
            embedding = embedding.to(torch_device)
            loss_value = train_step(
                model,
                optimizer,
                mixture,
                target,
                embedding,
                use_amp=use_amp,
                scaler=scaler,
                grad_clip=grad_clip,
                stft_weight=stft_weight,
                si_sdr_weight=si_sdr_weight,
            )
            if loss_value is not None:
                step_losses.append(loss_value)

        val_metrics = evaluate(
            model,
            val_dataset,
            device=torch_device,
            use_amp=use_amp,
            batch_size=batch_size,
            stft_weight=stft_weight,
            si_sdr_weight=si_sdr_weight,
        )
        # Test metrics are diagnostic only and never drive checkpoints.
        test_metrics = evaluate(
            model,
            test_dataset,
            device=torch_device,
            use_amp=use_amp,
            batch_size=batch_size,
            stft_weight=stft_weight,
            si_sdr_weight=si_sdr_weight,
        )

        saved = False
        if val_metrics["loss"] < best_val_loss and np.isfinite(val_metrics["loss"]):
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "manifest": str(manifest_path),
                    "manifest_digest": digest,
                    "dataset_a_root": str(dataset_a_resolved),
                    "dataset_a_used_for_training": False,
                    "data_boundary": (
                        "public-only; Dataset-A forbidden as a training source; "
                        "speaker-disjoint splits validated; target-present rows only"
                    ),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "val_metrics": val_metrics,
                    "test_metrics_diagnostic": test_metrics,
                },
                best_path,
            )
            saved = True

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(step_losses)) if step_losses else None,
            "val": val_metrics,
            "test": test_metrics,
            "best_val_loss": best_val_loss,
            "saved": saved,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    sources = sorted({row.source for row in rows})
    summary = {
        "manifest": str(manifest_path),
        "manifest_digest": digest,
        "dataset_a_root": str(dataset_a_resolved),
        "dataset_a_used_for_training": False,
        "data_boundary": (
            "public-only; Dataset-A forbidden as a training source; "
            "speaker-disjoint splits validated; target-present rows only"
        ),
        "source": sources,
        "model_name": model_name,
        "model_config": model_config,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "segment_seconds": segment_seconds,
        "segment_samples": segment_samples,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "stft_weight": stft_weight,
        "si_sdr_weight": si_sdr_weight,
        "device": str(torch_device),
        "amp": use_amp,
        "positive_split_rows": {
            split: len(split_rows[split]) for split in ("train", "val", "test")
        },
        "limit_per_split": limit_per_split,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "embedding_cache": str(cache_path),
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-a-root", default=DEFAULT_DATASET_A_ROOT)
    parser.add_argument("--model", default="chinese", help="frozen WeSpeaker model name")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--stft-weight", type=float, default=1.0)
    parser.add_argument("--si-sdr-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--channels", default="16,32,64")
    parser.add_argument("--gru-hidden", type=int, default=256)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="cap positive rows per split (smoke runs)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-reuse-cache", action="store_true")
    return parser.parse_args(argv)


def train(args: argparse.Namespace, *, embedding_provider: EmbeddingProvider | None = None) -> dict:
    """Argument.Namespace adapter around :func:`run_training`."""
    channels = tuple(int(part) for part in args.channels.split(",") if part.strip())
    return run_training(
        manifest=args.manifest,
        output_dir=args.output_dir,
        dataset_a_root=args.dataset_a_root,
        model_name=args.model,
        embedding_dim=args.embedding_dim,
        channels=channels,
        n_fft=512,
        hop_length=128,
        win_length=None,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        segment_seconds=args.segment_seconds,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        stft_weight=args.stft_weight,
        si_sdr_weight=args.si_sdr_weight,
        seed=args.seed,
        limit_per_split=args.limit_per_split,
        device=args.device,
        reuse_cache=not args.no_reuse_cache,
        num_workers=args.num_workers,
        embedding_provider=embedding_provider,
    )


if __name__ == "__main__":
    train(parse_args())
