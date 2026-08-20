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
    enhance_waveform_with_presence,
    multi_resolution_stft_loss,
    negative_si_sdr_loss,
)
from xh202615.speaker_score import (
    SCORE_VARIANTS,
    SpeakerEncoder,
    build_wespeaker_encoder,
    cosine_tensor,
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


def load_training_rows(
    manifest_path: str | Path, dataset_a_root: str | Path
) -> tuple[TrainingManifestRow, ...]:
    """Read and validate the manifest, returning ALL rows (present and absent).

    R6 joint training needs target-absent rows for the presence objective. The
    same fail-closed guards apply (Dataset-A containment, duplicate IDs, speaker
    split leakage, field ranges). Class balance (>=1 present and >=1 absent per
    split) is enforced separately by :func:`assert_class_balance`.
    """
    rows = read_training_manifest(manifest_path)
    rows = assert_valid_training_manifest(
        rows,
        manifest_path=manifest_path,
        forbidden_roots=(Path(dataset_a_root),),
    )
    return tuple(rows)


def assert_class_balance(
    rows: Iterable[TrainingManifestRow], *, splits: tuple[str, ...] = ("train", "val", "test")
) -> dict[str, dict[str, int]]:
    """Fail loudly if any split lacks present or absent rows.

    The presence objective is a binary classification target; a split with only
    one class cannot train or validate it. Returns per-split counts for audit.
    """
    counts: dict[str, dict[str, int]] = {}
    for split in splits:
        split_rows = [row for row in rows if row.split == split]
        pos = sum(1 for row in split_rows if row.target_present)
        neg = sum(1 for row in split_rows if not row.target_present)
        counts[split] = {"present": pos, "absent": neg, "total": len(split_rows)}
        if pos == 0 or neg == 0:
            raise ValueError(
                f"split {split!r} is class-imbalanced for presence: "
                f"{pos} present / {neg} absent (need >=1 of each)"
            )
    return counts


def _wav_sample_rate(path: Path) -> int:
    """Fast sample-rate read via the stdlib ``wave`` header (PCM fallback)."""
    try:
        with wave.open(str(path), "rb") as handle:
            return int(handle.getframerate())
    except (wave.Error, EOFError):
        return int(sf.info(str(path)).samplerate)


def check_audio(
    rows: Iterable[TrainingManifestRow], *, require_target_for_absent: bool = False
) -> None:
    """Fail closed on missing or non-16 kHz audio for the rows used.

    The enrollment and mixture fields are always checked (existence + 16 kHz
    sample rate). ``target_audio`` is checked for target-present rows; for
    target-absent rows it is only checked when ``require_target_for_absent`` is
    True, because the joint trainer synthesises a silence target for absent
    rows and must not depend on a (possibly empty) target file. The stdlib
    ``wave`` header read keeps this fast even for thousands of files;
    ``soundfile.info`` is the fallback for non-PCM WAVs.
    """
    missing: list[tuple[str, str, str]] = []
    bad_rate: list[tuple[str, str, int]] = []
    for row in rows:
        field_names = ["enrollment_audio", "mixture_audio"]
        if row.target_present or require_target_for_absent:
            field_names.append("target_audio")
        for field_name in field_names:
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
# Joint presence objective (R6)
# ---------------------------------------------------------------------------

_PRESENCE_BCE = torch.nn.BCEWithLogitsLoss()


def presence_bce_loss(presence_logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy on the presence logit (present=1, absent=0).

    ``presence_logit`` is ``[batch, 1]`` (or ``[batch]``); ``labels`` is
    ``[batch]`` float in {0, 1}. Positives and negatives are balanced in the
    public manifest so no ``pos_weight`` is applied.
    """
    logit = presence_logit.reshape(-1)
    label = labels.to(logit.dtype).reshape(-1)
    if logit.shape != label.shape:
        raise ValueError(
            f"presence logit/label shape mismatch: {tuple(logit.shape)} vs {tuple(label.shape)}"
        )
    return _PRESENCE_BCE(logit, label)


def joint_loss(
    enhanced: torch.Tensor,
    target: torch.Tensor,
    presence_logit: torch.Tensor,
    presence_label: torch.Tensor,
    *,
    present_mask: torch.Tensor,
    stft_weight: float = 1.0,
    si_sdr_weight: float = 1.0,
    presence_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reconstruction (present rows only) + presence BCE (all rows).

    Reconstruction never touches absent rows: their target is silence and the
    mask is free to do anything (the enhanced audio for absent rows is never
    transcribed because the gate rejects them). The presence head carries the
    reject signal. Returns the scalar total and a dict of float components.
    """
    if present_mask.any():
        recon = composite_loss(
            enhanced[present_mask],
            target[present_mask],
            stft_weight=stft_weight,
            si_sdr_weight=si_sdr_weight,
        )
    else:
        recon = enhanced.new_zeros(())
    bce = presence_bce_loss(presence_logit, presence_label)
    total = recon + presence_weight * bce
    return total, {
        "recon": float(recon.detach().item()),
        "presence": float(bce.detach().item()),
        "total": float(total.detach().item()),
    }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the rank-based Mann-Whitney U statistic (no sklearn).

    Returns 0.5 when one class is absent (degenerate); the caller fails loudly
    on class-imbalanced splits before this is used for decisions.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average ranks handle ties.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average rank
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_pos_ranks = float(ranks[labels == 1].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def calibrate_presence_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Pick a presence threshold on validation via Youden's J (TPR - FPR).

    A sample is accepted (present) when ``score >= threshold``. Candidates are
    the unique scores plus the two boundaries (all-accept, all-reject). Ties in
    J are broken toward the lower FPR (more conservative rejection) to protect
    RR. Fails loudly if only one class is present. Returns the threshold, AUC,
    and the confusion-derived rates at the chosen threshold.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"cannot calibrate presence threshold with {n_pos} pos / {n_neg} neg"
        )
    auc = _auc(scores, labels)
    candidates = np.unique(scores)
    # Boundaries: just above max (all reject) and at -inf (all accept).
    edges = np.concatenate(([-np.inf], candidates, [np.max(scores) + 1.0]))
    best_j = -2.0
    best_fpr = 2.0
    best: dict[str, float] = {}
    for thr in edges:
        predicted_present = scores >= thr
        tp = int((predicted_present & (labels == 1)).sum())
        fp = int((predicted_present & (labels == 0)).sum())
        fn = n_pos - tp
        tn = n_neg - fp
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        j = tpr - fpr
        # Tie-break: prefer lower FPR (protect RR / correct rejection).
        if (j > best_j) or (j == best_j and fpr < best_fpr):
            best_j = j
            best_fpr = fpr
            best = {
                "threshold": float(thr),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "rr_at_threshold": float(tn / n_neg if n_neg else 0.0),  # correct reject rate
                "false_reject_rate": float(fn / n_pos if n_pos else 0.0),
                "false_accept_rate": float(fpr),
                "accuracy": float((tp + tn) / (n_pos + n_neg)),
                "youden_j": float(j),
            }
    best["auc"] = float(auc)
    best["n_pos"] = float(n_pos)
    best["n_neg"] = float(n_neg)
    return best


def calibrate_speaker_youden(
    enhanced_scores: np.ndarray,
    mixture_scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Per-variant Youden-J calibration + AUC-based variant selection (surrogate).

    The trainer has no ASR, so it cannot optimise the official Overall. It
    therefore picks the speaker-score variant with the best public-val AUC and a
    Youden-J threshold as a justified surrogate - mirroring the R6 presence
    threshold, which was also a Youden-J surrogate stored in the checkpoint. The
    authoritative Overall-optimal variant+threshold is produced later by the
    ``--calibrate`` step (which has public-val ASR).

    ``enhanced_scores`` / ``mixture_scores`` are the cosines between the
    enrollment embedding and the enhanced / mixture embeddings; ``labels`` is
    {0,1} (absent/present). Returns the selected variant, its Youden threshold,
    AUC, and per-variant diagnostics. Fails loudly if a class is missing.
    """
    enhanced_scores = np.asarray(enhanced_scores, dtype=np.float64).reshape(-1)
    mixture_scores = np.asarray(mixture_scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if enhanced_scores.shape != labels.shape or mixture_scores.shape != labels.shape:
        raise ValueError("speaker scores and labels must share a shape")
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"cannot calibrate speaker score with {n_pos} pos / {n_neg} neg"
        )
    raw = {
        "enhanced_cosine": enhanced_scores,
        "mixture_cosine": mixture_scores,
        "max_cosine": np.maximum(enhanced_scores, mixture_scores),
    }
    per_variant: dict[str, dict] = {}
    for variant in SCORE_VARIANTS:
        per_variant[variant] = calibrate_presence_threshold(raw[variant], labels)
    # Select by AUC (higher is better); tie-break toward the enhanced variant
    # (it is the TSE-gating-aligned signal), then by Youden J.
    order = ["enhanced_cosine", "mixture_cosine", "max_cosine"]
    best_variant = max(
        order,
        key=lambda v: (per_variant[v]["auc"], per_variant[v]["youden_j"]),
    )
    chosen = per_variant[best_variant]
    return {
        "score_type": best_variant,
        "threshold": float(chosen["threshold"]),
        "threshold_source": "public_val_youden_j",
        "auc": float(chosen["auc"]),
        "youden_j": float(chosen["youden_j"]),
        "tpr": float(chosen["tpr"]),
        "fpr": float(chosen["fpr"]),
        "false_reject_rate": float(chosen["false_reject_rate"]),
        "false_accept_rate": float(chosen["false_accept_rate"]),
        "n_pos": float(n_pos),
        "n_neg": float(n_neg),
        "per_variant": {
            v: {
                "auc": per_variant[v]["auc"],
                "threshold": per_variant[v]["threshold"],
                "youden_j": per_variant[v]["youden_j"],
            }
            for v in SCORE_VARIANTS
        },
    }


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


class TSEJointDataset(Dataset):
    """Stream crops for joint reconstruction + presence training (R6).

    Like :class:`TSEDataset` but also yields a presence label (1.0 for
    target-present, 0.0 for absent). For absent rows the target is synthesised
    as silence (zeros) of the cropped mixture length, so the dataset does not
    depend on a target-audio file for absent rows and the reconstruction loss
    (applied to present rows only) is well-defined.
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
        if row.target_present:
            target = read_audio_mono(row.target_audio)
            mixture_crop, target_crop = crop_pair(
                mixture,
                target,
                self.segment_samples,
                seed=self.seed,
                epoch=self.epoch,
                index=index,
            )
            label = 1.0
        else:
            # Absent rows: crop the mixture only, synthesise a silence target.
            mixture_crop, _ = crop_pair(
                mixture,
                np.zeros_like(mixture),
                self.segment_samples,
                seed=self.seed,
                epoch=self.epoch,
                index=index,
            )
            target_crop = np.zeros_like(mixture_crop)
            label = 0.0
        embedding = self.embeddings[str(row.enrollment_audio)]
        return (
            torch.from_numpy(mixture_crop),
            torch.from_numpy(target_crop),
            torch.from_numpy(np.asarray(embedding, dtype=np.float32)),
            torch.tensor(label, dtype=torch.float32),
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


def train_step_joint(
    model: FiLMCRNExtractor,
    optimizer: torch.optim.Optimizer,
    mixture: torch.Tensor,
    target: torch.Tensor,
    embedding: torch.Tensor,
    presence_label: torch.Tensor,
    *,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
    stft_weight: float,
    si_sdr_weight: float,
    presence_weight: float,
) -> tuple[float | None, dict[str, float]]:
    """One joint reconstruction + presence training step.

    Returns ``(total_loss, components)`` where ``total_loss`` is ``None`` when a
    non-finite loss is skipped. AMP is CUDA-only; GradScaler skips overflow.
    """
    present_mask = presence_label.bool()
    optimizer.zero_grad(set_to_none=True)
    if use_amp:
        with torch.amp.autocast(device_type="cuda", enabled=True):
            enhanced, presence_logit = enhance_waveform_with_presence(
                model, mixture, embedding
            )
            loss, components = joint_loss(
                enhanced, target, presence_logit, presence_label,
                present_mask=present_mask, stft_weight=stft_weight,
                si_sdr_weight=si_sdr_weight, presence_weight=presence_weight,
            )
        if not torch.isfinite(loss):
            return None, components
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        enhanced, presence_logit = enhance_waveform_with_presence(
            model, mixture, embedding
        )
        loss, components = joint_loss(
            enhanced, target, presence_logit, presence_label,
            present_mask=present_mask, stft_weight=stft_weight,
            si_sdr_weight=si_sdr_weight, presence_weight=presence_weight,
        )
        if not torch.isfinite(loss):
            return None, components
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    return float(loss.item()), components


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


@torch.no_grad()
def evaluate_joint(
    model: FiLMCRNExtractor,
    dataset: TSEJointDataset,
    *,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
    stft_weight: float,
    si_sdr_weight: float,
    speaker_encoder: SpeakerEncoder | None = None,
) -> dict:
    """Evaluate reconstruction (present rows) + presence discrimination (all).

    Presence metrics: ROC-AUC, Youden-J threshold, RR (correct-reject rate) and
    false-accept/false-reject rates at that threshold. The threshold is a
    justified surrogate (maximises TPR-FPR on the public validation split) and
    is stored in the checkpoint so inference can gate without re-tuning.

    When ``speaker_encoder`` is given (R7), additionally compute the
    enrollment-conditioned speaker cosines (enhanced + mixture) for every row
    and calibrate a per-variant Youden-J threshold + AUC-selected variant as a
    surrogate reject score. This is the domain-invariant gating signal; the
    authoritative Overall-optimal threshold is produced later by ``--calibrate``.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_si_sdr = 0.0
    total_stft = 0.0
    count = 0
    scores: list[float] = []
    labels: list[float] = []
    enh_speaker: list[float] = []
    mix_speaker: list[float] = []
    speaker_labels: list[float] = []
    for mixture, target, embedding, label in loader:
        mixture = mixture.to(device)
        target = target.to(device)
        embedding = embedding.to(device)
        label = label.to(device)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                enhanced, presence_logit = enhance_waveform_with_presence(
                    model, mixture, embedding
                )
        else:
            enhanced, presence_logit = enhance_waveform_with_presence(
                model, mixture, embedding
            )
        present = label.bool()
        if present.any():
            stft = multi_resolution_stft_loss(enhanced[present], target[present])
            si_sdr_loss = negative_si_sdr_loss(enhanced[present], target[present])
            loss = stft_weight * stft + si_sdr_weight * si_sdr_loss
            batch = int(present.sum().item())
            total_loss += float(loss.item()) * batch
            total_si_sdr += float((-si_sdr_loss).item()) * batch
            total_stft += float(stft.item()) * batch
            count += batch
        scores.extend(torch.sigmoid(presence_logit.reshape(-1)).float().cpu().tolist())
        labels.extend(label.float().cpu().tolist())
        if speaker_encoder is not None:
            enh_emb = speaker_encoder.embed_tensor(enhanced)
            mix_emb = speaker_encoder.embed_tensor(mixture)
            enh_cos = cosine_tensor(enh_emb, embedding).float().cpu().tolist()
            mix_cos = cosine_tensor(mix_emb, embedding).float().cpu().tolist()
            enh_speaker.extend(enh_cos)
            mix_speaker.extend(mix_cos)
            speaker_labels.extend(label.float().cpu().tolist())
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    presence = calibrate_presence_threshold(scores_arr, labels_arr)
    recon = (
        {
            "loss": total_loss / count,
            "si_sdr": total_si_sdr / count,
            "stft": total_stft / count,
            "n": count,
        }
        if count
        else {"loss": float("nan"), "si_sdr": float("nan"), "stft": float("nan"), "n": 0}
    )
    result = {
        "recon": recon,
        "presence": presence,
        "n_rows": int(len(labels_arr)),
        "n_present": int(labels_arr.sum()),
        "n_absent": int(len(labels_arr) - labels_arr.sum()),
    }
    if speaker_encoder is not None:
        result["speaker"] = calibrate_speaker_youden(
            np.asarray(enh_speaker, dtype=np.float64),
            np.asarray(mix_speaker, dtype=np.float64),
            np.asarray(speaker_labels, dtype=np.float64),
        )
    return result


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
    with_presence: bool = False,
    presence_weight: float = 1.0,
    with_speaker_score: bool = False,
    speaker_encoder: SpeakerEncoder | None = None,
) -> dict:
    """Run a deterministic TSE training pilot and write checkpoint + summary.

    When ``embedding_provider`` is ``None`` a frozen WeSpeaker provider is built.
    Tests inject a mock provider so no WeSpeaker or network is required.

    When ``with_presence`` is ``True`` (R6) the model is built with a presence
    head, target-absent rows are loaded for a joint BCE presence objective, and
    the checkpoint stores a calibrated presence threshold (Youden J on the
    public val split) for inference-time rejection. The default ``False``
    reproduces the original pilot exactly (backward compatible).

    When ``with_speaker_score`` is ``True`` (R7, requires ``with_presence``) the
    val evaluation additionally computes the enrollment-conditioned speaker
    cosines (enhanced + mixture) and stores a Youden-J speaker threshold +
    AUC-selected variant in the checkpoint as the domain-invariant reject score.
    The authoritative Overall-optimal threshold is produced later by
    ``--calibrate``. A single frozen WeSpeaker encoder backs both the
    enrollment cache and the speaker score.
    """
    if with_speaker_score and not with_presence:
        raise ValueError("with_speaker_score=True requires with_presence=True")
    _seed_everything(seed)
    manifest_path = Path(manifest).expanduser().resolve(strict=False)
    output_path = Path(output_dir).expanduser().resolve(strict=False)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_a_resolved = Path(dataset_a_root).resolve(strict=False)

    digest = manifest_digest(manifest_path)
    if with_presence:
        rows = load_training_rows(manifest_path, dataset_a_resolved)
        class_balance = assert_class_balance(rows)
    else:
        rows = load_positive_rows(manifest_path, dataset_a_resolved)
        class_balance = None
    rows = tuple(sorted(rows, key=lambda row: row.row_id))
    rows = _apply_limit_per_split(rows, limit_per_split)
    if with_presence and limit_per_split is not None:
        # Re-check balance after a per-split cap (smoke runs may unbalance).
        class_balance = assert_class_balance(rows)
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

    if with_speaker_score:
        if speaker_encoder is None:
            speaker_encoder = build_wespeaker_encoder(model_name, torch_device)
        # One frozen WeSpeaker encoder backs both the enrollment cache and the
        # speaker score, so the two embeddings live in the same space.
        if embedding_provider is None:
            embedding_provider = speaker_encoder.embed_path
    elif embedding_provider is None:
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
        with_presence=with_presence,
    ).to(torch_device)
    model_config = {
        "embedding_dim": embedding_dim,
        "channels": list(channels_tuple),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": model.win_length,
        "gru_hidden": gru_hidden,
        "gru_layers": gru_layers,
        "with_presence": with_presence,
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    segment_samples = int(round(segment_seconds * TARGET_SAMPLE_RATE))

    dataset_cls = TSEJointDataset if with_presence else TSEDataset
    train_dataset = dataset_cls(split_rows["train"], embeddings, segment_samples, seed)
    val_dataset = dataset_cls(
        split_rows["val"], embeddings, segment_samples, seed, epoch=0
    )
    test_dataset = dataset_cls(
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
        if with_presence:
            for mixture, target, embedding, presence_label in train_loader:
                mixture = mixture.to(torch_device)
                target = target.to(torch_device)
                embedding = embedding.to(torch_device)
                presence_label = presence_label.to(torch_device)
                loss_value, _ = train_step_joint(
                    model,
                    optimizer,
                    mixture,
                    target,
                    embedding,
                    presence_label,
                    use_amp=use_amp,
                    scaler=scaler,
                    grad_clip=grad_clip,
                    stft_weight=stft_weight,
                    si_sdr_weight=si_sdr_weight,
                    presence_weight=presence_weight,
                )
                if loss_value is not None:
                    step_losses.append(loss_value)
            val_metrics = evaluate_joint(
                model,
                val_dataset,
                device=torch_device,
                use_amp=use_amp,
                batch_size=batch_size,
                stft_weight=stft_weight,
                si_sdr_weight=si_sdr_weight,
                speaker_encoder=speaker_encoder if with_speaker_score else None,
            )
            # Test metrics are diagnostic only and never drive checkpoints.
            test_metrics = evaluate_joint(
                model,
                test_dataset,
                device=torch_device,
                use_amp=use_amp,
                batch_size=batch_size,
                stft_weight=stft_weight,
                si_sdr_weight=si_sdr_weight,
                speaker_encoder=speaker_encoder if with_speaker_score else None,
            )
            val_selection_loss = val_metrics["recon"]["loss"]
        else:
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
            val_selection_loss = val_metrics["loss"]

        saved = False
        if val_selection_loss < best_val_loss and np.isfinite(val_selection_loss):
            best_val_loss = val_selection_loss
            best_epoch = epoch
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "manifest": str(manifest_path),
                "manifest_digest": digest,
                "dataset_a_root": str(dataset_a_resolved),
                "dataset_a_used_for_training": False,
                "data_boundary": (
                    "public-only; Dataset-A forbidden as a training source; "
                    "speaker-disjoint splits validated; "
                    + (
                        "target-present + target-absent rows (joint presence objective)"
                        if with_presence
                        else "target-present rows only"
                    )
                ),
                "seed": seed,
                "best_epoch": best_epoch,
                "val_metrics": val_metrics,
                "test_metrics_diagnostic": test_metrics,
            }
            if with_presence:
                checkpoint_payload["with_presence"] = True
                checkpoint_payload["presence_weight"] = presence_weight
                checkpoint_payload["presence_threshold"] = val_metrics["presence"]["threshold"]
                checkpoint_payload["presence_threshold_source"] = (
                    "public_val_youden_j"
                )
                checkpoint_payload["presence_auc"] = val_metrics["presence"]["auc"]
                checkpoint_payload["class_balance"] = class_balance
            if with_speaker_score and "speaker" in val_metrics:
                speaker = val_metrics["speaker"]
                checkpoint_payload["with_speaker_score"] = True
                checkpoint_payload["speaker_score_type"] = speaker["score_type"]
                checkpoint_payload["speaker_threshold"] = speaker["threshold"]
                checkpoint_payload["speaker_threshold_source"] = (
                    speaker["threshold_source"]
                )
                checkpoint_payload["speaker_auc"] = speaker["auc"]
                checkpoint_payload["speaker_score_variants"] = list(SCORE_VARIANTS)
            torch.save(checkpoint_payload, best_path)
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
            "speaker-disjoint splits validated; "
            + (
                "target-present + target-absent rows (joint presence objective)"
                if with_presence
                else "target-present rows only"
            )
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
            split: sum(1 for row in split_rows[split] if row.target_present)
            for split in ("train", "val", "test")
        },
        "limit_per_split": limit_per_split,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "embedding_cache": str(cache_path),
    }
    if with_presence:
        summary["with_presence"] = True
        summary["presence_weight"] = presence_weight
        summary["class_balance"] = class_balance
        if best_epoch is not None and history:
            best_record = next(
                (h for h in history if h["epoch"] == best_epoch), history[-1]
            )
            presence = best_record["val"]["presence"]
            summary["presence_threshold"] = presence["threshold"]
            summary["presence_threshold_source"] = "public_val_youden_j"
            summary["presence_auc"] = presence["auc"]
    if with_speaker_score:
        summary["with_speaker_score"] = True
        summary["speaker_score_variants"] = list(SCORE_VARIANTS)
        if best_epoch is not None and history:
            best_record = next(
                (h for h in history if h["epoch"] == best_epoch), history[-1]
            )
            if "speaker" in best_record["val"]:
                speaker = best_record["val"]["speaker"]
                summary["speaker_score_type"] = speaker["score_type"]
                summary["speaker_threshold"] = speaker["threshold"]
                summary["speaker_threshold_source"] = speaker["threshold_source"]
                summary["speaker_auc"] = speaker["auc"]
                summary["speaker_val_per_variant"] = speaker["per_variant"]
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
    parser.add_argument(
        "--with-presence",
        action="store_true",
        help="R6: train a joint presence/rejection head on present+absent rows "
        "and store a calibrated presence threshold in the checkpoint",
    )
    parser.add_argument(
        "--presence-weight",
        type=float,
        default=1.0,
        help="weight of the presence BCE objective (R6, used with --with-presence)",
    )
    parser.add_argument(
        "--with-speaker-score",
        action="store_true",
        help="R7: compute an enrollment-conditioned speaker-cosine reject score on "
        "val (requires --with-presence) and store a calibrated speaker threshold + "
        "variant in the checkpoint. The cosine is the domain-invariant gating signal; "
        "presence BCE remains as an auxiliary objective.",
    )
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
        with_presence=args.with_presence,
        presence_weight=args.presence_weight,
        with_speaker_score=args.with_speaker_score,
    )


if __name__ == "__main__":
    train(parse_args())
