"""Small enrollment-conditioned Personal VAD building blocks for R12."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised only in minimal CPU installs
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


PERSONAL_VAD_FEATURE_SCHEMA = (
    "target_speech_ratio",
    "target_speech_max",
    "target_longest_run_frames",
    "target_longest_run_seconds",
    "target_to_interferer_ratio",
    "overlap_probability",
    "non_target_speech_ratio",
)


@dataclass(frozen=True)
class PersonalVADConfig:
    mel_bins: int = 80
    embedding_dim: int = 192
    hidden_size: int = 128
    num_layers: int = 2
    sample_rate: int = 16_000

    @property
    def input_dim(self) -> int:
        return self.mel_bins + self.embedding_dim + 1

    def __post_init__(self) -> None:
        for name in ("mel_bins", "embedding_dim", "hidden_size", "num_layers", "sample_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def _finite_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite {ndim}D array")
    return array


def build_personal_vad_features(
    log_mel: np.ndarray,
    enrollment_embedding: np.ndarray,
    frame_cosine: np.ndarray,
) -> np.ndarray:
    """Concatenate acoustic, enrollment and frame-level speaker evidence."""
    mel = _finite_array(log_mel, name="log_mel", ndim=2)
    enrollment = _finite_array(enrollment_embedding, name="enrollment_embedding", ndim=1)
    cosine = _finite_array(frame_cosine, name="frame_cosine", ndim=1)
    if mel.shape[0] != cosine.shape[0]:
        raise ValueError("log_mel and frame_cosine must have the same number of frames")
    repeated = np.broadcast_to(enrollment, (mel.shape[0], enrollment.shape[0]))
    return np.ascontiguousarray(np.concatenate((mel, repeated, cosine[:, None]), axis=1))


def _longest_run(active: np.ndarray) -> int:
    longest = current = 0
    for value in active.tolist():
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def aggregate_personal_vad(
    frame_probabilities: np.ndarray, *, frame_seconds: float = 0.01
) -> dict[str, float]:
    """Convert [non-speech, target, non-target] frame probabilities to gate features."""
    probabilities = _finite_array(frame_probabilities, name="frame_probabilities", ndim=2)
    if probabilities.shape[1] != 3:
        raise ValueError("frame_probabilities must have three classes")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("frame_probabilities must be in [0, 1]")
    if not isinstance(frame_seconds, (int, float)) or not math.isfinite(frame_seconds) or frame_seconds <= 0:
        raise ValueError("frame_seconds must be positive and finite")

    target = probabilities[:, 1]
    non_target = probabilities[:, 2]
    active = target >= 0.5
    longest = _longest_run(active)
    target_mean = float(np.mean(target))
    non_target_mean = float(np.mean(non_target))
    return {
        "target_speech_ratio": target_mean,
        "target_speech_max": float(np.max(target)),
        "target_longest_run_frames": float(longest),
        "target_longest_run_seconds": float(longest * float(frame_seconds)),
        "target_to_interferer_ratio": float(target_mean / (non_target_mean + 1e-6)),
        "overlap_probability": float(np.mean(np.minimum(target, non_target))),
        "non_target_speech_ratio": non_target_mean,
    }


if torch is not None:

    class PersonalVADNet(nn.Module):
        """Bidirectional-free GRU model; its input can be built by the helper above."""

        def __init__(self, config: PersonalVADConfig) -> None:
            super().__init__()
            self.config = config
            self.gru = nn.GRU(
                input_size=config.input_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=0.1 if config.num_layers > 1 else 0.0,
            )
            self.classifier = nn.Linear(config.hidden_size, 3)

        def forward(self, features: Tensor) -> Tensor:
            if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
                raise ValueError(
                    f"features must have shape [batch, frames, {self.config.input_dim}]"
                )
            encoded, _ = self.gru(features)
            return self.classifier(encoded)


    def personal_vad_loss(
        logits: Tensor, targets: Tensor, *, class_weights: Tensor | None = None
    ) -> Tensor:
        """Three-class CE plus target-vs-rest and temporal smoothness terms."""
        if logits.ndim != 3 or logits.shape[-1] != 3:
            raise ValueError("logits must have shape [batch, frames, 3]")
        if targets.shape != logits.shape[:2]:
            raise ValueError("targets must align with logits frames")
        weights = class_weights
        if weights is None:
            weights = logits.new_tensor([0.5, 2.0, 1.0])
        ce = F.cross_entropy(logits.transpose(1, 2), targets.long(), weight=weights)
        target_logit = logits[..., 1] - torch.logsumexp(logits[..., (0, 2)], dim=-1)
        target_binary = (targets == 1).to(dtype=logits.dtype)
        target_bce = F.binary_cross_entropy_with_logits(target_logit, target_binary)
        smooth = logits.new_zeros(())
        if logits.shape[1] > 1:
            smooth = torch.mean(torch.abs(logits[:, 1:, 1] - logits[:, :-1, 1]))
        return ce + 0.5 * target_bce + 0.05 * smooth

else:

    class PersonalVADNet:  # type: ignore[no-redef]
        def __init__(self, config: PersonalVADConfig) -> None:
            del config
            raise RuntimeError("PyTorch is required for PersonalVADNet")


    def personal_vad_loss(*args: object, **kwargs: object) -> Any:  # type: ignore[no-redef]
        del args, kwargs
        raise RuntimeError("PyTorch is required for personal_vad_loss")


def _contains_forbidden_path(value: object) -> bool:
    return isinstance(value, str) and any(
        marker in value.lower() for marker in ("internal_test", "held_out")
    )


def write_personal_vad_lineage(source_manifest: Path, output_path: Path, *, seed: int) -> int:
    """Validate a train-only speaker manifest and write deterministic mixture lineage."""
    source_manifest, output_path = Path(source_manifest), Path(output_path)
    if not source_manifest.is_file():
        raise ValueError(f"source manifest does not exist: {source_manifest}")
    if output_path.exists():
        raise FileExistsError(output_path)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source_manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"source manifest row {line_number} must be an object")
        if any(_contains_forbidden_path(value) for value in raw.values()):
            raise ValueError("internal-test paths are forbidden in Personal VAD lineage")
        required = ("id", "parent_id", "enrollment_audio", "target_audio", "target_speaker_id")
        if any(not isinstance(raw.get(key), str) or not str(raw[key]).strip() for key in required):
            raise ValueError(f"source manifest row {line_number} has invalid required fields")
        sample_id = str(raw["id"])
        if sample_id in seen:
            raise ValueError(f"duplicate source manifest id: {sample_id}")
        seen.add(sample_id)
        digest = hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).hexdigest()
        rows.append({
            "id": sample_id,
            "parent_id": str(raw["parent_id"]),
            "enrollment_audio": str(raw["enrollment_audio"]),
            "target_audio": str(raw["target_audio"]),
            "target_speaker_id": str(raw["target_speaker_id"]),
            "mixture_seed": int(digest[:16], 16),
            "seed": seed,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)
