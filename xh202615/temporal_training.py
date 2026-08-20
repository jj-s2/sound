"""Feature and metric helpers for the frozen-encoder temporal experiment."""

from __future__ import annotations

import math

import numpy as np


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def build_pair_features(
    enrollment: np.ndarray,
    windows: np.ndarray,
    energy: np.ndarray,
) -> np.ndarray:
    """Build per-window features from a frozen enrollment/mixture embedding pair."""

    enrollment = np.asarray(enrollment, dtype=np.float32).reshape(-1)
    windows = np.asarray(windows, dtype=np.float32)
    energy = np.asarray(energy, dtype=np.float32).reshape(-1)
    if windows.ndim != 2 or windows.shape[1] != enrollment.shape[0]:
        raise ValueError("windows must have shape (time, embedding_dim)")
    if windows.shape[0] != energy.shape[0] or windows.shape[0] == 0:
        raise ValueError("energy must contain one finite value per window")
    if not np.isfinite(enrollment).all() or not np.isfinite(windows).all():
        raise ValueError("embeddings must be finite")
    if not np.isfinite(energy).all():
        raise ValueError("energy must be finite")

    enrollment_norm = enrollment / max(float(np.linalg.norm(enrollment)), 1e-8)
    windows_norm = _normalize_rows(windows)
    enrollment_repeated = np.repeat(enrollment_norm[None, :], windows.shape[0], axis=0)
    delta = windows_norm - enrollment_repeated
    similarity = np.sum(windows_norm * enrollment_repeated, axis=1, keepdims=True)
    return np.concatenate(
        (enrollment_repeated, windows_norm, delta, similarity, energy[:, None]), axis=1
    ).astype(np.float32)


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return deterministic binary metrics for a held-out split."""

    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape or labels.size == 0:
        raise ValueError("labels and probabilities must be non-empty and have the same shape")
    if not np.isfinite(probabilities).all() or not math.isfinite(float(threshold)):
        raise ValueError("probabilities and threshold must be finite")
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / labels.size
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def select_presence_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_recall: float = 0.95,
) -> float:
    """Select one public-validation threshold without consulting an audit set.

    Eligible thresholds must preserve at least ``min_recall`` of target-present
    examples. Among them, prefer higher negative specificity, then the higher
    threshold to minimize unnecessary raw-ASR rescues.
    """

    if not math.isfinite(float(min_recall)) or not 0.0 <= float(min_recall) <= 1.0:
        raise ValueError("min_recall must be finite and in [0, 1]")
    labels_array = np.asarray(labels).reshape(-1)
    probabilities_array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels_array.shape != probabilities_array.shape or labels_array.size == 0:
        raise ValueError("labels and probabilities must be non-empty and have the same shape")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("labels must contain only zero and one")
    if not np.any(labels_array == 1) or not np.any(labels_array == 0):
        raise ValueError("labels must contain both target-present and target-absent rows")
    if not np.isfinite(probabilities_array).all():
        raise ValueError("probabilities must be finite")

    candidates = sorted({0.0, 1.0, *(float(value) for value in probabilities_array)})
    eligible: list[tuple[float, dict[str, float]]] = []
    for threshold in candidates:
        metrics = binary_metrics(labels_array, probabilities_array, threshold=threshold)
        if metrics["recall"] >= float(min_recall):
            eligible.append((threshold, metrics))
    if not eligible:
        raise ValueError("no threshold met the requested minimum recall")
    return float(
        max(
            eligible,
            key=lambda item: (
                item[1]["tn"] / (item[1]["tn"] + item[1]["fp"]),
                item[0],
            ),
        )[0]
    )
