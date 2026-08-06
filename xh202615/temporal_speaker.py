"""Deterministic aggregation of temporal target-speaker evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .contracts import EvidenceWindow


@dataclass(frozen=True)
class TemporalAggregate:
    """Summary statistics computed from speaker-scored temporal windows."""

    topk_similarity: float | None
    temporal_coverage: float | None
    consistency: float | None
    quality: float | None


def aggregate_windows(
    windows: Iterable[EvidenceWindow],
    *,
    top_k: int,
    target_threshold: float,
) -> TemporalAggregate:
    """Aggregate valid temporal speaker windows without using labels.

    Windows are ordered by their start and end positions for the duration and
    adjacency calculations.  Quality is weighted only over windows that have
    a quality value; missing quality does not contribute to its denominator.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(target_threshold, bool) or not isinstance(
        target_threshold, (int, float)
    ) or not math.isfinite(float(target_threshold)):
        raise ValueError("target_threshold must be a finite number")

    ordered = sorted(windows, key=lambda item: (item.start_sec, item.end_sec))
    top = sorted((item.similarity for item in ordered), reverse=True)[: min(top_k, len(ordered))]
    topk_similarity = sum(top) / len(top) if top else None

    total_duration = sum(item.end_sec - item.start_sec for item in ordered)
    covered_duration = sum(
        item.end_sec - item.start_sec
        for item in ordered
        if item.similarity >= target_threshold
    )
    temporal_coverage = covered_duration / total_duration if total_duration else None

    deltas = [
        abs(right.similarity - left.similarity)
        for left, right in zip(ordered, ordered[1:])
    ]
    if deltas:
        consistency = max(0.0, min(1.0, 1.0 - sum(deltas) / len(deltas)))
    else:
        consistency = 1.0 if ordered else None

    quality_windows = [
        (item.end_sec - item.start_sec, item.quality)
        for item in ordered
        if item.quality is not None
    ]
    quality_duration = sum(duration for duration, _ in quality_windows)
    quality = (
        sum(duration * quality_value for duration, quality_value in quality_windows)
        / quality_duration
        if quality_duration
        else None
    )

    return TemporalAggregate(
        topk_similarity=topk_similarity,
        temporal_coverage=temporal_coverage,
        consistency=consistency,
        quality=quality,
    )
