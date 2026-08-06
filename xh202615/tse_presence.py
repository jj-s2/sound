"""Presence-gated ASR composition for the R6 joint TSE/rejection model.

Pure, provenance-neutral helpers that gate a frozen ASR transcript by a
calibrated presence score and score the result with the official
``Overall = ((1-CER)+RR)/2`` metric. The presence score is source-agnostic:
it may come from the TSE presence head (``audio_map.jsonl``) or from the
existing temporal-head path, so the same evaluator composes with either.

Data boundary
-------------
Threshold calibration uses ONLY public samples (the public manifest's val
split) with their public labels. Dataset-A labels are never read here; the
blind ``evaluate`` mode consumes Dataset-A samples (id + label) only to score
a frozen prediction set and never tunes on them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from .data import Sample
from .metrics import cer_stats, is_rejection
from .training_data import read_training_manifest

# Rejection is an empty normalised transcript (see xh202615.metrics.is_rejection).
REJECT_TEXT = ""


def overall_from_metrics(metrics: Mapping[str, float]) -> float:
    """Official competition Overall = ((1-CER) + RR) / 2."""
    return ((1.0 - float(metrics["avg_cer"])) + float(metrics["avg_rr"])) / 2.0


def _finite_score(value: object) -> float:
    if value is None:
        raise ValueError("presence_score is missing for a row")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"presence_score must be a number, got {value!r}") from exc
    if not math.isfinite(score):
        raise ValueError(f"presence_score must be finite, got {score}")
    return score


def load_presence(path: str | Path, *, id_field: str = "id") -> dict[str, float]:
    """Read ``{id: presence_score}`` from an audio map or presence JSONL.

    Fail-closed: every record must carry a finite numeric ``presence_score``.
    Duplicate ids raise. Used for the TSE presence head; the temporal-head path
    can write the same JSONL shape with its own probability field.
    """
    presence: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            sample_id = str(record[id_field])
            if sample_id in presence:
                raise ValueError(f"duplicate presence id {sample_id!r}")
            presence[sample_id] = _finite_score(record.get("presence_score"))
    if not presence:
        raise ValueError(f"no presence records in {path}")
    return presence


def load_asr_text(path: str | Path, *, id_field: str = "id") -> dict[str, str]:
    """Read ``{id: recognition_text}`` from a FunASR predictions JSONL.

    Accepts ``recognition_text`` or ``text`` aliases; missing values become "".
    Duplicate ids raise.
    """
    asr: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            sample_id = str(record[id_field])
            if sample_id in asr:
                raise ValueError(f"duplicate asr id {sample_id!r}")
            text = record.get("recognition_text", record.get("text", ""))
            asr[sample_id] = "" if text is None else str(text)
    return asr


def gate_predictions(
    asr_by_id: Mapping[str, str],
    presence_by_id: Mapping[str, float],
    threshold: float,
) -> list[dict]:
    """Gate ASR text by the presence score: keep text iff score >= threshold.

    A rejected row gets an empty transcript (the official rejection sentinel).
    Every id in ``asr_by_id`` must have a presence score; ids without one raise
    so a silent half-covered presence map can never look like a clean run.
    """
    predictions: list[dict] = []
    for sample_id, asr_text in asr_by_id.items():
        if sample_id not in presence_by_id:
            raise ValueError(f"missing presence score for id {sample_id!r}")
        accepted = presence_by_id[sample_id] >= threshold
        text = asr_text if accepted else REJECT_TEXT
        predictions.append({"id": sample_id, "recognition_text": text})
    return predictions


def samples_from_manifest(manifest_path: str | Path, split: str) -> list[Sample]:
    """Build evaluation :class:`Sample` objects from a public manifest split.

    Positive rows carry their transcript as the label; absent rows carry
    ``None`` (the rejection target). ``id`` is the manifest ``row_id``. Audio
    fields are populated from enrollment/mixture paths for provenance but are
    not used by the scorer.
    """
    rows = read_training_manifest(manifest_path)
    samples: list[Sample] = []
    for row in rows:
        if row.split != split:
            continue
        label = row.text if row.target_present else None
        samples.append(
            Sample(
                id=row.row_id,
                split=row.split,
                wakeup_audio=row.enrollment_audio,
                wakeup_text="",
                command_audio=row.mixture_audio or row.enrollment_audio,
                label=label,
            )
        )
    return samples


def _presence_for(samples: Iterable[Sample], presence_by_id: Mapping[str, float]) -> None:
    """Fail closed if any sample lacks a presence score."""
    missing = [str(s.id) for s in samples if str(s.id) not in presence_by_id]
    if missing:
        raise ValueError(f"missing presence scores for {len(missing)} sample(s): {missing[:5]}")


def overall_at_threshold(
    samples: Iterable[Sample],
    asr_by_id: Mapping[str, str],
    presence_by_id: Mapping[str, float],
    threshold: float,
) -> dict[str, float]:
    """Score (CER, RR, Overall) at a fixed presence threshold.

    Mirrors :func:`xh202615.evaluation._evaluate_subset` exactly: positives
    contribute pooled CER (a rejected positive scores as a full deletion),
    negatives contribute RR (correct-reject rate). Returns the official metric
    keys plus ``overall``.
    """
    pos_count = neg_count = 0
    correct_reject = false_reject = false_accept = 0
    substitutions = insertions = deletions = ref_chars = 0
    for sample in samples:
        sample_id = str(sample.id)
        if sample_id not in presence_by_id:
            raise ValueError(f"missing presence score for id {sample_id!r}")
        accepted = presence_by_id[sample_id] >= threshold
        hyp = asr_by_id.get(sample_id, "") if accepted else REJECT_TEXT
        if sample.label is None:
            neg_count += 1
            if is_rejection(hyp):
                correct_reject += 1
            else:
                false_accept += 1
        else:
            pos_count += 1
            if is_rejection(hyp):
                false_reject += 1
            stats = cer_stats(sample.label, hyp)
            substitutions += stats.substitutions
            insertions += stats.insertions
            deletions += stats.deletions
            ref_chars += stats.ref_chars
    avg_cer = (substitutions + insertions + deletions) / ref_chars if ref_chars else 0.0
    avg_rr = correct_reject / neg_count if neg_count else 0.0
    metrics = {
        "avg_cer": avg_cer,
        "avg_rr": avg_rr,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "missing_predictions": 0,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "ref_chars": ref_chars,
        "false_reject_rate": false_reject / pos_count if pos_count else 0.0,
        "false_accept_rate": false_accept / neg_count if neg_count else 0.0,
    }
    metrics["overall"] = overall_from_metrics(metrics)
    return metrics


def calibrate_threshold_overall(
    samples: Iterable[Sample],
    asr_by_id: Mapping[str, str],
    presence_by_id: Mapping[str, float],
    *,
    max_candidates: int = 2000,
) -> dict:
    """Sweep the presence threshold on public samples to maximise Overall.

    Candidates are the unique presence scores (plus the all-accept / all-reject
    boundaries), capped at ``max_candidates`` by quantile subsampling for very
    large sets. Ties in Overall are broken toward higher RR (protect rejection)
    then lower CER. Returns the chosen threshold, its metrics, and the source
    label. Fails loudly if samples lack both classes or any presence score.
    """
    materialized = list(samples)
    _presence_for(materialized, presence_by_id)
    pos = sum(1 for s in materialized if s.label is not None)
    neg = sum(1 for s in materialized if s.label is None)
    if pos == 0 or neg == 0:
        raise ValueError(
            f"calibration needs both classes; got {pos} pos / {neg} neg"
        )
    scores = sorted({presence_by_id[str(s.id)] for s in materialized})
    if len(scores) > max_candidates:
        # Deterministic quantile subsampling; always keep min and max.
        import numpy as np

        idx = np.linspace(0, len(scores) - 1, max_candidates).astype(int)
        scores = sorted({scores[i] for i in idx})
    candidates = [min(scores) - 1.0] + scores + [max(scores) + 1.0]
    best: dict | None = None
    for thr in candidates:
        metrics = overall_at_threshold(materialized, asr_by_id, presence_by_id, thr)
        key = (metrics["overall"], metrics["avg_rr"], -metrics["avg_cer"])
        if best is None or key > best[0]:
            best = (key, thr, metrics)
    assert best is not None
    _, threshold, metrics = best
    return {
        "threshold": float(threshold),
        "threshold_source": "public_val_max_overall",
        "metrics": metrics,
        "n_pos": pos,
        "n_neg": neg,
        "n_candidates": len(candidates),
    }
