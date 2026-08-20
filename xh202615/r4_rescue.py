"""R4 rescue router: public-only routing and threshold selection.

The rescue router uses a frozen presence head's probability to choose
between a raw-ASR transcript (clear-present) and a safe fallback (absent /
uncertain).  This is the routing-semantic fix for R3: the reject-mode gate
could only suppress existing fusion transcripts and therefore could not
improve CER.  Rescue routing sends clear-present samples to raw ASR, which
has materially lower CER on positives, while the fallback preserves the
baseline rejection behaviour.

Thresholds are selected on **public validation** by maximizing

    S_public = ((1 - CER) + RR) / 2

subject to the preserved rejection policy
(``false_reject_rate <= max_frr`` and ``reject_accuracy >= min_rr``).
Dataset-A is never consulted during selection.

CER/RR are computed with the project's own ``cer_stats`` / ``is_rejection``
helpers so that public numbers are directly comparable to the Dataset-A
audit.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .metrics import cer_stats, is_rejection

POLICY_MAX_FRR = 0.10
POLICY_MIN_RR = 0.85


def rescue_text(
    probability: float,
    raw_text: str,
    fallback_text: str,
    *,
    threshold: float,
) -> tuple[str, str]:
    """Route one sample with the rescue policy.

    A present decision (``probability >= threshold``) with a non-empty raw
    transcript uses the raw transcript; otherwise the safe fallback is kept.
    Returns ``(text, route)`` where ``route`` is ``"raw"`` or a fallback code.
    """

    if not math.isfinite(float(probability)):
        raise ValueError("probability must be finite")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if probability >= threshold and raw_text:
        return raw_text, "raw"
    if raw_text:
        return fallback_text, "fallback"
    return fallback_text, "fallback_empty_raw"


def safe_rescue_text(
    probability: float,
    raw_text: str,
    fallback_text: str,
    *,
    threshold: float,
) -> tuple[str, str]:
    """R3-P1 safe accepted-positive rescue (RR-preserving by construction).

    Use the raw transcript only when the head is confident (``probability >=
    threshold``) **and** the fallback (fusion) already accepted (non-empty).
    Samples the fallback rejected stay rejected, so this routing cannot create
    new false accepts and therefore cannot lower RR relative to the fallback.
    CER changes only on fallback-accepted present rows where the head agrees.
    """

    if not math.isfinite(float(probability)):
        raise ValueError("probability must be finite")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if probability >= threshold and fallback_text and raw_text:
        return raw_text, "safe_raw"
    return fallback_text, "fusion" if fallback_text else "fusion_reject"


def rescue_metrics(
    present_flags: Sequence[int],
    references: Sequence[str],
    outputs: Sequence[str],
) -> dict[str, float]:
    """Compute CER / RR / Overall / policy metrics for routed outputs.

    ``present_flags`` is 1 for target-present rows (CER contributors, using
    ``references``) and 0 for target-absent rows (RR contributors, where the
    reference is unused and a correct rejection is an empty output).  Rows are
    aggregated exactly like ``xh202615.evaluation._evaluate_subset``: CER is
    pooled over present rows, RR over absent rows.
    """

    flags = np.asarray(present_flags, dtype=np.int64).reshape(-1)
    refs = ["" if r is None else str(r) for r in references]
    outs = ["" if o is None else str(o) for o in outputs]
    if flags.shape[0] != len(refs) or flags.shape[0] != len(outs) or flags.size == 0:
        raise ValueError("present_flags, references, outputs must be non-empty and equal length")
    if not np.isin(flags, (0, 1)).all():
        raise ValueError("present_flags must contain only 0 and 1")

    substitutions = insertions = deletions = ref_chars = 0
    correct_reject = false_reject = false_accept = 0
    pos_count = neg_count = 0
    for flag, ref, out in zip(flags.tolist(), refs, outs):
        if flag == 1:
            pos_count += 1
            rejected = is_rejection(out)
            false_reject += int(rejected)
            stats = cer_stats(ref, out)
            substitutions += stats.substitutions
            insertions += stats.insertions
            deletions += stats.deletions
            ref_chars += stats.ref_chars
        else:
            neg_count += 1
            rejected = is_rejection(out)
            correct_reject += int(rejected)
            false_accept += int(not rejected)

    avg_cer = (substitutions + insertions + deletions) / ref_chars if ref_chars else 0.0
    avg_rr = correct_reject / neg_count if neg_count else 0.0
    false_reject_rate = false_reject / pos_count if pos_count else 0.0
    reject_accuracy = correct_reject / neg_count if neg_count else 0.0
    overall = ((1.0 - avg_cer) + avg_rr) / 2.0
    return {
        "avg_cer": float(avg_cer),
        "avg_rr": float(avg_rr),
        "overall": float(overall),
        "pos_count": int(pos_count),
        "neg_count": int(neg_count),
        "false_reject_rate": float(false_reject_rate),
        "reject_accuracy": float(reject_accuracy),
        "substitutions": int(substitutions),
        "insertions": int(insertions),
        "deletions": int(deletions),
        "ref_chars": int(ref_chars),
    }


def _routed_outputs(
    probabilities: np.ndarray,
    raw_texts: Sequence[str],
    fallback_texts: Sequence[str],
    threshold: float,
) -> list[str]:
    return [
        rescue_text(float(p), r, f, threshold=threshold)[0]
        for p, r, f in zip(probabilities.tolist(), raw_texts, fallback_texts)
    ]


def _precompute_contributions(
    present_flags: Sequence[int],
    references: Sequence[str],
    raw_texts: Sequence[str],
    fallback_texts: Sequence[str],
) -> dict:
    """Precompute per-row CER/rejection contributions once (cer_stats is the cost).

    For each row we need, independent of threshold:
      - err_raw / err_fall / ref_chars for present rows (CER contributors),
      - rej_raw / rej_fall for every row (a routed output is rejected iff the
        chosen transcript normalizes to empty).
    Threshold scans then become pure numpy boolean reductions.
    """

    flags = np.asarray(present_flags, dtype=np.int64).reshape(-1)
    refs = ["" if r is None else str(r) for r in references]
    raw = ["" if t is None else str(t) for t in raw_texts]
    fall = ["" if t is None else str(t) for t in fallback_texts]
    n = flags.shape[0]
    err_raw = np.zeros(n, dtype=np.int64)
    err_fall = np.zeros(n, dtype=np.int64)
    ref_chars = np.zeros(n, dtype=np.int64)
    rej_raw = np.zeros(n, dtype=np.int64)
    rej_fall = np.zeros(n, dtype=np.int64)
    for i, (flag, ref, r, f) in enumerate(zip(flags.tolist(), refs, raw, fall)):
        rej_raw[i] = int(is_rejection(r))
        rej_fall[i] = int(is_rejection(f))
        if flag == 1:
            s_raw = cer_stats(ref, r)
            s_fall = cer_stats(ref, f)
            err_raw[i] = s_raw.errors
            err_fall[i] = s_fall.errors
            ref_chars[i] = s_raw.ref_chars
    return {
        "flags": flags,
        "err_raw": err_raw,
        "err_fall": err_fall,
        "ref_chars": ref_chars,
        "rej_raw": rej_raw,
        "rej_fall": rej_fall,
        "total_ref": int(ref_chars.sum()),
        "num_present": int((flags == 1).sum()),
        "num_absent": int((flags == 0).sum()),
    }


def _metrics_from_contributions(pre: dict, routed_raw: np.ndarray) -> dict[str, float]:
    """Compute rescue metrics for a boolean ``routed_raw`` mask (present->raw)."""

    routed_raw = np.asarray(routed_raw, dtype=bool)
    flags = pre["flags"]
    present = flags == 1
    absent = ~present
    # Routed output is raw where routed_raw else fallback.
    routed_err = np.where(routed_raw, pre["err_raw"], pre["err_fall"])
    routed_rej = np.where(routed_raw, pre["rej_raw"], pre["rej_fall"])
    total_err = int(routed_err[present].sum())
    total_ref = pre["total_ref"]
    correct_reject = int(routed_rej[absent].sum())
    false_reject = int(routed_rej[present].sum())
    num_present = pre["num_present"]
    num_absent = pre["num_absent"]
    avg_cer = total_err / total_ref if total_ref else 0.0
    avg_rr = correct_reject / num_absent if num_absent else 0.0
    false_reject_rate = false_reject / num_present if num_present else 0.0
    reject_accuracy = avg_rr
    overall = ((1.0 - avg_cer) + avg_rr) / 2.0
    return {
        "avg_cer": float(avg_cer),
        "avg_rr": float(avg_rr),
        "overall": float(overall),
        "pos_count": int(num_present),
        "neg_count": int(num_absent),
        "false_reject_rate": float(false_reject_rate),
        "reject_accuracy": float(reject_accuracy),
        "substitutions": 0,
        "insertions": 0,
        "deletions": int(total_err),
        "ref_chars": int(total_ref),
    }


def select_rescue_threshold(
    present_flags: Sequence[int],
    probabilities: Sequence[float],
    references: Sequence[str],
    raw_texts: Sequence[str],
    fallback_texts: Sequence[str],
    *,
    max_frr: float = POLICY_MAX_FRR,
    min_rr: float = POLICY_MIN_RR,
    candidates: Sequence[float] | None = None,
) -> dict:
    """Select the public-validation rescue threshold.

    Maximizes ``overall`` (S_public) over eligible thresholds subject to the
    policy (``false_reject_rate <= max_frr`` and ``reject_accuracy >= min_rr``).
    Eligibility and selection use only the supplied (public) rows.  Raises
    ``ValueError`` (``no eligible public rescue threshold``) when the policy
    cannot be satisfied - the fail-closed behaviour preserved from the
    presence proxy.  ``cer_stats`` is run once per row (precomputed), so the
    threshold scan is O(n_candidates * n) numpy work, not Levenshtein-per-pair.
    """

    flags = np.asarray(present_flags, dtype=np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if flags.shape != probs.shape or flags.size == 0:
        raise ValueError("present_flags and probabilities must be non-empty and equal length")
    if len(references) != flags.size or len(raw_texts) != flags.size or len(fallback_texts) != flags.size:
        raise ValueError("references/raw_texts/fallback_texts must match present_flags length")
    if not np.isin(flags, (0, 1)).all():
        raise ValueError("present_flags must contain only 0 and 1")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities must be finite")
    if not (np.any(flags == 1) and np.any(flags == 0)):
        raise ValueError("present_flags must contain both present and absent rows")
    for name in ("max_frr", "min_rr"):
        value = locals()[name]
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be a finite value in [0, 1]")

    if candidates is None:
        candidate_set = sorted({0.0, 1.0, *(float(p) for p in probs.tolist())})
    else:
        candidate_set = sorted({float(c) for c in candidates})

    pre = _precompute_contributions(flags, references, raw_texts, fallback_texts)
    eligible: list[tuple[float, dict[str, float]]] = []
    best: tuple[float, dict[str, float]] | None = None
    for threshold in candidate_set:
        routed_raw = probs >= threshold
        metrics = _metrics_from_contributions(pre, routed_raw)
        if metrics["false_reject_rate"] <= float(max_frr) and metrics["reject_accuracy"] >= float(min_rr):
            eligible.append((threshold, metrics))
        if best is None or metrics["overall"] > best[1]["overall"] or (
            metrics["overall"] == best[1]["overall"] and threshold > best[0]
        ):
            best = (threshold, metrics)

    if not eligible:
        raise ValueError("no eligible public rescue threshold (policy fail-closed)")

    threshold, metrics = max(eligible, key=lambda item: (item[1]["overall"], item[0]))
    return {
        "threshold": float(threshold),
        "metrics": metrics,
        "threshold_source": "public_validation",
        "policy": {"max_false_reject_rate": float(max_frr), "min_reject_accuracy": float(min_rr)},
        "num_candidates": len(candidate_set),
        "num_eligible": len(eligible),
        "unconstrained_best": {"threshold": float(best[0]), "metrics": best[1]},
    }


def public_baselines(
    present_flags: Sequence[int],
    references: Sequence[str],
    raw_texts: Sequence[str],
    fallback_texts: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Return raw-alone and reject-alone (fallback-only) metrics for comparison.

    ``raw_alone`` routes every sample to the raw transcript (CER = raw CER on
    present, RR = 0 on absent).  ``fallback_only`` routes every sample to the
    fallback (CER = fallback CER on present, RR = fallback RR on absent).  For
    the public simulation the fallback is empty, so ``fallback_only`` is the
    reject-alone corner (CER = 1.0, RR = 1.0).
    """

    flags = np.asarray(present_flags, dtype=np.int64).reshape(-1)
    raw = ["" if t is None else str(t) for t in raw_texts]
    fall = ["" if t is None else str(t) for t in fallback_texts]
    refs = ["" if r is None else str(r) for r in references]
    return {
        "raw_alone": rescue_metrics(flags, refs, raw),
        "fallback_only": rescue_metrics(flags, refs, fall),
    }
