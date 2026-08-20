"""Pure, provenance-neutral evaluation of prediction rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .data import Sample
from .metrics import cer_stats, is_rejection


METRIC_KEYS = (
    "avg_cer",
    "avg_rr",
    "pos_count",
    "neg_count",
    "missing_predictions",
    "substitutions",
    "insertions",
    "deletions",
    "ref_chars",
    "false_reject_rate",
    "false_accept_rate",
)


@dataclass(frozen=True)
class EvaluationReport:
    metrics: dict
    per_sample: tuple[dict, ...]
    buckets: dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "metrics": dict(self.metrics),
            "per_sample": [dict(row) for row in self.per_sample],
            "buckets": {key: dict(value) for key, value in self.buckets.items()},
        }


def _evaluate_subset(
    samples: Iterable[Sample],
    predictions: Mapping[str, str],
    *,
    missing_policy: str,
) -> tuple[dict, tuple[dict, ...]]:
    pos_count = neg_count = missing = 0
    correct_reject = false_reject = false_accept = 0
    substitutions = insertions = deletions = ref_chars = 0
    per_sample: list[dict] = []

    for sample in samples:
        sample_id = str(sample.id)
        if sample_id not in predictions:
            missing += 1
            if missing_policy == "skip":
                continue

        hyp = predictions.get(sample_id, "")
        if sample.label is None:
            neg_count += 1
            rejected = is_rejection(hyp)
            correct_reject += int(rejected)
            false_accept += int(not rejected)
            per_sample.append(
                {"id": sample_id, "split": sample.split, "rr_correct": rejected}
            )
        else:
            pos_count += 1
            rejected = is_rejection(hyp)
            false_reject += int(rejected)
            stats = cer_stats(sample.label, hyp)
            substitutions += stats.substitutions
            insertions += stats.insertions
            deletions += stats.deletions
            ref_chars += stats.ref_chars
            per_sample.append(
                {
                    "id": sample_id,
                    "split": sample.split,
                    "cer": stats.cer,
                    "errors": stats.errors,
                }
            )

    metrics = {
        "avg_cer": (substitutions + insertions + deletions) / ref_chars
        if ref_chars
        else 0.0,
        "avg_rr": correct_reject / neg_count if neg_count else 0.0,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "missing_predictions": missing,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "ref_chars": ref_chars,
        "false_reject_rate": false_reject / pos_count if pos_count else 0.0,
        "false_accept_rate": false_accept / neg_count if neg_count else 0.0,
    }
    return metrics, tuple(per_sample)


def _prediction_map(rows: Iterable[dict]) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("prediction row must be a mapping")
        if "id" not in row:
            raise ValueError("prediction row is missing field 'id'")
        sample_id = str(row["id"])
        if sample_id in predictions:
            raise ValueError(f"duplicate prediction id {sample_id!r}")
        predictions[sample_id] = row.get("recognition_text", "")
        if predictions[sample_id] is None:
            predictions[sample_id] = ""
        elif not isinstance(predictions[sample_id], str):
            predictions[sample_id] = str(predictions[sample_id])
    return predictions


def evaluate_rows(
    samples: Iterable[Sample],
    rows: Iterable[dict],
    *,
    text_field: str = "recognition_text",
    missing_policy: str = "empty",
    metadata_by_id: Mapping[str, Mapping[str, object]] | None = None,
    bucket_fields: tuple[str, ...] = (),
) -> EvaluationReport:
    """Evaluate prediction rows without mutating inputs or routing decisions."""

    if missing_policy not in {"empty", "skip"}:
        raise ValueError("missing_policy must be 'empty' or 'skip'")

    ordered_samples = list(samples)
    sample_ids: set[str] = set()
    for sample in ordered_samples:
        sample_id = str(sample.id)
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample id {sample_id!r}")
        sample_ids.add(sample_id)

    # Normalize the caller-selected field while retaining the old CLI's text alias.
    normalized_rows = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("prediction row must be a mapping")
        normalized = dict(row)
        if text_field not in normalized:
            normalized[text_field] = normalized.get("text", "")
        normalized_rows.append(normalized)
    predictions = _prediction_map(normalized_rows)

    metrics, per_sample = _evaluate_subset(
        ordered_samples, predictions, missing_policy=missing_policy
    )

    buckets: dict[str, dict] = {}
    metadata = metadata_by_id or {}
    for field in bucket_fields:
        groups: dict[str, list[Sample]] = {}
        for sample in ordered_samples:
            sample_metadata = metadata.get(str(sample.id))
            if sample_metadata is None or field not in sample_metadata:
                continue
            key = f"{field}={sample_metadata[field]}"
            groups.setdefault(key, []).append(sample)
        for key, group in groups.items():
            bucket_metrics, _ = _evaluate_subset(
                group, predictions, missing_policy=missing_policy
            )
            buckets[key] = bucket_metrics

    return EvaluationReport(metrics, per_sample, buckets)
