"""Validation for competition-format submission payloads."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .contracts import RunTrace, ValidationIssue


def _issue(code: str, message: str, severity: str = "error") -> ValidationIssue:
    return ValidationIssue(code=code, message=message, severity=severity)


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def validate_competition_payload(
    payload: object,
    *,
    expected_ids: set[str],
    allowed_row_fields: set[str],
    trace: RunTrace | None = None,
    official: bool = False,
) -> tuple[ValidationIssue, ...]:
    """Return deterministic validation issues for a competition JSON payload.

    The validator deliberately does not infer optional template fields: callers must
    explicitly include every permitted row field in ``allowed_row_fields``.
    """
    issues: list[ValidationIssue] = []
    root = _mapping(payload)
    result = _mapping(root.get("result")) if root is not None else None
    if root is None:
        issues.append(_issue("invalid_payload", "payload must be an object"))
    elif result is None:
        issues.append(_issue("missing_result", "payload.result must be an object"))

    rows: object = result.get("results") if result is not None else None
    if not isinstance(rows, list):
        issues.append(_issue("invalid_results", "payload.result.results must be a list"))
        rows_list: list[object] = []
    else:
        rows_list = rows

    seen: list[str] = []
    for index, row_value in enumerate(rows_list):
        row = _mapping(row_value)
        if row is None:
            issues.append(_issue("invalid_row", f"result.results[{index}] must be an object"))
            continue
        for field in row:
            if field not in allowed_row_fields:
                issues.append(_issue(
                    "disallowed_row_field",
                    f"result.results[{index}] contains disallowed field {field!r}",
                ))
        if "id" not in row:
            issues.append(_issue("missing_id", f"result.results[{index}] is missing id"))
        else:
            row_id = row["id"]
            if not isinstance(row_id, str):
                issues.append(_issue("id_not_string", f"result.results[{index}].id must be a string"))
            else:
                seen.append(row_id)
        if "content" not in row:
            issues.append(_issue("missing_content", f"result.results[{index}] is missing content"))
        elif not isinstance(row["content"], str):
            issues.append(_issue(
                "content_not_string",
                f"result.results[{index}].content must be a string (empty string represents rejection)",
            ))

    counts = Counter(seen)
    for row_id, count in counts.items():
        if count > 1:
            issues.append(_issue("duplicate_id", f"submission id {row_id!r} occurs {count} times"))
    seen_set = set(seen)
    for row_id in sorted(expected_ids - seen_set):
        issues.append(_issue("missing_id", f"expected submission id {row_id!r} is missing"))
    for row_id in sorted(seen_set - expected_ids):
        issues.append(_issue("extra_id", f"unexpected submission id {row_id!r}"))

    duration = result.get("duration") if result is not None else None
    if isinstance(duration, bool) or not isinstance(duration, (int, float, str)):
        valid_duration = False
    else:
        try:
            valid_duration = math.isfinite(float(duration)) and float(duration) >= 0
        except (TypeError, ValueError, OverflowError):
            valid_duration = False
    if not valid_duration:
        issues.append(_issue("invalid_duration", "result.duration must be a finite non-negative number"))

    if official and (trace is None or trace.measurement_mode != "real"):
        issues.append(_issue(
            "official_duration_requires_real_trace",
            "official duration requires a real RunTrace; replay or missing traces are not accepted",
        ))
    return tuple(issues)
