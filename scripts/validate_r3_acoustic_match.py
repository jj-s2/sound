"""Aggregate-match validator for generated R3 counterfactual data (Task 4).

Profiles the generated public mixture audio, compares its aggregate acoustic
distribution to the target aggregate profile, and reports match quality plus
integrity checks. The report is aggregate-only: it never contains Dataset-A
paths, labels, texts, predictions, or bytes. A fail-closed guard aborts if any
generated audio path resolves under a Dataset-A root or if the report would
serialize a prohibited key.

Command::

    python scripts/validate_r3_acoustic_match.py --profile PROFILE.json \
        --manifest MANIFEST.jsonl --output REPORT.json [--dataset-a-root ROOT]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.acoustic_profile import (  # noqa: E402
    assert_aggregate_profile,
    compare_profiles,
    profile_audio_paths_safe,
    read_aggregate_profile,
)
from xh202615.r3_data import (  # noqa: E402
    R3MixtureRow,
    read_r3_manifest,
    validate_r3_manifest,
)

_PROHIBITED_KEYS = {
    "paths", "files", "file_paths", "ids", "transcript",
    "label", "labels", "text", "prediction", "predictions",
}


def _is_under(candidate: Path, root: Path) -> bool:
    candidate = candidate.resolve(strict=False)
    root = root.resolve(strict=False)
    return candidate == root or root in candidate.parents


def _assert_report_redacted(report: dict, dataset_a_root: Path | None) -> None:
    """Fail-closed: abort if the report would leak Dataset-A paths or prohibited keys."""
    for key in _PROHIBITED_KEYS:
        if key in report:
            raise ValueError(f"report contains prohibited key: {key!r}")
    text = json.dumps(report, ensure_ascii=False, sort_keys=True).lower()
    if dataset_a_root is not None:
        root = Path(dataset_a_root).resolve(strict=False)
        candidates = {
            str(root).lower(),
            root.as_posix().lower(),
            str(dataset_a_root).lower(),
            Path(dataset_a_root).as_posix().lower(),
        }
        for cand in candidates:
            if cand and cand in text:
                raise ValueError(
                    f"report would serialize a Dataset-A root path: {cand!r}"
                )


def _summary_distance(comparison_metrics: dict) -> dict:
    """Symmetric relative distance summary across all metric quantiles.

    Uses ``|candidate - reference| / max(|candidate|, |reference|, eps)`` so a
    near-zero reference (e.g. ``clipping_rate``) cannot inflate the distance.
    """
    rel_diffs: list[float] = []
    for per_q in comparison_metrics.values():
        for entry in per_q.values():
            r = float(entry.get("reference", 0.0))
            c = float(entry.get("candidate", 0.0))
            if not (math.isfinite(r) and math.isfinite(c)):
                continue
            denom = max(abs(r), abs(c), 1e-8)
            rel_diffs.append(abs(c - r) / denom)
    if not rel_diffs:
        return {"median_abs_rel_diff": 1.0, "max_abs_rel_diff": 1.0}
    return {
        "median_abs_rel_diff": float(sorted(rel_diffs)[len(rel_diffs) // 2]),
        "max_abs_rel_diff": float(max(rel_diffs)),
    }


def validate_match(
    profile_path: str | Path,
    manifest_path: str | Path,
    output: str | Path | None = None,
    *,
    dataset_a_root: str | Path | None = None,
) -> dict:
    """Validate a generated R3 manifest against an aggregate target profile."""
    profile_payload = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    assert_aggregate_profile(profile_payload)
    target = read_aggregate_profile(profile_path)

    rows = read_r3_manifest(manifest_path)
    issues = validate_r3_manifest(rows)
    error_issues = [i for i in issues if i.severity == "error"]

    da_root = Path(dataset_a_root).resolve(strict=False) if dataset_a_root else None
    # Fail-closed: no generated audio may resolve under Dataset-A.
    for row in rows:
        for field_name in ("enrollment_audio", "mixture_audio", "clean_target_audio"):
            path = Path(getattr(row, field_name))
            if da_root is not None and _is_under(path, da_root):
                raise ValueError(
                    f"Dataset-A containment violation: {field_name} for row "
                    f"{row.row_id!r} resolves under Dataset-A root {da_root}"
                )

    # Unique generated mixture paths only; profile readable ones, count the rest.
    seen: dict[str, None] = {}
    for row in rows:
        seen.setdefault(str(row.mixture_audio), None)
    mixture_paths = list(seen.keys())
    generated, unreadable = profile_audio_paths_safe(mixture_paths)

    comparison = compare_profiles(target, generated)
    checks = [
        {
            "name": "manifest_valid",
            "passed": not error_issues,
            "detail": f"{len(error_issues)} error-severity issue(s)",
        },
        {
            "name": "all_audio_readable",
            "passed": unreadable == 0,
            "detail": f"{unreadable} unreadable mixture file(s)",
        },
        {
            "name": "non_empty",
            "passed": len(rows) > 0,
            "detail": f"{len(rows)} row(s)",
        },
        {
            "name": "counterfactual_pairs_balanced",
            "passed": len(rows) % 2 == 0,
            "detail": f"{len(rows)} row(s); expected an even count (pos+neg pairs)",
        },
    ]
    if da_root is not None:
        checks.append(
            {
                "name": "no_dataset_a_bytes",
                "passed": True,
                "detail": "no generated audio path resolves under Dataset-A",
            }
        )

    report = {
        "profile_hash": target.hash,
        "generated_hash": generated.hash,
        "profile_file_count": target.file_count,
        "generated_file_count": generated.file_count,
        "matched_pair_count": len(rows) // 2,
        "row_count": len(rows),
        "unreadable_audio_count": unreadable,
        "manifest_issues": [issue.to_dict() for issue in issues],
        "comparison": comparison["metrics"],
        "aggregate_distance": _summary_distance(comparison["metrics"]),
        "checks": checks,
        "overall_passed": all(check["passed"] for check in checks),
    }
    _assert_report_redacted(report, da_root)

    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="aggregate target profile JSON")
    parser.add_argument("--manifest", required=True, help="generated R3 manifest JSONL")
    parser.add_argument("--output", required=True, help="output report JSON")
    parser.add_argument(
        "--dataset-a-root", default=None,
        help="optional Dataset-A root for fail-closed containment check",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_match(
        args.profile,
        args.manifest,
        args.output,
        dataset_a_root=args.dataset_a_root,
    )
    status = "PASS" if report["overall_passed"] else "FAIL"
    print(
        f"{status}: {report['matched_pair_count']} pairs, "
        f"{report['unreadable_audio_count']} unreadable, "
        f"median rel_diff={report['aggregate_distance']['median_abs_rel_diff']:.3f} "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
