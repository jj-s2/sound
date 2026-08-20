"""Validate a competition JSON submission against a dataset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_submission import submission_id
from xh202615.contracts import RunTrace
from xh202615.data import load_dataset
from xh202615.submission_validation import validate_competition_payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate a competition submission")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--splits", default="pos,neg")
    parser.add_argument("--id-source", choices=("sample_id", "command_audio_path", "command_audio_name", "command_audio_stem"), default="command_audio_name")
    parser.add_argument("--allow-field", action="append", default=[])
    parser.add_argument("--trace")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--report")
    return parser.parse_args(argv)


def _expected_ids(args) -> set[str]:
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    samples = load_dataset(args.dataset_root, splits)
    by_id = {str(sample.id): sample for sample in samples}
    return {
        submission_id(sample_id, sample, args.id_source, args.dataset_root)
        for sample_id, sample in by_id.items()
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        payload = json.loads(Path(args.submission).read_text(encoding="utf-8"))
        trace = None
        if args.trace:
            trace = RunTrace.from_dict(json.loads(Path(args.trace).read_text(encoding="utf-8")))
        issues = validate_competition_payload(
            payload,
            expected_ids=_expected_ids(args),
            allowed_row_fields=set(("id", "content", *args.allow_field)),
            trace=trace,
            official=args.official,
        )
    except Exception as exc:  # diagnostics should report failures, not crash
        from xh202615.contracts import ValidationIssue
        issues = (ValidationIssue("validator_error", str(exc)),)

    report = {"issues": [issue.to_dict() for issue in issues], "error_count": sum(issue.severity == "error" for issue in issues)}
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0 if report["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
