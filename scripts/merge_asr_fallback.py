"""Merge primary and fallback ASR maps with text-quality routing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import read_jsonl
from xh202615.robustness import should_enhance_for_robustness
from xh202615.text_router import analyze_text


def parse_args():
    parser = argparse.ArgumentParser(description="Merge fallback ASR results into a primary ASR map")
    parser.add_argument("--primary", required=True)
    parser.add_argument("--fallback", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-primary-length", type=int, default=8)
    parser.add_argument("--max-primary-domain-score", type=int, default=0)
    parser.add_argument("--use-robustness-trigger", action="store_true")
    parser.add_argument("--short-text-length", type=int, default=6)
    parser.add_argument("--enable-short-non-domain", action="store_true")
    parser.add_argument("--incomplete-text-length", type=int, default=12)
    parser.add_argument("--max-incomplete-domain-score", type=int, default=2)
    parser.add_argument("--long-text-length", type=int, default=14)
    parser.add_argument("--long-text-max-domain-score", type=int, default=1)
    parser.add_argument("--very-long-text-length", type=int, default=18)
    parser.add_argument("--min-length-reduction-ratio", type=float, default=0.25)
    parser.add_argument("--require-fallback-nonempty", action="store_true")
    parser.add_argument("--prefer-higher-domain-score", action="store_true")
    return parser.parse_args()


def load_map(path: Path) -> dict[str, dict]:
    rows = {}
    for row in read_jsonl(path):
        rows[str(row["id"])] = dict(row)
    return rows


def text_of(row: dict | None) -> str:
    if not row:
        return ""
    text = row.get("text", row.get("recognition_text", ""))
    return "" if text is None else str(text)


def should_use_fallback(primary_text: str, fallback_text: str, args) -> tuple[bool, str]:
    primary = analyze_text(primary_text)
    fallback = analyze_text(fallback_text)
    if args.require_fallback_nonempty and fallback.text_length == 0:
        return False, "fallback_empty"

    trigger_reason = ""
    if args.use_robustness_trigger:
        decision = should_enhance_for_robustness(
            primary_text,
            min_text_length=args.min_primary_length,
            max_domain_score=args.max_primary_domain_score,
            short_text_length=args.short_text_length,
            enable_short_non_domain=args.enable_short_non_domain,
            incomplete_text_length=args.incomplete_text_length,
            max_incomplete_domain_score=args.max_incomplete_domain_score,
            long_text_length=args.long_text_length,
            long_text_max_domain_score=args.long_text_max_domain_score,
            very_long_text_length=args.very_long_text_length,
        )
        primary_looks_bad = decision.enhance
        trigger_reason = decision.reason
    else:
        primary_looks_bad = (
            primary.text_length >= args.min_primary_length
            and primary.domain_score <= args.max_primary_domain_score
        )
        trigger_reason = f"primary_non_domain:len={primary.text_length},domain={primary.domain_score}"

    if primary_looks_bad:
        if fallback.domain_score <= primary.domain_score:
            min_reduction = primary.text_length * args.min_length_reduction_ratio
            length_reduction = primary.text_length - fallback.text_length
            if length_reduction < min_reduction:
                return (
                    False,
                    (
                        "fallback_not_better"
                        f":primary_len={primary.text_length},fallback_len={fallback.text_length},"
                        f"primary_domain={primary.domain_score},fallback_domain={fallback.domain_score}"
                    ),
                )
        return True, trigger_reason

    if args.prefer_higher_domain_score and fallback.domain_score > primary.domain_score:
        return True, f"fallback_domain_better:{primary.domain_score}->{fallback.domain_score}"

    return False, "keep_primary"


def main() -> None:
    args = parse_args()
    primary_rows = load_map(Path(args.primary))
    fallback_rows = load_map(Path(args.fallback))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    replaced = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for sample_id, primary_row in primary_rows.items():
            fallback_row = fallback_rows.get(sample_id)
            primary_text = text_of(primary_row)
            fallback_text = text_of(fallback_row)
            use_fallback, reason = should_use_fallback(primary_text, fallback_text, args)
            row = dict(fallback_row if use_fallback and fallback_row else primary_row)
            if use_fallback and fallback_row:
                replaced += 1
                row["primary_text"] = primary_text
                row["fallback_reason"] = reason
                row["asr_backend"] = "fallback_merge"
            else:
                row["fallback_reason"] = reason
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote merged ASR map to {out}; replaced={replaced}; primary={len(primary_rows)} fallback={len(fallback_rows)}")


if __name__ == "__main__":
    main()
