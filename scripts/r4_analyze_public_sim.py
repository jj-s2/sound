"""Analyze the public rescue simulation across thresholds (diagnostic, read-only).

Computes the public rescue router's CER/RR/Overall across candidate thresholds
on public val (selection) and test (validation), using the public ASR bakeoff
transcripts + AISHELL references + the frozen R3 head probabilities.  Reports
the strict-policy eligibility, the max-S_public operating point, the best-F1
threshold, and the raw-alone / reject-alone corners.  No Dataset-A is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import read_jsonl
from xh202615.r4_rescue import (
    _metrics_from_contributions,
    _precompute_contributions,
    public_baselines,
)
from scripts.r4_select_rescue_threshold import build_columns, head_probabilities


def best_f1_threshold(present, probs):
    y = np.asarray(present, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    cands = sorted(set([0.0, 1.0] + list(p)))
    best = (0.5, 0.0)
    for t in cands:
        pred = (p >= t).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[1]:
            best = (t, f1)
    return best


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bakeoff", required=True)
    parser.add_argument("--checkpoint", default="output/training/r3_temporal_head/best.pt")
    parser.add_argument("--feature-cache", default="output/training/r3_temporal_head/frozen_features.pt")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--max-frr", type=float, default=0.10)
    parser.add_argument("--min-rr", type=float, default=0.85)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    probs = head_probabilities(Path(args.checkpoint), Path(args.feature_cache), "cpu")
    rows = list(read_jsonl(Path(args.bakeoff)))
    val = build_columns(rows, probs, args.val_split)
    test = build_columns(rows, probs, args.test_split)

    val_pre = _precompute_contributions(val["present"], val["references"], val["raw_texts"], val["fallback_texts"])
    test_pre = _precompute_contributions(test["present"], test["references"], test["raw_texts"], test["fallback_texts"])
    val_base = public_baselines(val["present"], val["references"], val["raw_texts"], val["fallback_texts"])
    test_base = public_baselines(test["present"], test["references"], test["raw_texts"], test["fallback_texts"])

    val_probs = np.asarray(val["probabilities"], dtype=np.float64)
    cands = sorted({0.0, 1.0, *(float(x) for x in val_probs)})

    eligible = []
    all_thr = []
    for t in cands:
        m = _metrics_from_contributions(val_pre, val_probs >= t)
        all_thr.append((t, m))
        if m["false_reject_rate"] <= args.max_frr and m["reject_accuracy"] >= args.min_rr:
            eligible.append((t, m))

    max_s = max(all_thr, key=lambda x: (x[1]["overall"], x[0]))
    f1_t, f1_v = best_f1_threshold(val["present"], val["probabilities"])
    f1_m = _metrics_from_contributions(val_pre, val_probs >= f1_t)

    test_probs = np.asarray(test["probabilities"], dtype=np.float64)
    test_max_s = _metrics_from_contributions(test_pre, test_probs >= max_s[0])
    test_f1 = _metrics_from_contributions(test_pre, test_probs >= f1_t)

    result = {
        "val_count": len(val["present"]),
        "test_count": len(test["present"]),
        "strict_policy": {"max_frr": args.max_frr, "min_rr": args.min_rr, "num_eligible": len(eligible)},
        "val_baselines": val_base,
        "test_baselines": test_base,
        "val_max_s_public": {"threshold": max_s[0], "metrics": max_s[1]},
        "val_best_f1": {"threshold": f1_t, "f1": f1_v, "metrics": f1_m},
        "test_at_max_s_threshold": {"threshold": max_s[0], "metrics": test_max_s},
        "test_at_best_f1_threshold": {"threshold": f1_t, "metrics": test_f1},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
