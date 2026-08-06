"""Fuse multiple ASR maps with label-free text-quality rules."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import read_jsonl
from xh202615.text_router import analyze_text


def parse_args():
    parser = argparse.ArgumentParser(description="Select best ASR candidate without labels")
    parser.add_argument("--primary", required=True, help="Primary JSONL ASR map")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Repeatable candidate: name=path.jsonl. Primary is always included.",
    )
    parser.add_argument("--speaker-scores", default=None, help="Optional CSV for target_probability/similarity")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-priority", default="primary,hotword,energy,sensevoice,bss,tse")
    parser.add_argument("--max-reasonable-length", type=int, default=24)
    parser.add_argument("--min-nonempty-score-gap", type=float, default=0.20)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for row in read_jsonl(path):
        rows[str(row["id"])] = dict(row)
    return rows


def text_of(row: dict | None) -> str:
    if not row:
        return ""
    text = row.get("recognition_text", row.get("text", ""))
    return "" if text is None else str(text)


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name, Path(path)


def load_speaker_scores(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return {str(row["id"]): row for row in csv.DictReader(f)}


def get_float(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def source_bias(name: str, priority: list[str]) -> float:
    lowered = name.lower()
    for index, item in enumerate(priority):
        if item and item.lower() in lowered:
            return max(0.0, 0.30 - index * 0.04)
    return 0.0


def score_text(text: str, source_name: str, speaker_row: dict, priority: list[str], max_length: int) -> tuple[float, str]:
    evidence = analyze_text(text)
    if evidence.text_length == 0:
        return -10.0, "empty"

    score = 0.0
    reasons = []
    score += 1.0
    reasons.append("nonempty")

    if evidence.domain_score > 0:
        score += min(2.0, evidence.domain_score * 0.45)
        reasons.append(f"intent={evidence.domain_score}")
    else:
        score -= 0.50
        reasons.append("no_intent")

    if 2 <= evidence.text_length <= max_length:
        score += 0.35
        reasons.append(f"len_ok={evidence.text_length}")
    elif evidence.text_length > max_length:
        over = evidence.text_length - max_length
        score -= min(1.25, over * 0.08)
        reasons.append(f"long={evidence.text_length}")

    if evidence.text_length >= 18 and evidence.domain_score == 0:
        score -= 1.50
        reasons.append("possible_hallucination")

    target_prob = get_float(speaker_row, "target_probability")
    topk_similarity = get_float(speaker_row, "topk_similarity")
    global_similarity = get_float(speaker_row, "global_similarity")
    if target_prob is not None:
        score += min(0.30, max(0.0, target_prob - 0.50) * 0.30)
    if topk_similarity is not None or global_similarity is not None:
        sim = max(value for value in (topk_similarity, global_similarity) if value is not None)
        score += min(0.25, max(0.0, sim - 0.50) * 0.35)

    bias = source_bias(source_name, priority)
    if bias:
        score += bias
        reasons.append(f"source_bias={bias:.2f}")

    return score, ";".join(reasons)


def main() -> None:
    args = parse_args()
    priority = [item.strip() for item in args.source_priority.split(",") if item.strip()]
    sources: list[tuple[str, dict[str, dict]]] = [("primary", load_rows(Path(args.primary)))]
    sources.extend((name, load_rows(path)) for name, path in [parse_candidate(value) for value in args.candidate])
    speaker_scores = load_speaker_scores(args.speaker_scores)

    all_ids = list(sources[0][1].keys())
    for _, rows in sources[1:]:
        for sample_id in rows:
            if sample_id not in sources[0][1]:
                all_ids.append(sample_id)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    switched = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for sample_id in all_ids:
            speaker_row = speaker_scores.get(sample_id, {})
            ranked = []
            for source_name, rows in sources:
                row = rows.get(sample_id)
                text = text_of(row)
                score, reason = score_text(text, source_name, speaker_row, priority, args.max_reasonable_length)
                ranked.append((score, source_name, reason, row, text))
            ranked.sort(key=lambda item: item[0], reverse=True)
            best_score, best_source, best_reason, best_row, best_text = ranked[0]
            primary_score = next(item[0] for item in ranked if item[1] == "primary")
            if best_source != "primary" and best_score - primary_score < args.min_nonempty_score_gap:
                best_score, best_source, best_reason, best_row, best_text = next(item for item in ranked if item[1] == "primary")

            output_row = dict(best_row or {"id": sample_id})
            output_row["id"] = sample_id
            output_row["recognition_text"] = best_text
            output_row["text"] = best_text
            output_row["asr_backend"] = "candidate_fusion"
            output_row["candidate_source"] = best_source
            output_row["candidate_score"] = round(float(best_score), 6)
            output_row["candidate_reason"] = best_reason
            output_row["candidate_texts"] = {source_name: text_of(rows.get(sample_id)) for source_name, rows in sources}
            switched += int(best_source != "primary")
            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    print(f"Wrote fused ASR map to {out}; switched={switched}; total={len(all_ids)}")


if __name__ == "__main__":
    main()
