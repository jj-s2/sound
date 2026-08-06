"""Select enhancement candidates without using labels.

This is the deployable trigger for V4: it inspects primary ASR text, optional
speaker scores and audio duration, then writes ids that should receive
enhanced/fallback ASR.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.audio_features import read_wav_info
from xh202615.backends import ScoreCsvSpeakerBackend
from xh202615.data import load_dataset, read_jsonl
from xh202615.robustness import should_enhance_for_robustness


FIELDNAMES = [
    "id",
    "split",
    "enhance",
    "reason",
    "text_length",
    "domain_score",
    "action_hits",
    "device_hits",
    "setting_hits",
    "media_hits",
    "life_hits",
    "qa_hits",
    "tool_hits",
    "assistant_intent_score",
    "target_probability",
    "global_similarity",
    "topk_similarity",
    "duration_sec",
    "recognition_text",
    "command_audio",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Select V4 enhancement candidates without labels")
    parser.add_argument("--dataset-root", default="datasetA")
    parser.add_argument("--splits", default="pos,neg")
    parser.add_argument("--asr-map", required=True)
    parser.add_argument("--speaker-scores", default=None)
    parser.add_argument("--output", default="output/reports/v4_auto_candidates.csv")
    parser.add_argument("--ids-output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-split-limit", type=int, default=None)
    parser.add_argument("--min-text-length", type=int, default=8)
    parser.add_argument("--max-domain-score", type=int, default=0)
    parser.add_argument("--short-text-length", type=int, default=6)
    parser.add_argument("--enable-short-non-domain", action="store_true")
    parser.add_argument("--incomplete-text-length", type=int, default=12)
    parser.add_argument("--max-incomplete-domain-score", type=int, default=2)
    parser.add_argument("--long-text-length", type=int, default=14)
    parser.add_argument("--long-text-max-domain-score", type=int, default=1)
    parser.add_argument("--very-long-text-length", type=int, default=18)
    parser.add_argument("--min-audio-duration-sec", type=float, default=0.0)
    parser.add_argument("--speaker-similarity-max", type=float, default=None)
    parser.add_argument("--target-probability-max", type=float, default=None)
    parser.add_argument("--min-selected-ratio", type=float, default=None)
    parser.add_argument("--max-selected-ratio", type=float, default=None)
    parser.add_argument("--selected-only", action="store_true", help="Only write selected rows to --output")
    return parser.parse_args()


def load_asr_map(path: Path) -> dict[str, str]:
    values = {}
    for row in read_jsonl(path):
        text = row.get("text", row.get("recognition_text", ""))
        values[str(row["id"])] = "" if text is None else str(text)
    return values


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples = load_dataset(args.dataset_root, splits)
    if args.per_split_limit is not None:
        selected = []
        for split in splits:
            selected.extend([sample for sample in samples if sample.split == split][: args.per_split_limit])
        samples = selected
    if args.limit is not None:
        samples = samples[: args.limit]

    asr_map = load_asr_map(Path(args.asr_map))
    speaker = ScoreCsvSpeakerBackend(args.speaker_scores)

    rows = []
    selected_ids = []
    for sample in samples:
        sample_id = str(sample.id)
        text = asr_map.get(sample_id, "")
        scores = speaker.score(sample)
        info = read_wav_info(sample.command_audio)
        duration_sec = info.duration_sec if info.valid else 0.0
        decision = should_enhance_for_robustness(
            text,
            scores,
            min_text_length=args.min_text_length,
            max_domain_score=args.max_domain_score,
            short_text_length=args.short_text_length,
            enable_short_non_domain=args.enable_short_non_domain,
            incomplete_text_length=args.incomplete_text_length,
            max_incomplete_domain_score=args.max_incomplete_domain_score,
            long_text_length=args.long_text_length,
            long_text_max_domain_score=args.long_text_max_domain_score,
            very_long_text_length=args.very_long_text_length,
            min_audio_duration_sec=args.min_audio_duration_sec,
            audio_duration_sec=duration_sec,
            speaker_similarity_max=args.speaker_similarity_max,
            target_probability_max=args.target_probability_max,
        )
        if decision.enhance:
            selected_ids.append(sample_id)
        row = {
            "id": sample_id,
            "split": sample.split,
            "enhance": decision.enhance,
            "reason": decision.reason,
            "text_length": decision.evidence.text_length,
            "domain_score": decision.evidence.domain_score,
            "action_hits": decision.evidence.action_hits,
            "device_hits": decision.evidence.device_hits,
            "setting_hits": decision.evidence.setting_hits,
            "media_hits": decision.evidence.media_hits,
            "life_hits": decision.evidence.life_hits,
            "qa_hits": decision.evidence.qa_hits,
            "tool_hits": decision.evidence.tool_hits,
            "assistant_intent_score": decision.evidence.assistant_intent_score,
            "target_probability": "" if scores.target_probability is None else scores.target_probability,
            "global_similarity": "" if scores.global_similarity is None else scores.global_similarity,
            "topk_similarity": "" if scores.topk_similarity is None else scores.topk_similarity,
            "duration_sec": duration_sec,
            "recognition_text": text,
            "command_audio": str(sample.command_audio),
        }
        if decision.enhance or not args.selected_only:
            rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    ids_out = Path(args.ids_output) if args.ids_output else out.with_suffix(".ids.txt")
    ids_out.write_text("\n".join(selected_ids) + ("\n" if selected_ids else ""), encoding="utf-8")
    selected_ratio = len(selected_ids) / len(samples) if samples else 0.0
    print(f"Selected {len(selected_ids)} / {len(samples)} enhancement candidates ({selected_ratio:.2%})")
    if args.min_selected_ratio is not None and selected_ratio < args.min_selected_ratio:
        print(f"WARNING: selected ratio is below {args.min_selected_ratio:.2%}")
    if args.max_selected_ratio is not None and selected_ratio > args.max_selected_ratio:
        print(f"WARNING: selected ratio is above {args.max_selected_ratio:.2%}")
    print(f"Wrote report to {out}")
    print(f"Wrote ids to {ids_out}")


if __name__ == "__main__":
    main()
