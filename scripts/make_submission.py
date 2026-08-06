"""Create a compact submission file from prediction JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import load_dataset, read_jsonl
from xh202615.metrics import cer_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Create XH-202615 submission from predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="output/submissions/submission.jsonl")
    parser.add_argument("--format", choices=("jsonl", "json", "competition_json"), default="jsonl")
    parser.add_argument("--text-field", default="recognition_text")
    parser.add_argument("--dataset-root", default=None, help="Optional dataset root used for labels/audio-name ids")
    parser.add_argument("--splits", default="pos,neg")
    parser.add_argument(
        "--id-source",
        choices=("sample_id", "command_audio_path", "command_audio_name", "command_audio_stem"),
        default="sample_id",
        help="Id field for competition_json. The PDF says id is the test audio name.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Total batch=1 inference duration in seconds. Required for competition_json.",
    )
    parser.add_argument(
        "--allow-latency-duration",
        action="store_true",
        help="Diagnostic only: fill duration from sum(latency_ms)/1000 when --duration is omitted.",
    )
    return parser.parse_args()


def load_samples(args):
    if not args.dataset_root:
        return {}
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    return {str(sample.id): sample for sample in load_dataset(args.dataset_root, splits)}


def submission_id(sample_id: str, sample, id_source: str, dataset_root: str | None = None) -> str:
    if sample is None or id_source == "sample_id":
        return sample_id
    if id_source == "command_audio_path":
        if dataset_root:
            try:
                return sample.command_audio.relative_to(Path(dataset_root)).as_posix()
            except ValueError:
                pass
        return sample.command_audio.as_posix()
    if id_source == "command_audio_name":
        return sample.command_audio.name
    if id_source == "command_audio_stem":
        return sample.command_audio.stem
    return sample_id


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def main() -> None:
    args = parse_args()
    samples = load_samples(args)
    rows = []
    competition_rows = []
    total_latency_ms = 0.0
    total_errors = 0
    total_ref_chars = 0
    for row in read_jsonl(Path(args.predictions)):
        sample_id = str(row["id"])
        text = row.get(args.text_field, row.get("text", ""))
        text = "" if text is None else str(text)
        rows.append({"id": sample_id, "recognition_text": text})

        latency_ms = row.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            total_latency_ms += float(latency_ms)
        sample = samples.get(sample_id)
        label = "" if sample is None or sample.label is None else str(sample.label)
        cer = ""
        if label:
            stats = cer_stats(label, text)
            total_errors += stats.errors
            total_ref_chars += stats.ref_chars
            cer = format_float(stats.cer)
        competition_rows.append(
            {
                "id": submission_id(sample_id, sample, args.id_source, args.dataset_root),
                "content": text,
                "label": label,
                "cer": cer,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "competition_json":
        if args.duration is None and not args.allow_latency_duration:
            raise SystemExit(
                "competition_json requires explicit --duration from the real batch=1 inference run. "
                "Use --allow-latency-duration only for local diagnostics."
            )
        duration = args.duration if args.duration is not None else total_latency_ms / 1000
        payload = {
            "result": {
                "results": competition_rows,
                "final_cer": "" if total_ref_chars == 0 else format_float(total_errors / total_ref_chars),
                "duration": format_float(duration),
            }
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.format == "json":
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} submission rows to {out}")


if __name__ == "__main__":
    main()
