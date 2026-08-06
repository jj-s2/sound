"""Build one unified experiment table from metrics and optional predictions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import read_jsonl


FIELDNAMES = [
    "version",
    "cer",
    "rr",
    "false_reject_rate",
    "false_accept_rate",
    "mean_latency_ms",
    "p95_latency_ms",
    "candidate_ratio",
    "fallback_rate",
    "pos_count",
    "neg_count",
    "official_80_score",
    "notes",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize XH-202615 experiments into one CSV")
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Repeatable: name=metrics.json or name=metrics.json|predictions.jsonl",
    )
    parser.add_argument("--output", default="output/reports/experiment_summary.csv")
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Optional repeatable note: name=short note",
    )
    return parser.parse_args()


def parse_entry(value: str) -> tuple[str, Path, Path | None]:
    if "=" not in value:
        raise SystemExit(f"Bad --entry {value!r}; expected name=metrics.json[|predictions.jsonl]")
    name, rest = value.split("=", 1)
    parts = rest.split("|", 1)
    metrics = Path(parts[0])
    predictions = Path(parts[1]) if len(parts) == 2 and parts[1] else None
    return name, metrics, predictions


def load_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics", payload)


def load_prediction_stats(path: Path | None) -> dict[str, float | str]:
    if path is None or not path.exists():
        return {
            "mean_latency_ms": "",
            "p95_latency_ms": "",
            "candidate_ratio": "",
            "fallback_rate": "",
        }

    latencies = []
    total = 0
    candidates = 0
    fallbacks = 0
    for row in read_jsonl(path):
        total += 1
        latency = row.get("latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        backend = str(row.get("asr_backend", "")).lower()
        reason = str(row.get("fallback_reason", "")).lower()
        route_reason = str(row.get("route_reason", "")).lower()
        candidates += int("enhance" in backend or "fallback" in backend or "energy" in backend)
        fallbacks += int("fallback" in backend or "fallback" in reason or "fallback" in route_reason)

    if latencies:
        mean_latency = statistics.fmean(latencies)
        sorted_latency = sorted(latencies)
        p95_index = max(0, min(len(sorted_latency) - 1, int(len(sorted_latency) * 0.95) - 1))
        p95_latency = sorted_latency[p95_index]
    else:
        mean_latency = p95_latency = ""

    return {
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "candidate_ratio": candidates / total if total else "",
        "fallback_rate": fallbacks / total if total else "",
    }


def notes_by_name(values: list[str]) -> dict[str, str]:
    notes = {}
    for value in values:
        if "=" not in value:
            continue
        name, note = value.split("=", 1)
        notes[name] = note
    return notes


def fmt(value) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def main() -> None:
    args = parse_args()
    notes = notes_by_name(args.note)
    rows = []
    for entry in args.entry:
        name, metrics_path, predictions_path = parse_entry(entry)
        metrics = load_metrics(metrics_path)
        pred_stats = load_prediction_stats(predictions_path)
        cer = float(metrics.get("avg_cer", 0.0))
        rr = float(metrics.get("avg_rr", 0.0))
        rows.append(
            {
                "version": name,
                "cer": cer,
                "rr": rr,
                "false_reject_rate": metrics.get("false_reject_rate", ""),
                "false_accept_rate": metrics.get("false_accept_rate", ""),
                "mean_latency_ms": pred_stats["mean_latency_ms"],
                "p95_latency_ms": pred_stats["p95_latency_ms"],
                "candidate_ratio": pred_stats["candidate_ratio"],
                "fallback_rate": pred_stats["fallback_rate"],
                "pos_count": metrics.get("pos_count", ""),
                "neg_count": metrics.get("neg_count", ""),
                "official_80_score": (1.0 - cer) * 40.0 + rr * 40.0,
                "notes": notes.get(name, ""),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in FIELDNAMES})
    print(f"Wrote {len(rows)} experiment rows to {out}")


if __name__ == "__main__":
    main()
