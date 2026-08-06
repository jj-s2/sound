"""Run the Phase-1 replay-only ASR/speaker policy pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.contracts import RouteAction
from xh202615.data import load_dataset
from xh202615.instrumentation import RunTraceBuilder
from xh202615.policy import PolicyConfig, ThreeActionPolicy
from xh202615.replay_backends import GlobalScoreReplayBackend, TranscriptReplayBackend


FIXTURE_PURPOSE = "contract_fixture_only_not_competition_calibration"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--splits", default="pos,neg")
    parser.add_argument("--asr-map", required=True)
    parser.add_argument("--speaker-scores", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions-out", required=True)
    parser.add_argument("--evidence-out", required=True)
    parser.add_argument("--routes-out", required=True)
    parser.add_argument("--trace-out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _atomic_prepare(path: Path, content: str, temp_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(content, encoding="utf-8", newline="\n")
    temp_paths.append(temp_path)


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def run(args: argparse.Namespace) -> dict:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("purpose") != FIXTURE_PURPOSE:
        raise ValueError(
            "phase1 replay requires config purpose "
            f"{FIXTURE_PURPOSE!r}"
        )
    policy = ThreeActionPolicy(PolicyConfig.from_dict(config["policy"]))
    enhancement_available = bool(config.get("enhancement_available", False))

    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    samples = load_dataset(args.dataset_root, splits)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("limit must be non-negative")
        samples = samples[: args.limit]

    asr = TranscriptReplayBackend(args.asr_map)
    speaker = GlobalScoreReplayBackend(args.speaker_scores)
    asr.load()
    speaker.load()

    predictions: list[dict] = []
    evidence_rows: list[dict] = []
    route_rows: list[dict] = []
    trace_builder = RunTraceBuilder("phase1-replay", "cpu")
    run_start = time.perf_counter()
    for sample in samples:
        sample_start = time.perf_counter()
        with trace_builder.stage("speaker_replay", replay=True):
            evidence = speaker.score(sample)
        with trace_builder.stage("asr_replay", replay=True):
            asr_result = asr.transcribe(sample)
        decision = policy.decide(evidence, enhancement_available)
        if decision.action == RouteAction.REJECT:
            text = ""
        else:
            # Phase 1 has no real enhancement backend; enhanced is deliberately
            # an ASR replay fallback until a measured enhancer is integrated.
            text = asr_result.text
        sample_latency_ms = (time.perf_counter() - sample_start) * 1000.0
        trace_builder.record_sample_latency(sample_latency_ms)

        evidence_rows.append(evidence.to_dict())
        route_rows.append(decision.to_dict())
        predictions.append(
            {
                "id": str(sample.id),
                "recognition_text": text,
                "action": decision.action.value,
                "reason_code": decision.reason_code,
                "asr_backend": asr_result.metadata.to_dict() if asr_result.metadata else None,
                "speaker_backend": evidence.backend.to_dict(),
                "latency_ms": sample_latency_ms,
            }
        )

    total_sec = time.perf_counter() - run_start
    trace = trace_builder.finalize(
        measurement_mode="replay",
        batch_size=1,
        warmup_count=0,
        model_load_sec=0.0,
        inference_sec=total_sec,
        total_sec=total_sec,
    )

    output_paths = [
        Path(args.predictions_out),
        Path(args.evidence_out),
        Path(args.routes_out),
        Path(args.trace_out),
    ]
    contents = [
        _jsonl(predictions),
        _jsonl(evidence_rows),
        _jsonl(route_rows),
        json.dumps(trace.to_dict(), ensure_ascii=False, indent=2) + "\n",
    ]
    temp_paths: list[Path] = []
    try:
        for path, content in zip(output_paths, contents):
            _atomic_prepare(path, content, temp_paths)
        for temp_path, output_path in zip(temp_paths, output_paths):
            temp_path.replace(output_path)
    except Exception:
        for temp_path in temp_paths:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    return {
        "count": len(samples),
        "measurement_mode": trace.measurement_mode,
        "predictions_out": str(output_paths[0]),
        "evidence_out": str(output_paths[1]),
        "routes_out": str(output_paths[2]),
        "trace_out": str(output_paths[3]),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
