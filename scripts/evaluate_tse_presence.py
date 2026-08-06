"""Evaluate / calibrate the presence-gated TSE ASR composition (R6).

Two modes:

* ``--calibrate`` (public only): sweep the presence threshold on a public
  manifest split to maximise the public Overall proxy, and write the chosen
  threshold. Public labels are used here; Dataset-A is never read.
* ``--evaluate`` (default): apply a fixed threshold (``--threshold`` or read
  from a checkpoint via ``--threshold-from-checkpoint``) and score a frozen ASR
  set with the official ``Overall = ((1-CER)+RR)/2``. For Dataset-A this is the
  blind official evaluation; for a public split it is a reporting proxy.

The presence score is source-agnostic (``--presence-jsonl`` or
``--audio-map``), so the same evaluator composes with the TSE presence head or
the existing temporal-head path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import load_dataset  # noqa: E402
from xh202615.evaluation import evaluate_rows  # noqa: E402
from xh202615.tse_presence import (  # noqa: E402
    calibrate_threshold_overall,
    gate_predictions,
    load_asr_text,
    load_presence,
    overall_at_threshold,
    overall_from_metrics,
    samples_from_manifest,
)
from scripts.run_tse_inference import read_presence_metadata  # noqa: E402


def _load_samples(args: argparse.Namespace) -> list:
    if args.dataset_root:
        return load_dataset(args.dataset_root, ("pos", "neg"))
    if args.public_manifest:
        return samples_from_manifest(args.public_manifest, args.public_split)
    raise SystemExit("provide --dataset-root (blind eval) or --public-manifest (public proxy)")


def _resolve_threshold(args: argparse.Namespace) -> tuple[float, str]:
    """Resolve the gating threshold from an explicit value or a ``.pt`` checkpoint.

    ``--threshold`` (explicit numeric) always wins. ``--threshold-from-checkpoint``
    reads ``presence_threshold`` / ``presence_threshold_source`` from a binary
    PyTorch checkpoint via :func:`read_presence_metadata` (``torch.load`` with
    ``weights_only=False``). Fails closed on a missing file, an unloadable
    checkpoint, or a checkpoint without calibrated presence metadata. No labels
    are read and no threshold is tuned here.
    """
    if args.threshold is not None:
        return float(args.threshold), "explicit_override"
    if args.threshold_from_checkpoint:
        try:
            meta = read_presence_metadata(args.threshold_from_checkpoint)
        except Exception as exc:  # noqa: BLE001 - fail closed on any load error
            raise SystemExit(
                f"cannot read checkpoint {args.threshold_from_checkpoint!r}: {exc}"
            ) from exc
        if "presence_threshold" not in meta:
            raise SystemExit(
                f"checkpoint {args.threshold_from_checkpoint!r} has no presence_threshold"
            )
        return float(meta["presence_threshold"]), meta.get(
            "presence_threshold_source", "checkpoint"
        )
    raise SystemExit("--evaluate requires --threshold or --threshold-from-checkpoint")


def _presence_path(args: argparse.Namespace) -> str:
    if args.presence_jsonl:
        return args.presence_jsonl
    if args.audio_map:
        return args.audio_map
    raise SystemExit("provide --presence-jsonl or --audio-map")


def calibrate(args: argparse.Namespace) -> dict:
    samples = _load_samples(args)
    presence = load_presence(_presence_path(args))
    asr = load_asr_text(args.asr_predictions)
    result = calibrate_threshold_overall(samples, asr, presence)
    out = {
        "threshold": result["threshold"],
        "threshold_source": result["threshold_source"],
        "metrics": result["metrics"],
        "n_pos": result["n_pos"],
        "n_neg": result["n_neg"],
        "n_candidates": result["n_candidates"],
        "asr_predictions": str(Path(args.asr_predictions).resolve(strict=False)),
        "presence_source": str(Path(_presence_path(args)).resolve(strict=False)),
        "dataset_a_used_for_training": False,
    }
    _write_output(args, out)
    print(
        f"calibrated threshold={result['threshold']:.6f} "
        f"public_overall={result['metrics']['overall']:.4f} "
        f"(CER={result['metrics']['avg_cer']:.4f}, RR={result['metrics']['avg_rr']:.4f})",
        flush=True,
    )
    return out


def evaluate(args: argparse.Namespace) -> dict:
    samples = _load_samples(args)
    presence = load_presence(_presence_path(args))
    asr = load_asr_text(args.asr_predictions)
    threshold, threshold_source = _resolve_threshold(args)
    metrics = overall_at_threshold(samples, asr, presence, threshold)
    out = {
        "threshold": threshold,
        "threshold_source": threshold_source,
        "metrics": metrics,
        "asr_predictions": str(Path(args.asr_predictions).resolve(strict=False)),
        "presence_source": str(Path(_presence_path(args)).resolve(strict=False)),
        "dataset_a_used_for_training": False,
        "label_usage": "audit_only" if args.dataset_root else "public_proxy",
    }
    if args.gated_predictions:
        predictions = gate_predictions(asr, presence, threshold)
        pred_path = Path(args.gated_predictions).expanduser().resolve(strict=False)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        out["gated_predictions"] = str(pred_path)
        # Cross-check the gated file against the official evaluator.
        report = evaluate_rows(samples, predictions, missing_policy="empty")
        official = dict(report.metrics)
        official["overall"] = overall_from_metrics(official)
        out["official_evaluator_crosscheck"] = official
    _write_output(args, out)
    print(
        f"overall={metrics['overall']:.4f} CER={metrics['avg_cer']:.4f} "
        f"RR={metrics['avg_rr']:.4f} FAR={metrics['false_accept_rate']:.4f} "
        f"FRR={metrics['false_reject_rate']:.4f} (threshold={threshold:.6f})",
        flush=True,
    )
    return out


def _write_output(args: argparse.Namespace, payload: dict) -> None:
    if not args.output:
        return
    out_path = Path(args.output).expanduser().resolve(strict=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-predictions", required=True, help="FunASR predictions JSONL (id->text)")
    parser.add_argument("--presence-jsonl", default=None, help="JSONL with id + presence_score")
    parser.add_argument("--audio-map", default=None, help="TSE audio_map.jsonl carrying presence_score")
    parser.add_argument("--output", default=None, help="write metrics/threshold JSON here")
    parser.add_argument("--gated-predictions", default=None, help="write gated prediction JSONL (evaluate mode)")
    parser.add_argument("--calibrate", action="store_true", help="sweep threshold on public samples to maximise Overall")
    parser.add_argument("--threshold", type=float, default=None, help="fixed threshold (evaluate mode)")
    parser.add_argument(
        "--threshold-from-checkpoint",
        default=None,
        help="read presence_threshold from this TSE checkpoint (evaluate mode)",
    )
    parser.add_argument("--dataset-root", default=None, help="Dataset-A root for blind official evaluation")
    parser.add_argument("--public-manifest", default=None, help="public TrainingManifest JSONL")
    parser.add_argument("--public-split", default="val", help="split to use from the public manifest")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.calibrate:
        return calibrate(args)
    return evaluate(args)


if __name__ == "__main__":
    main()
