"""R11 E0: reproducible cached gate-oracle OOF CLI and artifact writer.

This is a thin, diagnostic-only wrapper around the R11 gate-oracle evaluation.
It validates Dataset-A labels and source coverage, runs a frozen CPU model grid,
checks official-evaluator parity, and writes exactly five reproducibility
artifacts under the output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import Sample, load_dataset
from xh202615.evaluation import evaluate_rows
from xh202615.r10_selector import CandidateRow, load_candidate_bundle
from xh202615.r11_gate_oracle import (
    GATE_FEATURE_SCHEMA,
    GateModelSpec,
    build_oracle_contributions,
    default_model_specs,
    evaluate_e0,
)


# Evaluator-only sentinel for accepted negatives. It must never leave this module.
_SENTINEL_ACCEPTED_NEGATIVE = "__accepted_negative__"

# JSON-safe marker for the reject-all threshold (positive infinity).
_REJECT_ALL_THRESHOLD_MARKER = "__reject_all__"

_ARTIFACT_NAMES = {
    "manifest": "r11_e0_manifest.json",
    "summary": "r11_e0_summary.json",
    "scores": "r11_e0_scores.jsonl",
    "frontier": "r11_e0_frontier.jsonl",
    "report": "r11_e0_report.md",
}

_FIXED_DECISION_GATES = {
    "overall_high_water": 0.81,
    "worst_fold_overall_high_water": 0.77,
    "bootstrap_ci_high_falsification": 0.80,
    "rr_floor": 0.93,
}


class _NonFiniteValueError(ValueError):
    """Raised when a value that must be JSON-serializable is non-finite."""


class _StrictJSONEncoder(json.JSONEncoder):
    """JSON encoder that rejects NaN and converts infinity to a safe marker."""

    def encode(self, obj: Any) -> str:
        def visit(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, (tuple, list)):
                return [visit(item) for item in value]
            if isinstance(value, dict):
                return {str(k): visit(v) for k, v in value.items()}
            if is_dataclass(value) and not isinstance(value, type):
                return visit(asdict(value))
            if isinstance(value, float):
                if math.isnan(value):
                    raise _NonFiniteValueError("NaN is not JSON-serializable")
                if math.isinf(value):
                    if value > 0:
                        return _REJECT_ALL_THRESHOLD_MARKER
                    raise _NonFiniteValueError("negative infinity is not JSON-serializable")
            if isinstance(value, Path):
                return str(value)
            return value

        return super().encode(visit(obj))


def _jsonify(obj: Any) -> Any:
    """Recursively convert dataclasses, tuples, NumPy arrays, and Paths."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (tuple, list)):
        return [_jsonify(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonify(asdict(obj))
    if isinstance(obj, float):
        if math.isnan(obj):
            raise _NonFiniteValueError("NaN is not JSON-serializable")
        if math.isinf(obj):
            if obj > 0:
                return _REJECT_ALL_THRESHOLD_MARKER
            raise _NonFiniteValueError("negative infinity is not JSON-serializable")
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _canonical_json(obj: Any, *, allow_nan: bool = False) -> str:
    return json.dumps(
        _jsonify(obj),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=allow_nan,
        cls=_StrictJSONEncoder if not allow_nan else json.JSONEncoder,
    )


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _config_hash(
    n_outer: int,
    n_boot: int,
    seed: int,
    rr_floor: float,
    feature_schema: Sequence[str],
    model_specs: Sequence[GateModelSpec],
    fixed_gates: dict[str, Any],
) -> str:
    payload = {
        "n_outer": n_outer,
        "n_boot": n_boot,
        "seed": seed,
        "rr_floor": rr_floor,
        "feature_schema": list(feature_schema),
        "model_specs": _jsonify(tuple(model_specs)),
        "fixed_gates": fixed_gates,
    }
    canonical = _canonical_json(payload, allow_nan=False)
    return _sha256_hex(canonical.encode("utf-8"))


def _source_digest(paths: dict[str, Path]) -> str:
    """Hash all consumed source bytes (candidate sources, manifest, and dataset)."""

    digest_parts: dict[str, str] = {}
    source_files = [
        ("candidate_fusion", paths["candidate_fusion"]),
        ("tse_asr", paths["tse_asr"]),
        ("audio_map", paths["audio_map"]),
        ("r3_predictions", paths["r3_predictions"]),
        ("group_manifest", paths["group_manifest"]),
    ]
    dataset_root = Path(paths["dataset_root"])
    for split in ("pos", "neg"):
        source_files.append((f"dataset_{split}", dataset_root / f"{split}.jsonl"))

    for key, path in source_files:
        data = Path(path).read_bytes()
        digest_parts[key] = _sha256_hex(data)

    canonical = json.dumps(digest_parts, sort_keys=True, ensure_ascii=False)
    return _sha256_hex(canonical.encode("utf-8"))


def _read_jsonl_ids(path: Path) -> set[str]:
    """Return the exact unique IDs from a JSONL file."""

    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sid = str(record.get("id", ""))
            if not sid:
                raise ValueError(f"{path}:{line_no} is missing an id")
            if sid in seen:
                raise ValueError(f"duplicate id {sid!r} in {path}")
            seen.add(sid)
    return seen


def _load_manifest_labels_and_groups(path: Path) -> tuple[dict[str, str | None], dict[str, str], list[str]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    rows = manifest.get("rows", [])
    if not rows:
        raise ValueError("group manifest contains no rows")

    labels: dict[str, str | None] = {}
    groups: dict[str, str] = {}
    seen_ids: set[str] = set()
    for row in rows:
        sid = str(row["id"])
        if sid in seen_ids:
            raise ValueError(f"duplicate manifest id {sid!r}")
        seen_ids.add(sid)
        labels[sid] = row.get("label")
        wake_component = row.get("wake_component")
        if wake_component is None or str(wake_component) == "":
            raise ValueError(f"missing or empty wake_component for id {sid!r}")
        groups[sid] = str(wake_component)

    sample_ids = sorted(labels, key=lambda x: int(x) if x.isdigit() else x)
    return labels, groups, sample_ids


def _build_samples(
    rows: Sequence[CandidateRow],
    labels: dict[str, str | None],
) -> list[Sample]:
    return [
        Sample(
            id=row.id,
            split=row.split,
            wakeup_audio=Path("."),
            wakeup_text="",
            command_audio=row.original_command_audio or Path("."),
            label=labels[row.id],
        )
        for row in rows
    ]


def _build_official_predictions(
    rows: Sequence[CandidateRow],
    labels: dict[str, str | None],
    scores: np.ndarray,
    threshold: float,
    contributions,
) -> list[dict[str, str]]:
    """Construct official-evaluator predictions for the selected gate-oracle point.

    Accepted positives use the oracle-selected candidate. Rejected positives and
    rejected negatives use empty text. Accepted negatives use the evaluator-only
    sentinel so that they register as false accepts.
    """

    accepted = np.asarray(scores, dtype=np.float64) >= threshold
    predictions: list[dict[str, str]] = []
    for row, is_accepted, action in zip(rows, accepted, contributions.chosen_actions):
        label = labels[row.id]
        if label is None:
            text = _SENTINEL_ACCEPTED_NEGATIVE if is_accepted else ""
        else:
            text = row.texts.get(action, "") if is_accepted else ""
        predictions.append({"id": row.id, "recognition_text": text})
    return predictions


def _check_evaluator_parity(
    rows: Sequence[CandidateRow],
    labels: dict[str, str | None],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Verify that the selected point matches the official evaluator metrics."""

    selected = result["selected_point"]
    model_name = str(selected["model"])
    threshold = float(selected["threshold"])
    scores = result["scores_by_model"][model_name]
    contributions = build_oracle_contributions(rows, labels)
    predictions = _build_official_predictions(rows, labels, scores, threshold, contributions)
    samples = _build_samples(rows, labels)
    official = evaluate_rows(samples, predictions, missing_policy="empty").metrics

    tolerance = 1e-9
    if abs(float(selected["cer"]) - official["avg_cer"]) > tolerance:
        raise ValueError(
            f"evaluator parity failed on cer: "
            f"custom={selected['cer']!r}, official={official['avg_cer']!r}"
        )
    if abs(float(selected["rr"]) - official["avg_rr"]) > tolerance:
        raise ValueError(
            f"evaluator parity failed on rr: "
            f"custom={selected['rr']!r}, official={official['avg_rr']!r}"
        )

    expected_overall = ((1.0 - official["avg_cer"]) + official["avg_rr"]) / 2.0
    if abs(float(selected["overall"]) - expected_overall) > tolerance:
        raise ValueError(
            f"evaluator parity failed on overall: "
            f"custom={selected['overall']!r}, official={expected_overall!r}"
        )

    negative_count = int((~np.asarray(contributions.is_positive)).sum())
    accepted_negative_count = int(selected["accepted_negatives"])
    if negative_count:
        expected_far = accepted_negative_count / negative_count
        if abs(official["false_accept_rate"] - expected_far) > tolerance:
            raise ValueError(
                f"accepted negatives did not register as false accepts: "
                f"far={official['false_accept_rate']!r}, expected={expected_far!r}"
            )

    return official


def _next_branch(decision: str) -> str:
    if decision == "continue_cached":
        return "E1 — complete cached negative control"
    if decision == "falsified_cached":
        return "E2 — FireRed zero-shot and fused gate-oracle"
    return "E2/E3 — FireRed zero-shot and fused gate-oracle"


def write_e0_artifacts(
    result: dict[str, Any],
    rows: Sequence[CandidateRow],
    groups: Sequence[object],
    paths: dict[str, Any],
    output_root: Path,
) -> dict[str, Path]:
    """Write the five fixed E0 reproducibility artifacts.

    Returns a mapping of artifact role to written path.
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    n_outer = int(paths["n_outer"])
    n_boot = int(paths["n_boot"])
    seed = int(paths["seed"])
    rr_floor = float(paths["rr_floor"])

    config_hash = _config_hash(
        n_outer=n_outer,
        n_boot=n_boot,
        seed=seed,
        rr_floor=rr_floor,
        feature_schema=GATE_FEATURE_SCHEMA,
        model_specs=result["model_specs"],
        fixed_gates=_FIXED_DECISION_GATES,
    )
    source_digest = _source_digest({k: Path(v) for k, v in paths.items() if k not in {"n_outer", "n_boot", "seed", "rr_floor"}})

    selected = dict(result["selected_point"])
    selected_model = str(selected["model"])
    selected_threshold = float(selected["threshold"])
    selected_threshold_display = (
        _REJECT_ALL_THRESHOLD_MARKER if math.isinf(selected_threshold) else selected_threshold
    )

    bootstrap = result["bootstrap"]
    bootstrap_summary = {
        "n_boot": int(bootstrap["n_boot"]),
        "n_groups": int(bootstrap["n_groups"]),
        "max_attempts": int(bootstrap["max_attempts"]),
        "attempted_replicates": int(bootstrap["attempted_replicates"]),
        "rejected_replicates": int(bootstrap["rejected_replicates"]),
        "overall_mean": float(bootstrap["overall_mean"]),
        "ci_low": float(bootstrap["ci_low"]),
        "ci_high": float(bootstrap["ci_high"]),
    }

    resolved_paths = {
        key: str(Path(value).resolve())
        for key, value in paths.items()
        if key not in {"n_outer", "n_boot", "seed", "rr_floor"}
    }

    manifest = {
        "config_hash": config_hash,
        "source_digest": source_digest,
        "resolved_paths": resolved_paths,
        "n_outer": n_outer,
        "n_boot": n_boot,
        "seed": seed,
        "rr_floor": rr_floor,
        "feature_schema": list(GATE_FEATURE_SCHEMA),
        "model_specs": _jsonify(result["model_specs"]),
        "decision": result["decision"],
        "diagnostic_only": result["diagnostic_only"],
        "global_threshold_deployable": result["global_threshold_deployable"],
        "selected_point": _jsonify(selected),
        "worst_fold": _jsonify(result["worst_fold"]),
        "fold_metrics": _jsonify(result["fold_metrics"]),
        "bootstrap_summary": bootstrap_summary,
        "input_validation": {
            "dataset_manifest_label_parity": True,
            "exact_source_id_sets": True,
            "evaluator_parity": True,
        },
    }

    summary = {
        "status": "success",
        "decision": result["decision"],
        "diagnostic_only": result["diagnostic_only"],
        "global_threshold_deployable": result["global_threshold_deployable"],
        "selected_model": selected_model,
        "selected_threshold": selected_threshold_display,
        "cer": float(selected["cer"]),
        "rr": float(selected["rr"]),
        "overall": float(selected["overall"]),
        "accepted_positives": float(selected["accepted_positives"]),
        "accepted_negatives": float(selected["accepted_negatives"]),
        "worst_fold": {
            "fold_index": int(result["worst_fold"]["fold_index"]),
            "overall": float(result["worst_fold"]["overall"]),
            "cer": float(result["worst_fold"]["cer"]),
            "rr": float(result["worst_fold"]["rr"]),
        },
        "bootstrap_ci": bootstrap_summary,
        "config_hash": config_hash,
        "source_digest": source_digest,
        "output_files": [],
    }

    manifest_path = output_root / _ARTIFACT_NAMES["manifest"]
    summary_path = output_root / _ARTIFACT_NAMES["summary"]
    scores_path = output_root / _ARTIFACT_NAMES["scores"]
    frontier_path = output_root / _ARTIFACT_NAMES["frontier"]
    report_path = output_root / _ARTIFACT_NAMES["report"]

    manifest_path.write_text(
        _canonical_json(manifest, allow_nan=False),
        encoding="utf-8",
    )
    summary["output_files"] = [
        str(manifest_path.resolve()),
        str(summary_path.resolve()),
        str(scores_path.resolve()),
        str(frontier_path.resolve()),
        str(report_path.resolve()),
    ]
    summary_path.write_text(
        _canonical_json(summary, allow_nan=False),
        encoding="utf-8",
    )

    scores_by_model = result["scores_by_model"]
    model_names = sorted(scores_by_model)
    fold_assignments = np.asarray(result["fold_assignments"], dtype=np.int64)
    with scores_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record = {
                "id": row.id,
                "group": str(groups[index]),
                "fold": int(fold_assignments[index]),
            }
            for model_name in model_names:
                score = float(scores_by_model[model_name][index])
                if not math.isfinite(score) or score < 0.0 or score > 1.0:
                    raise ValueError(
                        f"invalid OOF probability for {model_name} at {row.id}: {score}"
                    )
                record[model_name] = score
            handle.write(_canonical_json(record, allow_nan=False) + "\n")

    with frontier_path.open("w", encoding="utf-8") as handle:
        for point in result["frontier"]:
            handle.write(_canonical_json(point, allow_nan=False) + "\n")

    report_lines = [
        "# R11 E0 Gate-Oracle OOF Report",
        "",
        f"Config hash: `{config_hash}`",
        f"Source digest: `{source_digest}`",
        f"Outer folds: {n_outer}, bootstrap resamples: {n_boot}, seed: {seed}",
        "",
        "## Selected point",
        f"- Decision: {result['decision']}",
        f"- Next branch: {_next_branch(result['decision'])}",
        f"- Model: `{selected_model}`",
        f"- Threshold: `{selected_threshold_display}`",
        f"- CER: {float(selected['cer']):.6f}",
        f"- RR: {float(selected['rr']):.6f}",
        f"- Overall: {float(selected['overall']):.6f}",
        f"- Accepted positives: {int(selected['accepted_positives'])}",
        f"- Accepted negatives: {int(selected['accepted_negatives'])}",
        "",
        "## Worst outer fold",
        f"- Fold {int(result['worst_fold']['fold_index'])}: Overall {float(result['worst_fold']['overall']):.6f}, CER {float(result['worst_fold']['cer']):.6f}, RR {float(result['worst_fold']['rr']):.6f}",
        "",
        "## Group-bootstrap 95% CI for Overall",
        f"- Mean: {bootstrap_summary['overall_mean']:.6f}",
        f"- CI: [{bootstrap_summary['ci_low']:.6f}, {bootstrap_summary['ci_high']:.6f}]",
        "",
        "## Model grid",
    ]
    for spec in result["model_specs"]:
        report_lines.append(f"- `{spec.name}` ({spec.family})")
    report_lines.extend([
        "",
        "## Data boundary",
        "- Labels used only for target generation, oracle candidate selection, and held-out scoring.",
        "- Gate input features are inference-only post-ASR and acoustic values.",
        "- No Dataset-B, hidden labels, or leaderboard feedback used.",
        "- Reference label text is absent from score and frontier artifacts.",
        "",
        f"Artifacts: {manifest_path}, {summary_path}, {scores_path}, {frontier_path}, {report_path}",
    ])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "scores": scores_path,
        "frontier": frontier_path,
        "report": report_path,
    }


def _validate_inputs(args: argparse.Namespace) -> tuple[list[CandidateRow], dict[str, str | None], list[str], dict[str, Any]]:
    """Load and validate all inputs before any output is created."""

    required_files = [
        args.candidate_fusion,
        args.tse_asr,
        args.audio_map,
        args.r3_predictions,
        args.group_manifest,
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"required input file not found: {path}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"required dataset root not found: {args.dataset_root}")
    for split in ("pos", "neg"):
        split_path = args.dataset_root / f"{split}.jsonl"
        if not split_path.is_file():
            raise FileNotFoundError(f"required split file not found: {split_path}")

    samples = load_dataset(args.dataset_root, splits=("pos", "neg"))
    dataset_labels = {str(s.id): s.label for s in samples}
    dataset_ids = set(dataset_labels)

    manifest_labels, manifest_groups, manifest_ids = _load_manifest_labels_and_groups(
        args.group_manifest
    )
    if set(manifest_ids) != dataset_ids:
        raise ValueError(
            f"manifest IDs do not match dataset IDs: "
            f"manifest={len(set(manifest_ids))}, dataset={len(dataset_ids)}"
        )
    for sid in manifest_ids:
        if dataset_labels[sid] != manifest_labels[sid]:
            raise ValueError(
                f"dataset label for {sid!r} disagrees with frozen manifest label"
            )

    rows_by_id, loaded_groups, loaded_labels = load_candidate_bundle(
        args.candidate_fusion,
        args.tse_asr,
        args.audio_map,
        args.group_manifest,
        r3_predictions_path=args.r3_predictions,
    )
    if dataset_labels != loaded_labels:
        raise ValueError("group manifest labels disagree with dataset loader labels")

    # Exact unique source ID sets.
    fusion_ids = _read_jsonl_ids(args.candidate_fusion)
    tse_ids = _read_jsonl_ids(args.tse_asr)
    audio_ids = _read_jsonl_ids(args.audio_map)
    r3_ids = _read_jsonl_ids(args.r3_predictions)
    expected_ids = set(manifest_ids)
    for name, ids in [
        ("candidate_fusion", fusion_ids),
        ("tse_asr", tse_ids),
        ("audio_map", audio_ids),
        ("r3_predictions", r3_ids),
    ]:
        if ids != expected_ids:
            raise ValueError(
                f"{name} IDs do not match manifest IDs exactly: "
                f"{name}={len(ids)}, manifest={len(expected_ids)}"
            )

    # Reject missing/empty wake_component fallback.
    for sid in manifest_ids:
        if loaded_groups[sid] != manifest_groups[sid]:
            raise ValueError(
                f"loaded group for {sid!r} does not match manifest wake_component"
            )

    sample_ids = sorted(rows_by_id, key=lambda x: int(x) if x.isdigit() else x)
    rows = [rows_by_id[sid] for sid in sample_ids]
    groups = [manifest_groups[sid] for sid in sample_ids]
    labels = dataset_labels

    paths = {
        "dataset_root": args.dataset_root,
        "candidate_fusion": args.candidate_fusion,
        "tse_asr": args.tse_asr,
        "audio_map": args.audio_map,
        "r3_predictions": args.r3_predictions,
        "group_manifest": args.group_manifest,
        "n_outer": args.n_outer,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "rr_floor": args.rr_floor,
    }

    return rows, labels, groups, paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="R11 E0 reproducible cached gate-oracle OOF evaluation"
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasetA" / "datasetA")
    parser.add_argument("--candidate-fusion", type=Path, default=REPO_ROOT / "output" / "asr" / "candidate_fusion_smoke.jsonl")
    parser.add_argument("--tse-asr", type=Path, default=REPO_ROOT / "output" / "training_r9" / "datasetA_tse" / "asr_predictions.jsonl")
    parser.add_argument("--audio-map", type=Path, default=REPO_ROOT / "output" / "training_r9" / "datasetA_tse" / "audio_map.jsonl")
    parser.add_argument("--r3-predictions", type=Path, default=REPO_ROOT / "output" / "evaluations" / "r3_temporal_on_datasetA_gated.jsonl")
    parser.add_argument("--group-manifest", type=Path, default=REPO_ROOT / ".superpowers" / "sdd" / "2026-08-07-r9-overall-08-arena" / "datasetA_group_manifest_v1.json")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output" / "r11_gate_oracle_e0")
    parser.add_argument("--n-outer", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--rr-floor", type=float, default=0.93)
    args = parser.parse_args(argv)

    rows, labels, groups, paths = _validate_inputs(args)

    result = evaluate_e0(
        rows,
        labels,
        groups,
        n_splits=args.n_outer,
        seed=args.seed,
        rr_floor=args.rr_floor,
        n_boot=args.n_boot,
    )

    _check_evaluator_parity(rows, labels, result)

    files = write_e0_artifacts(result, rows, groups, paths, args.output_root)

    print(f"R11 E0 gate-oracle complete. Decision={result['decision']}")
    print(f"  Selected model: {result['selected_point']['model']}")
    print(f"  Selected threshold: {result['selected_point']['threshold']}")
    print(f"  Overall: {result['selected_point']['overall']:.6f}")
    print(f"  RR: {result['selected_point']['rr']:.6f}")
    print(f"  CER: {result['selected_point']['cer']:.6f}")
    print(f"  Summary: {files['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
