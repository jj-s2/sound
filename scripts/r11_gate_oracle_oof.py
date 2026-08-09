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
import os
import secrets
import shutil
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import Sample, load_dataset, load_split
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

_ARTIFACT_KIND = "r11_e0_gate_oracle"
_SCHEMA_VERSION = "v1"

_ARTIFACT_NAMES = {
    "manifest": "e0_manifest.json",
    "summary": "e0_summary.json",
    "scores": "e0_oof_scores.jsonl",
    "frontier": "e0_frontier.jsonl",
    "report": "e0_report.md",
}

_FIXED_DECISION_GATES = {
    "overall_high_water": 0.81,
    "worst_fold_overall_high_water": 0.77,
    "bootstrap_ci_high_falsification": 0.80,
    "rr_floor": 0.93,
}


class _NonFiniteValueError(ValueError):
    """Raised when a value that must be JSON-serializable is non-finite."""


def _digestable_float(value: float) -> float | str:
    """Represent a float for canonical hashing; NaN becomes a stable marker."""
    if isinstance(value, float):
        if math.isnan(value):
            return "__nan__"
        if math.isinf(value):
            return "__inf__" if value > 0 else "__neg_inf__"
    return value


def _jsonify(obj: Any) -> Any:
    """Recursively convert dataclasses, tuples, NumPy arrays, and Paths.

    Unlike a generic JSON encoder, this function rejects NaN and treats any
    non-threshold infinity as an error. Threshold fields are markerized by the
    caller before serialization.
    """
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
            raise _NonFiniteValueError(
                "non-threshold infinity is not JSON-serializable"
            )
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _apply_threshold_markers(obj: Any) -> Any:
    """Replace every ``threshold`` field with the safe marker when it is +inf."""
    if isinstance(obj, dict):
        return {
            k: _serialize_threshold(v)
            if k == "threshold"
            else _apply_threshold_markers(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_apply_threshold_markers(item) for item in obj]
    return obj


def _serialize_threshold(value: Any) -> Any:
    """Encode a standalone reject-all threshold as a JSON-safe boundary marker."""
    if isinstance(value, float) and math.isinf(value) and value > 0:
        return _REJECT_ALL_THRESHOLD_MARKER
    return value


def _canonical_json(obj: Any, *, allow_nan: bool = False) -> str:
    return json.dumps(
        _jsonify(obj),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=allow_nan,
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


def _source_digest(paths: dict[str, Path]) -> dict[str, Any]:
    """Return per-source and aggregate SHA-256 of all consumed raw bytes."""

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

    aggregate = _sha256_hex(
        json.dumps(digest_parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    return {"per_source": digest_parts, "aggregate": aggregate}


def _feature_state_digest(rows: Sequence[CandidateRow]) -> dict[str, Any]:
    """Canonical digest of the fully joined feature-bearing row state.

    Covers every candidate text and every cached acoustic feature (including
    ``cmd_duration_sec`` derived from the WAV header), so a mutation that
    changes gate inputs changes this digest even when source file bytes are
    unchanged.
    """
    per_id: dict[str, str] = {}
    for row in rows:
        state = {
            "id": row.id,
            "texts": {k: row.texts[k] for k in sorted(row.texts)},
            "audio_features": {
                k: _digestable_float(row.audio_features[k])
                for k in sorted(row.audio_features)
            },
        }
        canonical = json.dumps(state, sort_keys=True, ensure_ascii=False)
        per_id[row.id] = _sha256_hex(canonical.encode("utf-8"))

    aggregate = _sha256_hex(
        json.dumps(per_id, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    return {"aggregate": aggregate, "per_id": per_id}


def _read_jsonl_ids(path: Path) -> list[str]:
    """Return the exact unique IDs from a JSONL file in file order."""

    seen: set[str] = set()
    ids: list[str] = []
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
            ids.append(sid)
    return ids


def _source_id_sets(
    paths: dict[str, Path],
    dataset_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Return counts and sorted-ID digests for every consumed source."""

    def _id_digest(ids: Sequence[str]) -> str:
        canonical = json.dumps(
            sorted(ids, key=lambda x: int(x) if x.isdigit() else x),
            ensure_ascii=False,
        )
        return _sha256_hex(canonical.encode("utf-8"))

    def _ids_for_jsonl(path: Path) -> tuple[int, str]:
        ids = _read_jsonl_ids(path)
        return len(ids), _id_digest(ids)

    def _ids_for_manifest(path: Path) -> tuple[int, str]:
        manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        ids = [str(row["id"]) for row in manifest.get("rows", [])]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate ids in group manifest")
        return len(ids), _id_digest(ids)

    cf_count, cf_digest = _ids_for_jsonl(paths["candidate_fusion"])
    tse_count, tse_digest = _ids_for_jsonl(paths["tse_asr"])
    audio_count, audio_digest = _ids_for_jsonl(paths["audio_map"])
    r3_count, r3_digest = _ids_for_jsonl(paths["r3_predictions"])
    manifest_count, manifest_digest = _ids_for_manifest(paths["group_manifest"])

    return {
        "candidate_fusion": {"count": cf_count, "id_digest": cf_digest},
        "tse_asr": {"count": tse_count, "id_digest": tse_digest},
        "audio_map": {"count": audio_count, "id_digest": audio_digest},
        "r3_predictions": {"count": r3_count, "id_digest": r3_digest},
        "group_manifest": {"count": manifest_count, "id_digest": manifest_digest},
        "dataset": {"count": len(dataset_ids), "id_digest": _id_digest(dataset_ids)},
    }


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


def _validate_dataset_uniqueness(root: Path) -> list[Sample]:
    """Load Dataset-A and reject any duplicate ID, including across splits."""

    pos_samples = load_split(str(root), "pos")
    neg_samples = load_split(str(root), "neg")
    samples = pos_samples + neg_samples
    seen: set[str] = set()
    for sample in samples:
        sid = str(sample.id)
        if sid in seen:
            raise ValueError(f"duplicate Dataset-A id {sid!r}")
        seen.add(sid)
    return samples


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
    samples: Sequence[Sample] | None = None,
) -> dict[str, Any]:
    """Verify that the selected point matches the official evaluator metrics."""

    selected = result["selected_point"]
    model_name = str(selected["model"])
    threshold = float(selected["threshold"])
    scores = result["scores_by_model"][model_name]
    contributions = build_oracle_contributions(rows, labels)
    predictions = _build_official_predictions(rows, labels, scores, threshold, contributions)
    official_samples = _build_samples(rows, labels) if samples is None else list(samples)
    official = dict(evaluate_rows(official_samples, predictions, missing_policy="empty").metrics)
    official["overall"] = ((1.0 - official["avg_cer"]) + official["avg_rr"]) / 2.0

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
        return (
            "E1 — complete cached negative control: compare positive-only "
            "expected-CER regression and LambdaMART; do not tune cached-gate "
            "hyperparameters further."
        )
    if decision == "falsified_cached":
        return (
            "Stop cached-gate tuning; go directly to E2 — FireRedChat-pVAD "
            "zero-shot and fused gate-oracle."
        )
    return "Begin E2 — FireRedChat-pVAD zero-shot and fused gate-oracle."


def _validate_result(
    result: dict[str, Any],
    rows: Sequence[CandidateRow],
    groups: Sequence[object],
) -> None:
    """Fail-closed validation of the evaluation result before any file is written."""

    if len(rows) != len(groups):
        raise ValueError("rows and groups must have equal lengths")

    n_rows = len(rows)
    required_keys = (
        "decision",
        "selected_point",
        "worst_fold",
        "fold_metrics",
        "frontier",
        "bootstrap",
        "model_specs",
        "scores_by_model",
        "fold_assignments",
        "fold_metadata",
        "official_metrics",
    )
    for key in required_keys:
        if key not in result:
            raise ValueError(f"result is missing required key {key!r}")

    official = result["official_metrics"]
    for key in ("avg_cer", "avg_rr", "overall", "false_reject_rate", "false_accept_rate"):
        if key not in official:
            raise ValueError(f"official_metrics is missing {key!r}")
        if not math.isfinite(float(official[key])):
            raise ValueError(f"official metric {key} is not finite: {official[key]!r}")
    if float(official["avg_cer"]) < 0.0:
        raise ValueError(f"official metric avg_cer is negative: {official['avg_cer']!r}")
    for key in ("avg_rr", "false_reject_rate", "false_accept_rate"):
        value = float(official[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"official metric {key} is outside [0, 1]: {official[key]!r}")

    selected = result["selected_point"]
    for key, custom_key in (("avg_cer", "cer"), ("avg_rr", "rr"), ("overall", "overall")):
        delta = abs(float(selected[custom_key]) - float(official[key]))
        if delta > 1e-9:
            raise ValueError(
                f"custom/official {key} disagreement: custom={selected[custom_key]!r}, "
                f"official={official[key]!r}, delta={delta}"
            )

    scores_by_model = result["scores_by_model"]
    if not scores_by_model:
        raise ValueError("scores_by_model must not be empty")
    for model_name, scores in scores_by_model.items():
        arr = np.asarray(scores, dtype=np.float64)
        if arr.shape != (n_rows,):
            raise ValueError(
                f"scores for {model_name} have shape {arr.shape}, expected ({n_rows},)"
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"scores for {model_name} contain non-finite values")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError(f"scores for {model_name} are outside [0, 1]")

    fold_assignments = np.asarray(result["fold_assignments"], dtype=np.int64)
    if fold_assignments.shape != (n_rows,):
        raise ValueError(
            f"fold_assignments have shape {fold_assignments.shape}, expected ({n_rows},)"
        )
    if np.any(fold_assignments < 0):
        raise ValueError("fold_assignments contains negative fold indices")
    n_folds = int(fold_assignments.max()) + 1
    for fold_index in range(n_folds):
        if not np.any(fold_assignments == fold_index):
            raise ValueError(f"fold {fold_index} has no assigned rows")
    if len(np.unique(fold_assignments)) != n_folds:
        raise ValueError("fold_assignments contains unexpected fold indices")

    required_frontier_keys = {
        "model",
        "threshold",
        "cer",
        "rr",
        "overall",
        "accepted_positives",
        "accepted_negatives",
    }
    for index, point in enumerate(result["frontier"]):
        if not required_frontier_keys.issubset(point):
            missing = required_frontier_keys - set(point)
            raise ValueError(f"frontier point {index} is missing keys {missing}")
        for key in ("cer", "rr", "overall"):
            value = float(point[key])
            if not math.isfinite(value):
                raise ValueError(
                    f"frontier point {index} has non-finite {key}: {value}"
                )

    required_selected_keys = {
        "model",
        "threshold",
        "cer",
        "rr",
        "overall",
        "accepted_positives",
        "accepted_negatives",
        "diagnostic_only",
        "deployable",
    }
    if not required_selected_keys.issubset(result["selected_point"]):
        missing = required_selected_keys - set(result["selected_point"])
        raise ValueError(f"selected_point is missing keys {missing}")

    for key in ("cer", "rr", "overall"):
        value = float(result["selected_point"][key])
        if not math.isfinite(value):
            raise ValueError(f"selected_point has non-finite {key}: {value}")

    # Fold-metadata validation: exact once-only coverage, bounds, assignment
    # agreement, train/test complement, row-derived group sets, zero intersections.
    fold_metadata = result["fold_metadata"]
    required_meta_keys = {
        "fold_index",
        "train_indices",
        "test_indices",
        "train_groups",
        "test_groups",
    }
    seen_fold_indices: set[int] = set()
    all_test_indices: list[int] = []
    all_row_indices = set(range(n_rows))

    for fold in fold_metadata:
        if not required_meta_keys.issubset(fold):
            missing = required_meta_keys - set(fold)
            raise ValueError(f"fold metadata missing keys {missing}")

        fold_index = int(fold["fold_index"])
        if fold_index < 0 or fold_index >= n_folds:
            raise ValueError(f"fold_index {fold_index} out of range")
        if fold_index in seen_fold_indices:
            raise ValueError(f"duplicate fold_index {fold_index}")
        seen_fold_indices.add(fold_index)

        train = np.asarray(fold["train_indices"], dtype=np.int64)
        test = np.asarray(fold["test_indices"], dtype=np.int64)
        if train.ndim != 1 or test.ndim != 1:
            raise ValueError("fold train/test indices must be one-dimensional")
        if not np.all((train >= 0) & (train < n_rows)):
            raise ValueError(f"fold {fold_index} train indices out of bounds")
        if not np.all((test >= 0) & (test < n_rows)):
            raise ValueError(f"fold {fold_index} test indices out of bounds")

        train_set = set(train.tolist())
        test_set = set(test.tolist())
        if len(train_set) != len(train):
            raise ValueError(f"fold {fold_index} has duplicate train indices")
        if len(test_set) != len(test):
            raise ValueError(f"fold {fold_index} has duplicate test indices")
        if train_set & test_set:
            raise ValueError(f"fold {fold_index} train/test indices overlap")
        if train_set | test_set != all_row_indices:
            raise ValueError(f"fold {fold_index} train/test indices do not cover all rows")

        if not np.all(fold_assignments[test] == fold_index):
            raise ValueError(
                f"fold {fold_index} test_indices disagree with fold_assignments"
            )

        expected_train_groups = {groups[i] for i in train_set}
        expected_test_groups = {groups[i] for i in test_set}
        if set(fold["train_groups"]) != expected_train_groups:
            raise ValueError(f"fold {fold_index} train_groups do not match row-derived groups")
        if set(fold["test_groups"]) != expected_test_groups:
            raise ValueError(f"fold {fold_index} test_groups do not match row-derived groups")
        if expected_train_groups & expected_test_groups:
            raise ValueError(f"fold {fold_index} train/test groups intersect")

        all_test_indices.extend(test.tolist())

    if set(seen_fold_indices) != set(range(n_folds)):
        raise ValueError("fold_metadata fold indices do not match fold_assignments")
    if len(all_test_indices) != n_rows or len(set(all_test_indices)) != n_rows:
        raise ValueError("test indices do not cover each row exactly once")


def _build_coverage(rows: Sequence[CandidateRow], result: dict[str, Any]) -> dict[str, Any]:
    fold_assignments = np.asarray(result["fold_assignments"], dtype=np.int64)
    return {
        "n_rows_total": len(rows),
        "n_rows_covered": len(rows),
        "once_only": True,
        "n_folds": int(fold_assignments.max()) + 1,
        "sample_ids": [row.id for row in rows],
    }


def _build_fold_metadata_evidence(result: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for fold in result["fold_metadata"]:
        train_groups = set(fold["train_groups"])
        test_groups = set(fold["test_groups"])
        evidence.append(
            {
                "fold_index": int(fold["fold_index"]),
                "n_train": len(fold["train_indices"]),
                "n_test": len(fold["test_indices"]),
                "train_groups": sorted(train_groups, key=str),
                "test_groups": sorted(test_groups, key=str),
                "group_disjoint": len(train_groups & test_groups) == 0,
            }
        )
    return evidence


def _is_recognized_output(path: Path) -> bool:
    """True only if the directory contains exactly this CLI's five artifact files
    with the approved identity/schema markers in manifest and summary."""
    if not path.is_dir():
        return False
    expected_names = set(_ARTIFACT_NAMES.values())
    if {p.name for p in path.iterdir()} != expected_names:
        return False
    for name in expected_names:
        child = path / name
        if not child.is_file():
            return False

    try:
        manifest = json.loads((path / _ARTIFACT_NAMES["manifest"]).read_text(encoding="utf-8"))
        summary = json.loads((path / _ARTIFACT_NAMES["summary"]).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    for obj in (manifest, summary):
        if obj.get("artifact_kind") != _ARTIFACT_KIND:
            return False
        if obj.get("schema_version") != _SCHEMA_VERSION:
            return False
    return True


def _unique_sibling(parent: Path, prefix: str) -> Path:
    """Return a non-existent path under ``parent`` with the given prefix."""
    while True:
        candidate = parent / f"{prefix}.{secrets.token_hex(8)}"
        if not candidate.exists():
            return candidate


def write_e0_artifacts(
    result: dict[str, Any],
    rows: Sequence[CandidateRow],
    groups: Sequence[object],
    paths: dict[str, Any],
    output_root: Path,
) -> dict[str, Path]:
    """Write the five fixed E0 reproducibility artifacts atomically.

    The result is fully validated before any file is written. All artifacts are
    staged in a uniquely named sibling directory. If a recognizable prior output
    root exists, it is moved to a unique backup and restored if publication fails.
    """

    output_root = Path(output_root)
    _validate_result(result, rows, groups)

    n_outer = int(paths["n_outer"])
    n_boot = int(paths["n_boot"])
    seed = int(paths["seed"])
    rr_floor = float(paths["rr_floor"])
    path_values = {k: Path(v) for k, v in paths.items() if k not in {"n_outer", "n_boot", "seed", "rr_floor"}}

    official = result["official_metrics"]
    config_hash = _config_hash(
        n_outer=n_outer,
        n_boot=n_boot,
        seed=seed,
        rr_floor=rr_floor,
        feature_schema=GATE_FEATURE_SCHEMA,
        model_specs=result["model_specs"],
        fixed_gates=_FIXED_DECISION_GATES,
    )
    source_digest = _source_digest(path_values)
    feature_state_digest = _feature_state_digest(rows)
    source_id_sets = _source_id_sets(path_values, [row.id for row in rows])
    coverage = _build_coverage(rows, result)
    fold_metadata_evidence = _build_fold_metadata_evidence(result)

    selected = dict(result["selected_point"])
    selected_model = str(selected["model"])
    selected_threshold = float(selected["threshold"])
    selected_threshold_display = _serialize_threshold(selected_threshold)

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

    custom = {
        "avg_cer": float(selected["cer"]),
        "avg_rr": float(selected["rr"]),
        "overall": float(selected["overall"]),
        "accepted_positives": float(selected["accepted_positives"]),
        "accepted_negatives": float(selected["accepted_negatives"]),
    }
    deltas = {
        "avg_cer": abs(custom["avg_cer"] - float(official["avg_cer"])),
        "avg_rr": abs(custom["avg_rr"] - float(official["avg_rr"])),
        "overall": abs(custom["overall"] - float(official["overall"])),
    }
    parity = {key: value <= 1e-9 for key, value in deltas.items()}

    manifest = {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "config_hash": config_hash,
        "source_digest": source_digest,
        "feature_state_digest": feature_state_digest,
        "resolved_paths": resolved_paths,
        "source_id_sets": source_id_sets,
        "coverage": coverage,
        "fold_metadata": _apply_threshold_markers(fold_metadata_evidence),
        "n_outer": n_outer,
        "n_boot": n_boot,
        "seed": seed,
        "rr_floor": rr_floor,
        "feature_schema": list(GATE_FEATURE_SCHEMA),
        "model_specs": _jsonify(result["model_specs"]),
        "decision": result["decision"],
        "diagnostic_only": result["diagnostic_only"],
        "global_threshold_deployable": result["global_threshold_deployable"],
        "selected_point": _jsonify(_apply_threshold_markers(selected)),
        "worst_fold": _jsonify(_apply_threshold_markers(result["worst_fold"])),
        "fold_metrics": _jsonify(_apply_threshold_markers(result["fold_metrics"])),
        "bootstrap_summary": bootstrap_summary,
        "official_parity": {
            "custom": custom,
            "official": {
                "avg_cer": float(official["avg_cer"]),
                "avg_rr": float(official["avg_rr"]),
                "overall": float(official["overall"]),
                "false_reject_rate": float(official["false_reject_rate"]),
                "false_accept_rate": float(official["false_accept_rate"]),
            },
            "deltas": deltas,
            "parity": parity,
        },
    }

    summary = {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
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
        "source_digest": source_digest["aggregate"],
        "feature_state_digest": feature_state_digest["aggregate"],
        "output_files": [],
    }

    final_paths = {
        "manifest": output_root / _ARTIFACT_NAMES["manifest"],
        "summary": output_root / _ARTIFACT_NAMES["summary"],
        "scores": output_root / _ARTIFACT_NAMES["scores"],
        "frontier": output_root / _ARTIFACT_NAMES["frontier"],
        "report": output_root / _ARTIFACT_NAMES["report"],
    }

    manifest_json = _canonical_json(manifest, allow_nan=False)
    summary["output_files"] = [str(p.resolve()) for p in final_paths.values()]
    summary_json = _canonical_json(summary, allow_nan=False)

    scores_by_model = result["scores_by_model"]
    model_names = sorted(scores_by_model)
    fold_assignments = np.asarray(result["fold_assignments"], dtype=np.int64)
    score_lines: list[str] = []
    for index, row in enumerate(rows):
        record = {
            "id": row.id,
            "group": str(groups[index]),
            "fold": int(fold_assignments[index]),
        }
        for model_name in model_names:
            record[model_name] = float(scores_by_model[model_name][index])
        score_lines.append(_canonical_json(record, allow_nan=False))

    frontier_lines: list[str] = []
    for point in result["frontier"]:
        frontier_lines.append(_canonical_json(_apply_threshold_markers(point), allow_nan=False))

    report_lines = [
        "# R11 E0 Gate-Oracle OOF Report",
        "",
        f"Config hash: `{config_hash}`",
        f"Source digest: `{source_digest['aggregate']}`",
        f"Feature state digest: `{feature_state_digest['aggregate']}`",
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
        f"Artifacts: {final_paths['manifest']}, {final_paths['summary']}, {final_paths['scores']}, {final_paths['frontier']}, {final_paths['report']}",
    ])
    report_text = "\n".join(report_lines)

    # Serialize all staging, replacement, rollback, and cleanup for this root.
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock = output_root.with_name(output_root.name + ".publish.lock")
    owner_metadata = _canonical_json({"pid": os.getpid(), "token": secrets.token_hex(16)})
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"publication lock already exists for {output_root}: {lock}; refusing to modify it"
        ) from exc
    (lock / "owner.json").write_text(owner_metadata, encoding="utf-8")

    staging = _unique_sibling(parent, output_root.name + ".staging")
    backup: Path | None = None
    publish_succeeded = False
    restore_succeeded = False
    try:
        staging.mkdir()
        (staging / _ARTIFACT_NAMES["manifest"]).write_text(manifest_json, encoding="utf-8")
        (staging / _ARTIFACT_NAMES["summary"]).write_text(summary_json, encoding="utf-8")
        with (staging / _ARTIFACT_NAMES["scores"]).open("w", encoding="utf-8") as handle:
            for line in score_lines:
                handle.write(line + "\n")
        with (staging / _ARTIFACT_NAMES["frontier"]).open("w", encoding="utf-8") as handle:
            for line in frontier_lines:
                handle.write(line + "\n")
        (staging / _ARTIFACT_NAMES["report"]).write_text(report_text, encoding="utf-8")

        if output_root.exists():
            if not _is_recognized_output(output_root):
                raise ValueError(
                    f"existing output root {output_root} is not a recognizable E0 artifact directory"
                )
            backup = _unique_sibling(parent, output_root.name + ".backup")
            os.rename(output_root, backup)

        os.rename(staging, output_root)
        publish_succeeded = True
    except Exception as publish_exc:
        if backup is not None and backup.exists():
            if output_root.exists():
                raise RuntimeError(
                    f"publish failed and unexpected output root was preserved at {output_root}; "
                    f"recovery backup preserved at {backup}"
                ) from publish_exc
            try:
                os.rename(backup, output_root)
                restore_succeeded = True
            except Exception as restore_exc:
                raise RuntimeError(
                    f"publish failed and restore failed; recovery backup at {backup}: {restore_exc}"
                ) from publish_exc
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and (publish_succeeded or restore_succeeded):
            shutil.rmtree(backup)
        owner = lock / "owner.json"
        if owner.is_file() and owner.read_text(encoding="utf-8") == owner_metadata:
            owner.unlink()
            lock.rmdir()

    return final_paths


def _validate_inputs(args: argparse.Namespace) -> tuple[list[CandidateRow], dict[str, str | None], list[str], dict[str, Any], list[Sample]]:
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

    samples = _validate_dataset_uniqueness(args.dataset_root)
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
        if set(ids) != expected_ids:
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
    ordered_samples = [next(s for s in samples if str(s.id) == sid) for sid in sample_ids]

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

    return rows, labels, groups, paths, ordered_samples


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
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output" / "r11_gate_oracle")
    parser.add_argument("--n-outer", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--rr-floor", type=float, default=0.93)
    args = parser.parse_args(argv)

    rows, labels, groups, paths, ordered_samples = _validate_inputs(args)

    result = evaluate_e0(
        rows,
        labels,
        groups,
        n_splits=args.n_outer,
        seed=args.seed,
        rr_floor=args.rr_floor,
        n_boot=args.n_boot,
    )

    official = _check_evaluator_parity(rows, labels, result, samples=ordered_samples)
    result = {**result, "official_metrics": official}

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
