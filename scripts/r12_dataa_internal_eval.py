"""Staged R12 Dataset-A augmented training, selection, and internal evaluation.

``select`` accepts only train and validation labels. ``evaluate`` verifies the
frozen selection, refits on train plus raw validation, produces internal-test
predictions, and only then reads internal-test labels once for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.r11_pvad_oracle_oof import _cache, load_canonical_rows
from xh202615.data import Sample
from xh202615.evaluation import evaluate_rows
from xh202615.metrics import cer_stats
from xh202615.r11_pvad_oracle import JoinedPvadRow, join_pvad_e0_rows
from xh202615.r12_calibrated_gate import (
    OracleContributions,
    _bootstrap_point_stats,
    fit_train_calibrated_gate,
    predict_with_selection,
    select_on_validation,
)
from xh202615.r12_candidate_router import fit_train_candidate_router, predict_router_actions
from xh202615.r12_dataa_augmentation import load_lineage
from xh202615.r12_dataa_augmented_split import load_augmented_internal_split
from xh202615.r12_text_presence import fit_train_text_presence, predict_text_presence


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_labels(path: Path, expected_ids: Sequence[str], role: str) -> dict[str, str | None]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or set(raw) != set(expected_ids):
        raise ValueError(f"{role} labels must exactly cover its expected IDs")
    if any(value is not None and not isinstance(value, str) for value in raw.values()):
        raise ValueError(f"{role} labels must be strings or null")
    return {str(key): value for key, value in raw.items()}


def validate_role_ids(role: str, ids: Sequence[str]) -> None:
    if role in {"validation", "internal_test"} and any("__aug_" in sample_id for sample_id in ids):
        raise ValueError(f"{role} must contain original IDs only")


def expand_train_labels(
    parent_labels: Mapping[str, str | None], rows: Mapping[str, tuple[str, str, str]]
) -> dict[str, str | None]:
    """Inherit private labels only from train-role parent samples."""
    result: dict[str, str | None] = {}
    for sample_id, (parent_id, augmentation_id, role) in rows.items():
        if role != "train":
            continue
        if parent_id not in parent_labels:
            raise ValueError("train child has no train parent label")
        if augmentation_id not in {"original", "aug_a", "aug_b"}:
            raise ValueError("invalid train augmentation ID")
        result[sample_id] = parent_labels[parent_id]
    if set(parent_labels) != {parent for parent, augmentation, role in rows.values() if role == "train" and augmentation == "original"}:
        raise ValueError("train labels must exactly cover original train parents")
    return result


def _common(args: argparse.Namespace):
    rows = load_canonical_rows(args.canonical_input_jsonl)
    lineage = load_lineage(args.lineage)
    if {row.id for row in rows} != set(lineage):
        raise ValueError("canonical IDs must exactly cover lineage IDs")
    original_ids = [sample_id for sample_id, item in lineage.items() if item.augmentation_id == "original"]
    split = load_augmented_internal_split(args.split_manifest, original_ids)
    for sample_id in original_ids:
        item = lineage[sample_id]
        if split.roles_by_id[sample_id] != item.role or split.groups_by_id[sample_id] != item.group:
            raise ValueError("lineage original role/group differs from frozen split")
    role_by_id = {sample_id: item.role for sample_id, item in lineage.items()}
    group_by_id = {sample_id: item.group for sample_id, item in lineage.items()}
    for role in ("validation", "internal_test"):
        validate_role_ids(role, [sample_id for sample_id, assigned in role_by_id.items() if assigned == role])
    records, cache_manifest = _cache(args.cache_root)
    return rows, lineage, split, role_by_id, group_by_id, records, cache_manifest


def _join(rows, group_by_id, records, cache_manifest, labels):
    return join_pvad_e0_rows(rows, labels, group_by_id, records, cache_manifest)


def _rows_for(rows, roles: Mapping[str, str], role: str):
    return [row for row in rows if roles[row.id] == role]


def _fit(rows, joined, labels, roles, *, include_validation: bool):
    allowed = {"train", "validation"} if include_validation else {"train"}
    fit_rows = [row for row in rows if roles[row.id] in allowed]
    fit_joined_by_id = {item.id: item for item in joined}
    fit_joined = [fit_joined_by_id[row.id] for row in fit_rows]
    fit_labels = {row.id: labels[row.id] for row in fit_rows}
    gate = fit_train_calibrated_gate(fit_joined, seed=20260812)
    router = fit_train_candidate_router(fit_rows, fit_joined, fit_labels, seed=20260812)
    text = fit_train_text_presence(fit_rows, fit_labels, seed=20260812)
    return gate, router, text, fit_rows, fit_joined


def _selection_payload(selection, router, text, provenance: Mapping[str, object]) -> dict[str, object]:
    choice = selection.to_dict()
    choice["router"] = router.to_public_dict()
    choice["text_presence"] = text.to_public_dict()
    payload = {
        "artifact_kind": "r12_dataa_augmented_selection",
        "schema_version": "v1",
        "selection": choice,
        "provenance": dict(provenance),
    }
    payload["selection_sha256"] = _json_sha(payload["selection"])
    payload["provenance_sha256"] = _json_sha(payload["provenance"])
    return payload


def _load_selection(path: Path) -> dict[str, object]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or raw.get("artifact_kind") != "r12_dataa_augmented_selection" or raw.get("schema_version") != "v1":
        raise ValueError("selection artifact kind/schema is invalid")
    if raw.get("selection_sha256") != _json_sha(raw.get("selection")) or raw.get("provenance_sha256") != _json_sha(raw.get("provenance")):
        raise ValueError("selection artifact digest mismatch")
    if not isinstance(raw.get("selection"), dict) or not isinstance(raw.get("provenance"), dict):
        raise ValueError("selection artifact payload is invalid")
    return raw


def _provenance(args: argparse.Namespace, *, fit_ids: Sequence[str], validation_ids: Sequence[str], train_labels: Path, validation_labels: Path) -> dict[str, object]:
    return {
        "canonical_sha256": _sha(args.canonical_input_jsonl),
        "lineage_sha256": _sha(args.lineage),
        "split_sha256": _sha(args.split_manifest),
        "cache_features_sha256": _sha(args.cache_root / "pvad_features.jsonl"),
        "cache_manifest_sha256": _sha(args.cache_root / "pvad_manifest.json"),
        "train_labels_sha256": _sha(train_labels),
        "validation_labels_sha256": _sha(validation_labels),
        "fit_ids": list(fit_ids),
        "validation_ids": list(validation_ids),
        "seed": 20260812,
    }


def select(args: argparse.Namespace) -> int:
    rows, lineage, split, roles, groups, records, cache_manifest = _common(args)
    train_parents = [sample_id for sample_id in split.roles_by_id if split.roles_by_id[sample_id] == "train"]
    validation_rows = _rows_for(rows, roles, "validation")
    train_labels = expand_train_labels(_load_labels(args.train_labels, train_parents, "train"), {sid: (item.parent_id, item.augmentation_id, item.role) for sid, item in lineage.items()})
    validation_labels = _load_labels(args.validation_labels, [row.id for row in validation_rows], "validation")
    all_labels = {row.id: None for row in rows}
    all_labels.update(train_labels)
    all_labels.update(validation_labels)
    joined = _join(rows, groups, records, cache_manifest, all_labels)
    joined_by_id = {row.id: row for row in joined}
    gate, router, text, fit_rows, _ = _fit(rows, joined, all_labels, roles, include_validation=False)
    validation_joined = [joined_by_id[row.id] for row in validation_rows]
    selection = select_on_validation(gate, validation_joined, validation_rows, validation_labels, n_boot=args.bootstrap_count, seed=20260812, accepted_actions=predict_router_actions(router, validation_rows, validation_joined), text_scores=predict_text_presence(text, validation_rows))
    result = _selection_payload(selection, router, text, _provenance(args, fit_ids=[row.id for row in fit_rows], validation_ids=[row.id for row in validation_rows], train_labels=args.train_labels, validation_labels=args.validation_labels))
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    if args.selection_output.exists():
        raise ValueError("selection output already exists")
    args.selection_output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


def _test_contributions(rows, labels, actions) -> OracleContributions:
    substitutions = np.zeros(len(rows), dtype=np.int64)
    insertions = np.zeros(len(rows), dtype=np.int64)
    deletions = np.zeros(len(rows), dtype=np.int64)
    ref_chars = np.zeros(len(rows), dtype=np.int64)
    positive = np.zeros(len(rows), dtype=np.bool_)
    choices: list[str] = []
    for index, (row, action) in enumerate(zip(rows, actions)):
        label = labels[row.id]
        if label is None:
            choices.append("reject")
            continue
        stat = cer_stats(label, row.texts[action])
        substitutions[index], insertions[index], deletions[index], ref_chars[index] = stat.substitutions, stat.insertions, stat.deletions, stat.ref_chars
        positive[index] = True
        choices.append(action)
    return OracleContributions(substitutions, insertions, deletions, ref_chars, positive, tuple(choices))


def evaluate(args: argparse.Namespace) -> int:
    if args.evaluation_output.exists():
        raise ValueError("internal test result output already exists; evaluation is one-time")
    artifact = _load_selection(args.selection_input)
    rows, lineage, split, roles, groups, records, cache_manifest = _common(args)
    train_parents = [sample_id for sample_id in split.roles_by_id if split.roles_by_id[sample_id] == "train"]
    validation_rows = _rows_for(rows, roles, "validation")
    expected = _provenance(args, fit_ids=list(artifact["provenance"]["fit_ids"]), validation_ids=[row.id for row in validation_rows], train_labels=args.train_labels, validation_labels=args.validation_labels)
    for key in ("canonical_sha256", "lineage_sha256", "split_sha256", "cache_features_sha256", "cache_manifest_sha256", "train_labels_sha256", "validation_labels_sha256", "fit_ids", "validation_ids", "seed"):
        if artifact["provenance"].get(key) != expected[key]:
            raise ValueError(f"selection provenance mismatch for {key}")
    train_labels = expand_train_labels(_load_labels(args.train_labels, train_parents, "train"), {sid: (item.parent_id, item.augmentation_id, item.role) for sid, item in lineage.items()})
    validation_labels = _load_labels(args.validation_labels, [row.id for row in validation_rows], "validation")
    all_labels = {row.id: None for row in rows}
    all_labels.update(train_labels)
    all_labels.update(validation_labels)
    joined = _join(rows, groups, records, cache_manifest, all_labels)
    gate, router, text, _, _ = _fit(rows, joined, all_labels, roles, include_validation=True)
    test_rows = _rows_for(rows, roles, "internal_test")
    test_joined_by_id = {row.id: row for row in joined}
    test_joined = [test_joined_by_id[row.id] for row in test_rows]
    actions = predict_router_actions(router, test_rows, test_joined)
    selection = artifact["selection"]
    from xh202615.r12_calibrated_gate import FrozenGateSelection
    frozen = FrozenGateSelection(
        base_model_names=tuple(selection["base_model_names"]), base_parameters_digest=str(selection["base_parameters_digest"]), feature_schema_digest=str(selection["feature_schema_digest"]), calibrator_coefficients=tuple(selection["calibrator"]["coefficients"]), calibrator_intercept=float(selection["calibrator"]["intercept"]), blend_definition=dict(selection["blend_definition"]), selected_model_name=str(selection["selected_model_name"]), selected_blend_weight=float(selection["selected_blend_weight"]), threshold=selection["threshold"], validation_raw_metrics=dict(selection["validation_raw_metrics"]), validation_bootstrapped_metrics=dict(selection["validation_bootstrapped_metrics"]), provenance=dict(selection["provenance"]),
    )
    decisions = predict_with_selection(gate, frozen, test_joined, text_scores=predict_text_presence(text, test_rows))
    predictions = [{"id": row.id, "group": groups[row.id], "accepted": bool(decision), "threshold": frozen.threshold, "router_action": action, "recognition_text": row.texts[action] if decision else ""} for row, decision, action in zip(test_rows, decisions, actions)]
    # Internal-test labels are deliberately opened only after predictions exist.
    test_labels = _load_labels(args.internal_test_labels, [row.id for row in test_rows], "internal_test")
    samples = [Sample(row.id, "internal_test", Path("."), "", Path("."), test_labels[row.id]) for row in test_rows]
    metrics = dict(evaluate_rows(samples, predictions, missing_policy="empty").metrics)
    metrics["overall"] = ((1.0 - metrics["avg_cer"]) + metrics["avg_rr"]) / 2.0
    bootstrap = _bootstrap_point_stats(np.asarray(decisions, dtype=np.float64), _test_contributions(test_rows, test_labels, actions), [groups[row.id] for row in test_rows], 0.5, args.bootstrap_count, 20260812)
    args.evaluation_output.mkdir(parents=True)
    (args.evaluation_output / "r12_internal_predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions), encoding="utf-8")
    summary = {"metrics": metrics, "bootstrap": bootstrap, "internal_test_label_read_count": 1, "scope": "Dataset-A group-disjoint internal test; not independent blind-test evidence"}
    (args.evaluation_output / "r12_summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest = {"selection_sha256": _sha(args.selection_input), "prediction_sha256": _sha(args.evaluation_output / "r12_internal_predictions.jsonl"), "summary_sha256": _sha(args.evaluation_output / "r12_summary.json"), "internal_test_label_read_count": 1}
    (args.evaluation_output / "r12_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("select", "evaluate"):
        item = sub.add_parser(command)
        item.add_argument("--canonical-input-jsonl", type=Path, required=True)
        item.add_argument("--lineage", type=Path, required=True)
        item.add_argument("--split-manifest", type=Path, required=True)
        item.add_argument("--cache-root", type=Path, required=True)
        item.add_argument("--train-labels", type=Path, required=True)
        item.add_argument("--validation-labels", type=Path, required=True)
        item.add_argument("--bootstrap-count", type=int, default=2000)
    sub.choices["select"].add_argument("--selection-output", type=Path, required=True)
    sub.choices["evaluate"].add_argument("--selection-input", type=Path, required=True)
    sub.choices["evaluate"].add_argument("--internal-test-labels", type=Path, required=True)
    sub.choices["evaluate"].add_argument("--evaluation-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return select(args) if args.command == "select" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
