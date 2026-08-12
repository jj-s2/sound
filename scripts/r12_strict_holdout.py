"""Staged R12 train/validation selection and held-out evaluation CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.r11_pvad_oracle_oof import _cache, _mapping_file, load_canonical_rows
from xh202615.artifact_publish import ArtifactContract, publish_text_package
from xh202615.data import Sample
from xh202615.evaluation import evaluate_rows
from xh202615.r10_selector import CandidateRow
from xh202615.r11_pvad_oracle import JoinedPvadRow, join_pvad_e0_rows
from xh202615.r12_calibrated_gate import (
    FrozenGateSelection,
    BASE_MODELS,
    BLEND_WEIGHTS,
    TrainCalibratedGate,
    fit_train_calibrated_gate,
    predict_with_selection,
    select_on_validation,
)
from xh202615.r12_split import R12SplitManifest, load_r12_split


_ARTIFACT_KIND = "r12_strict_selection"
_SCHEMA_VERSION = "v1"
_EVAL_CONTRACT = ArtifactContract(
    "r12_strict_holdout", "v1",
    ("r12_manifest.json", "r12_selection.json", "r12_held_out_predictions.jsonl", "r12_summary.json", "r12_report.md"),
    ("r12_manifest.json", "r12_summary.json"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_source_digests(args: argparse.Namespace) -> dict[str, str]:
    fields = ("candidate_fusion", "tse_asr", "audio_map", "r3_predictions", "group_manifest")
    paths = {name: getattr(args, name, None) for name in fields}
    supplied = [name for name, path in paths.items() if path is not None]
    if not supplied:
        raise ValueError("R10 candidate source paths are required for strict provenance")
    if len(supplied) != len(fields):
        raise ValueError("all R10 candidate source paths must be supplied together")
    return {name: _sha(path) for name, path in paths.items()}


def _json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_mapping(path: Path) -> dict[str, str | None]:
    value = _mapping_file(path, "label")
    if any(item is not None and not isinstance(item, str) for item in value.values()):
        raise ValueError(f"{path} contains a non-string label")
    return {str(key): item for key, item in value.items()}


def _load_groups(path: Path) -> dict[str, str]:
    value = _mapping_file(path, "group")
    if any(not isinstance(item, str) or not item for item in value.values()):
        raise ValueError(f"{path} contains an invalid group")
    return {str(key): str(item) for key, item in value.items()}


def _validate_role_labels(
    labels: Mapping[str, str | None], ids: Sequence[str], role: str
) -> dict[str, str | None]:
    expected = set(ids)
    if set(labels) != expected:
        raise ValueError(f"{role} labels must exactly cover its split IDs")
    return dict(labels)


def _load_joined(
    rows: Sequence[CandidateRow],
    groups: Mapping[str, str],
    records: Sequence[Mapping[str, object]],
    cache_manifest: Mapping[str, object],
    labels: Mapping[str, str | None],
) -> list[JoinedPvadRow]:
    return join_pvad_e0_rows(rows, labels, groups, records, cache_manifest)


def _rows_by_role(rows: Sequence[CandidateRow], split: R12SplitManifest, role: str) -> list[CandidateRow]:
    return [row for row in rows if split.roles_by_id[row.id] == role]


def _selection_dict(selection: FrozenGateSelection, provenance: Mapping[str, object]) -> dict[str, object]:
    payload = selection.to_dict()
    frozen_provenance = {
        **dict(provenance),
        "selection_payload_sha256": _json_sha(payload),
    }
    frozen_provenance["provenance_payload_sha256"] = _json_sha(frozen_provenance)
    return {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "selection": payload,
        "provenance": frozen_provenance,
    }


def _selection_from_dict(data: Mapping[str, object]) -> FrozenGateSelection:
    if data.get("artifact_kind") != _ARTIFACT_KIND or data.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("selection artifact kind/schema is invalid")
    selection = data.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("selection artifact has no selection object")
    outer = data.get("provenance")
    if not isinstance(outer, Mapping):
        raise ValueError("selection artifact has no provenance")
    if outer.get("selection_payload_sha256") != _json_sha(selection):
        raise ValueError("selection payload digest mismatch")
    provenance_without_digest = dict(outer)
    provenance_digest = provenance_without_digest.pop("provenance_payload_sha256", None)
    if provenance_digest != _json_sha(provenance_without_digest):
        raise ValueError("selection provenance digest mismatch")
    nested = selection.get("provenance")
    if not isinstance(nested, Mapping):
        raise ValueError("selection has no validation provenance")
    calibrator = selection.get("calibrator")
    if not isinstance(calibrator, Mapping):
        raise ValueError("selection must contain one two-column calibrator")
    record = calibrator
    if not isinstance(record, Mapping):
        raise ValueError("selection calibrator is invalid")
    coefficients = record.get("coefficients")
    if not isinstance(coefficients, list) or len(coefficients) != 2:
        raise ValueError("selection calibrator must have two coefficients")
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in coefficients):
        raise ValueError("selection calibrator coefficients must be finite")
    intercept = record.get("intercept")
    if type(intercept) not in (int, float) or not math.isfinite(float(intercept)):
        raise ValueError("selection calibrator intercept must be finite")
    model_names = selection.get("base_model_names")
    if tuple(model_names or ()) != tuple(BASE_MODELS):
        raise ValueError("selection base models are invalid")
    selected_model_name = selection.get("selected_model_name")
    if selected_model_name not in {*BASE_MODELS, "blend", "reject_all"}:
        raise ValueError("selection model name is invalid")
    selected_weight = selection.get("selected_blend_weight")
    if type(selected_weight) not in (int, float) or float(selected_weight) not in BLEND_WEIGHTS:
        raise ValueError("selection blend weight is invalid")
    threshold = selection.get("threshold")
    if threshold != "reject_all" and (
        type(threshold) not in (int, float) or not math.isfinite(float(threshold))
    ):
        raise ValueError("selection threshold is invalid")
    return FrozenGateSelection(
        base_model_names=tuple(str(item) for item in model_names),
        base_parameters_digest=str(selection["base_parameters_digest"]),
        feature_schema_digest=str(selection["feature_schema_digest"]),
        calibrator_coefficients=tuple(float(item) for item in coefficients),
        calibrator_intercept=float(intercept),
        blend_definition=dict(selection["blend_definition"]),
        selected_model_name=str(selected_model_name),
        selected_blend_weight=float(selected_weight),
        threshold=threshold,
        validation_raw_metrics=dict(selection["validation_raw_metrics"]),
        validation_bootstrapped_metrics=dict(selection["validation_bootstrapped_metrics"]),
        provenance=dict(nested),
    )


def _common_inputs(args: argparse.Namespace) -> tuple[list[CandidateRow], dict[str, str], R12SplitManifest, list[dict[str, object]], dict[str, object]]:
    rows = load_canonical_rows(args.canonical_input_jsonl)
    ids = [row.id for row in rows]
    groups = _load_groups(args.groups)
    if set(groups) != set(ids):
        raise ValueError("groups do not exactly cover canonical IDs")
    split = load_r12_split(args.split_manifest, ids)
    if dict(split.groups_by_id) != groups:
        raise ValueError("groups do not match the frozen split manifest")
    records, manifest = _cache(args.cache_root)
    return rows, groups, split, records, manifest


def _fit_from_train(
    rows: Sequence[CandidateRow], groups: Mapping[str, str], split: R12SplitManifest,
    records: Sequence[Mapping[str, object]], cache_manifest: Mapping[str, object],
    train_labels: Mapping[str, str | None], other_labels: Mapping[str, str | None] | None = None,
) -> tuple[TrainCalibratedGate, list[JoinedPvadRow], list[JoinedPvadRow]]:
    other_labels = other_labels or {}
    all_labels = {row.id: other_labels.get(row.id) for row in rows}
    all_labels.update(train_labels)
    joined = _load_joined(rows, groups, records, cache_manifest, all_labels)
    by_id = {row.id: row for row in joined}
    train_joined = [by_id[row.id] for row in _rows_by_role(rows, split, "train")]
    validation_joined = [by_id[row.id] for row in _rows_by_role(rows, split, "validation")]
    return fit_train_calibrated_gate(train_joined, seed=20260807), train_joined, validation_joined


def _validate_selection_against_trained(
    selection: FrozenGateSelection, trained: TrainCalibratedGate
) -> None:
    if selection.base_model_names != trained.base_model_names:
        raise ValueError("selection base models do not match trained models")
    if selection.base_parameters_digest != trained.base_parameters_digest:
        raise ValueError("selection base parameter digest mismatch")
    if selection.feature_schema_digest != trained.feature_schema_digest:
        raise ValueError("selection feature schema digest mismatch")
    coefficients = tuple(float(value) for value in trained.calibrator.coef_.ravel())
    intercept = float(trained.calibrator.intercept_.ravel()[0])
    if selection.calibrator_coefficients != coefficients or selection.calibrator_intercept != intercept:
        raise ValueError("selection calibrator does not match the train refit")
    if selection.blend_definition.get("weights") != list(BLEND_WEIGHTS):
        raise ValueError("selection blend definition is invalid")


def _recompute_and_verify_selection(
    selection: FrozenGateSelection,
    trained: TrainCalibratedGate,
    validation_joined: Sequence[JoinedPvadRow],
    validation_rows: Sequence[CandidateRow],
    validation_labels: Mapping[str, str | None],
) -> None:
    n_boot = selection.provenance.get("validation_n_boot")
    seed = selection.provenance.get("validation_seed")
    accepted_action = selection.provenance.get("accepted_action")
    if (
        type(n_boot) is not int
        or n_boot <= 0
        or type(seed) is not int
        or accepted_action != "primary"
    ):
        raise ValueError("selection validation provenance is invalid")
    expected = select_on_validation(
        trained,
        validation_joined,
        validation_rows,
        validation_labels,
        n_boot=n_boot,
        seed=seed,
        accepted_action=accepted_action,
    )
    if expected.to_dict() != selection.to_dict():
        raise ValueError("frozen selection does not match validation recomputation")


def _select(args: argparse.Namespace) -> int:
    rows, groups, split, records, cache_manifest = _common_inputs(args)
    train_ids = [row.id for row in _rows_by_role(rows, split, "train")]
    validation_rows = _rows_by_role(rows, split, "validation")
    validation_ids = [row.id for row in validation_rows]
    train_labels = _validate_role_labels(_load_mapping(args.train_labels), train_ids, "train")
    validation_labels = _validate_role_labels(_load_mapping(args.validation_labels), validation_ids, "validation")
    trained, _, validation_joined = _fit_from_train(
        rows, groups, split, records, cache_manifest, train_labels, validation_labels
    )
    selection = select_on_validation(
        trained, validation_joined, validation_rows,
        validation_labels,
        n_boot=args.bootstrap_count, seed=args.seed,
        accepted_action="primary",
    )
    provenance = {
        "canonical_sha256": _sha(args.canonical_input_jsonl),
        "groups_sha256": _sha(args.groups),
        "split_sha256": _sha(args.split_manifest),
        "cache_records_sha256": _sha(args.cache_root / "pvad_features.jsonl"),
        "cache_manifest_sha256": _sha(args.cache_root / "pvad_manifest.json"),
        "train_labels_sha256": _sha(args.train_labels),
        "validation_labels_sha256": _sha(args.validation_labels),
        "train_ids_sha256": _json_sha(train_ids),
        "validation_ids_sha256": _json_sha(validation_ids),
        "seed": args.seed,
        "candidate_source_digests": _candidate_source_digests(args),
        "model_identity": {
            "module": "xh202615.r12_calibrated_gate",
            "base_models": list(selection.base_model_names),
            "base_parameters_digest": selection.base_parameters_digest,
            "feature_schema_digest": selection.feature_schema_digest,
        },
    }
    Path(args.selection_output).write_text(
        json.dumps(_selection_dict(selection, provenance), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.selection_output)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    rows, groups, split, records, cache_manifest = _common_inputs(args)
    selection_path = Path(args.selection_input)
    if not selection_path.is_file():
        raise ValueError(f"selection artifact does not exist: {selection_path}")
    selection_data = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    selection = _selection_from_dict(selection_data)
    outer_provenance = selection_data["provenance"]
    train_ids = [row.id for row in _rows_by_role(rows, split, "train")]
    validation_ids = [row.id for row in _rows_by_role(rows, split, "validation")]
    actual = {
        "canonical_sha256": _sha(args.canonical_input_jsonl),
        "groups_sha256": _sha(args.groups),
        "split_sha256": _sha(args.split_manifest),
        "cache_records_sha256": _sha(args.cache_root / "pvad_features.jsonl"),
        "cache_manifest_sha256": _sha(args.cache_root / "pvad_manifest.json"),
        "train_labels_sha256": _sha(args.train_labels),
        "validation_labels_sha256": _sha(args.validation_labels),
        "candidate_source_digests": _candidate_source_digests(args),
        "seed": 20260807,
        "train_ids_sha256": _json_sha(train_ids),
        "validation_ids_sha256": _json_sha(validation_ids),
        "model_identity": {
            "module": "xh202615.r12_calibrated_gate",
            "base_models": list(selection.base_model_names),
            "base_parameters_digest": selection.base_parameters_digest,
            "feature_schema_digest": selection.feature_schema_digest,
        },
    }
    for key, value in actual.items():
        if outer_provenance.get(key) != value:
            raise ValueError(f"selection provenance mismatch for {key}")

    train_rows = _rows_by_role(rows, split, "train")
    validation_rows = _rows_by_role(rows, split, "validation")
    held_out_rows = _rows_by_role(rows, split, "held_out_test")
    train_labels = _validate_role_labels(_load_mapping(args.train_labels), [row.id for row in train_rows], "train")
    validation_labels = _validate_role_labels(
        _load_mapping(args.validation_labels),
        [row.id for row in validation_rows],
        "validation",
    )
    trained, _, validation_joined = _fit_from_train(
        rows, groups, split, records, cache_manifest, train_labels, validation_labels
    )
    _validate_selection_against_trained(selection, trained)
    expected_model_identity = {
        "module": "xh202615.r12_calibrated_gate",
        "base_models": list(selection.base_model_names),
        "base_parameters_digest": selection.base_parameters_digest,
        "feature_schema_digest": selection.feature_schema_digest,
    }
    if outer_provenance.get("model_identity") != expected_model_identity:
        raise ValueError("selection model identity mismatch")
    _recompute_and_verify_selection(
        selection, trained, validation_joined, validation_rows, validation_labels
    )
    all_labels = {row.id: None for row in rows}
    all_labels.update(train_labels)
    joined = _load_joined(rows, groups, records, cache_manifest, all_labels)
    joined_by_id = {row.id: row for row in joined}
    held_out_joined = [joined_by_id[row.id] for row in held_out_rows]
    decisions = predict_with_selection(trained, selection, held_out_joined)

    held_out_labels = _validate_role_labels(
        _load_mapping(args.held_out_labels),
        [row.id for row in held_out_rows],
        "held_out_test",
    )
    official_predictions: list[dict[str, str]] = []
    prediction_lines: list[str] = []
    for row, joined_row, decision in (
        zip(held_out_rows, held_out_joined, decisions)
    ):
        accepted = bool(decision)
        text = row.primary_text if accepted else ""
        official_predictions.append({"id": row.id, "recognition_text": text})
        prediction_lines.append(json.dumps({
            "id": row.id,
            "group": joined_row.group,
            "accepted": accepted,
            "threshold": selection.threshold,
            "action": "accept" if accepted else "reject",
            "recognition_text": text,
        }, ensure_ascii=False, sort_keys=True))
    samples = [
        Sample(
            id=row.id,
            split=row.split,
            wakeup_audio=Path("."),
            wakeup_text="",
            command_audio=row.original_command_audio or Path("."),
            label=held_out_labels[row.id],
        )
        for row in held_out_rows
    ]
    official_metrics = dict(evaluate_rows(samples, official_predictions, missing_policy="empty").metrics)
    official_metrics["overall"] = ((1.0 - official_metrics["avg_cer"]) + official_metrics["avg_rr"]) / 2.0
    metrics = official_metrics
    output = Path(args.evaluation_output)
    selection_text = json.dumps(selection_data, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    predictions_text = "\n".join(prediction_lines) + "\n"
    summary_text = json.dumps({
        "artifact_kind": _EVAL_CONTRACT.artifact_kind,
        "schema_version": _EVAL_CONTRACT.schema_version,
        "metrics": metrics,
        "held_out_count": len(held_out_rows),
        "selection_is_frozen": True,
    }, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    report_text = "# R12 strict held-out evaluation\n\nHeld-out labels were used only for official scoring of fixed primary-text predictions; no model, transcript source, or threshold selection was performed in this stage.\n"
    output_digests = {
        "r12_selection.json": hashlib.sha256(selection_text.encode("utf-8")).hexdigest(),
        "r12_held_out_predictions.jsonl": hashlib.sha256(predictions_text.encode("utf-8")).hexdigest(),
        "r12_summary.json": hashlib.sha256(summary_text.encode("utf-8")).hexdigest(),
        "r12_report.md": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
    }
    contents = {
        "r12_manifest.json": json.dumps({"artifact_kind": _EVAL_CONTRACT.artifact_kind, "schema_version": _EVAL_CONTRACT.schema_version, "held_out_count": len(held_out_rows), "provenance": actual, "output_digests": output_digests}, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        "r12_selection.json": selection_text,
        "r12_held_out_predictions.jsonl": predictions_text,
        "r12_summary.json": summary_text,
        "r12_report.md": report_text,
    }
    publish_text_package(output, _EVAL_CONTRACT, contents)
    print(output / "r12_summary.json")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--canonical-input-jsonl", type=Path, required=True)
    common.add_argument("--groups", type=Path, required=True)
    common.add_argument("--split-manifest", type=Path, required=True)
    common.add_argument("--cache-root", type=Path, required=True)
    common.add_argument("--candidate-fusion", type=Path)
    common.add_argument("--tse-asr", type=Path)
    common.add_argument("--audio-map", type=Path)
    common.add_argument("--r3-predictions", type=Path)
    common.add_argument("--group-manifest", type=Path)
    select = sub.add_parser("select", parents=[common])
    select.add_argument("--train-labels", type=Path, required=True)
    select.add_argument("--validation-labels", type=Path, required=True)
    select.add_argument("--selection-output", type=Path, required=True)
    select.add_argument("--bootstrap-count", type=int, default=2000)
    select.add_argument("--seed", type=int, default=20260807)
    evaluate = sub.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--train-labels", type=Path, required=True)
    evaluate.add_argument("--validation-labels", type=Path, required=True)
    evaluate.add_argument("--held-out-labels", type=Path, required=True)
    evaluate.add_argument("--selection-input", type=Path, required=True)
    evaluate.add_argument("--evaluation-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "select":
        if args.bootstrap_count <= 0 or args.seed != 20260807:
            raise ValueError("bootstrap-count must be positive and seed must be 20260807")
        return _select(args)
    return _evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())

