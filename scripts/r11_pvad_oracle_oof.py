"""Publish the frozen R11 E2 FireRed pVAD grouped-OOF evidence package.

The input rows intentionally contain only inference-time candidate data. Labels
are loaded separately and are introduced only after Task 5 has built its OOF
score banks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from xh202615.artifact_publish import ArtifactContract, publish_text_package
from xh202615.r10_selector import CandidateRow
from xh202615.r11_gate_oracle import (
    _best_model_frontier,
    _fixed_threshold_point,
    _subset_contributions,
    build_oracle_contributions,
    evaluate_e0,
    gate_oracle_frontier,
    group_bootstrap_best_frontier,
)
from xh202615.r11_pvad_oracle import canonical_json, join_pvad_e0_rows, run_pvad_oracle
from scripts.r11_gate_oracle_oof import _check_evaluator_parity

_KIND, _VERSION = "r11_e2_pvad_oracle", "v1"
_NAMES = ("e2_manifest.json", "e2_oof_scores.jsonl", "e2_frontier.jsonl", "e2_summary.json", "e2_report.md")
_CONTRACT = ArtifactContract(_KIND, _VERSION, _NAMES, ("e2_manifest.json", "e2_summary.json"))
_REJECT_ALL = "reject_all"
_PRIVATE = {"label", "labels", "reference", "reference_text", "transcript", "recognition_text", "candidate_cer", "optimal_action", "speaker_embedding", "speaker_embeddings", "embedding", "frame_arrays", "frames"}


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: object) -> Any:
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("published values must be finite")
        return value
    if type(value) is int or isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, list) or isinstance(value, tuple):
        return [_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    raise ValueError(f"published value is not JSON-safe: {type(value).__name__}")


def _threshold(value: object) -> float | str:
    numeric = float(value)
    return _REJECT_ALL if math.isinf(numeric) and numeric > 0 else _safe(numeric)


def _reject_private(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _PRIVATE or any(token in lowered for token in ("label", "reference", "transcript", "candidate_cer", "optimal_action", "embedding", "frame")):
                raise ValueError("published artifact contains a forbidden private field")
            _reject_private(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private(item)


def _load_json(path: Path) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _mapping_file(path: Path, field: str) -> dict[str, object]:
    value = _load_json(path)
    if isinstance(value, Mapping):
        result = dict(value)
    elif isinstance(value, list):
        result = {}
        for item in value:
            if not isinstance(item, Mapping) or set(item) != {"id", field} or not isinstance(item["id"], str) or item["id"] in result:
                raise ValueError(f"{path} must be an exact id/{field} mapping")
            result[item["id"]] = item[field]
    else:
        raise ValueError(f"{path} must be a JSON object or id/{field} list")
    if not result or any(not isinstance(key, str) or not key for key in result):
        raise ValueError(f"{path} contains an invalid ID")
    return result


def load_canonical_rows(path: Path) -> list[CandidateRow]:
    """Load an explicit R10 row projection without accepting labels or references."""
    rows: list[CandidateRow] = []
    seen: set[str] = set()
    required = {"id", "split", "r3_text", "primary_text", "energy_text", "tse_text", "audio_features", "source_digest"}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line:
            raise ValueError(f"canonical input has an empty line at {number}")
        record = json.loads(line)
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValueError("canonical row schema must be exact and label-free")
        sid = record["id"]
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("canonical rows contain missing or duplicate IDs")
        if not isinstance(record["audio_features"], Mapping) or not isinstance(record["source_digest"], str):
            raise ValueError("canonical row has invalid feature provenance")
        seen.add(sid)
        rows.append(CandidateRow(sid, str(record["split"]), None, str(record["r3_text"]), str(record["primary_text"]), str(record["energy_text"]), str(record["tse_text"]), dict(record["audio_features"]), None, record["source_digest"], {}))
    if not rows:
        raise ValueError("canonical input contains no rows")
    return rows


def _cache(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    expected = {"pvad_features.jsonl", "pvad_manifest.json", "pvad_report.md"}
    if not root.is_dir() or root.is_symlink() or {item.name for item in root.iterdir()} != expected or any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ValueError("cache root is not an exact regular Task 4 package")
    records = [json.loads(line) for line in (root / "pvad_features.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    manifest = _load_json(root / "pvad_manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("cache manifest must be an object")
    return records, manifest


def _validate_task5_result(result: Mapping[str, object], rows: Sequence[CandidateRow]) -> None:
    """Validate the public Task 5 shape before deriving publication evidence."""
    if result.get("diagnostic_only") is not True or result.get("deployable") is not False:
        raise ValueError("Task 5 result is not diagnostic-only")
    state = result.get("source_joined_state")
    if not isinstance(state, Mapping) or set(state) != {
        "cache_joined_state_sha256", "cache_source_projection_sha256",
        "cache_model_identity_sha256", "joined_rows_sha256",
    }:
        raise ValueError("Task 5 result has no complete source_joined_state")
    for key in state:
        if not isinstance(state[key], str) or len(state[key]) != 64:
            raise ValueError("Task 5 source_joined_state contains an invalid digest")
    families = result.get("families")
    expected = {"firered_scalar", "firered_crossfit", "firered_fused_crossfit"}
    if not isinstance(families, Mapping) or set(families) != expected:
        raise ValueError("Task 5 result has an invalid family set")
    expected_ids = [row.id for row in rows]
    for name, family in families.items():
        if not isinstance(family, Mapping) or not isinstance(family.get("rows"), list) or not isinstance(family.get("folds"), list) or not isinstance(family.get("score_bank"), Mapping):
            raise ValueError(f"Task 5 family {name} has an invalid schema")
        if [item.get("id") for item in family["rows"] if isinstance(item, Mapping)] != expected_ids:
            raise ValueError(f"Task 5 family {name} does not cover canonical IDs in order")
        if len(family["folds"]) != 5:
            raise ValueError(f"Task 5 family {name} does not have five folds")
        for model, scores in family["score_bank"].items():
            values = np.asarray(scores, dtype=np.float64)
            if values.shape != (len(rows),) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError(f"Task 5 score bank {name}:{model} is invalid")


def _family_banks(result: Mapping[str, object]) -> dict[str, np.ndarray]:
    banks: dict[str, np.ndarray] = {}
    for family_name, family in result["families"].items():
        assert isinstance(family, Mapping)
        for model, scores in family["score_bank"].items():
            banks[f"{family_name}:{model}"] = np.asarray(scores, dtype=np.float64)
    return banks


def _selected(scores: Mapping[str, np.ndarray], contributions: object) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected, frontier = _best_model_frontier(scores, contributions, 0.93)
    return selected, [{**point, "threshold": _threshold(point["threshold"])} for point in frontier]


def _decision(selected: Mapping[str, object], worst: Mapping[str, object], improvement_low: float, pvad_removed_drop: float, inner_ok: bool) -> str:
    if (float(selected["overall"]) >= .81 and float(selected["rr"]) >= .93 and float(worst["overall"]) >= .77 and float(worst["rr"]) >= .90 and inner_ok and improvement_low > 0):
        return "continue_ranker"
    if float(selected["overall"]) >= .78 and float(selected["rr"]) >= .90 and improvement_low > 0 and pvad_removed_drop >= .01:
        return "consider_custom_pvad"
    return "falsified_firered"


def build_e2_evidence(rows: Sequence[CandidateRow], labels: Mapping[str, str | None], groups: Mapping[str, str], records: Sequence[Mapping[str, object]], cache_manifest: Mapping[str, object], *, n_boot: int, seed: int) -> dict[str, object]:
    """Run Task 5 and construct public E2 selection/bootstrap evidence."""
    joined = join_pvad_e0_rows(rows, labels, groups, records, cache_manifest)
    if [record["id"] for record in records] != [row.id for row in rows]:
        raise ValueError("pVAD cache records do not match canonical order")
    task5 = run_pvad_oracle(rows, labels, groups, records, cache_manifest)
    _validate_task5_result(task5, rows)
    banks = _family_banks(task5)
    contribution = build_oracle_contributions(rows, labels)
    selected, frontier = _selected(banks, contribution)
    selected_model = str(selected["model"])
    selected_scores = banks[selected_model]
    folds = task5["families"]["firered_scalar"]["folds"]
    fold_metrics: list[dict[str, object]] = []
    for fold in folds:
        indexes = np.asarray([next(i for i, row in enumerate(rows) if row.id == sid) for sid in fold["test_ids"]], dtype=np.int64)
        threshold = float(selected["threshold"])
        point = _fixed_threshold_point(selected_scores[indexes], _subset_contributions(contribution, indexes), threshold)
        fold_metrics.append({"fold": int(fold["fold"]), "test_groups": list(fold["test_groups"]), **point})
    worst = min(fold_metrics, key=lambda point: (float(point["overall"]), int(point["fold"])))
    boot = group_bootstrap_best_frontier(banks, contribution, [groups[row.id] for row in rows], rr_floor=.93, n_boot=n_boot, seed=seed)
    # E0's outer split is frozen independently from the paired bootstrap RNG.
    e0 = evaluate_e0(rows, labels, [groups[row.id] for row in rows], n_splits=5, seed=20260807, rr_floor=.93, n_boot=n_boot)
    e0_selected = e0["selected_point"]
    # Pair identical group draws by subtracting the deterministic bootstrap samples.
    e0_boot = group_bootstrap_best_frontier(e0["scores_by_model"], contribution, [groups[row.id] for row in rows], rr_floor=.93, n_boot=n_boot, seed=seed)
    deltas = np.asarray(boot["overall_samples"], dtype=np.float64) - np.asarray(e0_boot["overall_samples"], dtype=np.float64)
    improvement = {"seed": seed, "n_boot": n_boot, "mean": float(deltas.mean()), "ci_low": float(np.quantile(deltas, .025)), "ci_high": float(np.quantile(deltas, .975))}
    fused = _best_model_frontier({key: value for key, value in banks.items() if key.startswith("firered_fused_crossfit:")}, contribution, .93)[0]
    # The ablation is the frozen fused family against the same E0-only gate,
    # not the best pVAD family against E0.
    drop = float(fused["overall"]) - float(e0_selected["overall"])
    decision = _decision(selected, worst, float(improvement["ci_low"]), drop, all(bool(fold["inner_search_feasible"]) for family in task5["families"].values() for fold in family["folds"]))
    parity_input = {"selected_point": selected, "scores_by_model": {selected_model: selected_scores}}
    official = _check_evaluator_parity(rows, labels, parity_input)
    return {"task5": task5, "selected": selected, "frontier": frontier, "fold_metrics": fold_metrics, "worst": worst, "bootstrap": boot, "improvement": improvement, "e0": {"selected": e0_selected, "bootstrap": e0_boot}, "pvad_removed_overall_drop": drop, "fused_selected": fused, "official": official, "decision": decision, "joined": joined, "contribution": contribution}


def write_e2_artifacts(evidence: Mapping[str, object], rows: Sequence[CandidateRow], groups: Mapping[str, str], *, input_paths: Mapping[str, Path], source_digests: Mapping[str, str], output_root: Path, seed: int, n_boot: int, elapsed_seconds: float) -> dict[str, Path]:
    """Validate public projections and atomically publish the exact five files."""
    selected = evidence["selected"]
    task5 = evidence["task5"]
    source = {("target_boundary" if name == "labels" else name): _file_sha(path) for name, path in input_paths.items()}
    if set(source_digests) != {"cache_records", "cache_manifest"} or any(not isinstance(value, str) or len(value) != 64 for value in source_digests.values()):
        raise ValueError("source_digests must contain exact cache raw SHA-256 values")
    oof: list[dict[str, object]] = []
    rows_by_id = {row.id: row for row in rows}
    joined_state = task5.get("source_joined_state")
    if not isinstance(joined_state, Mapping):
        raise ValueError("Task 5 source_joined_state is required for publication")
    for family_name, family in task5["families"].items():
        for record in family["rows"]:
            # One score record per ID: include all model scores below, and retain the
            # selected family/fold evidence from the frozen outer split.
            if family_name != "firered_scalar":
                continue
            sid = record["id"]
            oof.append({"id": sid, "group": groups[sid], "group_sha256": _sha(groups[sid]), "fold": int(record["fold"]), "selected_outer_model": record["model"], "selected_outer_threshold": record["threshold"], "source_digest": rows_by_id[sid].source_digest, "cache_joined_state_sha256": joined_state["cache_joined_state_sha256"]})
    banks = _family_banks(task5)
    for index, record in enumerate(oof):
        record["scores"] = {name: float(values[index]) for name, values in sorted(banks.items())}
    if [item["id"] for item in oof] != [row.id for row in rows]:
        raise ValueError("public OOF projection lost canonical order")
    cache_manifest = _load_json(input_paths["cache_manifest"])
    assert isinstance(cache_manifest, Mapping)
    manifest = {"artifact_kind": _KIND, "schema_version": _VERSION, "digest_algorithms": {"canonical_json_sha256": "sha256(UTF-8 canonical JSON)", "raw_file_sha256": "sha256(raw input bytes)"}, "source_sha256": {**source, "cache_records": source_digests["cache_records"], "cache_manifest": source_digests["cache_manifest"]}, "cache": {"records_sha256": source_digests["cache_records"], "manifest_sha256": source_digests["cache_manifest"], "joined_state_sha256": joined_state["cache_joined_state_sha256"], "model_identity_sha256": joined_state["cache_model_identity_sha256"], "runtime_config_sha256": cache_manifest["runtime_config_sha256"], "feature_schema_sha256": cache_manifest["feature_schema_sha256"]}, "feature_schema_sha256": _sha([family["feature_allowlist"] for family in task5["families"].values()]), "seed": seed, "outer_folds": 5, "rr_floor": .93, "bootstrap": {key: evidence["bootstrap"][key] for key in ("n_boot", "n_groups", "max_attempts", "attempted_replicates", "rejected_replicates")}, "model_families": list(task5["families"]), "coverage": {"ids": [row.id for row in rows], "groups": [groups[row.id] for row in rows], "ids_sha256": _sha([row.id for row in rows])}, "selected": {**selected, "threshold": _threshold(selected["threshold"])}, "decision": evidence["decision"], "official_evaluator": {"implementation": "xh202615.evaluation.evaluate_rows", "metrics": ["avg_cer", "avg_rr", "false_reject_rate", "false_accept_rate", "overall"]}, "timing": {"cache_timing": cache_manifest["timing"]}, "joined_state_sha256": _sha({"task5": joined_state, "oof": oof})}
    summary = {"artifact_kind": _KIND, "schema_version": _VERSION, "decision": evidence["decision"], "pooled": {**selected, "threshold": _threshold(selected["threshold"])}, "worst_fold": {**evidence["worst"], "threshold": _threshold(evidence["worst"]["threshold"])}, "paired_grouped_bootstrap_improvement": evidence["improvement"], "official_parity": evidence["official"], "pvad_removed_overall_drop": evidence["pvad_removed_overall_drop"], "latency_memory": {"cache_timing": "recorded in provenance-bound Task 4 manifest"}}
    front = []
    for point in evidence["frontier"]:
        front.append({"scope": "pooled", **point})
    for family_name, family in task5["families"].items():
        for model_name, values in family["score_bank"].items():
            for fold in family["folds"]:
                indexes = np.asarray([next(i for i, row in enumerate(rows) if row.id == sid) for sid in fold["test_ids"]], dtype=np.int64)
                points = gate_oracle_frontier(np.asarray(values, dtype=np.float64)[indexes], _subset_contributions(evidence["contribution"], indexes))
                for point in points:
                    front.append({"scope": "fold", "family": family_name, "model": model_name, "fold": int(fold["fold"]), **point, "threshold": _threshold(point["threshold"])})
    contents = {"e2_manifest.json": canonical_json(_safe(manifest)) + "\n", "e2_oof_scores.jsonl": "".join(canonical_json(_safe(record)) + "\n" for record in oof), "e2_frontier.jsonl": "".join(canonical_json(_safe(point)) + "\n" for point in front), "e2_summary.json": canonical_json(_safe(summary)) + "\n", "e2_report.md": f"# R11 E2 FireRed pVAD Oracle\n\n- Decision: {evidence['decision']}\n- Overall: {float(selected['overall']):.6f}\n- RR: {float(selected['rr']):.6f}\n- Bootstrap improvement 95%: [{evidence['improvement']['ci_low']:.6f}, {evidence['improvement']['ci_high']:.6f}]\n"}
    for name, text in contents.items():
        if name.endswith(".json"):
            _reject_private(json.loads(text))
        elif name.endswith(".jsonl"):
            for line in text.splitlines():
                _reject_private(json.loads(line))
    return publish_text_package(output_root, _CONTRACT, contents)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset-A root; used only to bind the invocation, never read for features")
    parser.add_argument("--canonical-input-jsonl", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "output" / "r11_e2_firered_cache")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output" / "r11_e2_pvad_oracle")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--bootstrap-count", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    parser.add_argument("--limit", type=int, help="explicit noncanonical smoke-run limit")
    args = parser.parse_args(argv)
    if not args.dataset_root.is_dir() or args.dataset_root.is_symlink():
        raise ValueError("dataset root must be a regular directory")
    if args.seed != 20260807:
        raise ValueError("the frozen split seed is 20260807")
    if args.bootstrap_count <= 0 or args.limit is not None and args.limit <= 0:
        raise ValueError("bootstrap count and limit must be positive")
    started = time.monotonic()
    rows = load_canonical_rows(args.canonical_input_jsonl)
    labels = _mapping_file(args.labels, "label")
    groups = _mapping_file(args.groups, "group")
    if set(labels) != {row.id for row in rows} or set(groups) != {row.id for row in rows}:
        raise ValueError("canonical rows, labels, and groups must have exact ID equality")
    if any(value is not None and not isinstance(value, str) for value in labels.values()) or any(not isinstance(value, str) or not value for value in groups.values()):
        raise ValueError("labels/groups have invalid values")
    if args.limit is not None:
        rows = rows[:args.limit]
        labels, groups = ({row.id: labels[row.id] for row in rows}, {row.id: groups[row.id] for row in rows})
    records, cache_manifest = _cache(args.cache_root)
    if args.limit is not None:
        record_ids = [record.get("id") for record in records if isinstance(record, Mapping)]
        if record_ids[:args.limit] != [row.id for row in rows]:
            raise ValueError("--limit must select the canonical cache prefix")
        records = records[:args.limit]
    evidence = build_e2_evidence(rows, labels, groups, records, cache_manifest, n_boot=args.bootstrap_count, seed=args.bootstrap_seed)
    paths: dict[str, Path] = {
        "canonical": args.canonical_input_jsonl,
        "target_boundary": args.labels,
        "groups": args.groups,
        "cache_manifest": args.cache_root / "pvad_manifest.json",
    }
    source_digests = {"cache_records": _file_sha(args.cache_root / "pvad_features.jsonl"), "cache_manifest": _file_sha(args.cache_root / "pvad_manifest.json")}
    files = write_e2_artifacts(evidence, rows, groups, input_paths=paths, source_digests=source_digests, output_root=args.output_root, seed=args.seed, n_boot=args.bootstrap_count, elapsed_seconds=time.monotonic() - started)
    print(files["e2_summary.json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
