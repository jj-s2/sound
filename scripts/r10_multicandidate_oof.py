"""R10: multi-candidate grouped OOF selector evaluation CLI.

Outputs pooled/per-fold official metrics, group-bootstrap CIs, oracle ceilings,
and a promotion decision under output/r10_multicandidate_oof/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import load_dataset
from xh202615.evaluation import evaluate_rows
from xh202615.r10_selector import (
    ACTION_ORDER,
    bootstrap_grouped_ci,
    load_candidate_bundle,
    run_grouped_nested_oof,
)


def _config_hash(
    n_outer: int,
    n_inner: int,
    C_values: tuple[float, ...],
    tau_values: tuple[float, ...],
    feature_schema: list[str],
    seed: int,
) -> str:
    payload = {
        "n_outer": n_outer,
        "n_inner": n_inner,
        "C_values": list(C_values),
        "tau_values": list(tau_values),
        "feature_schema": feature_schema,
        "seed": seed,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_digest(rows_by_id: dict) -> str:
    parts = {sid: row.source_digest for sid, row in rows_by_id.items()}
    canonical = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="R10 multi-candidate grouped OOF selector")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasetA" / "datasetA")
    parser.add_argument("--candidate-fusion", type=Path, default=REPO_ROOT / "output" / "asr" / "candidate_fusion_smoke.jsonl")
    parser.add_argument("--tse-asr", type=Path, default=REPO_ROOT / "output" / "training_r9" / "datasetA_tse" / "asr_predictions.jsonl")
    parser.add_argument("--audio-map", type=Path, default=REPO_ROOT / "output" / "training_r9" / "datasetA_tse" / "audio_map.jsonl")
    parser.add_argument("--r3-predictions", type=Path, default=REPO_ROOT / "output" / "evaluations" / "r3_temporal_on_datasetA_gated.jsonl")
    parser.add_argument("--group-manifest", type=Path, default=REPO_ROOT / ".superpowers" / "sdd" / "2026-08-07-r9-overall-08-arena" / "datasetA_group_manifest_v1.json")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output" / "r10_multicandidate_oof")
    parser.add_argument("--n-outer", type=int, default=5)
    parser.add_argument("--n-inner", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    samples = load_dataset(args.dataset_root, splits=("pos", "neg"))
    labels = {str(s.id): s.label for s in samples}

    rows_by_id, groups, loaded_labels = load_candidate_bundle(
        args.candidate_fusion,
        args.tse_asr,
        args.audio_map,
        args.group_manifest,
        r3_predictions_path=args.r3_predictions,
    )
    if labels != loaded_labels:
        raise ValueError("group manifest labels disagree with dataset loader labels")

    sample_ids = sorted(rows_by_id, key=lambda x: int(x) if x.isdigit() else x)
    missing_ids = [sid for sid in sample_ids if sid not in rows_by_id]
    if missing_ids:
        raise ValueError(f"missing joined rows for {len(missing_ids)} sample(s): {missing_ids[:5]}")

    C_values = (0.01, 0.1, 1.0, 10.0)
    tau_values = (0.3, 0.5, 0.7, 0.9)

    result = run_grouped_nested_oof(
        rows_by_id,
        labels,
        groups,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        C_values=C_values,
        tau_values=tau_values,
        seed=args.seed,
    )

    # Coverage and evaluator equivalence checks.
    oof_ids = [p["id"] for p in result["oof_predictions"]]
    if sorted(oof_ids) != sorted(sample_ids):
        raise ValueError("OOF predictions do not cover all sample IDs exactly once")
    if len(oof_ids) != len(set(oof_ids)):
        raise ValueError("duplicate IDs in OOF predictions")

    official = evaluate_rows(
        samples,
        [{"id": p["id"], "recognition_text": p["recognition_text"]} for p in result["oof_predictions"]],
        missing_policy="empty",
    )
    official_metrics = dict(official.metrics)
    official_metrics["overall"] = ((1.0 - official_metrics["avg_cer"]) + official_metrics["avg_rr"]) / 2.0
    pooled = result["pooled_metrics"]
    for key in ("avg_cer", "avg_rr", "substitutions", "insertions", "deletions", "ref_chars", "false_reject_rate", "false_accept_rate"):
        if abs(pooled[key] - official_metrics[key]) > 1e-9:
            raise ValueError(f"pooled metrics disagree with official evaluator on {key}: {pooled[key]} vs {official_metrics[key]}")

    ci = bootstrap_grouped_ci(
        result["oof_predictions"],
        labels,
        groups,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    fold_overalls = [f["metrics"]["overall"] for f in result["fold_reports"]]
    worst_fold = result["fold_reports"][int(np.argmin(fold_overalls))]

    cfg_hash = _config_hash(
        args.n_outer,
        args.n_inner,
        C_values,
        tau_values,
        ACTION_ORDER,
        args.seed,
    )

    # Promotion gates: derived from concrete diagnostics; fail closed when unavailable.
    gate_coverage = len(oof_ids) == len(sample_ids) and len(set(oof_ids)) == len(sample_ids)
    gate_evaluator = all(
        abs(pooled[key] - official_metrics[key]) <= 1e-9
        for key in ("avg_cer", "avg_rr", "substitutions", "insertions", "deletions", "ref_chars", "false_reject_rate", "false_accept_rate")
    )
    gate_rr = pooled["avg_rr"] >= 0.95
    gate_overall = pooled["overall"] >= 0.80
    gate_worst_fold = worst_fold["metrics"]["overall"] >= 0.77
    gate_leakage = all(f.get("group_disjoint") is True for f in result["fold_reports"])
    gate_fallback = all(p.get("recognition_text") is not None for p in result["oof_predictions"])
    gate_feasible = result.get("n_infeasible_folds", len(result["fold_reports"])) == 0
    promoted = (
        gate_evaluator and gate_coverage and gate_rr and gate_overall and gate_worst_fold
        and gate_leakage and gate_fallback and gate_feasible
    )

    manifest = {
        "config_hash": cfg_hash,
        "source_digest": _source_digest(rows_by_id),
        "dataset_root": str(args.dataset_root.resolve()),
        "candidate_fusion": str(args.candidate_fusion.resolve()),
        "tse_asr": str(args.tse_asr.resolve()),
        "audio_map": str(args.audio_map.resolve()),
        "r3_predictions": str(args.r3_predictions.resolve()),
        "group_manifest": str(args.group_manifest.resolve()),
        "n_outer": args.n_outer,
        "n_inner": args.n_inner,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "pooled_metrics": pooled,
        "r3_metrics": result["r3_metrics"],
        "agreement_rescue_metrics": result["agreement_rescue_metrics"],
        "oracle_2_metrics": result["oracle_2_metrics"],
        "oracle_all_metrics": result["oracle_all_metrics"],
        "n_infeasible_folds": result.get("n_infeasible_folds", 0),
        "bootstrap_ci": ci,
        "worst_fold": {
            "outer_idx": worst_fold["outer_idx"],
            "metrics": worst_fold["metrics"],
        },
        "fold_reports": result["fold_reports"],
        "promotion": {
            "promoted": promoted,
            "gate_evaluator": gate_evaluator,
            "gate_coverage": gate_coverage,
            "gate_rr": gate_rr,
            "gate_overall": gate_overall,
            "gate_worst_fold": gate_worst_fold,
            "gate_leakage": gate_leakage,
            "gate_fallback": gate_fallback,
            "gate_feasible": gate_feasible,
        },
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "r10_oof_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    preds_path = args.output_root / "r10_oof_predictions.jsonl"
    with preds_path.open("w", encoding="utf-8") as handle:
        for p in result["oof_predictions"]:
            handle.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_infeasible = result.get("n_infeasible_folds", 0)
    summary = {
        "status": "success",
        "promoted": promoted,
        "n_infeasible_folds": n_infeasible,
        "pooled": {
            "cer": pooled["avg_cer"],
            "rr": pooled["avg_rr"],
            "overall": pooled["overall"],
            "frr": pooled["false_reject_rate"],
            "far": pooled["false_accept_rate"],
        },
        "oracle": {
            "two_candidate_overall": result["oracle_2_metrics"]["overall"],
            "all_candidate_overall": result["oracle_all_metrics"]["overall"],
        },
        "bootstrap_ci": ci,
        "worst_fold": {
            "outer_idx": worst_fold["outer_idx"],
            "overall": worst_fold["metrics"]["overall"],
            "cer": worst_fold["metrics"]["avg_cer"],
            "rr": worst_fold["metrics"]["avg_rr"],
        },
        "r3_baseline": result["r3_metrics"]["overall"],
        "agreement_rescue": result["agreement_rescue_metrics"]["overall"],
        "config_hash": cfg_hash,
        "output_files": [str(manifest_path), str(preds_path)],
    }
    summary_path = args.output_root / "r10_oof_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# R10 Multi-Candidate Grouped OOF Report",
        "",
        f"Config hash: `{cfg_hash}`",
        f"Source digest: `{_source_digest(rows_by_id)}`",
        f"Outer folds: {args.n_outer}, inner folds: {args.n_inner}, bootstrap resamples: {args.n_boot}",
        "",
        "## Pooled OOF metrics (learned selector, R3 fallback where infeasible)",
        f"- CER: {pooled['avg_cer']:.6f}",
        f"- RR: {pooled['avg_rr']:.6f}",
        f"- Overall: {pooled['overall']:.6f}",
        f"- S/I/D: {pooled['substitutions']} / {pooled['insertions']} / {pooled['deletions']}",
        f"- FRR: {pooled['false_reject_rate']:.6f}",
        f"- FAR: {pooled['false_accept_rate']:.6f}",
        "",
        "## Oracle ceilings",
        f"- Two-candidate (R3 + TSE) Overall: {result['oracle_2_metrics']['overall']:.6f}",
        f"- All-candidate Overall: {result['oracle_all_metrics']['overall']:.6f}",
        "",
        "## Comparison policies",
        f"- R3 unchanged Overall: {result['r3_metrics']['overall']:.6f}",
        f"- Agreement-rescue Overall: {result['agreement_rescue_metrics']['overall']:.6f}",
        "",
        "## Group-bootstrap 95% CI for Overall",
        f"- Mean: {ci['overall_mean']:.6f}",
        f"- CI: [{ci['ci_low']:.6f}, {ci['ci_high']:.6f}]",
        "",
        "## Worst outer fold",
        f"- Fold {worst_fold['outer_idx']}: Overall {worst_fold['metrics']['overall']:.6f}, CER {worst_fold['metrics']['avg_cer']:.6f}, RR {worst_fold['metrics']['avg_rr']:.6f}",
        "",
        "## Per-fold Overall",
    ]
    for f in result["fold_reports"]:
        line = f"- Fold {f['outer_idx']}: {f['metrics']['overall']:.6f}"
        if f.get("selected_C") is not None:
            line += f" (C={f['selected_C']}, tau={f['selected_tau']})"
        if f.get("fallback"):
            line += f" [fallback={f['fallback']}]"
        report_lines.append(line)
    report_lines.extend([
        "",
        "## Feasibility under RR >= 0.95",
        f"- Outer folds with no feasible learned policy: {n_infeasible} / {len(result['fold_reports'])}",
    ])
    if n_infeasible:
        report_lines.append(
            "- The learned selector was not evaluable/promotable under the RR >= 0.95 constraint; "
            "OOF falls back to exact R3 for those folds."
        )
    report_lines.extend([
        "",
        "## Promotion gates",
        f"- Promoted: {promoted}",
        f"- Evaluator equivalence: {gate_evaluator}",
        f"- Complete coverage: {gate_coverage}",
        f"- RR >= 0.95: {gate_rr} ({pooled['avg_rr']:.6f})",
        f"- Overall >= 0.80: {gate_overall} ({pooled['overall']:.6f})",
        f"- Worst fold >= 0.77: {gate_worst_fold} ({worst_fold['metrics']['overall']:.6f})",
        f"- Leakage checks: {gate_leakage}",
        f"- Fallback checks: {gate_fallback}",
        f"- Feasible inner policy in every fold: {gate_feasible}",
        "",
        "## Data boundary",
        "- Labels used only for fold-local target generation and held-out scoring.",
        "- Model input features are inference-only post-ASR and acoustic values.",
        "- No Dataset-B, hidden labels, or leaderboard feedback used.",
        "",
        f"Artifacts: {manifest_path}, {preds_path}, {summary_path}",
    ])
    report_path = args.output_root / "r10_oof_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"R10 OOF complete. Overall={pooled['overall']:.6f} RR={pooled['avg_rr']:.6f} CER={pooled['avg_cer']:.6f}")
    print(f"  Infeasible folds: {n_infeasible} / {len(result['fold_reports'])}")
    print(f"  95% CI: [{ci['ci_low']:.6f}, {ci['ci_high']:.6f}]")
    print(f"  Worst fold: {worst_fold['metrics']['overall']:.6f}")
    print(f"  Promoted: {promoted}")
    print(f"  Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
