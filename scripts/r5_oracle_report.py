"""Compute the R5 ASR oracle report from verified manifest/index evidence.

Evidence is derived from the verified manifest (loaded via the index, not
reconstructed from ASR records). Every selected manifest row must have exactly
one successful mixture and one successful clean-target ASR record with matching
metadata, digests, and config; missing evidence or ASR errors fail closed. Then
computes pooled/bucketed CER for clean and mixture by seed/SNR/overlap, bucket
gaps G_b, H_ASR with predeclared uniform weights, a stratified bootstrap 95% CI
per seed, ASR-segment batch-1 latency (labeled NOT full-pipeline), and an
inconclusive-aware branch recommendation. Writes JSON + markdown.

No Dataset-A file is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.r5_oracle import (  # noqa: E402
    ALL_BUCKETS,
    BUCKET_WEIGHTS,
    OVERLAP_GRID,
    SNR_GRID,
    bootstrap_h_asr_ci,
    branch_recommendation,
    compute_h_asr,
    pooled_cer,
    read_r5_manifest,
    validate_asr_evidence,
)

REPORT_DIR = Path(".superpowers/sdd/2026-08-05-r3-domain-matched-tse-pilot")


def load_asr_records(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_manifest_rows(*, index: str | None, manifest: list[str] | None) -> list:
    """Load verified manifest rows from the index (preferred) or explicit files."""
    rows = []
    if index:
        idx = json.loads(Path(index).read_text(encoding="utf-8"))
        for seed, info in idx.get("seeds", {}).items():
            rows.extend(read_r5_manifest(info["manifest_path"]))
        return rows, idx
    if manifest:
        for m in manifest:
            rows.extend(read_r5_manifest(m))
        return rows, None
    raise SystemExit("provide --index or at least one --manifest")


def latency_stats(records: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for role_key in ("mixture", "clean_target"):
        vals = np.array(
            [r["latency_ms"] for r in records
             if r.get("path_role") == role_key and not r.get("error")
             and r.get("latency_ms", 0.0) > 0.0],
            dtype=np.float64,
        )
        if vals.size == 0:
            out[role_key] = {"count": 0}
            continue
        out[role_key] = {
            "count": int(vals.size),
            "mean_ms": float(np.mean(vals)),
            "p50_ms": float(np.median(vals)),
            "p95_ms": float(np.percentile(vals, 95)),
            "total_s": float(vals.sum() / 1000.0),
        }
    return out


def per_seed_report(manifest_rows: list, asr_by_row: dict, *, seed: str) -> dict:
    seed_rows = [r for r in manifest_rows if r.seed == seed]
    test_rows = [r for r in seed_rows if r.split == "test"]
    val_rows = [r for r in seed_rows if r.split == "val"]
    test_dicts = [r.to_dict() for r in test_rows]
    val_dicts = [r.to_dict() for r in val_rows]
    all_dicts = [r.to_dict() for r in seed_rows]
    h_test = compute_h_asr(test_dicts, asr_by_row, expected_buckets=ALL_BUCKETS)
    ci_test = bootstrap_h_asr_ci(
        test_dicts, asr_by_row, n_boot=2000, expected_buckets=ALL_BUCKETS,
        rng=np.random.default_rng(20260806 + (0 if seed == "A" else 1)),
    )
    # Stringify tuple bucket keys for JSON serialization.
    ci_test = dict(ci_test)
    ci_test["bucket_sizes"] = {
        f"{k[0]}|{k[1]}": v for k, v in ci_test["bucket_sizes"].items()
    }
    h_val = compute_h_asr(val_dicts, asr_by_row, expected_buckets=ALL_BUCKETS)
    h_pooled = compute_h_asr(all_dicts, asr_by_row, expected_buckets=ALL_BUCKETS)
    return {
        "seed": seed,
        "n_test": len(test_rows),
        "n_val": len(val_rows),
        "h_asr_test": h_test["h_asr"],
        "h_asr_val": h_val["h_asr"],
        "h_asr_pooled": h_pooled["h_asr"],
        "ci_test": ci_test,
        "buckets_test": list(h_test["buckets"].values()),
        "pooled_cer_test": {
            "mixture": pooled_cer(test_dicts, asr_by_row, "mixture"),
            "clean": pooled_cer(test_dicts, asr_by_row, "clean"),
        },
    }


def gpu_identity() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            }
    except Exception:  # noqa: BLE001
        pass
    return {"device": "unavailable"}


def render_markdown(report: dict) -> str:
    lines = ["# R5 Overall-first ASR oracle report", ""]
    lines.append("**Diagnostic, public-only.** `H_ASR` is an Overall ceiling for CER-only "
                 "recovery, not an achieved gain. No Dataset-A artifact was read or executed.")
    lines.append("")
    gpu = report["gpu"]
    lines.append(f"GPU: {gpu.get('device')} | torch {gpu.get('torch')} | CUDA {gpu.get('cuda')} "
                 f"| {gpu.get('mem_gb')} GiB")
    lines.append(f"ASR config digest: `{report.get('config_digest', '')}`")
    lines.append("")
    lines.append("## Headline (test split)")
    lines.append("")
    lines.append("| seed | n_test | H_ASR | 95% CI | clean CER | mix CER |")
    lines.append("|---|---|---|---|---|---|")
    for r in report["per_seed"]:
        ci = r["ci_test"]
        lines.append(
            f"| {r['seed']} | {r['n_test']} | {r['h_asr_test']:.4f} | "
            f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] | "
            f"{r['pooled_cer_test']['clean']:.4f} | {r['pooled_cer_test']['mixture']:.4f} |"
        )
    lines.append("")
    br = report["branch"]
    lines.append(f"**Branch recommendation:** {br['recommendation']}")
    lines.append("")
    reasons = br["reasons"]
    lines.append(f"Inconclusive = {br['inconclusive']} "
                 f"(CI crosses boundary = {reasons.get('ci_crosses_boundary')}; "
                 f"different bands = {reasons.get('different_bands')} "
                 f"bands={reasons.get('bands')}; "
                 f"CI overlap = {reasons.get('ci_overlap')}).")
    lines.append("")
    lines.append("## Bucket gaps G_b = CER_mix - CER_clean (test split, signed)")
    lines.append("")
    for r in report["per_seed"]:
        lines.append(f"### seed {r['seed']}")
        lines.append("| snr_db | overlap | n | clean CER | mix CER | gap G_b | weight |")
        lines.append("|---|---|---|---|---|---|---|")
        for b in r["buckets_test"]:
            lines.append(
                f"| {b['snr_db']} | {b['overlap_ratio']} | {b['count']} | "
                f"{b['cer_clean']:.4f} | {b['cer_mix']:.4f} | {b['gap']:.4f} | "
                f"{b['weight']:.4f} |"
            )
        lines.append(f"H_ASR (test) = 0.5 * Σ w_b * G_b = **{r['h_asr_test']:.4f}** "
                     f"(95% CI [{r['ci_test']['ci_low']:.4f}, {r['ci_test']['ci_high']:.4f}])")
        lines.append("")
    lines.append("## Latency (ASR-segment batch-1, NOT full-pipeline)")
    lines.append("")
    lat = report["latency"]
    lines.append("| path | count | mean (ms) | p50 (ms) | p95 (ms) | total (s) |")
    lines.append("|---|---|---|---|---|---|")
    for role in ("mixture", "clean_target"):
        s = lat.get(role, {})
        if s.get("count", 0) == 0:
            lines.append(f"| {role} | 0 | - | - | - | - |")
        else:
            lines.append(f"| {role} | {s['count']} | {s['mean_ms']:.1f} | {s['p50_ms']:.1f} | "
                         f"{s['p95_ms']:.1f} | {s['total_s']:.1f} |")
    lines.append("")
    lines.append("_Latency excludes warm-up and uses CUDA synchronization; it does NOT include "
                 "enrollment, TSE, or routing overhead, so it is not full-pipeline latency._")
    lines.append("")
    if report.get("counts"):
        lines.append("## Public set counts and digests (from index)")
        lines.append("")
        lines.append("| seed | rows | val | test | manifest digest |")
        lines.append("|---|---|---|---|---|")
        for seed, info in report["counts"].items():
            sr = info.get("split_rows", {})
            lines.append(f"| {seed} | {info['row_count']} | {sr.get('val', 0)} | "
                         f"{sr.get('test', 0)} | `{info['manifest_digest'][:16]}…` |")
        lines.append("")
    lines.append("## Decision gates")
    lines.append("")
    lines.append("- `H_ASR < +0.020`: close custom TSE.")
    lines.append("- `+0.020 <= H_ASR < +0.040`: pretrained / very short TSE mini-pilot only.")
    lines.append("- `H_ASR >= +0.040`: scratch TSE eligible (not auto-approved).")
    lines.append("")
    lines.append("Inconclusive if any seed's 95% CI crosses a gate boundary, the seeds occupy "
                 "different bands, or the seeds' CIs do not overlap.")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr", required=True, help="ASR output JSONL")
    parser.add_argument("--index", default=None, help="r5_oracle_v1/index.json")
    parser.add_argument("--manifest", action="append", default=None,
                        help="R5 manifest.jsonl (repeatable, if no --index)")
    parser.add_argument("--output-json", default="output/r5_oracle/r5_oracle_report.json")
    parser.add_argument("--output-md", default=str(REPORT_DIR / "r5-oracle-report.md"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_rows, index = load_manifest_rows(index=args.index, manifest=args.manifest)
    records = load_asr_records(args.asr)
    # Fail-closed evidence validation against the verified manifest.
    asr_by_row, config_digest = validate_asr_evidence(manifest_rows, records, recheck_files=True)

    seeds = sorted({r.seed for r in manifest_rows})
    per_seed = [per_seed_report(manifest_rows, asr_by_row, seed=s) for s in seeds]
    branch = branch_recommendation(per_seed)
    counts = index.get("seeds", {}) if index else {}

    report = {
        "gpu": gpu_identity(),
        "config_digest": config_digest,
        "weights": {f"{s}|{o}": BUCKET_WEIGHTS[(float(s), float(o))]
                    for s in SNR_GRID for o in OVERLAP_GRID},
        "n_records": len(records),
        "n_manifest_rows": len(manifest_rows),
        "counts": counts,
        "per_seed": per_seed,
        "branch": branch,
        "latency": latency_stats(records),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "branch_recommendation": branch["recommendation"],
        "inconclusive": branch["inconclusive"],
        "per_seed": [{"seed": r["seed"], "h_asr_test": r["h_asr_test"],
                      "ci_test": r["ci_test"]} for r in per_seed],
        "output_json": str(out_json),
        "output_md": str(out_md),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
