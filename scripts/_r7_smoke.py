"""Bounded R7 CUDA smoke: train -> infer -> calibrate -> evaluate.

Runs on CUDA with the REAL public R7 manifest audio (r3_r7_speaker_v1, which
contains impostor + counterfactual hard negatives) and the REAL frozen WeSpeaker
``chinese`` encoder for both enrollment and the speaker-cosine reject score.
FunASR is NOT run here; a MOCK ASR transcript is used for the CER/Overall leg so
the smoke stays bounded. The CER/Overall numbers are DIAGNOSTIC ONLY.

The go/no-go signal from this smoke is the **public-val speaker-score AUC per
variant** and the **present-vs-absent cosine separation**: these validate the
core R7 thesis (an enrollment-conditioned cosine is a domain-invariant, speaker-
identity reject signal) on real audio with real WeSpeaker, independent of TSE
training quality (the mixture_cosine variant does not depend on the extractor).

Outputs go to output/tse_r7_smoke/. Not a full training run.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train_tse import run_training  # noqa: E402
from scripts.run_tse_inference import run_inference  # noqa: E402
from xh202615.data import Sample  # noqa: E402
from xh202615.speaker_score import (  # noqa: E402
    SCORE_VARIANTS,
    select_score_variant,
)
from xh202615.tse_presence import (  # noqa: E402
    load_all_score_fields,
    overall_at_threshold,
    overall_from_metrics,
)

MANIFEST = REPO / "data/synthetic/r3_r7_speaker_v1/manifest.jsonl"
OUT = REPO / "output/tse_r7_smoke"
POS_PER = 8
NEG_PER = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "chinese"


def _subset(rows: list[dict]) -> list[dict]:
    subset = []
    for split in ("train", "val", "test"):
        pos = [r for r in rows if r["split"] == split and r["target_present"]][:POS_PER]
        neg = [r for r in rows if r["split"] == split and not r["target_present"]][:NEG_PER]
        subset.extend(pos + neg)
    return subset


def _mock_asr(row_ids: list[str], present_ids: set[str]) -> dict[str, str]:
    # Perfect ASR for present rows (isolates gating from ASR quality); a fixed
    # non-empty string for absent rows (so a false-accept is detectable).
    return {rid: ("你好世界" if rid in present_ids else "干扰语音") for rid in row_ids}


def main() -> None:
    assert MANIFEST.is_file(), f"manifest not found: {MANIFEST} (run prepare_r7_manifest.py)"
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        subset = _subset(rows)
        sub_man = tmp / "subset.jsonl"
        sub_man.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in subset) + "\n", encoding="utf-8"
        )
        dataset_a = REPO / "datasetA/datasetA"
        present_ids = {r["row_id"] for r in subset if r["target_present"]}

        # 1) Joint training on CUDA with the REAL WeSpeaker speaker encoder.
        print(f"[smoke] training on {DEVICE} with real WeSpeaker ({len(subset)} rows)...",
              flush=True)
        summary = run_training(
            manifest=sub_man, output_dir=OUT / "train", dataset_a_root=dataset_a,
            model_name=MODEL_NAME, embedding_dim=256, channels=(16, 32, 64),
            gru_hidden=128, gru_layers=2, epochs=2, batch_size=4, segment_seconds=2.0,
            seed=20260806, device=DEVICE, reuse_cache=False,
            with_presence=True, presence_weight=1.0, with_speaker_score=True,
        )
        best = next(h for h in summary["history"] if h["epoch"] == summary["best_epoch"])
        spk = best["val"]["speaker"]
        pres = best["val"]["presence"]
        print("[smoke] === VAL SPEAKER-SCORE DIAGNOSTICS (real WeSpeaker) ===", flush=True)
        print(f"  presence_auc         = {pres['auc']:.4f}", flush=True)
        for v in SCORE_VARIANTS:
            pv = spk["per_variant"][v]
            print(f"  {v:18s} auc={pv['auc']:.4f} youden_thr={pv['threshold']:.4f}", flush=True)
        print(f"  selected variant     = {spk['score_type']}", flush=True)
        print(f"  speaker_val_auc      = {spk['auc']:.4f}  youden_thr={spk['threshold']:.4f}",
              flush=True)

        # 2) Inference on val rows -> enhanced audio + real speaker cosines.
        eval_rows = [r for r in subset if r["split"] == "val"]
        inj = tmp / "input.jsonl"
        inj.write_text(
            "\n".join(json.dumps({
                "id": r["row_id"], "split": r["split"],
                "wakeup_audio": r["enrollment_audio"], "command_audio": r["mixture_audio"],
            }, ensure_ascii=False) for r in eval_rows) + "\n",
            encoding="utf-8",
        )
        print(f"[smoke] inference on {len(eval_rows)} val rows (real WeSpeaker)...", flush=True)
        from scripts.run_tse_inference import read_input_jsonl  # noqa: E402
        input_rows = read_input_jsonl(inj)
        run_inference(
            input_rows, checkpoint=OUT / "train" / "best.pt",
            output_root=OUT / "enhanced", output_map=OUT / "audio_map.jsonl",
            embedding_cache=OUT / "enrollment_embeddings.pt",
            device=DEVICE, model_name=MODEL_NAME,
        )

        # 3) Present-vs-absent cosine separation on val (real WeSpeaker scores).
        all_scores = load_all_score_fields(OUT / "audio_map.jsonl")
        val_present = {r["row_id"] for r in eval_rows if r["target_present"]}
        val_absent = {r["row_id"] for r in eval_rows if not r["target_present"]}
        print("[smoke] === VAL COSINE SEPARATION (present vs absent) ===", flush=True)
        for v in SCORE_VARIANTS:
            if v not in all_scores:
                continue
            m = all_scores[v]
            p = np.mean([m[i] for i in val_present]) if val_present else float("nan")
            a = np.mean([m[i] for i in val_absent]) if val_absent else float("nan")
            print(f"  {v:18s} present_mean={p:.4f}  absent_mean={a:.4f}  gap={p - a:+.4f}",
                  flush=True)

        # 4) In-process variant selection on val (mock ASR) -> wiring + threshold.
        samples = [
            Sample(id=rid, split="val", wakeup_audio=".", wakeup_text="",
                   command_audio=".", label=("你好世界" if rid in val_present else None))
            for rid in (list(val_present) + list(val_absent))
        ]
        asr = _mock_asr(list(val_present) + list(val_absent), val_present)
        variant_scores = {v: all_scores[v] for v in SCORE_VARIANTS if v in all_scores}
        calib = select_score_variant(
            samples, asr, variant_scores,
            overall_at_threshold=overall_at_threshold,
            overall_from_metrics=overall_from_metrics,
        )
        print("[smoke] === CALIBRATE (val, mock ASR) ===", flush=True)
        print(f"  selected variant = {calib['score_type']}", flush=True)
        print(f"  threshold         = {calib['threshold']:.4f}", flush=True)
        print(f"  public_overall    = {calib['metrics']['overall']:.4f} "
              f"(CER={calib['metrics']['avg_cer']:.4f}, RR={calib['metrics']['avg_rr']:.4f})",
              flush=True)
        for v, pv in calib["per_variant"].items():
            print(f"    {v:18s} overall={pv['overall']:.4f}", flush=True)

        # 5) GO/NO-GO. The smoke uses a barely-trained TSE (2 epochs, tiny data),
        # so enhanced_cosine is not expected to discriminate yet. The TSE-
        # independent mixture_cosine is the thesis test: a domain-invariant
        # speaker signal must separate present from absent even without a good
        # extractor. Full Overall improvement still requires Codex's trained run.
        mix_auc = spk["per_variant"].get("mixture_cosine", {}).get("auc", 0.0)
        enh_auc = spk["per_variant"].get("enhanced_cosine", {}).get("auc", 0.0)
        verdict = "GO" if mix_auc > 0.60 else "NO-GO"
        print(f"[smoke] === VERDICT: {verdict} ===", flush=True)
        print(f"  mixture_cosine_auc = {mix_auc:.4f}  (TSE-independent thesis test; >0.60 = GO)",
              flush=True)
        print(f"  enhanced_cosine_auc= {enh_auc:.4f}  (expected ~0.5 with an untrained TSE; "
              "needs Codex's full run)", flush=True)
        print("[smoke] NOTE: CER/Overall are mock-ASR DIAGNOSTIC ONLY; the official Dataset-A",
              flush=True)
        print("[smoke] Overall is not claimed and must be run by Codex.", flush=True)


if __name__ == "__main__":
    main()
