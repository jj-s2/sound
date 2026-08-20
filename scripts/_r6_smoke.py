"""Bounded R6 CUDA smoke: train -> infer -> calibrate -> evaluate.

Runs on CUDA with the REAL public manifest audio (aishell1_phase2_v2) but uses
a MOCK 256-D enrollment-embedding provider and MOCK ASR text, because WeSpeaker
and FunASR are not installed in this environment. It exercises the full R6
wiring (joint presence training, presence-score inference, presence-gated
Overall evaluation) on a tiny balanced subset. Metrics are DIAGNOSTIC ONLY.

Outputs go to output/tse_r6_smoke/.
"""

from __future__ import annotations

import hashlib
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
from xh202615.tse_presence import (  # noqa: E402
    calibrate_threshold_overall,
    overall_at_threshold,
    samples_from_manifest,
)

MANIFEST = REPO / "data/synthetic/aishell1_phase2_v2/manifest.jsonl"
OUT = REPO / "output/tse_r6_smoke"
DIM = 256
POS_PER = 8
NEG_PER = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mock_provider(path: Path) -> np.ndarray:
    b = hashlib.sha256(str(path).encode()).digest()  # 32 bytes
    return np.tile(np.frombuffer(b, dtype=np.uint8), 8)[:DIM].astype(np.float32) / 255.0 - 0.5


def main() -> None:
    assert MANIFEST.is_file(), f"manifest not found: {MANIFEST}"
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        subset = []
        for split in ("train", "val", "test"):
            pos = [r for r in rows if r["split"] == split and r["target_present"]][:POS_PER]
            neg = [r for r in rows if r["split"] == split and not r["target_present"]][:NEG_PER]
            subset.extend(pos + neg)
        sub_man = tmp / "subset.jsonl"
        sub_man.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in subset) + "\n", encoding="utf-8")
        dataset_a = REPO / "datasetA/datasetA"

        # 1) Joint training on CUDA (mock embeddings).
        summary = run_training(
            manifest=sub_man, output_dir=OUT / "train", dataset_a_root=dataset_a,
            embedding_dim=DIM, channels=(16, 32, 64), gru_hidden=128, gru_layers=2,
            epochs=2, batch_size=4, segment_seconds=2.0, seed=20260806,
            device=DEVICE, reuse_cache=False, embedding_provider=mock_provider,
            with_presence=True, presence_weight=1.0,
        )
        print("TRAIN done. device=", summary["device"],
              "val_auc=", round(summary["presence_auc"], 4),
              "val_threshold=", round(summary["presence_threshold"], 4),
              "class_balance=", summary["class_balance"], flush=True)

        # 2) Inference on val+test rows -> enhanced audio + presence_score.
        eval_rows = [r for r in subset if r["split"] in ("val", "test")]
        inj = tmp / "input.jsonl"
        inj.write_text(
            "\n".join(json.dumps({
                "id": r["row_id"], "split": r["split"],
                "wakeup_audio": r["enrollment_audio"], "command_audio": r["mixture_audio"],
            }, ensure_ascii=False) for r in eval_rows) + "\n",
            encoding="utf-8",
        )
        from scripts.run_tse_inference import read_input_jsonl
        inrows = read_input_jsonl(inj)
        inf = run_inference(
            inrows, checkpoint=OUT / "train/best.pt", output_root=OUT / "enhanced",
            output_map=OUT / "audio_map.jsonl", embedding_cache=OUT / "cache.pt",
            device=DEVICE, model_name="mock", embedding_provider=mock_provider,
            manifest_digest="r6-smoke",
        )
        print("INFER done. rows=", inf["rows"], "with_presence=", inf.get("with_presence"),
              "threshold=", round(inf.get("presence_threshold", 0.0), 4), flush=True)
        presence = {rec["id"]: rec["presence_score"]
                    for rec in (json.loads(l) for l in (OUT / "audio_map.jsonl").read_text(encoding="utf-8").splitlines())}

        # 3) Mock ASR: positives -> perfect transcript; negatives -> non-empty
        #    (tests that the presence gate rejects transcribed absent rows).
        asr = {}
        for r in eval_rows:
            if r["target_present"]:
                asr[r["row_id"]] = r["text"] or ""
            else:
                asr[r["row_id"]] = "噪声干扰语音"
        asr_path = OUT / "mock_asr.jsonl"
        asr_path.write_text("\n".join(json.dumps({"id": k, "text": v}, ensure_ascii=False) for k, v in asr.items()) + "\n", encoding="utf-8")

        # 4) Calibrate threshold on val, evaluate on test (public proxy).
        val_samples = samples_from_manifest(sub_man, "val")
        test_samples = samples_from_manifest(sub_man, "test")
        cal = calibrate_threshold_overall(val_samples, asr, presence)
        test_metrics = overall_at_threshold(test_samples, asr, presence, cal["threshold"])
        result = {
            "device": summary["device"],
            "manifest": str(MANIFEST),
            "subset_rows": len(subset),
            "train_val_auc": summary["presence_auc"],
            "train_val_threshold": summary["presence_threshold"],
            "calibrated_threshold": cal["threshold"],
            "calibrate_val_overall": cal["metrics"]["overall"],
            "test_overall": test_metrics["overall"],
            "test_cer": test_metrics["avg_cer"],
            "test_rr": test_metrics["avg_rr"],
            "test_far": test_metrics["false_accept_rate"],
            "test_frr": test_metrics["false_reject_rate"],
            "note": "DIAGNOSTIC ONLY; mock embeddings + mock ASR; not an official Overall.",
        }
        (OUT / "smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("SMOKE RESULT:", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
