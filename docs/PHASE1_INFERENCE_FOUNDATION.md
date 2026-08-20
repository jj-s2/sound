# Phase 1 inference foundation

This phase makes the XH-202615 pipeline replayable and auditable before any new model is allowed
to affect the competition score. It separates ASR text, temporal speaker evidence, routing policy,
resource measurement, evaluation, and submission validation.

## Honesty boundary

Replay reads frozen ASR and speaker artifacts. It proves that the contracts, provenance fields,
missing-evidence fallback, policy decisions, and output schemas compose correctly. It does not prove
that a new ASR, TSE, enhancement model, or official runtime is better. Replay timings are marked
`measurement_mode: replay` and must never be submitted as the competition duration. Dataset-A labels
are evaluation-only: do not use them to calibrate thresholds, make phrase rules, select templates,
or route inference.

The competition score remains `Overall = ((1 - CER) + RR) / 2`, with CER and RR each weighted 40%
and efficiency weighted 20% by the rules document.

## Environment

The reproducible Windows environment is `.venv`, created with `--system-site-packages` so the
existing CUDA Torch installation is reused. The checked runtime is Python 3.12.10, Torch
2.4.1+cu121, CUDA 12.1, RTX 4060 Laptop GPU, FunASR 1.4.1, ModelScope 1.39.1, WeSpeaker 0.0.0
from the pinned upstream commit, and ONNX Runtime GPU 1.28.0.

```powershell
. .\scripts\setup_runtime.ps1 -SkipInstall
# or install/reconcile pinned packages:
. .\scripts\setup_runtime.ps1
python .\scripts\environment_doctor.py `
  --package torch --package torchaudio --package funasr `
  --package modelscope --package wespeaker --package onnxruntime-gpu `
  --artifact datasetA.zip
```

Activate an already-created environment with `. .\scripts\activate_runtime.ps1`.

This venv intentionally reuses the host CUDA Torch installation. A raw `pip check` may therefore
also inspect unrelated host packages (for example OpenXLab or a host OpenCV build) and report their
constraints against the venv's NumPy 1.26 pin required by WeSpeaker's HDBSCAN dependency. The
project-level authority is the doctor report plus the import/GPU smoke checks above; no project
runtime import is failing.

## Phase-1 replay

```powershell
python .\scripts\run_phase1_replay.py `
  --dataset-root datasetA `
  --asr-map output/asr/funasr_full_hotword_safe.jsonl `
  --speaker-scores output/speaker/wespeaker_scores_full.csv `
  --config configs/phase1_replay.json `
  --predictions-out tmp/phase1_smoke/predictions.jsonl `
  --evidence-out tmp/phase1_smoke/evidence.jsonl `
  --routes-out tmp/phase1_smoke/routes.jsonl `
  --trace-out tmp/phase1_smoke/trace.json
```

The fixture config is deliberately named
`contract_fixture_only_not_competition_calibration`. Missing speaker evidence routes to raw ASR;
the enhancement action currently also falls back to replay ASR because no measured enhancer is
integrated in Phase 1.

## Evaluation and validation

Evaluate a prediction JSONL without changing the frozen artifacts:

```powershell
python -m xh202615.evaluate_predictions `
  --dataset-root datasetA `
  --predictions tmp/phase1_smoke/predictions.jsonl `
  --output tmp/phase1_smoke/metrics.json
```

Validate a diagnostic competition JSON. Optional fields are explicit; the validator defaults to
only `id` and `content`:

```powershell
python .\scripts\validate_competition_submission.py `
  --submission output/submissions/v4_equal_weight_best_competition_audio_name.json `
  --dataset-root datasetA `
  --allow-field label --allow-field cer
```

For official mode, provide a real (not replay) `RunTrace`:

```powershell
python .\scripts\validate_competition_submission.py `
  --submission path/to/submission.json `
  --dataset-root datasetA `
  --trace path/to/real_trace.json --official
```

The validator rejects missing or replay traces for official duration. It never treats a replay
latency as a valid efficiency measurement.

## Frozen baselines

The pure evaluator reproduces both frozen artifact metric files within `1e-12` for every metric:

| Artifact | CER | RR | Overall |
|---|---:|---:|---:|
| `stage_asr_only_full` | 0.3904541632 | 0.0021097046 | 0.3058277707 |
| `stage_dynamic_tse_fusion_full` | 0.5078847771 | 0.9388185654 | 0.7154668940 |

These are regression anchors, not new leaderboard claims. The screenshot’s approximate rank-7
reference was CER 0.4341, RR 0.8376, Overall 0.7018; future changes must report all three metrics
and the reject/false-accept tradeoff.

## Next experiment gates

1. **ASR bakeoff:** run FunASR variants on a fixed, label-free manifest; compare CER and latency
   with the same normalization and replay-compatible metadata.
2. **Temporal speaker evidence:** replace global-score replay with measured windows, then calibrate
   target/overlap routing only on an explicitly designated development split—not Dataset-A labels.
3. **Conditional enhancement:** integrate a real TSE/enhancement backend only after evidence shows
   enough overlap/gray-zone mass to justify its latency and memory cost.
4. **Official gate:** require real-run trace, exact submission IDs, no duplicate rows, no label
   leakage, and frozen-metric regression before generating a competition submission.
