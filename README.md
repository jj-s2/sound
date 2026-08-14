# sound

Research and evaluation code for robust Chinese command speech recognition in complex
interactive scenes. The repository contains the model/data contracts, synthetic-mixture
renderers, ASR evidence validation, scoring utilities, and regression tests used by the
XH-202615 experiments.

## Scope

- Public-data-only R5 Overall oracle generation and evaluation.
- Deterministic mixture rendering with manifest/file-digest validation.
- CUDA FunASR evidence collection with resumable output handling.
- CER/RR/Overall-related evaluation and training utilities.
- Public-only TSE (target speaker extraction) pilot trainer.
- R7 enrollment-conditioned speaker-cosine rejection with deterministic
  impostor hard negatives and public AISHELL transcript-backed calibration.

## Public-only data boundary

The TSE trainers train only on public/synthetic manifests. R7 uses
`scripts/prepare_r7_manifest.py` to combine R3 counterfactual negatives with
deterministic different-speaker impostors; the optional AISHELL transcript only
supplies public positive references for calibration. Dataset-A audio, IDs, and
labels are never placed in manifests or checkpoints and are rejected as training
paths. After a public baseline is recorded, the competition owner has authorized
read-only Dataset-A metric use for threshold/routing/hyperparameter tuning; such
tuning must be logged separately and must not alter the training corpus. The
frozen WeSpeaker encoder carries no gradient. Generated checkpoints, audio, and
caches are gitignored and must not be uploaded.

## TSE pilot command

```powershell
# Controlled short pilot (defaults: 2 epochs, small batch, CUDA + AMP when available)
.venv\Scripts\python.exe scripts\train_tse.py --output-dir output\tse_pilot

# CUDA smoke (<=16 positive rows, 1 epoch) for code review before the full pilot
.venv\Scripts\python.exe scripts\train_tse.py `
  --output-dir output\tse_pilot_smoke --limit-per-split 5 --epochs 1 --batch-size 4
```

`--limit-per-split` caps the positive rows per split (smoke runs). The best
validation checkpoint is written to `best.pt` and a full audit (manifest digest,
data boundary, model config, seed, training history) to `summary.json`.

## R7 speaker-score pilot

Build a public R7 manifest (the default transcript path is auto-detected):

```powershell
.venv\Scripts\python.exe scripts\prepare_r7_manifest.py `
  --r3-manifest data\synthetic\r3_public_pilot_v1\manifest.jsonl `
  --output data\synthetic\r3_r7_speaker_v1\manifest.jsonl `
  --dataset-a-root datasetA\datasetA --impostor-fraction 0.5 --seed 20260806 `
  --transcript "data_aishell (1)\data_aishell\transcript\aishell_transcript_v0.8.txt"

.venv\Scripts\python.exe scripts\train_tse.py `
  --manifest data\synthetic\r3_r7_speaker_v1\manifest.jsonl `
  --output-dir output\tse_r7_joint --with-presence --with-speaker-score `
  --epochs 8 --batch-size 8 --segment-seconds 2.0 --device cuda
```

R7 inference writes enhanced/mixture/max cosine scores. Calibrate the variant
and threshold on the public val split with
`scripts/evaluate_tse_presence.py --calibrate`; use the resulting fixed
configuration for the official evaluation.


Datasets, generated audio, model checkpoints, ASR outputs, and local experiment artifacts are
intentionally excluded from this repository. Obtain the permitted public datasets separately and
pass their locations through the command-line options; no Dataset-A labels are required by the R5
oracle.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-runtime-windows.txt
.venv\Scripts\python -m pytest -q
```

The R5 pipeline is documented in `docs/` and implemented by:

```text
scripts/prepare_r5_oracle.py
scripts/r5_oracle_asr.py
scripts/r5_oracle_report.py
xh202615/r5_oracle.py
```

## TSE inference adapter

After the public-only pilot, run the trained checkpoint to create enhanced
command audio for the existing FunASR runner:

```powershell
.venv\Scripts\python.exe scripts\run_tse_inference.py `
  --checkpoint output\tse_pilot\best.pt `
  --dataset-root datasetA\datasetA `
  --splits pos,neg `
  --output-root output\tse_inference\enhanced `
  --output-map output\tse_inference\audio_map.jsonl `
  --embedding-cache output\tse_inference\enrollment_embeddings.pt `
  --device cuda --resume
```

Pass the resulting map to `scripts\run_funasr_asr.py` with
`--command-audio-map`. The adapter reads only input-side audio fields and keeps
generated audio, caches, predictions, and evaluation artifacts outside Git.

Use local, explicitly supplied paths for AISHELL and any permitted RIR/noise assets. Do not commit
datasets, credentials, or generated outputs.
# R12 speech-command pipeline

For the reproducible Dataset-A train-only Paraformer preparation, training dry run, test boundary, and GitHub export policy, see [R12 training and publishing](docs/r12/r12-train-and-publish.md).
