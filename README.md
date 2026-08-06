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

## Public-only data boundary

The TSE pilot trainer (`scripts/train_tse.py`) trains on the existing public
synthetic manifest only (`data/synthetic/aishell1_phase2_v2/manifest.jsonl`). It
never reads Dataset-A audio, labels, IDs, paths, predictions, or metrics.
Dataset-A is used solely as a forbidden containment root and is rejected if any
training audio path resolves beneath it. Only `target_present == true` rows feed
the reconstruction objective; speaker-disjoint splits are validated before
training, and the frozen WeSpeaker enrollment encoder carries no gradient. No
Dataset-A data is used in checkpoint selection, threshold fitting, or early
stopping. Generated checkpoints, audio, and caches are gitignored and must not be
uploaded.

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

Use local, explicitly supplied paths for AISHELL and any permitted RIR/noise assets. Do not commit
datasets, credentials, or generated outputs.
