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
