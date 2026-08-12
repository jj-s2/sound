# Task 2 Report: Train-only Calibrated Leaf7/Leaf15 Gate and Robust Validation Selection

## Summary

Implemented `xh202615/r12_calibrated_gate.py` and `tests/test_r12_calibrated_gate.py`
per the Task 2 brief. The module reuses frozen R11 gate pieces, builds train-only
3-fold group-disjoint OOF probabilities for the two approved HGB base models, fits
per-model deterministic balanced Platt calibrators, refits the bases on the whole
train partition, and selects a frozen validation configuration under strict raw-RR
and grouped-bootstrap-RR floors.

## Files changed

- `xh202615/r12_calibrated_gate.py`
- `tests/test_r12_calibrated_gate.py`
- `.superpowers/sdd/2026-08-11-r12-strict-holdout-calibration/task-2-report.md`

## RED (tests before production code)

Command:

```powershell
python -m pytest tests\test_r12_calibrated_gate.py -q
```

Result:

```
FFFFFFFFFFFFFF  [100%]
14 failed in 7.11s
```

All failures were `ModuleNotFoundError` / `ImportError` because
`xh202615/r12_calibrated_gate.py` did not yet exist.

## GREEN (implementation complete)

Focused Task 2 tests:

```powershell
python -m pytest tests\test_r12_calibrated_gate.py -q
```

Result:

```
....................  [100%]
20 passed in 28.19s
```

R11 E2 regression tests:

```powershell
python -m pytest tests\test_r11_pvad_oracle_oof.py tests\test_r11_pvad_oracle.py -q
```

Result:

```
...............................................  [100%]
47 passed in 161.93s (0:02:41)
```

## Key implementation notes

- Public constants `BASE_MODELS` and `BLEND_WEIGHTS` exactly match the brief.
- `fit_train_calibrated_gate` takes only `joined_train` and `seed`; no label or
  validation argument is accepted, satisfying the "no held-out labels" rule.
- Feature matrix is built from `JoinedPvadRow.e0` using the frozen
  `E0_FITTING_FEATURE_SCHEMA`; no feature extraction is duplicated.
- 3-fold group-disjoint OOF uses `cross_fit_gate_models` with
  `StratifiedGroupKFold(3, shuffle=True, random_state=seed)` and the two fixed
  HGB specs (`leaf_7`, `leaf_15`).
- A deterministic balanced `LogisticRegression` calibrator is fit per base model
  on its own train OOF score column and `target_present`.
- Base pipelines are refit on the complete training partition only after the OOF
  calibration bank is complete.
- `select_on_validation` scores validation with the refit bases and calibrators,
  evaluates the two calibrated base variants plus five fixed blends (weight is
  leaf15), enumerates tied-score threshold boundaries plus reject-all, and
  filters by raw RR >= 0.95 and grouped-bootstrap 5th-percentile RR >= 0.93.
- Selection tie-breaker is: max bootstrap-median Overall, then higher raw RR,
  lower CER, lower threshold, lexical model name, then lower blend weight.
- Group bootstrap resamples whole validation groups only, redraws degenerate
  class samples, and is deterministic from `seed`.
- `FrozenGateSelection.to_dict()` exposes the required serial fields (base model
  names/parameters digest, feature schema digest, calibrator coefficient/intercept,
  blend definition, threshold, validation raw/bootstrapped metrics, provenance)
  and excludes fitted sklearn objects, labels, references, candidate-CER,
  optimal-action, embedding, and frame fields.
- `predict_with_selection` returns binary accept/reject predictions for any joined
  rows using the frozen selected model/blend and threshold.

## Commit

`feat: add R12 train-only calibrated gate`
