# Task 2a Report

This is **Task 2a** only: the R12 train-only calibrated base-gate core.
Validation selection, threshold/frontier search, blending, and bootstrap
evaluation are deliberately excluded from this changeset.

## Files changed

- `xh202615/r12_calibrated_gate.py` — production code
- `tests/test_r12_calibrated_gate.py` — contract tests
- `task-2-report.md` — this report

## RED / GREEN commands

```powershell
# RED: run the new Task 2a tests before production code exists
pytest tests/test_r12_calibrated_gate.py -v

# GREEN: run the same tests after implementing the production code
pytest tests/test_r12_calibrated_gate.py -v
```

## GREEN result

```
14 passed in ~15s
```

## Scope confirmation

- Uses only `E0_FITTING_FEATURE_SCHEMA` and `JoinedPvadRow` from R11.
- Reuses `_fit_gate_pipeline` and `cross_fit_gate_models` from R11 without modification.
- Defines exactly two HGB specs: `hist_gradient_boosting_leaf_7` and `hist_gradient_boosting_leaf_15`.
- Computes 3-fold group-disjoint train-only OOF probabilities via `StratifiedGroupKFold(3, shuffle=True, random_state=seed)`.
- Fits one deterministic balanced `LogisticRegression` calibrator per base model on its own OOF column and train `target_present` labels.
- Refits each base pipeline on all training rows.
- `predict_calibrated` accepts joined rows and returns an `(n_rows, 2)` array of calibrated scores in `[0, 1]`.
- Rejects empty inputs, non-binary targets, single-class training sets, and insufficient groups.
- No validation data, labels parameter, selection, or threshold logic is present.
