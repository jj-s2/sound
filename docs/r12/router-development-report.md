# R12 deployable candidate-router development report

## Boundary

The existing 368-row Dataset-A held-out partition is **contaminated** because its labels were opened on 2026-08-12. It is not used for selection, router training, hyperparameter choice, threshold choice, or promotion. The figures below are train-to-validation development evidence only, not an independent final result.

## Run

- Canonical input: 1,838 Dataset-A rows rebuilt from the raw candidate sources and CPU pVAD cache.
- Roles: 1,103 train, 367 validation, 368 contaminated held-out rows (not loaded by the command).
- Router: candidate-error regressor trained on train-role positive rows only; inference inputs are candidate agreement/length features plus pVAD and E0 features.
- Gate: train-only leaf-7/leaf-15 calibrated gate; model/blend/finite threshold selected on validation with 2,000 grouped bootstrap replicates.
- Router action order for deterministic ties: `primary`, `r3`, `tse`, `energy`.

## Result

| Metric | Validation value |
| --- | ---: |
| CER | 0.4731416169 |
| RR | 0.9684210526 |
| Overall | 0.7476397179 |
| Bootstrap Overall median | 0.7477106491 |
| Bootstrap RR p05 | 0.9368085106 |

The selected policy was leaf-15 blend weight 1.0 at threshold 0.2737489523. Its validation action mix was primary 171, r3 167, tse 21, energy 8. This implementation is reproducible and deployable, but it does not meet the 0.8 development target and must not be promoted as such.

Selection artifact SHA-256: `61f09fa9eaad2ba2fd30dffaaa906a979f232806e01a1bc71492485cb5465284`.

