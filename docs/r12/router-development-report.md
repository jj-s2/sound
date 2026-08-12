# R12 deployable candidate-router development report

## Boundary

The existing 368-row Dataset-A held-out partition is **contaminated** because its labels were opened on 2026-08-12. It is not used for selection, router training, hyperparameter choice, threshold choice, or promotion. The figures below are train-to-validation development evidence only, not an independent final result.

## Run

- Canonical input: 1,838 Dataset-A rows rebuilt from the raw candidate sources and CPU pVAD cache.
- Roles: 1,103 train, 367 validation, 368 contaminated held-out rows (not loaded by the command).
- Router: candidate-error regressor trained on train-role positive rows only; inference inputs are candidate agreement/length features plus pVAD and E0 features.
- Gate: train-only leaf-7/leaf-15 calibrated gate; model/blend/finite threshold selected on validation with 2,000 grouped bootstrap replicates.
- Router action order for deterministic ties: `primary`, `r3`, `tse`, `energy`.

## Candidate-router-only result

| Metric | Validation value |
| --- | ---: |
| CER | 0.4731416169 |
| RR | 0.9684210526 |
| Overall | 0.7476397179 |
| Bootstrap Overall median | 0.7477106491 |
| Bootstrap RR p05 | 0.9368085106 |

The selected policy was leaf-15 blend weight 1.0 at threshold 0.2737489523. Its validation action mix was primary 171, r3 167, tse 21, energy 8. This candidate-router-only implementation is reproducible and deployable, but it does not meet the 0.8 development target.

Selection artifact SHA-256: `61f09fa9eaad2ba2fd30dffaaa906a979f232806e01a1bc71492485cb5465284`.

## Text-presence fusion result

A separate train-only text-presence model was then added. It uses only `r3_text` and `primary_text` at inference, represented by character 1–3-gram TF–IDF and a balanced logistic classifier. Its score is fused with the frozen leaf-15 acoustic gate at weight 0.5. The model itself, input fields, parameters, and source/cache/split provenance are frozen in the selection artifact and rebuilt before any scoring label is opened.

| Metric | Validation value |
| --- | ---: |
| CER | 0.3288117200 |
| RR | 0.9684210526 |
| Overall | 0.8198046663 |
| Bootstrap Overall median | 0.8206409053 |
| Bootstrap RR p05 | 0.9368421053 |

The selected deployable policy is `text_gate_fusion`, text weight 0.5, at threshold 0.3804691416. Its strict development selection artifact SHA-256 is `dc55262f94719c9d8faeafa80e65daa985fcfb107bd6074942fb69c3499110ea`.

This clears the 0.8 target only on the validation role used for model-family selection. The contaminated Dataset-A held-out labels remain not used for selection or promotion; a newly blind partition or Dataset-B is required before claiming independent Overall ≥0.8.
