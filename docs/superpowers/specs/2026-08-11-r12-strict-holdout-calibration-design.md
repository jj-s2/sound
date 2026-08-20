# R12 Strict Holdout Calibration Design

## Goal

Produce a minimal reproducible R12 pipeline that rebuilds its inputs from raw Dataset-A audio, trains gate models only on designated development groups, and reports a final score only on a group-disjoint held-out test set. The acceptance target is Overall > 0.80 with RR >= 0.93.

## Baseline and evidence boundary

- The pre-cleanup source snapshot is the local Git tag `r12-before-structure-cleanup-20260811` at commit `cf5cce3`.
- The current frozen R12 grouped-OOF evidence package reached Overall 0.8026902726469429, but it is an oracle/OOF diagnostic and not an independent test result.
- The fixed Fold-4 probe reached Overall 0.7851155036913551 with the validation-selected threshold. Fold 4 has now been inspected, including diagnostic threshold scans, so it is a development diagnostic set and cannot be used as the final blind test set.
- The existing oracle metric chooses the lowest-CER candidate among `r3`, `primary`, `energy`, and `tse` for a positive row. This remains an explicit diagnostic metric; it must not be presented as a deployable end-to-end ASR result.

## Data and split contract

1. Build a deterministic, group-disjoint split manifest from Dataset-A IDs and wake groups.
2. The manifest contains exactly three roles: `train`, `validation`, and `held_out_test`; every canonical ID appears exactly once.
3. The target sizes are 60% train groups, 20% validation groups, and 20% held-out-test groups. Split generation uses a fixed seed recorded in the manifest, with class stratification by target presence.
4. The manifest records the ordered IDs, group IDs, counts, SHA-256 digests, and seed. It is generated before model experiments and is immutable for a run.
5. Training and model-selection commands receive only train and validation labels. The held-out labels are accepted only by the final evaluator command.
6. A new held-out manifest must be reserved before final reporting. The already-inspected Fold-4 manifest remains available only for diagnostics and regression tests.

## Reproducible input pipeline

The retained pipeline must rebuild these artifacts from raw Dataset-A audio:

1. Four candidate transcripts per ID: `r3`, `primary`, `energy`, and `tse`.
2. Label-free `canonical_input.jsonl`, containing candidate texts, inference-time acoustic features, source digests, and no labels or references.
3. `labels.json` and `groups.json`, emitted separately from the canonical input.
4. A CPU-only FireRed pVAD cache. It uses a wake utterance for a normalized ECAPA speaker embedding, runs the command utterance through recurrent ONNX pVAD, and stores the fixed 55 aggregate pVAD features with provenance hashes.

The canonical path remains CPU-only because prior CUDA/CPU core-feature parity did not hold.

## Development model protocol

The first improvement is a calibrated fused gate, without changing the frozen ASR or FireRed pVAD weights.

1. Start from the 96 fused inference-time features: 55 pVAD aggregates and 41 pre-existing E0 gate features. Exclude latency fields exactly as the current R12 fused family does.
2. Use the two strongest existing base specifications: `hist_gradient_boosting_leaf_7` and `hist_gradient_boosting_leaf_15`.
3. Within train groups, create 3 deterministic `StratifiedGroupKFold` splits. Fit each base model on each inner-train partition and collect once-only train OOF scores.
4. Fit a Platt-style logistic calibration layer from the two train OOF score columns to target presence. The calibration layer uses train labels only.
5. Refit each base model on all train groups. Apply the two models and the calibration layer to validation groups.
6. Evaluate blend weights `{0.0, 0.25, 0.5, 0.75, 1.0}`, where 1.0 is leaf-15 and 0.0 is leaf-7. For each weight, enumerate finite validation score thresholds.
7. A threshold is eligible only when validation RR is at least 0.95 and the 5th percentile of a grouped bootstrap RR distribution is at least 0.93. Select the eligible candidate with the highest bootstrap median Overall; break ties by higher median RR, lower median CER, then lower threshold.
8. Freeze the selected blend, calibration, and threshold before executing the held-out evaluator. The held-out evaluator may calculate metrics but may not alter any selection.

## Fallback: utility-aware gate

If the calibrated ensemble cannot beat the existing development-rotation baseline without violating the RR constraint, add a separate train-only utility model.

- Its target is the per-row CER reduction achieved by accepting a positive candidate, with zero utility for rejection and an explicit false-accept penalty for negatives.
- Inputs remain label-free inference-time features.
- Model and threshold selection remain validation-only under the same group-bootstrap RR contract.
- This is a diagnostic gate improvement only while the candidate choice remains oracle-derived.

## Evaluation gates

Before opening the new held-out test, each candidate must pass all of these development checks:

1. Every split has exact ID coverage and no train/validation group overlap.
2. The train OOF scores are once-only and finite.
3. Validation selection uses no held-out labels, references, candidate CER, or oracle action.
4. Across five diagnostic rotations, mean Overall exceeds 0.7900480643992768, no diagnostic fold has RR below 0.93, and at least three rotations exceed 0.80.
5. The original scoped R12 test suite passes.

The final held-out test passes only when Overall is strictly greater than 0.80 and RR is at least 0.93. A failed held-out test is reported as failed; it is not used to retune the model or threshold.

## Publication and cleanup

The final run publishes the exact five public evidence files: manifest, OOF/held-out scores, frontier, summary, and report. Public artifacts exclude labels, references, candidate texts, candidate CER, oracle actions, raw embeddings, and frame arrays.

Only after a strict pipeline is frozen will the project be reduced to the source, tests, scripts, model-asset checks, and documentation required for raw Dataset-A input generation, CPU pVAD caching, gate fitting, and evaluation. Legacy code remains recoverable through the snapshot tag.

## Verification

- Unit tests cover split exclusivity, source-artifact reconstruction, calibration OOF coverage, validation-only selection, held-out immutability, and forbidden-field rejection.
- Integration tests run a small explicit noncanonical subset with a matching small cache.
- The canonical CPU run verifies 1838-row coverage, provenance digests, output contract, and official evaluator parity.
