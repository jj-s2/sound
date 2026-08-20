# R12 Deployable Candidate Router Design

## Goal

Raise the deployable Dataset-A development score toward Overall 0.80 without
using reference text at inference time or using the already-opened R12
held-out partition for feature, model, router, or threshold selection.

The development promotion target is grouped validation/OOF Overall >= 0.80,
RR >= 0.95, and CER <= 0.35. A final Overall claim requires a new blind
Dataset-B or a newly reserved Dataset-A partition that has never been opened.

## Evidence and framing

The current strict run uses a fixed `primary_text` after a presence gate and
scores Overall 0.703117, CER 0.593766, and RR 1.0 on the now-contaminated
368-row held-out partition. The newly rebuilt CPU pVAD cache is byte-identical
at the record level to the previous complete cache, so cache drift is not the
cause.

On that partition, perfect presence decisions with fixed primary text have an
Overall ceiling of about 0.80694. Perfect presence plus oracle four-candidate
routing has an Overall ceiling of about 0.84892. These numbers are diagnostic
only and must not be used for selection. They show that threshold tuning alone
has too little margin and that deployable candidate routing is the required
capability.

## Approaches considered

### A. Lower only the presence threshold

This is the smallest change, but validation evidence places the best raw
RR>=0.95 primary-only point near Overall 0.738. It cannot plausibly close the
full gap and risks trading false rejects for false accepts.

### B. One multiclass action classifier

Train a single classifier over `reject`, `primary`, `r3`, `tse`, and `energy`.
This directly predicts the final action but entangles presence detection with
text quality, makes RR constraints harder to enforce, and is sensitive to ties
between candidates.

### C. Two-stage presence gate plus candidate-quality router

Keep presence rejection and transcript routing separate. The gate enforces the
RR floor. For accepted rows, a candidate-quality model predicts each
candidate's normalized edit cost from inference-visible features and chooses
the lowest predicted cost. This is the selected design because it is
deployable, auditable, and lets each failure mode be measured independently.

## Architecture

### Candidate feature builder

For each row and each candidate action (`primary`, `r3`, `tse`, `energy`),
build only inference-visible features:

- candidate source one-hot encoding;
- empty flag and normalized character length;
- pairwise normalized edit similarity to the other three candidates;
- exact-agreement count and non-empty candidate count;
- length median, range, and deviation from the candidate consensus;
- existing label-free audio features;
- frozen pVAD/E0 fitting features already allowed by the gate contract.

Reference text, candidate CER, oracle action, labels, and held-out-derived
statistics are forbidden in serialized features and prediction artifacts.

### Candidate-quality model

Train one shared `HistGradientBoostingRegressor` on row-expanded training data.
The target is the candidate's clipped character error rate against the private
training label. Negative rows are excluded from router fitting because routing
is only invoked after acceptance. Candidate rows from one sample always stay
together in grouped folds.

Use grouped out-of-fold predictions on the train role to validate the router
implementation and refit one model on all train positives for validation and
deployment. Deterministic ties resolve in the frozen order `primary`, `r3`,
`tse`, `energy`.

### Presence gate

Reuse the R12 fused pVAD/E0 gate family and train calibration only on the train
role. The router does not alter gate probabilities. Infinite/reject-all
thresholds remain forbidden as deployable fallbacks.

### Joint validation selection

On validation only, enumerate finite gate thresholds for each calibrated base
or blend score. For accepted validation rows, use the router-selected text;
for rejected rows, emit empty text. Select by:

1. raw RR >= 0.95;
2. grouped-bootstrap RR p05 >= 0.93;
3. highest grouped-bootstrap median Overall;
4. higher raw RR;
5. lower raw CER;
6. lower finite threshold and deterministic model name.

If no candidate is feasible, fail closed with `BootstrapFeasibilityError` and
publish no selection.

## Data boundaries

- Train role: fit gate base models, calibrator, and candidate router.
- Validation role: select gate model, blend, threshold, and router
  hyperparameters from a small frozen grid.
- Existing held-out role: contaminated as of 2026-08-12; diagnostics may be
  retained, but it cannot influence implementation or promotion.
- New blind set: run exactly once after freezing all code and artifacts.

The candidate source files, canonical projection, group manifest, split,
cache, model identity, train/validation labels, router schema, router
parameters, and selected payload all require SHA-256 provenance.

## Interfaces and files

- Create `xh202615/r12_candidate_router.py` for feature construction, grouped
  router fitting, prediction, serialization, and validation.
- Create `tests/test_r12_candidate_router.py` for feature privacy, grouped
  folds, deterministic routing, source coverage, and serialization tests.
- Modify `xh202615/r12_calibrated_gate.py` so validation selection consumes a
  fixed deployable action per row instead of a single global transcript source.
- Modify `scripts/r12_strict_holdout.py` to fit/freeze the router during
  `select`, verify it during `evaluate`, and publish its provenance.
- Extend strict hold-out tests to prove held-out labels cannot change gate
  decisions, router actions, or recognition text.

## Error handling

Fail closed on missing or duplicate IDs, non-finite features or predictions,
candidate-source coverage mismatch, schema or digest drift, label leakage,
group leakage, unsupported router parameters, infeasible RR floors, and any
selection recomputation mismatch.

No partial evaluation package is published after a validation failure.

## Test strategy

1. Unit-test the candidate feature vector against hand-computed examples.
2. Prove all four candidates from one sample share a grouped fold.
3. Prove labels and held-out mutations do not alter serialized inference
   features or frozen predictions.
4. Prove deterministic tie-breaking and empty-candidate handling.
5. Prove joint selection uses router actions and refuses reject-all fallback.
6. Prove exact provenance recomputation on evaluation.
7. Run focused R12 tests, then the full repository suite.
8. Run grouped train/validation development evaluation without reading the
   contaminated held-out labels.

## Falsification and stopping rules

The design is falsified for this release if any of the following occurs:

- grouped validation/OOF Overall remains below 0.80;
- RR is below 0.95 or bootstrap RR p05 is below 0.93;
- CER remains above 0.35;
- router improvement is not positive under paired grouped bootstrap;
- provenance/privacy tests or existing repository tests fail;
- the router requires held-out-derived features or thresholds.

If falsified, retain the strict primary-only baseline and investigate new ASR
candidate generation or additional independently labeled development data;
do not tune against the contaminated held-out partition.
