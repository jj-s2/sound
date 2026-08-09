# R11 E0 Cached Gate-Oracle Design

**Status:** Approved by the user's 2026-08-10 instruction to implement the
previously presented R11 decision.

**Parent decision:** `docs/r11-router-literature-decision.md`

## Purpose

Build the cheapest honest falsification test for the cached target-presence
features. E0 must answer whether any small multivariate gate using current
inference-side features has enough CER/RR operating-point headroom for an
eventual Overall above 0.8 when candidate selection is made artificially
perfect on accepted positive rows.

E0 is a diagnostic upper bound, not a deployable selector. Candidate quality is
allowed to use labels only to construct the positive oracle contribution. Gate
features and cross-fitted gate scores must never use reference text, label text,
candidate CER, or action-oracle labels.

## Inputs

Reuse the frozen R10 joins and exact Dataset-A leakage groups:

- `output/asr/candidate_fusion_smoke.jsonl`
- `output/training_r9/datasetA_tse/asr_predictions.jsonl`
- `output/training_r9/datasetA_tse/audio_map.jsonl`
- `output/evaluations/r3_temporal_on_datasetA_gated.jsonl`
- `.superpowers/sdd/2026-08-07-r9-overall-08-arena/datasetA_group_manifest_v1.json`
- `datasetA/datasetA`

Every ID must occur exactly once, and every outer train/test split must be
disjoint by `wake_component`.

## Gate features

One feature row per utterance, never one row per action:

- cached presence/cosine values and missingness;
- command duration and cached latency;
- source-specific hypothesis length, emptiness, digit and Chinese ratios;
- six pairwise hypothesis CER distances, exact agreement count, unique and
  non-empty candidate counts;
- simple cross-source length dispersion.

The feature function accepts only `CandidateRow`; it must not accept a reference
label argument. NaN is permitted at the feature boundary and is imputed inside
each fitted fold.

## Model grid and OOF contract

Use a small frozen CPU grid:

- balanced logistic regression with `C` in `{0.01, 0.1, 1.0, 10.0}`;
- balanced histogram gradient boosting with `max_leaf_nodes` in `{3, 7}`, fixed
  `learning_rate=0.05`, `max_iter=150`, and `l2_regularization=1.0`.

Use five deterministic `StratifiedGroupKFold` outer folds with
`shuffle=True` and `random_state=20260807`. Preprocessing is fitted separately
inside each outer training fold. Output one target-present probability per ID
and model specification. These OOF probabilities may be swept globally because
E0 is explicitly an optimistic upper-bound diagnostic; the selected threshold
must be marked non-deployable.

## Gate-oracle scoring

- If a positive row is accepted, use the current candidate with the smallest
  official normalized character error; ties follow `r3, primary, energy, tse`.
- If a positive row is rejected, charge full deletion error.
- If a negative row is rejected, count a correct reject.
- If a negative row is accepted, count a false accept regardless of whether a
  downstream candidate happens to contain an empty string. This preserves the
  factorization between gate and positive-only ranker.
- Compute `Overall=((1-CER)+RR)/2` from pooled S/I/D/ref-character and negative
  counts.

For each model, sweep all unique score boundaries plus reject-all. Select the
highest Overall point satisfying `RR >= 0.93`, breaking ties by higher RR, then
lower CER, then stable model name and threshold ordering.

## Uncertainty and decision

Sample complete leakage groups with replacement 2,000 times. In each bootstrap
replicate, reselect the best model and threshold under `RR >= 0.93`; report the
2.5/97.5 percentiles of the maximum feasible Overall.

- `continue_cached`: best OOF Overall >= 0.81, RR >= 0.93, and worst-fold
  Overall at the selected point >= 0.77.
- `falsified_cached`: bootstrap upper 95% bound < 0.80.
- Otherwise `proceed_pvad`: cached features lack required promotion headroom or
  the result is too uncertain; begin FireRedChat-pVAD E2 rather than tuning R10.

## Artifacts

Write under `output/r11_gate_oracle/`:

- `e0_manifest.json`: resolved input paths/digests, config hash, feature schema,
  model specifications, fold groups and coverage/evaluator checks;
- `e0_oof_scores.jsonl`: ID, group, fold and per-model OOF probabilities; no
  reference label text;
- `e0_frontier.jsonl`: all model/threshold metric points;
- `e0_summary.json`: selected point, bootstrap interval, worst fold, decision;
- `e0_report.md`: concise human-readable result and the next mandated branch.

The CLI must fail closed on missing/duplicate IDs, non-disjoint groups,
non-finite output probabilities, metric disagreement, or missing feasible
points. It must not modify Dataset-A, candidate files, or model checkpoints.

## Tests

Use real `CandidateRow` fixtures and hand-derived metric examples. Tests must
cover label-free feature construction, missingness, oracle tie-breaking,
accepted-negative accounting, threshold boundaries, exact OOF coverage/group
disjointness, deterministic grouped bootstrap, artifact schema, and source
digest/config reproducibility.
