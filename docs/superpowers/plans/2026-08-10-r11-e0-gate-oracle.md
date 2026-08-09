# R11 E0 Cached Gate-Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a reproducible grouped-OOF upper-bound test that decides whether current cached gate features can support Overall above 0.8.

**Architecture:** A focused `r11_gate_oracle` module builds one label-free feature row per utterance, generates probabilities with a frozen group-disjoint CPU model grid, and computes an optimistic candidate-oracle CER/RR frontier. A thin CLI validates inputs, cross-checks metrics, writes digested artifacts, and emits `continue_cached`, `falsified_cached`, or `proceed_pvad`.

**Tech Stack:** Python 3.12, NumPy, scikit-learn 1.9, existing `xh202615` metrics/evaluator and R10 candidate loader, pytest.

## Global Constraints

- E0 is a diagnostic upper bound, not a deployable selector; globally selected OOF model/threshold must be marked non-deployable.
- Gate features and fitted probabilities must not use label/reference text, candidate CER, or optimal-action labels.
- Use exactly five `StratifiedGroupKFold` folds, `shuffle=True`, `random_state=20260807` by default.
- Use RR floor `0.93`, continuation Overall `0.81`, worst-fold Overall `0.77`, falsification Overall `0.80`, and 2,000 grouped bootstrap replicates by default.
- Accepted negatives are false accepts even when a downstream candidate text is empty.
- Positive accepted rows receive the best current candidate by normalized character error with tie order `r3, primary, energy, tse`; rejected positives receive full deletion error.
- Preserve all existing files and tests; do not modify Dataset-A, candidate outputs, checkpoints, R10 behavior, or official metric definitions.
- Follow strict red-green-refactor: each production behavior must first have a focused test that fails for the expected missing-behavior reason.
- Claude Code writes implementation and tests; Codex performs review, verification, and real-data execution.

---

### Task 1: Label-free row features and gate-oracle metric kernel

**Files:**
- Create: `xh202615/r11_gate_oracle.py`
- Create: `tests/test_r11_gate_oracle.py`

**Interfaces:**
- Consumes: `xh202615.r10_selector.CandidateRow`, `CANDIDATE_ACTIONS`; `xh202615.metrics.cer_stats`; `xh202615.text.normalize_text`.
- Produces: `GATE_FEATURE_SCHEMA: tuple[str, ...]`; `build_gate_feature_matrix(rows: Sequence[CandidateRow]) -> numpy.ndarray`; `build_oracle_contributions(rows, labels) -> OracleContributions`; `gate_oracle_frontier(scores, contributions) -> list[dict[str, float]]`; `select_frontier_point(points, rr_floor) -> dict[str, float] | None`.

- [ ] **Step 1: Write failing feature tests**

Create real `CandidateRow` fixtures and tests which prove that changing only
`row.label` cannot change the feature vector, missing values remain represented
through explicit missing flags after fold preprocessing, and one row yields one
feature vector rather than five action rows. Name the realistic mutation each
test catches in a short comment.

- [ ] **Step 2: Run the feature tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r11_gate_oracle.py -k "feature" -q
```

Expected: collection/import failure because `xh202615.r11_gate_oracle` does not
exist, not a fixture or syntax error.

- [ ] **Step 3: Implement the minimal feature builder**

Define a stable schema containing cached acoustic values/missing flags,
per-source text shape, pairwise hypothesis distances, candidate counts and
length dispersion. The production feature function must have no label/reference
parameter and must return `float64` with shape `(len(rows), len(schema))`.

- [ ] **Step 4: Run feature tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass with no warnings.

- [ ] **Step 5: Write failing metric/frontier tests**

Use literal hand-calculated fixtures to cover:

```python
# Scores: accepted perfect positive first, then accepted negative.
# At threshold 0.9: CER=0, RR=1, Overall=1.
# At threshold 0.8: CER=0, RR=0, Overall=0.5.
```

Also require accepted negatives to reduce RR even if every candidate text is
empty, rejected positives to charge full deletion, and candidate-CER ties to
choose source order `r3, primary, energy, tse`.

- [ ] **Step 6: Run metric tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r11_gate_oracle.py -k "oracle or frontier or accepted_negative" -q
```

Expected: failures for unimplemented contribution/frontier behavior.

- [ ] **Step 7: Implement the metric kernel**

Use pooled integer S/I/D/ref-character contributions. The frontier must include
reject-all plus every unique `score >= threshold` boundary, keep tied scores
together, and return CER, RR, Overall, accepted positive/negative counts and
threshold. `select_frontier_point` must enforce the RR floor and deterministic
ties.

- [ ] **Step 8: Run Task 1 tests and existing metric tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r11_gate_oracle.py tests\test_metrics.py tests\test_r10_selector.py -q
```

Expected: all pass.

- [ ] **Step 9: Self-review and commit**

Confirm every new function has a behavior test, no test derives its expectation
with production helpers, and the feature schema contains no label/oracle field.
Commit only Task 1 files with message `feat: add R11 gate-oracle metric core`.

### Task 2: Group-disjoint OOF scoring and bootstrap decision

**Files:**
- Modify: `xh202615/r11_gate_oracle.py`
- Modify: `tests/test_r11_gate_oracle.py`

**Interfaces:**
- Consumes: Task 1 feature matrix and contributions.
- Produces: `default_model_specs() -> tuple[GateModelSpec, ...]`; `cross_fit_gate_models(X, target_present, groups, *, n_splits, seed, specs) -> CrossFitResult`; `group_bootstrap_best_frontier(scores_by_model, contributions, groups, *, rr_floor, n_boot, seed) -> dict`; `evaluate_e0(rows, labels, groups, *, n_splits=5, seed=20260807, rr_floor=0.93, n_boot=2000) -> dict`.

- [ ] **Step 1: Write failing OOF tests**

Create a deterministic synthetic dataset with at least ten distinct groups and
both target classes. Assert exact once-only coverage, finite probabilities in
`[0,1]`, identical fold assignment across model specs, and zero train/test group
intersection for every fold.

- [ ] **Step 2: Run OOF tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r11_gate_oracle.py -k "cross_fit or group_disjoint" -q
```

Expected: missing OOF implementation failures.

- [ ] **Step 3: Implement frozen model grid and OOF scoring**

Use fold-local `SimpleImputer(add_indicator=True)` plus `StandardScaler` for
balanced logistic regression `C={0.01,0.1,1,10}`. Use fold-local imputation and
balanced `HistGradientBoostingClassifier` for `max_leaf_nodes={3,7}`, fixed
learning rate, iterations and L2 from the design. Use
`StratifiedGroupKFold(shuffle=True, random_state=seed)` and fail closed when a
fold loses either class.

- [ ] **Step 4: Run OOF tests and verify GREEN**

Run Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Write failing bootstrap/decision tests**

Use at least four literal groups and two model-score vectors. Assert that each
bootstrap replicate reselects the best feasible model/threshold, results are
bitwise deterministic for a fixed seed, a CI high below 0.80 produces
`falsified_cached`, and a best point below 0.81 produces `proceed_pvad` even
when not statistically falsified.

- [ ] **Step 6: Run bootstrap tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r11_gate_oracle.py -k "bootstrap or decision" -q
```

Expected: failures for missing bootstrap/decision behavior.

- [ ] **Step 7: Implement grouped bootstrap and E0 evaluation**

Sample full group index arrays with replacement, recompute each model frontier,
and reselect under RR>=0.93 inside every replicate. `evaluate_e0` must also
compute the selected point on each held-out fold and return the worst-fold
metrics, model/fold metadata, the full frontier, and one of the three exact
decisions.

- [ ] **Step 8: Run Task 2 and regression tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r11_gate_oracle.py tests\test_r10_selector.py tests\test_r8_router.py -q
```

Expected: all pass.

- [ ] **Step 9: Self-review and commit**

Check the global threshold is explicitly described as diagnostic/non-deployable,
all preprocessing is fold-local, tied score boundaries remain atomic, and no
reference text enters model fitting. Commit Task 2 files with message
`feat: add grouped OOF gate diagnostics`.

### Task 3: Reproducible E0 CLI and artifact contract

**Files:**
- Create: `scripts/r11_gate_oracle_oof.py`
- Modify: `tests/test_r11_gate_oracle.py`
- Modify: `docs/r11-router-literature-decision.md`

**Interfaces:**
- Consumes: existing R10 candidate bundle paths and `evaluate_e0`.
- Produces: `main(argv: Sequence[str] | None = None) -> int`; `write_e0_artifacts(result, rows, groups, paths, output_root) -> dict[str, Path]`; five artifacts defined by the design.

- [ ] **Step 1: Write failing artifact tests**

Use `tmp_path` with a small already-evaluated result. Execute the real artifact
writer and assert exact filenames, required manifest/summary keys, UTF-8 JSONL
round-trip, no reference label text in OOF score rows, deterministic source and
configuration hashes, and a report containing the selected decision and next
branch.

- [ ] **Step 2: Run artifact tests and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r11_gate_oracle.py -k "artifact or digest" -q
```

Expected: import or behavior failure because the CLI/writer is absent.

- [ ] **Step 3: Implement the thin CLI and writer**

Match the existing R10 path defaults, expose `--n-outer`, `--n-boot`, `--seed`,
`--rr-floor`, and `--output-root`, and validate dataset labels against the frozen
group manifest. Cross-check selected custom metrics by constructing official
evaluator predictions with a non-empty sentinel for gate-accepted negatives;
the sentinel is evaluator-only and must not appear in score artifacts.

- [ ] **Step 4: Run artifact tests and verify GREEN**

Run Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Update the decision document with the exact command**

Add:

```powershell
Set-Location 'F:\XH-202615\XH-202615'
.\.venv\Scripts\python.exe .\scripts\r11_gate_oracle_oof.py
```

State that E0 runs on CPU and does not start neural training.

- [ ] **Step 6: Run all scoped tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r11_gate_oracle.py tests\test_r10_selector.py tests\test_datasetA_group_manifest.py tests\test_metrics.py tests\test_r8_router.py -q
```

Expected: all pass with no warnings from new code.

- [ ] **Step 7: Self-review and commit**

Verify CLI `--help`, missing-file failure, no writes outside output root, digest
stability and absence of Dataset-A label text from score/frontier artifacts.
Commit Task 3 files with message `feat: add R11 E0 gate-oracle CLI`.

### Task 4: Real Dataset-A E0 execution and evidence review

**Files:**
- Generate only: `output/r11_gate_oracle/e0_manifest.json`
- Generate only: `output/r11_gate_oracle/e0_oof_scores.jsonl`
- Generate only: `output/r11_gate_oracle/e0_frontier.jsonl`
- Generate only: `output/r11_gate_oracle/e0_summary.json`
- Generate only: `output/r11_gate_oracle/e0_report.md`
- Append: `.superpowers/sdd/2026-08-10-r11-e0-gate-oracle/progress.md`

**Interfaces:**
- Consumes: Task 3 CLI and frozen real artifacts.
- Produces: verified E0 decision that controls whether implementation proceeds to cached E1 or FireRed pVAD E2.

- [ ] **Step 1: Run a fast smoke execution**

Run with `--n-boot 50 --output-root output/r11_gate_oracle_smoke`; inspect only
the structured summary and manifest. Expected: exact coverage, all folds group
disjoint, finite scores, and successful artifact creation.

- [ ] **Step 2: Run the canonical execution**

Run the default command from Task 3. Expected: 5 folds, 2,000 grouped bootstrap
replicates and five canonical artifacts.

- [ ] **Step 3: Independently verify outputs**

Re-run official metric parity, recompute SHA-256 digests, confirm OOF ID coverage,
group disjointness, finite probabilities and bootstrap determinism with the same
seed. Compare selected Overall/RR/worst fold to exact decision gates.

- [ ] **Step 4: Record the branch decision**

If `continue_cached`, queue E1 positive-only expected-CER/LambdaMART. Otherwise
queue FireRedChat-pVAD E2; if `falsified_cached`, explicitly prohibit more cached
gate hyperparameter tuning.
