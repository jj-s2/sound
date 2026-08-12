# R12 Strict Holdout Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` to implement one task at a time. Each task must follow test-first RED/GREEN and receive a Codex review before the next task.

**Goal:** Build a reproducible Dataset-A group-disjoint train/validation/held-out-test pipeline for the R12 fused FireRed pVAD gate. The test partition remains unread until a frozen validation-derived selection is evaluated exactly once. The development target is Overall > 0.8 with RR >= 0.93; failure must be recorded faithfully rather than tuned on held-out labels.

**Architecture:** Preserve R11 E2 as an oracle evidence baseline. Add a separate R12 module family: a deterministic 60/20/20 group manifest, two base HGB gates (leaf 7 and 15), fold-local train-only OOF scores for Platt calibration, a validation-only blend/threshold selector with grouped-bootstrap RR safety, and two CLI stages. `select` may consume train/validation labels only and writes a frozen selection package. `evaluate` consumes the frozen package and the separately supplied held-out labels, writes metrics and predictions, and refuses any model/threshold search.

**Tech Stack:** Python 3.12, NumPy, scikit-learn 1.9, pytest, existing `xh202615` CandidateRow/pVAD/cache/evaluator modules.

## Global constraints

- This is a gate experiment, not ASR model training. Candidate transcripts remain the frozen four inference-time fields `r3`, `primary`, `energy`, `tse`.
- The pVAD cache is rebuilt only by `scripts/cache_firered_pvad_features.py` using `CPUExecutionProvider`; CUDA is not canonical.
- The split is at frozen `wake_component` group level. No group may occur in more than one role. IDs, group IDs, role counts, source digests, and a manifest digest must be persisted.
- Roles are 60% train, 20% validation, and 20% held-out test under `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260807)`: train folds 0-2, validation fold 3, held-out fold 4. The implementation must accept only this frozen seed and role mapping.
- The split manifest contains no reference text or target label values. Labels are supplied separately for stratification creation, training/selection, and final evaluation.
- `select` must not open the held-out-label path and must not produce test scores. `evaluate` must load no train/validation label path and must reject a selection package whose input/cache/split digests differ.
- Base fitting uses only train rows. Calibration fits only train-local three-fold group OOF base scores and train labels. Validation is used only for model/blend/threshold selection. There is no refit after validation selection.
- Eligible validation points require raw RR >= 0.95 and 5th-percentile grouped-bootstrap RR >= 0.93. Among eligible points, select maximum bootstrap-median Overall, then higher raw RR, lower CER, lower threshold, lexical model name, then lower blend weight.
- The candidate-oracle CER contribution remains diagnostic-only. Published public artifacts must not contain labels, references, candidate CER, optimal action, embeddings, or frame arrays.
- Never overwrite R11 outputs, raw Dataset-A data, model files, or the R12-before-cleanup tag. Do not commit generated cache/output artifacts.
- Project cleanup is explicitly deferred until a strict pipeline result is frozen and independently reproduced.

## File map

- Create `xh202615/r12_split.py`: exact group-disjoint manifest construction, canonical JSON/digest, serialization, and validation.
- Create `scripts/r12_prepare_split.py`: build the label-free R12 split manifest from canonical IDs, labels, and groups.
- Create `xh202615/r12_calibrated_gate.py`: train-only inner OOF probabilities, calibrated leaf7/leaf15 ensemble, validation frontier, grouped bootstrap, frozen selection schema, and prediction helpers.
- Create `scripts/r12_strict_holdout.py`: separate `select` and `evaluate` command stages plus exact artifact/provenance checks.
- Create `tests/test_r12_split.py`, `tests/test_r12_calibrated_gate.py`, and `tests/test_r12_strict_holdout.py`.
- Modify `README.md` only after a reproducible strict run exists, to point to the final canonical commands and clarify diagnostic oracle versus deployable gate behavior.

---

### Task 1: Frozen label-free 60/20/20 group split manifest

**Files:**
- Create `xh202615/r12_split.py`
- Create `scripts/r12_prepare_split.py`
- Create `tests/test_r12_split.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class R12SplitManifest:
    schema_version: str
    seed: int
    roles_by_id: Mapping[str, Literal["train", "validation", "held_out_test"]]
    groups_by_id: Mapping[str, str]
    source_digests: Mapping[str, str]
    manifest_sha256: str

def build_r12_split(
    ids_in_order: Sequence[str],
    labels: Mapping[str, str | None],
    groups: Mapping[str, str],
    *,
    seed: int = 20260807,
) -> R12SplitManifest: ...

def write_r12_split(path: Path, manifest: R12SplitManifest) -> None: ...
def load_r12_split(path: Path, expected_ids: Sequence[str]) -> R12SplitManifest: ...
```

- [ ] **Step 1: Write failing split tests**

Test literal small grouped fixtures and a full-size-shaped fixture. Assert exact ID coverage/order, stable bytes across reruns, required role mapping from folds 0/1/2/3/4, both classes in every role, and no group crosses roles. Add mutations for duplicate IDs, an unknown ID, seed != `20260807`, incomplete mappings, tampered digest, and labels accidentally serialized.

- [ ] **Step 2: Verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r12_split.py -q
```

Expected: import/collection failure because the R12 split module does not exist.

- [ ] **Step 3: Implement the strict manifest and thin CLI**

Use `StratifiedGroupKFold` over a zero feature column. Validate all five folds before assigning roles. Serialize only schema, fixed seed, role mapping, group mapping, counts, source digests, and canonical digest; do not serialize labels. The CLI receives `--canonical-input-jsonl`, `--labels`, `--groups`, `--output` and must reuse `load_canonical_rows` plus exact mapping checks.

- [ ] **Step 4: Verify GREEN and CLI determinism**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r12_split.py -q
.venv\Scripts\python.exe scripts\r12_prepare_split.py --help
```

- [ ] **Step 5: Commit Task 1 only**

```powershell
git add xh202615/r12_split.py scripts/r12_prepare_split.py tests/test_r12_split.py
git commit -m "feat: add frozen R12 group split manifest"
```

### Task 2: Train-only calibrated leaf7/leaf15 gate and robust validation selection

**Files:**
- Create `xh202615/r12_calibrated_gate.py`
- Create `tests/test_r12_calibrated_gate.py`

**Interfaces:**

```python
BASE_MODELS = (
    "hist_gradient_boosting_leaf_7",
    "hist_gradient_boosting_leaf_15",
)
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)

@dataclass(frozen=True)
class FrozenGateSelection:
    base_models: tuple[str, str]
    calibration: Mapping[str, float]
    blend_weight_leaf15: float
    threshold: float | str
    validation_metrics: Mapping[str, float]
    validation_bootstrap: Mapping[str, float]
    provenance: Mapping[str, str]

def fit_train_calibrated_gate(
    joined_train: Sequence[JoinedPvadRow],
    *,
    seed: int,
) -> TrainCalibratedGate: ...

def select_on_validation(
    trained: TrainCalibratedGate,
    joined_validation: Sequence[JoinedPvadRow],
    validation_rows: Sequence[CandidateRow],
    validation_labels: Mapping[str, str | None],
    *,
    n_boot: int, seed: int,
) -> FrozenGateSelection: ...

def predict_with_selection(
    trained: TrainCalibratedGate,
    selection: FrozenGateSelection,
    joined_rows: Sequence[JoinedPvadRow],
) -> np.ndarray: ...
```

- [ ] **Step 1: Write failing calibration/selection tests**

Build a realistic 96-column joined-row fixture with group structure. Require three-fold group-disjoint train OOF probability banks for both HGB specs; force a test sentinel so validation rows cannot be present in any calibration fit. Verify calibration uses only the two train OOF columns and train target labels, all score probabilities are finite in `[0,1]`, and validation selection inspects the two calibrated bases plus all five declared blend weights.

Use hand-calculated contribution fixtures to test candidate threshold boundaries, the raw RR >= .95 filter, bootstrap 5th-percentile RR >= .93 filter, deterministic tie breaking, and reject-all marker serialization. Add a mutation test proving test labels do not appear in any function signature or serialized selection package.

- [ ] **Step 2: Verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r12_calibrated_gate.py -q
```

- [ ] **Step 3: Implement the minimal R12 gate module**

Reuse `join_pvad_e0_rows`, `_fit_gate_pipeline`, the frozen R11 schemas, and `build_oracle_contributions`; do not duplicate feature extraction or alter R11 defaults. Build HGB leaf7/leaf15 train OOF scores with `StratifiedGroupKFold(3, shuffle=True, random_state=seed)`. Fit a deterministic balanced `LogisticRegression` calibrator on the two OOF score columns, then fit each base HGB on all train rows. Score validation with the refit bases, calibrate, enumerate individual calibrated bases and the fixed five blends, and select a validation-only point using grouped bootstrap samples of whole groups.

The serialized selection must include canonical feature schema digest, base model parameter digest, calibration coefficients/intercept, blend definition, selected threshold, raw/bootstrapped validation metrics, and no fitted sklearn pickle. `evaluate` will deterministically re-fit from those frozen fields and train inputs.

- [ ] **Step 4: Verify GREEN and preserve R11 behavior**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r12_calibrated_gate.py tests\test_r11_pvad_oracle.py tests\test_r11_pvad_oracle_oof.py -q
```

- [ ] **Step 5: Commit Task 2 only**

```powershell
git add xh202615/r12_calibrated_gate.py tests/test_r12_calibrated_gate.py
git commit -m "feat: add R12 train-only calibrated gate"
```

### Task 3: Staged selection and exactly-once held-out evaluator

**Files:**
- Create `scripts/r12_strict_holdout.py`
- Create `tests/test_r12_strict_holdout.py`

**Interfaces:**

```text
python scripts/r12_strict_holdout.py select \\
  --canonical-input-jsonl ... --groups ... --split-manifest ... \\
  --train-labels ... --validation-labels ... --cache-root ... \\
  --selection-output ... --bootstrap-count 2000

python scripts/r12_strict_holdout.py evaluate \\
  --canonical-input-jsonl ... --groups ... --split-manifest ... \\
  --train-labels ... --cache-root ... --selection-input ... \\
  --held-out-labels ... --evaluation-output ...
```

- [ ] **Step 1: Write failing stage-boundary tests**

Create temporary canonical/cache fixtures based on `tests.test_r11_pvad_oracle._fixture`. Test that `select` succeeds when the held-out-label path is a nonexistent sentinel, and that it writes a deterministic `r12_selection.json` with train/validation ID/group digests and no predictions for held-out IDs. Test that `evaluate` fails without held-out labels, fails if selection provenance/canonical/cache/split digests mismatch, and does not accept flags that can select a new model, blend, or threshold.

Test final evaluation publishes exactly five files: `r12_manifest.json`, `r12_selection.json`, `r12_held_out_predictions.jsonl`, `r12_summary.json`, `r12_report.md`. The predictions may include only id, group, score, threshold, action, and selected model definition; no label/reference/oracle fields. Check official evaluator parity from public decisions plus the private supplied held-out labels.

- [ ] **Step 2: Verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r12_strict_holdout.py -q
```

- [ ] **Step 3: Implement select/evaluate CLI stages**

Use `load_canonical_rows`, `_mapping_file`, `_cache`, `join_pvad_e0_rows`, and `publish_text_package`. Partition only after validating the R12 manifest. `select` must read labels only for IDs whose role is train or validation and reject a labels file that supplies held-out values. `evaluate` must refit only the frozen leaf models/calibrator using train rows and labels, then apply the immutable selection to held-out rows. It may compute contributions/metrics only after decisions exist. No evaluator command line option may change models, blend weights, threshold, seed, or bootstrap count.

Use a new artifact kind/schema (`r12_strict_holdout`, `v1`) and fail closed on output identity/package errors. The report must state that held-out test is an evaluation, not a tuning source.

- [ ] **Step 4: Verify GREEN plus complete relevant regression suite**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r12_split.py tests\test_r12_calibrated_gate.py tests\test_r12_strict_holdout.py tests\test_r11_pvad_oracle.py tests\test_r11_pvad_oracle_oof.py -q
.venv\Scripts\python.exe scripts\r12_strict_holdout.py --help
```

- [ ] **Step 5: Commit Task 3 only**

```powershell
git add scripts/r12_strict_holdout.py tests/test_r12_strict_holdout.py
git commit -m "feat: add staged R12 held-out evaluator"
```

### Task 4: Raw Dataset-A rebuild contract and development stress test

**Files:**
- Modify `scripts/r12_prepare_split.py`
- Modify `scripts/r12_strict_holdout.py`
- Create `tests/test_r12_rebuild_contract.py`
- Modify `README.md` only after a successful reproducible run

- [ ] **Step 1: Write failing rebuild-contract tests**

Require an explicit manifest section documenting canonical input command inputs, candidate source digests, CPU pVAD manifest/records digests, model identity, and output digests. Test that changing a canonical candidate/cache byte after selection makes evaluation fail. Test the existing CPU cache CLI with `--help` and a small fake package only; do not invoke models in unit tests.

- [ ] **Step 2: Implement source/provenance binding**

Document and enforce the executable rebuild sequence:

```powershell
# Generate/freeze the four R10 candidate transcript sources from raw Dataset-A.
# Construct label-free canonical_input.jsonl from those sources.
.venv\Scripts\python.exe scripts\cache_firered_pvad_features.py --dataset-root datasetA\datasetA --ecapa-device cpu ...
.venv\Scripts\python.exe scripts\r12_prepare_split.py ...
.venv\Scripts\python.exe scripts\r12_strict_holdout.py select ...
.venv\Scripts\python.exe scripts\r12_strict_holdout.py evaluate ...
```

The committed code must identify the precise existing R10 candidate-producing source files and bind their SHA-256 values in selection/evaluation artifacts. Do not claim a raw rebuild is complete unless all commands succeed from raw Dataset-A into a new output root.

- [ ] **Step 3: Execute development rotations (diagnostic only)**

Use five rotations of the role assignment as a development stress test. For each rotation run train/validation selection and evaluate its rotation test. Record results outside Git. A candidate is eligible for one new untouched held-out evaluation only if mean Overall exceeds `0.7900480643992768`, every rotation has RR >= .93, and at least three rotations exceed .8. Do not replace the reserved final test with any already inspected fold.

- [ ] **Step 4: Verify and commit code/documentation only**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r12_split.py tests\test_r12_calibrated_gate.py tests\test_r12_strict_holdout.py tests\test_r12_rebuild_contract.py -q
.venv\Scripts\python.exe -m pytest -q
git add scripts/r12_prepare_split.py scripts/r12_strict_holdout.py tests/test_r12_rebuild_contract.py README.md
git commit -m "docs: bind R12 strict rebuild provenance"
```

### Task 5: Independent final test, then surgical repository cleanup

**Files:**
- Modify `README.md`
- Create `docs/r12/strict-evaluation.md`
- Delete/move only files proven unreachable after the strict pipeline freezes (separate review and explicit file list required)

- [ ] **Step 1: Reserve a new held-out manifest**

Create a fresh group-disjoint final-test manifest whose labels are inaccessible to tuning. Record its digest and access boundary. If no genuinely unseen Data-A labels exist because historical R11 diagnostics used all groups, report that limitation; do not relabel an already scanned fold as independent.

- [ ] **Step 2: Run select once, evaluate once**

Regenerate `canonical_input.jsonl` and CPU pVAD cache from raw Dataset-A in a new output directory. Run `select`; review the immutable selection package; then authorize and run `evaluate` exactly once against final labels. Record Overall, RR, CER, false accepts/rejects, output digests, official parity, and the interpretation of whether >.8/RR>=.93 was met.

- [ ] **Step 3: Cleanup proposal, not an automatic deletion**

Produce an explicit reachability table of entry points, imports, tests, and candidate/raw-rebuild dependencies. Keep the minimal runnable pipeline, its test fixtures, requirements, and docs. Present every proposed deletion/move to the user for approval before changing files. Keep tag `r12-before-structure-cleanup-20260811` as a restore point.

## Final verification checklist

- `git status --short` contains only intended source/test/doc changes; no Dataset-A, model, cache, output, or credential artifacts.
- R11 regression artifacts are byte-identical where its tests establish that guarantee.
- The complete test suite passes under `.venv\\Scripts\\python.exe -m pytest -q`.
- `select` never opens held-out labels and `evaluate` performs no selection; tests demonstrate both boundaries.
- The published package has exact names, deterministic bytes on identical inputs, ID/group/digest coverage, and no private fields.
- Any claim about Overall > .8 is tied to the split/role manifest and marked either development rotation or truly untouched held-out evaluation.
