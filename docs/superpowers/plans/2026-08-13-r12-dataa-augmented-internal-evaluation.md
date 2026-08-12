# R12 Dataset-A Augmented Internal Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible R12 gate/text-model training pipeline that augments only the group-disjoint Dataset-A training role, selects on untouched validation data, and runs one final internal test.

**Architecture:** Preserve frozen `r12_split.py` and `r12_strict_holdout.py`. Add a parallel `r12_dataa_augmented_*` pipeline: an immutable split builder, a deterministic audio/lineage builder, label-free feature reconstruction, and a staged selection/evaluation CLI.

**Tech Stack:** Python 3.11, NumPy, SoundFile, scikit-learn, PyTorch/WeSpeaker, FireRed CPU ONNX pVAD, FunASR, pytest.

## Global Constraints

- Dataset-A root is `datasetA/datasetA`; its labels must not train or modify ASR, TSE, R3, or pVAD base models.
- Split with `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=20260812)`; folds `0..13=train`, `14..16=validation`, and `17..19=internal_test`.
- Group key is `wake_component`. Original samples and all children retain their parent group and never cross roles.
- Train keeps originals and creates exactly `aug_a` (speed 0.95 or 1.05 plus -3..+3 dB gain) and `aug_b` (-3..+3 dB gain plus deterministic white noise at 18..25 dB SNR). Validation and test remain original-only.
- All transforms derive from `(parent_id, augmentation_id, seed=20260812)`, output finite mono 16-kHz WAV, and retain 1 dB peak headroom.
- Augmented samples must regenerate ASR, TSE, R3, and pVAD from their own command-audio SHA-256. Original feature reuse is forbidden.
- All new output resides under `output/r12_dataa_augmented_internal_v1/`; raw Dataset-A and frozen R12 artifacts are never overwritten.
- `selection_artifact.json` is complete and digest-verified before evaluation may open internal-test labels. Evaluation must not alter selected parameters.

---

## File Structure

- Create `xh202615/r12_dataa_augmented_split.py`: immutable 70/15/15 group manifest.
- Create `xh202615/r12_dataa_augmentation.py`: deterministic waveform transforms and lineage.
- Create `xh202615/r12_dataa_canonical.py`: digest-safe candidate join to label-free canonical rows.
- Create `scripts/r12_dataa_prepare_split.py`, `scripts/r12_dataa_augment_audio.py`, `scripts/run_temporal_head_inference.py`, `scripts/r12_dataa_rebuild_features.py`, and `scripts/r12_dataa_internal_eval.py`.
- Create `tests/test_r12_dataa_augmented_split.py`, `tests/test_r12_dataa_augmentation.py`, `tests/test_r12_dataa_canonical.py`, `tests/test_temporal_head_inference.py`, and `tests/test_r12_dataa_internal_eval.py`.
- Create `docs/r12/dataa-augmented-internal-runbook.md`; amend `docs/r12/strict-evaluation.md` to distinguish this internal evaluation from independent blind testing.

### Task 1: Frozen 70/15/15 group split

**Files:**
- Create: `xh202615/r12_dataa_augmented_split.py`
- Create: `scripts/r12_dataa_prepare_split.py`
- Test: `tests/test_r12_dataa_augmented_split.py`

**Interfaces:**
- Consumes: `ids: Sequence[str]`, `labels: Mapping[str, str | None]`, `groups: Mapping[str, str]`.
- Produces: `AugmentedInternalSplitManifest`, `build_augmented_internal_split(ids, labels, groups)`, `write_augmented_internal_split(path, manifest)`, and `load_augmented_internal_split(path, expected_ids)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_twenty_fold_mapping_is_group_disjoint():
    manifest = build_augmented_internal_split(ids, labels, groups)
    assert role_counts(manifest) == {"train": 28, "validation": 6, "internal_test": 6}
    assert all(one_role_per_group(manifest))

def test_serialized_manifest_has_no_private_fields(tmp_path):
    write_augmented_internal_split(tmp_path / "split.json", manifest)
    text = (tmp_path / "split.json").read_text().lower()
    assert all(word not in text for word in ("label", "reference", "target", "text"))
```

Also cover fixed seed rejection, duplicate JSON keys, changed digest/counts, both target classes per role, and exact ID/group coverage.

- [ ] **Step 2: Run the tests to verify failure**

Run: `pytest tests/test_r12_dataa_augmented_split.py -q`

Expected: import failure for `xh202615.r12_dataa_augmented_split`.

- [ ] **Step 3: Implement the split and CLI**

```python
_N_SPLITS, _SEED = 20, 20260812
_FOLD_TO_ROLE = (*(["train"] * 14), *(["validation"] * 3), *(["internal_test"] * 3))

def build_augmented_internal_split(ids, labels, groups, *, seed=_SEED):
    # Validate exact coverage; run StratifiedGroupKFold; require class and group invariants.
    ...
```

Canonical JSON contains only schema version, seed, role/group mappings, counts, source digests and manifest SHA-256. The loader rejects unknown/private keys, invalid roles, malformed hashes, changed counts and cross-role groups. The CLI validates canonical IDs, private labels, and private groups before it writes.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_r12_dataa_augmented_split.py -q && python scripts/r12_dataa_prepare_split.py --help`

Expected: all tests pass; help exits without neural imports.

- [ ] **Step 5: Commit**

```bash
git add xh202615/r12_dataa_augmented_split.py scripts/r12_dataa_prepare_split.py tests/test_r12_dataa_augmented_split.py
git commit -m "feat: add R12 Dataset-A 70-15-15 group split"
```

### Task 2: Deterministic train-only augmentation and lineage

**Files:**
- Create: `xh202615/r12_dataa_augmentation.py`
- Create: `scripts/r12_dataa_augment_audio.py`
- Test: `tests/test_r12_dataa_augmentation.py`

**Interfaces:**
- Consumes: raw Dataset-A JSONL, `AugmentedInternalSplitManifest`, and an output root.
- Produces: `build_augmented_dataset(dataset_root, split, output_root) -> AugmentationSummary` and `load_lineage(path) -> dict[str, LineageRow]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_only_train_parents_receive_two_children(tmp_path):
    summary = build_augmented_dataset(raw_root, split, tmp_path / "derived")
    rows = load_lineage(summary.lineage_path)
    assert child_kinds(rows, "train-1") == {"original", "aug_a", "aug_b"}
    assert child_kinds(rows, "validation-1") == {"original"}

def test_same_seed_produces_same_lineage_digest(tmp_path):
    a = build_augmented_dataset(raw_root, split, tmp_path / "a")
    b = build_augmented_dataset(raw_root, split, tmp_path / "b")
    assert a.lineage_digest == b.lineage_digest
```

Also test role/group inheritance, original wake audio, finite mono 16-kHz/headroom output, and an invalid input WAV becoming one exclusion record with no child lineage row.

- [ ] **Step 2: Run the tests to verify failure**

Run: `pytest tests/test_r12_dataa_augmentation.py -q`

Expected: import failure for `xh202615.r12_dataa_augmentation`.

- [ ] **Step 3: Implement transforms and materialization**

```python
def augmentation_rng(parent_id: str, augmentation_id: str, seed: int = 20260812) -> np.random.Generator: ...
def augment_a(audio: np.ndarray, rate: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]: ...
def augment_b(audio: np.ndarray, rate: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]: ...
def build_augmented_dataset(dataset_root: Path, split: AugmentedInternalSplitManifest, output_root: Path) -> AugmentationSummary: ...
```

Use linear interpolation for speed perturbation, amplitude-domain gain, seeded Gaussian white noise scaled from measured RMS, and global downscaling to `10 ** (-1 / 20)`. The derived `pos.jsonl`/ `neg.jsonl` preserve only ID/split/wakeup/command input fields. Lineage records role, group, parent/child IDs, transform parameters, input/output audio SHA-256 and paths; no labels/reference text are serialized.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_r12_dataa_augmentation.py -q && python scripts/r12_dataa_augment_audio.py --help`

Expected: all tests pass; output never overwrites a raw path.

- [ ] **Step 5: Commit**

```bash
git add xh202615/r12_dataa_augmentation.py scripts/r12_dataa_augment_audio.py tests/test_r12_dataa_augmentation.py
git commit -m "feat: add deterministic R12 train audio augmentation"
```

### Task 3: Label-free temporal/R3 candidate inference

**Files:**
- Create: `scripts/run_temporal_head_inference.py`
- Test: `tests/test_temporal_head_inference.py`

**Interfaces:**
- Consumes: `--input-jsonl`, `--candidate-asr`, temporal checkpoint and device.
- Produces: exact-ID JSONL records `{id, recognition_text, temporal_probability, accepted, route, command_audio_sha256}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_inference_reads_only_input_and_candidate_map(monkeypatch, tmp_path):
    rows = run_inference(input_jsonl, candidate_map, checkpoint, device="cpu")
    assert [row["id"] for row in rows] == ["a", "b"]
    assert "label" not in json.dumps(rows, ensure_ascii=False).lower()

def test_missing_or_duplicate_candidate_ids_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="cover"):
        run_inference(input_jsonl, incomplete_map, checkpoint, device="cpu")
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `pytest tests/test_temporal_head_inference.py -q`

Expected: script/module import failure.

- [ ] **Step 3: Implement label-free inference**

Extract only checkpoint loading, frozen encoder preparation and `_feature_sequence` from `scripts/evaluate_temporal_head.py`. Read input via `scripts.run_tse_inference.read_input_jsonl`; calculate command SHA-256; gate the candidate map with the checkpoint threshold. Do not call `load_dataset` or `evaluate_rows`.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_temporal_head_inference.py -q && python scripts/run_temporal_head_inference.py --help && python scripts/evaluate_temporal_head.py --help`

Expected: tests pass and both CLIs print help.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_temporal_head_inference.py tests/test_temporal_head_inference.py
git commit -m "feat: add label-free temporal candidate inference"
```

### Task 4: Rebuild contract and canonical join

**Files:**
- Create: `xh202615/r12_dataa_canonical.py`
- Create: `scripts/r12_dataa_rebuild_features.py`
- Test: `tests/test_r12_dataa_canonical.py`

**Interfaces:**
- Consumes: lineage, pVAD cache, primary/energy ASR maps, TSE audio/ASR maps and temporal/R3 map.
- Produces: `build_augmented_canonical(...) -> CanonicalBuildSummary` and existing-schema `canonical_input.jsonl`.

- [ ] **Step 1: Write the failing provenance tests**

```python
def test_canonical_rejects_audio_digest_mismatch(tmp_path):
    with pytest.raises(ValueError, match="command_audio_sha256"):
        build_augmented_canonical(lineage, mismatched_sources, tmp_path / "canonical.jsonl")

def test_validation_and_test_are_original_only(tmp_path):
    summary = build_augmented_canonical(lineage, sources, tmp_path / "canonical.jsonl")
    assert all("__aug_" not in row["id"] for row in summary.rows if row["split"] != "train")
```

Also test exact source coverage, one exclusion per incomplete child, absence of labels/reference text, and deterministic source digest.

- [ ] **Step 2: Run the tests to verify failure**

Run: `pytest tests/test_r12_dataa_canonical.py -q`

Expected: import failure for `xh202615.r12_dataa_canonical`.

- [ ] **Step 3: Implement canonical builder and safe orchestrator**

The rebuild CLI runs existing CPU pVAD, FunASR primary/energy with `--command-audio-map`, candidate fusion, TSE then FunASR, and `run_temporal_head_inference.py` against the derived root. It must attach/validate `command_audio_sha256` at every source and publish from a staging directory only when all non-excluded lineage IDs have exactly one record per required source.

```python
{"id": sid, "split": role, "r3_text": r3, "primary_text": primary,
 "energy_text": energy, "tse_text": tse, "audio_features": features,
 "source_digest": digest}
```

Include the lineage command digest and all four candidate texts in `source_digest`.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_r12_dataa_canonical.py tests/test_r12_rebuild_contract.py -q`

Expected: all tests pass without neural inference.

- [ ] **Step 5: Commit**

```bash
git add xh202615/r12_dataa_canonical.py scripts/r12_dataa_rebuild_features.py tests/test_r12_dataa_canonical.py
git commit -m "feat: add R12 augmented feature rebuild contract"
```

### Task 5: Train-only selection and one-time internal evaluation

**Files:**
- Create: `scripts/r12_dataa_internal_eval.py`
- Test: `tests/test_r12_dataa_internal_eval.py`

**Interfaces:**
- Consumes: canonical rows/cache/lineage/split manifest, role-scoped private labels and candidate-source digests.
- Produces: `selection_artifact.json` and a single `internal_test_result/` package.

- [ ] **Step 1: Write the failing protocol tests**

```python
def test_select_uses_train_augmented_and_raw_validation_only(tmp_path):
    main(select_args)
    artifact = json.loads(selection.read_text())
    assert artifact["provenance"]["fit_ids"] == ["train", "train__aug_a", "train__aug_b"]
    assert artifact["provenance"]["validation_ids"] == ["validation"]

def test_evaluate_requires_frozen_artifact_and_refits_train_plus_validation(tmp_path):
    main(select_args)
    main(evaluate_args)
    assert (result / "r12_summary.json").is_file()
    with pytest.raises(ValueError, match="selection provenance"):
        main(changed_canonical_args)
```

Also reject test labels during select, altered source/split digests, augmented validation/test IDs, missing selection, and a second result under the same directory.

- [ ] **Step 2: Run the tests to verify failure**

Run: `pytest tests/test_r12_dataa_internal_eval.py -q`

Expected: script import failure.

- [ ] **Step 3: Implement staged selection and evaluation**

For `select`, use full train role (original+children) with `fit_train_calibrated_gate`, `fit_train_candidate_router` and `fit_train_text_presence`; call `select_on_validation` only for raw validation rows. Freeze selected threshold/fusion/action order, validation metrics, ordered IDs, source/lineage/cache/canonical/split digests and model public digests.

For `evaluate`, verify all selection digests and a nonexistent result directory; refit the three models on train plus raw validation (and only train augmentations), generate raw internal-test predictions, then load `--internal-test-labels` exactly once for `evaluate_rows` and grouped bootstrap. Atomically write predictions, summary, manifest and report with `internal_test_label_read_count: 1`; never recompute selection.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_r12_dataa_internal_eval.py tests/test_r12_calibrated_gate.py tests/test_r12_candidate_router.py tests/test_r12_text_presence.py tests/test_r12_strict_holdout.py -q`

Expected: all tests pass; frozen R12 behavior remains unchanged.

- [ ] **Step 5: Commit**

```bash
git add scripts/r12_dataa_internal_eval.py tests/test_r12_dataa_internal_eval.py
git commit -m "feat: add staged R12 augmented internal evaluation"
```

### Task 6: End-to-end preflight and controlled training run

**Files:**
- Create: `docs/r12/dataa-augmented-internal-runbook.md`
- Modify: `docs/r12/strict-evaluation.md`
- Test: the five new test modules plus `tests/test_r12_rebuild_contract.py`.

**Interfaces:**
- Consumes: committed Tasks 1-5 and approved raw/model paths.
- Produces: an auditable runbook, validation selection artifact, and one internal-test result package.

- [ ] **Step 1: Write the runbook**

Document commands for: private label/group maps; split; augmentation; feature rebuild; coverage checks; selection; selection hash inspection; one evaluation; archiving. State stop conditions: any missing/digest-mismatched feature stops before selection; any result-driven change requires a new experiment name and seed.

- [ ] **Step 2: Run preflight verification**

Run: `pytest tests/test_r12_dataa_augmented_split.py tests/test_r12_dataa_augmentation.py tests/test_r12_dataa_canonical.py tests/test_temporal_head_inference.py tests/test_r12_dataa_internal_eval.py tests/test_r12_rebuild_contract.py -q`

Expected: all tests pass. Then run every new CLI with `--help` and `git diff --check`.

- [ ] **Step 3: Build non-neural artifacts and assert invariants**

Run split/augmentation into `output/r12_dataa_augmented_internal_v1/`. Verify raw role counts `1286/275/277`, exactly two children for each non-excluded train parent, no child in validation/test, and no output under the raw Dataset-A root.

- [ ] **Step 4: Rebuild features and select**

Run CPU pVAD and all candidate regeneration. Proceed only if each non-excluded row has primary, energy, TSE, R3 and pVAD records whose command digest equals lineage. Run `select`, persist its digest and validation score.

- [ ] **Step 5: Run the single internal test and report accurately**

Run `evaluate` exactly once. Report CER/RR/Overall/bootstrap as “Dataset-A group-disjoint internal test”; state explicitly that historically opened Dataset-A labels mean it is not independent blind-test evidence. If Overall is below 0.8, do not tune on this test package.

- [ ] **Step 6: Commit verified documentation**

```bash
git add docs/r12/dataa-augmented-internal-runbook.md docs/r12/strict-evaluation.md
git commit -m "docs: add R12 augmented internal evaluation runbook"
```

## Self-Review

- The split, train-only augmentation, exact feature regeneration, staged selection, train+validation refit, single label read, output isolation and accurate reporting requirements each have a dedicated task.
- The plan contains concrete paths, tests, commands, functions and artifact names; no undefined cross-task interfaces are used.
- 已检查计划中是否含有未落实的占位描述；未发现。
