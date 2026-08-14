# R12 Training Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one validated local command that derives all Dataset-A train-only ASR prerequisites from raw data, labels, and groups.

**Architecture:** A bootstrap module composes the existing split, augmentation, manifest, fold, and hotword modules behind one fail-closed output root. It deterministically selects group-disjoint inner validation from train parents. The CLI exposes safe dry-run and explicit materialization modes.

**Tech Stack:** Python 3.11, existing R12 JSON/JSONL modules, pytest, no GPU.

## Global Constraints

- Work only in the existing R12 isolated worktree.
- No Dataset-A audio, labels, group maps, run roots, model artifacts, or outputs are committed.
- The bootstrap accepts only private raw Dataset-A inputs and writes only under a non-existent output root.
- Internal-test and validation groups are excluded from augmentation, ASR manifests, and hotwords.
- Inner validation is a deterministic 10% group-disjoint subset of train parents, seed `20260814`.
- Dry-run allocates no GPU and does not generate audio or write artifacts.

---

### Task 1: Bootstrap module and CLI

**Files:**
- Create: `xh202615/r12_training_bootstrap.py`
- Create: `scripts/r12_bootstrap_training.py`
- Test: `tests/test_r12_training_bootstrap.py`

**Interfaces:**
- `BootstrapConfig(dataset_root, labels_path, groups_path, output_root, inner_valid_fraction=0.1, seed=20260814)`.
- `plan_bootstrap(config) -> BootstrapPlan` validates input maps/raw IDs and returns split and selected inner-valid parent IDs without writes.
- `materialize_bootstrap(config) -> BootstrapResult` creates split, augmentation lineage, private ASR manifests, folds and hotwords under `output_root`.

- [ ] **Step 1: Write failing synthetic end-to-end tests**

```python
def test_plan_selects_group_disjoint_inner_valid_from_train_only(tmp_path: Path) -> None:
    plan = plan_bootstrap(config_with_synthetic_dataset(tmp_path))
    assert plan.inner_valid_parent_ids <= plan.train_parent_ids
    assert plan.inner_valid_groups.isdisjoint(plan.fit_groups)

def test_materialize_excludes_internal_test_from_private_asr_rows(tmp_path: Path) -> None:
    result = materialize_bootstrap(config_with_synthetic_dataset(tmp_path))
    rendered = result.train_jsonl.read_text(encoding="utf-8")
    assert "held-out-command" not in rendered
```

Also test: dry-run leaves output root absent; output root existing fails; same seed is deterministic; all required artifact paths exist after materialization; split/lineage/folds summaries do not expose label text publicly.

- [ ] **Step 2: Run RED**

Run: `F:\\XH-202615\\XH-202615\\.venv\\Scripts\\python.exe -m pytest tests\\test_r12_training_bootstrap.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement composition**

Build the split with `build_augmented_internal_split` and write it below `output_root/split.json`. Derive train parent IDs from `roles_by_id`; select sorted groups using SHA-256 of `seed:group`, choosing `max(1, round(0.1 * train_group_count))` when at least two train groups exist, otherwise zero. Materialize audio with `build_augmented_dataset`; pass train labels and inner-valid parents to `prepare_asr_manifests`; write folds with `build_asr_folds`; build capacities `(10,)` only when at least ten phrases exist, otherwise use the exact phrase count. Stage into a sibling temporary directory and rename once after all parts succeed.

- [ ] **Step 4: Run GREEN**

Run: `F:\\XH-202615\\XH-202615\\.venv\\Scripts\\python.exe -m pytest tests\\test_r12_training_bootstrap.py tests\\test_r12_asr_manifest.py tests\\test_r12_asr_folds.py tests\\test_r12_asr_hotword.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit only module, CLI and tests with `feat: add raw Dataset-A R12 training bootstrap`.

### Task 2: Documentation, snapshot allowlist, verification and publishing

**Files:**
- Modify: `docs/r12/r12-train-and-publish.md`
- Modify: `scripts/export_r12_github_snapshot.ps1`
- Modify: `tests/test_export_r12_github_snapshot.py`

- [ ] **Step 1: Add failing allowlist test**

```python
def test_snapshot_includes_training_bootstrap_sources() -> None:
    script = Path("scripts/export_r12_github_snapshot.ps1").read_text(encoding="utf-8")
    assert "r12_training_bootstrap.py" in script
    assert "r12_bootstrap_training.py" in script
```

- [ ] **Step 2: Run RED**

Run: `F:\\XH-202615\\XH-202615\\.venv\\Scripts\\python.exe -m pytest tests\\test_export_r12_github_snapshot.py -q`

Expected: FAIL.

- [ ] **Step 3: Update docs and allowlist**

Document dry run then materialization command with placeholders. Add exactly the bootstrap module, CLI and test to the copy-only exporter allowlist; do not widen any directory glob.

- [ ] **Step 4: Verify, snapshot, and publish**

Run the full focused R12 suite, export into a new empty temporary directory, run all exported tests, verify the exported file list, and replace remote `main` only after confirming its current SHA. Use GitHub API only if the verified Git transport remains unavailable.
