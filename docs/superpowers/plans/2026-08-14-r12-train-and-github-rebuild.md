# R12 Training Entry and GitHub Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real, Dataset-A-train-only Paraformer training launcher with CPU-safe smoke coverage, then publish a minimal R12-only snapshot to GitHub without deleting local workspace files.

**Architecture:** A small training-plan module validates private manifests, fold role and output location, then constructs an explicit FunASR `train_ds` argv. The CLI supports a CPU-only dry run that never imports FunASR. A separate export manifest defines the only tracked paths copied into a temporary publish repository; the source worktree is never cleaned.

**Tech Stack:** Python 3.11, FunASR 1.4.1 `funasr.bin.train_ds`, JSON/JSONL, pytest, git, GitHub CLI.

## Global Constraints

- Work only in `F:\\XH-202615\\XH-202615\\.worktrees\\r12-dataa-augmented-internal` on `codex/r12-dataa-augmented-internal`.
- Real training accepts only a private train manifest and private inner-validation manifest derived from Dataset-A train parents; it rejects internal-test labels/paths and never evaluates internal test.
- Default `dry-run` is CPU-only, imports no FunASR, allocates no GPU, reads no audio, and creates no checkpoint.
- Actual training is explicit (`train`), requires `--device cuda:0`, an empty output directory, and calls FunASR `train_ds` only after all validation passes.
- No raw data, labels, audio paths, generated outputs, models, caches, `.arena/`, `.superpowers/`, or `.pytest_cache/` is committed or published.
- The source worktree and primary checkout are never deleted or cleaned. GitHub replacement happens from a newly-created temporary export repository.
- Force-pushing the remote default branch is authorized only to replace GitHub code; do not rewrite any local branch or delete any local file.

---

### Task 1: Validated formal training launcher

**Files:**
- Create: `xh202615/r12_asr_train.py`
- Modify: `scripts/r12_asr_train.py`
- Test: `tests/test_r12_asr_train.py`

**Interfaces:**
- Produces `TrainingConfig(train_manifest: Path, valid_manifest: Path, output_dir: Path, model: str, device: str, mode: Literal["lora", "freeze_encoder"], seed: int = 20260814)`.
- Produces `build_train_argv(config) -> tuple[str, ...]` and `run_training(config, *, runner: Callable[[Sequence[str]], int] | None = None) -> TrainingResult`.
- `run_training` invokes `sys.executable -m funasr.bin.train_ds` only after a `train` request has passed validation.

- [ ] **Step 1: Write failing tests**

```python
def test_dry_run_returns_train_ds_command_without_importing_funasr(tmp_path: Path) -> None:
    result = run_training(config(tmp_path), runner=None, dry_run=True)
    assert result.executed is False
    assert result.argv[1:3] == ("-m", "funasr.bin.train_ds")

def test_training_rejects_internal_test_path_before_runner(tmp_path: Path) -> None:
    bad = config(tmp_path, train_manifest=tmp_path / "internal_test.jsonl")
    with pytest.raises(ValueError, match="internal-test"):
        run_training(bad, runner=fail_if_called)
```

Also test: private JSONL rows have exactly Task-1 ASR keys; output directory must not exist; LoRA arguments include only model/device/lora configuration and no VAD/punctuation; freeze mode uses `freeze_param=encoder`; runner receives executable `-m funasr.bin.train_ds`; CLI rejects `--internal-test-labels` and default dry-run creates no directory.

- [ ] **Step 2: Run RED**

Run: `F:\\XH-202615\\XH-202615\\.venv\\Scripts\\python.exe -m pytest tests\\test_r12_asr_train.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement minimal launcher**

```python
def build_train_argv(config: TrainingConfig) -> tuple[str, ...]:
    return (sys.executable, "-m", "funasr.bin.train_ds", f"model={config.model}",
            f"device={config.device}", f"output_dir={config.output_dir}",
            f"dataset_conf.data_list={config.train_manifest}",
            f"dataset_conf.data_list_valid={config.valid_manifest}")
```

Validate path names/row shapes before returning argv. In LoRA mode append `lora_only=true` and `decoder_conf.lora_list=[q,k,v,o]`; in freeze mode append `freeze_param=encoder`. Use a subprocess runner only in `train`, never in dry-run.

- [ ] **Step 4: Run GREEN and regressions**

Run: `F:\\XH-202615\\XH-202615\\.venv\\Scripts\\python.exe -m pytest tests\\test_r12_asr_train.py tests\\test_r12_asr_smoke.py tests\\test_r12_asr_manifest.py tests\\test_r12_asr_folds.py tests\\test_r12_asr_hotword.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit only Task 1 code/tests with `feat: add validated R12 Paraformer training launcher`.

### Task 2: Minimal repository contract and documentation

**Files:**
- Create: `configs/r12_paraformer_train.example.yaml`
- Create: `docs/r12/r12-train-and-publish.md`
- Modify: `README.md`
- Test: `tests/test_r12_publish_contract.py`

**Interfaces:**
- The example config uses placeholder local paths only, never real Dataset-A paths.
- `r12-train-and-publish.md` documents manifest generation, dry-run, explicit GPU training, validation-only selection, and a publish allowlist.

- [ ] **Step 1: Write failing contract test**

```python
def test_publish_contract_lists_only_committable_source_paths() -> None:
    text = Path("docs/r12/r12-train-and-publish.md").read_text(encoding="utf-8")
    assert "internal test" in text.lower()
    assert "not committed" in text.lower()
    assert "output/" in text
```

- [ ] **Step 2: Run RED**

Run: `F:\\XH-202615\\XH-202615\\.venv\\Scripts\\python.exe -m pytest tests\\test_r12_publish_contract.py -q`

Expected: FAIL because the document/config do not exist.

- [ ] **Step 3: Write minimal docs/config**

Document exact script commands using placeholders (`<PRIVATE_TRAIN_LABELS>`, `<LINEAGE>`, `<RUN_ROOT>`), state the test boundary, and list publish paths: `xh202615/`, `scripts/r12_*.py`, `tests/test_r12_*.py`, `configs/r12_*.yaml|json`, docs, README, and requirements. State explicitly that the export operation copies rather than deletes local files.

- [ ] **Step 4: Run GREEN**

Run: `F:\\XH-202615\\XH-202615\\.venv\\Scripts\\python.exe -m pytest tests\\test_r12_publish_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit Task 2 only with `docs: add R12 training and publish contract`.

### Task 3: Verify and replace GitHub contents from export snapshot

**Files:**
- Create: `scripts/export_r12_github_snapshot.ps1`
- Test: `tests/test_export_r12_github_snapshot.py`

**Interfaces:**
- `export_r12_github_snapshot.ps1 -SourceRoot <repo> -Destination <empty-dir>` copies only allowlisted source paths into a new destination and initializes no source-side deletion.
- It rejects a nonempty destination and rejects any source path outside the documented allowlist.

- [ ] **Step 1: Write failing test**

```python
def test_export_script_has_copy_only_contract() -> None:
    script = Path("scripts/export_r12_github_snapshot.ps1").read_text(encoding="utf-8")
    assert "Copy-Item" in script
    assert "Remove-Item" not in script
    assert "output" not in script.lower()
```

- [ ] **Step 2: Run RED**

Run: `F:\\XH-202615\\XH-202615\\.venv\\Scripts\\python.exe -m pytest tests\\test_export_r12_github_snapshot.py -q`

Expected: FAIL because the export script does not exist.

- [ ] **Step 3: Implement exporter and publish**

The script builds an explicit file list with `git ls-files`, filters to the documented source allowlist, and copies those files into an empty temporary destination. It never calls `Remove-Item`. After tests pass, initialize Git only in the temporary destination, create one fresh commit, set the GitHub origin, and force-push that new commit to the remote default branch. The push replaces remote code only; it never changes the local worktree contents.

- [ ] **Step 4: Verify and publish**

Run focused training/doc/export tests plus M0 tests, then inspect the export file list and commit tree before `git push --force-with-lease origin <temp-branch>:<default-branch>`. Verify the remote tree with `gh api repos/jj-s2/sound/git/trees/<sha>?recursive=1` and report only source paths.
