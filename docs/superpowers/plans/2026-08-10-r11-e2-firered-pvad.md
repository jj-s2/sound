# R11 E2 FireRedChat-pVAD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the pinned FireRedChat-pVAD model, cache deterministic enrollment-conditioned temporal features for Dataset-A, and run a reproducible grouped-OOF FireRed-only/fused gate-oracle that decides whether to proceed toward Overall above 0.8.

**Architecture:** A small offline adapter uses SpeechBrain ECAPA to encode wake audio and ONNX Runtime to stream 160-sample command frames through FireRed pVAD. A label-free atomic cache feeds a new R11 E2 grouped-OOF module with direct scalar, pVAD-only cross-fit, and pVAD-plus-cached cross-fit families. A thin CLI publishes a five-file evidence package and a frozen `continue_ranker`, `consider_custom_pvad`, or `falsified_firered` decision.

**Tech Stack:** Python 3.12, NumPy 1.26.4, SciPy 1.17.1, SoundFile 0.14.0, PyTorch/Torchaudio 2.4.1 CUDA 12.1, ONNX Runtime GPU 1.28.0 with CPU provider for pVAD, SpeechBrain 1.0.3, scikit-learn 1.9, Hugging Face Hub 0.36.2, pytest.

## Global Constraints

- Implement the approved design at `docs/superpowers/specs/2026-08-10-r11-e2-firered-pvad-design.md`; do not widen E2 into neural training or candidate ranking.
- Pin `FireRedTeam/FireRedChat-pvad` to revision `74561b17a50fbe9d8f84dacc453f175cb97f567c` and never use `trust_remote_code=True`.
- The canonical pVAD ONNX provider is `CPUExecutionProvider`; CUDA may accelerate ECAPA only after the fixed 32-item CPU/CUDA parity gate (`max_abs_feature_delta <= 1e-4`).
- Use mono 16 kHz float32 audio, 160-sample command frames, speaker embedding `(1,192)`, mel state `(1,80,15)`, and GRU state `(2,1,256)`.
- Use exactly five `StratifiedGroupKFold` folds with `shuffle=True`, `random_state=20260807`, frozen `wake_component` groups, RR floor `0.93`, and canonical bootstrap count `2000`.
- Dataset-A labels may appear only in fold-local target construction/fitting, oracle candidate assignment, and held-out evaluation. Cache extraction and published cache/score artifacts are label-free.
- The E2 selected point is diagnostic and non-deployable. The default upstream threshold `0.5` is not the competition decision threshold.
- Preserve unrelated untracked files. Do not commit models, generated audio, embeddings, cache records, Dataset-A content, credentials, or output artifacts.
- Every task uses TDD, records actual RED/GREEN output in `.superpowers/sdd/2026-08-10-r11-e2-firered-pvad/progress.md`, and receives a fresh Codex specification and quality review before the next task.

---

## File map

- Create `xh202615/artifact_publish.py`: generic fail-closed atomic package publisher extracted from the reviewed E0 writer.
- Modify `scripts/r11_gate_oracle_oof.py`: delegate package replacement to the shared publisher without changing E0 bytes or behavior.
- Create `xh202615/firered_model_assets.py`: pinned model specification, download verification, ONNX interface audit, and model manifest.
- Create `scripts/download_firered_pvad.py`: thin model-download/preflight CLI.
- Modify `requirements-runtime-windows.txt`: pin SpeechBrain and its currently
  missing direct/config-parser dependencies without changing Torch/CUDA.
- Create `xh202615/firered_pvad.py`: offline audio preprocessing, ECAPA enrollment, pVAD recurrent inference, and fixed feature aggregation.
- Create `xh202615/pvad_cache.py`: exact-ID cache construction, restart records, cache schemas, audit summaries, and atomic three-file publisher.
- Create `scripts/cache_firered_pvad_features.py`: feature-cache CLI.
- Create `xh202615/r11_pvad_oracle.py`: E2 feature joins, model families, paired group bootstrap, gates, and branch decision.
- Create `scripts/r11_pvad_oracle_oof.py`: real-input validation, official parity, evidence serialization, and E2 CLI.
- Create `tests/test_artifact_publish.py`, `tests/test_firered_model_assets.py`, `tests/test_firered_pvad.py`, `tests/test_pvad_cache.py`, and `tests/test_r11_pvad_oracle.py`.
- Create `tests/test_firered_pvad_integration.py`: opt-in real-model generated-audio smoke, skipped when the model root is absent.

### Task 1: Extract the reviewed atomic artifact publisher

**Files:**
- Create: `xh202615/artifact_publish.py`
- Create: `tests/test_artifact_publish.py`
- Modify: `scripts/r11_gate_oracle_oof.py`
- Modify: `tests/test_r11_gate_oracle.py`

**Interfaces:**
- Consumes: prepared UTF-8 artifact contents and an exact artifact identity contract.
- Produces:
  ```python
  @dataclass(frozen=True)
  class ArtifactContract:
      artifact_kind: str
      schema_version: str
      required_names: tuple[str, ...]
      identity_json_names: tuple[str, ...]

  def publish_text_package(
      output_root: Path,
      contract: ArtifactContract,
      contents: Mapping[str, str],
  ) -> dict[str, Path]: ...
  ```

- [ ] **Step 1: Write failing generic publisher tests**

Create tests for exact-name rejection before writes, recognized replacement,
foreign-root preservation, held-lock preservation, single-rename rollback,
double-rename backup retention, competing-output preservation, and
byte-identical replacement. The competing test must distinguish the competing
package from the backup:

```python
def test_competing_output_and_old_backup_both_survive(tmp_path, monkeypatch):
    publish_text_package(tmp_path / "out", CONTRACT, package("old"))
    competing = package("competing")
    inject_competing_root_on_publish_rename(monkeypatch, tmp_path, competing)
    with pytest.raises(RuntimeError, match="unexpected output root"):
        publish_text_package(tmp_path / "out", CONTRACT, package("new"))
    assert read_marker(tmp_path / "out") == "competing"
    assert read_marker(single_backup(tmp_path)) == "old"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_artifact_publish.py -q
```

Expected: collection failure because `xh202615.artifact_publish` does not exist.

- [ ] **Step 3: Implement the shared publisher**

Move the fixed sibling lock, owner metadata, unique staging/backup, recognized
identity checks, safe rollback, and owner-checked lock cleanup into the new
module. Validate `set(contents) == set(contract.required_names)` before creating
the parent or lock. A pre-existing lock and an unexpected rollback destination
must be preserved.

- [ ] **Step 4: Adapt E0 to the shared publisher**

Replace only E0's package-publication block. Keep E0 artifact names, canonical
JSON, manifest/summary bytes, error messages covered by tests, and CLI defaults
stable. Remove publisher helpers from the script only after all callers use the
shared module.

- [ ] **Step 5: Run focused and E0 regressions**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_artifact_publish.py tests\test_r11_gate_oracle.py -q
.venv\Scripts\python.exe scripts\r11_gate_oracle_oof.py --help
```

Expected: all pass; help exits 0.

- [ ] **Step 6: Commit**

```powershell
git add xh202615/artifact_publish.py scripts/r11_gate_oracle_oof.py tests/test_artifact_publish.py tests/test_r11_gate_oracle.py
git commit -m "refactor: share fail-closed artifact publisher"
```

### Task 2: Add pinned FireRed model acquisition and interface preflight

**Files:**
- Create: `xh202615/firered_model_assets.py`
- Create: `scripts/download_firered_pvad.py`
- Create: `tests/test_firered_model_assets.py`
- Modify: `requirements-runtime-windows.txt`

**Interfaces:**
- Consumes: Hugging Face cache/download function and optional local model root.
- Produces:
  ```python
  FIRERED_REPO_ID = "FireRedTeam/FireRedChat-pvad"
  FIRERED_REVISION = "74561b17a50fbe9d8f84dacc453f175cb97f567c"

  @dataclass(frozen=True)
  class FireRedModelPaths:
      root: Path
      pvad_onnx: Path
      ecapa_root: Path
      manifest: Path

  def download_and_verify_model(root: Path, *, downloader=snapshot_download) -> FireRedModelPaths: ...
  def verify_onnx_contract(session: object) -> dict[str, object]: ...
  ```

- [ ] **Step 1: Write failing model-asset tests**

Use a fake downloader and fake ONNX metadata objects. Assert the exact repo and
revision, no remote-code flag, required asset existence, per-file SHA-256,
aggregate digest, and rejection of wrong/missing ONNX names or shapes.

```python
def test_download_is_revision_pinned(tmp_path):
    calls = []
    paths = download_and_verify_model(tmp_path, downloader=fake_downloader(calls))
    assert calls[0]["repo_id"] == FIRERED_REPO_ID
    assert calls[0]["revision"] == FIRERED_REVISION
    assert "trust_remote_code" not in calls[0]
    assert paths.pvad_onnx.is_file()
```

- [ ] **Step 2: Run tests to verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_firered_model_assets.py -q
```

Expected: import failure for the missing module.

- [ ] **Step 3: Implement model asset verification**

Call `snapshot_download` with only `repo_id`, `revision`, `local_dir`, and a
fixed allow-list covering `pvad.onnx`, `NOTICE`, `README.md`, and
`spkrec-ecapa-voxceleb/**`. Verify regular files and write canonical
`model_manifest.json` containing upstream identity, dependency versions, raw
digests, and ONNX metadata. Never overwrite a foreign recognized model root.

- [ ] **Step 4: Implement the CLI and dependency pins**

Add these exact pins to `requirements-runtime-windows.txt`:

```text
speechbrain==1.0.3
hyperpyyaml==1.2.3
joblib==1.5.3
packaging==24.2
PyYAML==6.0.3
ruamel.yaml==0.18.16
ruamel.yaml.clib==0.2.15
sentencepiece==0.2.2
tqdm==4.65.2
```

Keep Torch 2.4.1+cu121 and Torchaudio 2.4.1+cu121 unchanged. The CLI defaults to
`output/models/FireRedChat-pvad/74561b17a50fbe9d8f84dacc453f175cb97f567c`
and supports `--model-root` only.

- [ ] **Step 5: Run focused tests and CLI help**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_firered_model_assets.py -q
.venv\Scripts\python.exe scripts\download_firered_pvad.py --help
```

- [ ] **Step 6: Commit**

```powershell
git add requirements-runtime-windows.txt xh202615/firered_model_assets.py scripts/download_firered_pvad.py tests/test_firered_model_assets.py
git commit -m "feat: add pinned FireRed model preflight"
```

### Task 3: Implement offline FireRed pVAD inference and aggregates

**Files:**
- Create: `xh202615/firered_pvad.py`
- Create: `tests/test_firered_pvad.py`

**Interfaces:**
- Consumes: `FireRedModelPaths`, wake/command audio paths, injected ECAPA
  encoder and ONNX session.
- Produces:
  ```python
  @dataclass(frozen=True)
  class PvadRuntimeConfig:
      sample_rate: int = 16000
      frame_samples: int = 160
      enrollment_cap_seconds: float = 5.0
      minimum_audio_seconds: float = 0.25
      ema_alpha: float = 0.8
      onnx_provider: str = "CPUExecutionProvider"
      ecapa_device: str = "cpu"

  @dataclass(frozen=True)
  class PvadUtteranceFeatures:
      sample_id: str
      values: Mapping[str, float | int]
      audit: Mapping[str, float | int | str]

  class FireRedPvadRuntime:
      def extract(self, sample_id: str, wake_path: Path, command_path: Path) -> PvadUtteranceFeatures: ...
  ```

- [ ] **Step 1: Write failing preprocessing and runtime tests**

Generate mono/stereo WAVs in temporary directories. Inject a fake encoder that
returns a known non-unit vector and a fake ONNX session that validates input
shapes and returns deterministic probabilities/states. Cover mono conversion,
16 kHz polyphase resampling, clipping, short-audio rejection, 5-second wake cap,
tail dropping, one embedding per sample, state reset between utterances, and
state carry within one utterance.

```python
def test_recurrent_state_resets_between_commands(runtime, wav_pair):
    first = runtime.extract("a", *wav_pair)
    second = runtime.extract("b", *wav_pair)
    assert first.values == second.values
    assert runtime.fake_session.first_state_for("a") == runtime.zero_state
    assert runtime.fake_session.first_state_for("b") == runtime.zero_state
```

- [ ] **Step 2: Write failing aggregate tests**

For fixed probability vector `[0.1, 0.4, 0.8, 0.8, 0.2]`, assert exact EMA,
quantiles, fractions at 0.1/0.3/0.5/0.7/0.9, longest runs, crossings, active
spans, and transitions. Assert timing/memory keys are audit-only and absent
from `PVAD_GATE_FEATURE_SCHEMA`.

- [ ] **Step 3: Run tests to verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_firered_pvad.py -q
```

Expected: import failure for the missing module.

- [ ] **Step 4: Implement audio and neural runtime**

Use `soundfile.read(always_2d=True)`, channel mean, float64-safe clipping, and
`scipy.signal.resample_poly` with integer GCD factors before conversion to
contiguous float32. Normalize ECAPA output with a finite nonzero L2 check. Run
ONNX in complete 160-sample frames with zero mel/GRU states per utterance and
validate every raw probability is finite and in `[0,1]`.

- [ ] **Step 5: Implement the frozen feature aggregator**

Expose `PVAD_GATE_FEATURE_SCHEMA` in the exact design order and construct every
feature from raw/EMA arrays without labels. Record cold/warm elapsed time,
audio seconds, RTF, RSS delta, and optional CUDA peak memory only in `audit`.

- [ ] **Step 6: Run focused tests and label-independence mutation**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_firered_pvad.py -q
```

The label-independence test constructs two external sample metadata objects
with different labels but the same audio paths and asserts byte-identical
canonical feature JSON.

- [ ] **Step 7: Commit**

```powershell
git add xh202615/firered_pvad.py tests/test_firered_pvad.py
git commit -m "feat: add offline FireRed pVAD features"
```

### Task 4: Build the resumable label-free pVAD feature cache

**Files:**
- Create: `xh202615/pvad_cache.py`
- Create: `scripts/cache_firered_pvad_features.py`
- Create: `tests/test_pvad_cache.py`

**Interfaces:**
- Consumes: Dataset-A input JSONL/audio paths, verified model paths, and
  `FireRedPvadRuntime`.
- Produces:
  ```python
  def build_pvad_cache(
      dataset_root: Path,
      model_paths: FireRedModelPaths,
      output_root: Path,
      *,
      resume_root: Path,
      ecapa_device: str,
      limit: int | None = None,
  ) -> dict[str, Path]: ...
  ```

- [ ] **Step 1: Write failing cache contract tests**

Assert duplicate IDs within/across splits, missing audio, model digest mismatch,
resume-record source digest mismatch, incomplete coverage, non-finite features,
foreign output root, and concurrent lock all fail closed. Assert final cache
contains exactly `pvad_features.jsonl`, `pvad_manifest.json`, and
`pvad_report.md`.

```python
def test_published_cache_is_label_free(tmp_path, fake_runtime):
    files = build_pvad_cache(bundle_with_labels(tmp_path), MODEL, tmp_path / "out",
                             resume_root=tmp_path / "resume", ecapa_device="cpu")
    text = files["features"].read_text("utf-8")
    assert "label" not in text and "reference" not in text
```

- [ ] **Step 2: Run tests to verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_pvad_cache.py -q
```

- [ ] **Step 3: Implement exact input and resume validation**

Use the existing dataset loader only after independently rejecting duplicate
raw JSONL IDs. Resume records live under `tmp/r11_e2_firered_frames`, include
audio/model/config digests, and are reused only after all identity fields and
feature schema match. Sort final records by the same numeric-aware canonical ID
order used by E0.

- [ ] **Step 4: Implement cache manifest and atomic publish**

Record exact coverage, source/model/config digests, schema, environment,
provider/device, dependency versions, CPU/CUDA parity result, timing percentiles,
RTF, RSS/VRAM summaries, and deterministic record digest. Publish through
`publish_text_package` with artifact kind `r11_e2_firered_cache` and schema
version `v1`.

- [ ] **Step 5: Implement CLI**

Defaults:

```text
--dataset-root datasetA/datasetA
--model-root output/models/FireRedChat-pvad/74561b17a50fbe9d8f84dacc453f175cb97f567c
--output-root output/r11_e2_firered_cache
--resume-root tmp/r11_e2_firered_frames
--ecapa-device cpu
```

Also support `--limit` and `--parity-reference`. A recognized cache root is
atomically replaced by the generic publisher; a foreign root is always
preserved and rejected. There is no unsafe overwrite switch.

- [ ] **Step 6: Run focused tests and help**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_pvad_cache.py tests\test_artifact_publish.py -q
.venv\Scripts\python.exe scripts\cache_firered_pvad_features.py --help
```

- [ ] **Step 7: Commit**

```powershell
git add xh202615/pvad_cache.py scripts/cache_firered_pvad_features.py tests/test_pvad_cache.py
git commit -m "feat: add resumable FireRed feature cache"
```

### Task 5: Implement E2 grouped-OOF families and paired decisions

**Files:**
- Create: `xh202615/r11_pvad_oracle.py`
- Create: `tests/test_r11_pvad_oracle.py`

**Interfaces:**
- Consumes: canonical R11 `CandidateRow` values, labels, frozen groups, exact
  pVAD feature mapping, and E0 feature functions.
- Produces:
  ```python
  @dataclass(frozen=True)
  class E2DecisionGates:
      preferred_overall: float = 0.81
      rr_floor: float = 0.93
      worst_overall_floor: float = 0.77
      worst_rr_floor: float = 0.90
      adaptation_overall_floor: float = 0.78
      adaptation_rr_floor: float = 0.90
      ablation_delta_floor: float = 0.01

  def evaluate_e2(
      rows: Sequence[CandidateRow],
      labels: Mapping[str, str | None],
      groups: Sequence[object],
      pvad_features: Mapping[str, Mapping[str, float]],
      *, n_splits: int = 5, seed: int = 20260807,
      n_boot: int = 2000,
  ) -> dict[str, object]: ...
  ```

- [ ] **Step 1: Write failing join and leakage tests**

Require exact ID equality and exact finite schema. Mutate labels/reference text
while keeping audio-derived pVAD features fixed and assert the feature matrices
are unchanged. Assert each group appears in only one outer test fold and each
row appears exactly once.

- [ ] **Step 2: Write failing family tests**

Use literal synthetic rows and pVAD features to assert these predeclared
families exist:

```python
{
  "firered_scalar",
  "firered_crossfit",
  "firered_fused_crossfit",
  "cached_e0_baseline",
}
```

Scalar score names are frozen to raw/EMA max, q90, q95, and fractions at
0.3/0.5/0.7. Cross-fit models are logistic C 0.01/0.1/1/10 and HGB leaf 3/7;
no other hyperparameters may enter the grid.

- [ ] **Step 3: Write failing paired-bootstrap and branch tests**

Every replicate must resample whole groups once and reselect the best feasible
point independently for E0, pVAD-only, and fused families using the same sampled
indices. Assert deterministic replicate deltas and all decisions:

```python
assert decide_e2(passing_fixture) == "continue_ranker"
assert decide_e2(useful_but_narrow_fixture) == "consider_custom_pvad"
assert decide_e2(no_signal_fixture) == "falsified_firered"
assert decide_e2(ci_high_below_point_eight) == "falsified_firered"
```

- [ ] **Step 4: Run tests to verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r11_pvad_oracle.py -q
```

- [ ] **Step 5: Implement feature matrices and OOF model families**

Reuse E0 metrics, candidate-oracle assignments, fold generation, and model
factories. Keep pVAD timing/audit values out of matrices. Export OOF probability
vectors by stable family/model name and explicit fold metadata.

- [ ] **Step 6: Implement paired group bootstrap and frozen decisions**

Return compact summaries plus in-memory replicate arrays for parity tests. Do
not serialize raw bootstrap arrays. Compute paired E2-minus-E0 and
fused-minus-pVAD-only intervals and the pVAD ablation point delta.

- [ ] **Step 7: Run focused and E0 regression tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r11_pvad_oracle.py tests\test_r11_gate_oracle.py tests\test_metrics.py -q
```

- [ ] **Step 8: Commit**

```powershell
git add xh202615/r11_pvad_oracle.py tests/test_r11_pvad_oracle.py
git commit -m "feat: add FireRed fused gate oracle"
```

### Task 6: Add the E2 CLI and fail-closed evidence package

**Files:**
- Create: `scripts/r11_pvad_oracle_oof.py`
- Modify: `tests/test_r11_pvad_oracle.py`

**Interfaces:**
- Consumes: the exact E0 real inputs plus
  `output/r11_e2_firered_cache/pvad_features.jsonl` and its manifest.
- Produces: exactly five artifacts under `output/r11_e2_pvad_oracle`.

- [ ] **Step 1: Write failing CLI and artifact tests**

Cover help, missing inputs, duplicate cache IDs, source/model/config digest
disagreement, incomplete coverage, invalid fold audit, non-finite scores,
official metric disagreement, strict five-file identity, foreign-root
preservation, deterministic rerun, and forbidden-field scans.

The score JSONL may contain only `id`, `group`, `fold`, and stable model
probabilities. It must not contain `label`, `reference`, candidate text, CER,
speaker embedding, raw frames, or chosen actions.

- [ ] **Step 2: Run tests to verify RED**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_r11_pvad_oracle.py -k "cli or artifact" -q
```

- [ ] **Step 3: Implement real-input validation and official parity**

Reuse E0's exact duplicate/ID/label/group checks. Additionally verify cache
model revision/digest, feature schema/digest, source audio digests, coverage,
provider policy, and CPU/CUDA parity evidence before evaluation.

- [ ] **Step 4: Implement five-file serialization**

Publish:

```text
e2_manifest.json
e2_oof_scores.jsonl
e2_frontier.jsonl
e2_summary.json
e2_report.md
```

The manifest contains per-source digests, cache/model/config hashes, exact ID
sets, fold train/test groups, coverage/disjointness audits, official parity,
paired CI summaries, RTF/memory audit, and branch gates. Threshold infinities
use the same explicit reject-all marker as E0. Publish through the shared
artifact publisher with identity `r11_e2_pvad_oracle/v1`.

- [ ] **Step 5: Implement CLI defaults**

Copy E0 defaults for dataset/candidate/TSE/audio/R3/group paths and add:

```text
--pvad-cache output/r11_e2_firered_cache/pvad_features.jsonl
--pvad-manifest output/r11_e2_firered_cache/pvad_manifest.json
--output-root output/r11_e2_pvad_oracle
--n-outer 5 --n-boot 2000 --seed 20260807 --rr-floor 0.93
```

- [ ] **Step 6: Run focused and full scoped tests**

```powershell
.venv\Scripts\python.exe -m pytest tests\test_artifact_publish.py tests\test_firered_model_assets.py tests\test_firered_pvad.py tests\test_pvad_cache.py tests\test_r11_pvad_oracle.py tests\test_r11_gate_oracle.py tests\test_metrics.py -q
.venv\Scripts\python.exe scripts\r11_pvad_oracle_oof.py --help
```

- [ ] **Step 7: Commit**

```powershell
git add scripts/r11_pvad_oracle_oof.py tests/test_r11_pvad_oracle.py
git commit -m "feat: add reproducible R11 E2 CLI"
```

### Task 7: Add and run the opt-in real-model integration gate

**Files:**
- Create: `tests/test_firered_pvad_integration.py`
- Modify only if the real gate exposes a verified defect: the owning Task 2-4 file and its focused test.

**Interfaces:**
- Consumes: downloaded pinned model root and generated sine/silence/speech-like WAV fixtures.
- Produces: deterministic interface/finite-probability evidence and CPU/CUDA parity audit.

- [ ] **Step 1: Install the pinned dependency without replacing Torch**

```powershell
.venv\Scripts\python.exe -m pip install --no-deps speechbrain==1.0.3 hyperpyyaml==1.2.3 ruamel.yaml==0.18.16 ruamel.yaml.clib==0.2.15
```

Verify imports and record all dependency versions. Existing exact pins for
SentencePiece, Joblib, NumPy, Packaging, SciPy, Torch, Torchaudio, tqdm, and
Hugging Face Hub satisfy SpeechBrain's remaining direct dependencies. If an
additional runtime dependency is actually missing, first add a failing import
test, then pin only that named dependency in a reviewed fix commit.

- [ ] **Step 2: Download and verify the pinned model**

```powershell
.venv\Scripts\python.exe scripts\download_firered_pvad.py
```

Expected: exit 0, exact revision in `model_manifest.json`, verified ONNX input
and output metadata, and raw model digests.

- [ ] **Step 3: Write the opt-in integration test**

Skip unless `FIRERED_PVAD_MODEL_ROOT` points to a verified model root. Generate
local WAV fixtures, run two identical inferences after reset, assert identical
feature JSON, finite probabilities, correct frame counts, and no provider
fallback.

- [ ] **Step 4: Run real CPU integration and 32-item CPU cache smoke**

```powershell
$env:FIRERED_PVAD_MODEL_ROOT = (Resolve-Path 'output\models\FireRedChat-pvad\74561b17a50fbe9d8f84dacc453f175cb97f567c')
.venv\Scripts\python.exe -m pytest tests\test_firered_pvad_integration.py -q
.venv\Scripts\python.exe scripts\cache_firered_pvad_features.py --limit 32 --output-root output\r11_e2_firered_cache_cpu32 --ecapa-device cpu
```

- [ ] **Step 5: Run 32-item CUDA parity cache smoke**

```powershell
.venv\Scripts\python.exe scripts\cache_firered_pvad_features.py --limit 32 --output-root output\r11_e2_firered_cache_cuda32 --ecapa-device cuda:0 --parity-reference output\r11_e2_firered_cache_cpu32\pvad_features.jsonl
```

Use CUDA for the full ECAPA cache only when the manifest reports
`max_abs_feature_delta <= 1e-4`; otherwise use CPU. In either case pVAD remains
on ONNX CPU.

- [ ] **Step 6: Commit the integration test**

```powershell
git add tests/test_firered_pvad_integration.py
git commit -m "test: add FireRed real-model smoke"
```

### Task 8: Run the full cache, canonical E2 oracle, and independent audit

**Files:**
- Append: `.superpowers/sdd/2026-08-10-r11-e2-firered-pvad/progress.md`
- Do not modify production code unless a failing test first reproduces a verified defect.

**Interfaces:**
- Consumes: all reviewed Task 1-7 code, pinned model, Dataset-A input audio,
  frozen R11 sources, and group manifest.
- Produces: verified cache/evaluation artifacts and one frozen branch decision.

- [ ] **Step 1: Run the full label-free pVAD cache**

Use the device selected by Task 7 parity:

```powershell
.venv\Scripts\python.exe scripts\cache_firered_pvad_features.py --ecapa-device cuda:0
```

or, when CUDA parity failed:

```powershell
.venv\Scripts\python.exe scripts\cache_firered_pvad_features.py --ecapa-device cpu
```

Require 1,838 unique IDs, finite schema-complete features, exact source/model
digests, no labels, and complete timing/memory evidence.

- [ ] **Step 2: Run the fast E2 smoke**

```powershell
.venv\Scripts\python.exe scripts\r11_pvad_oracle_oof.py --n-boot 50 --output-root output\r11_e2_pvad_oracle_smoke
```

Inspect only the structured summary and manifest. Require exact coverage,
group-disjoint folds, finite unit probabilities, all official parity flags,
and a valid frozen decision.

- [ ] **Step 3: Run the canonical E2 evaluation**

```powershell
.venv\Scripts\python.exe scripts\r11_pvad_oracle_oof.py
```

Require `n_boot=2000` and exactly five canonical artifacts.

- [ ] **Step 4: Perform an independent evidence audit**

Recompute every raw source/model/cache/config digest; parse every score/frontier
row; verify 1,838 unique IDs, 1,830 frozen groups, exact once-only test
coverage, zero train/test group intersections, finite `[0,1]` probabilities,
official formula parity, paired CI arithmetic, and absence of forbidden fields,
locks, staging, and backups.

- [ ] **Step 5: Verify determinism**

Record SHA-256 for all five artifacts, rerun the canonical command with the same
seed, and require byte-identical hashes. Any mismatch is a failure, not a
warning.

- [ ] **Step 6: Record the branch decision and report metrics**

Append exact CER, RR, Overall, 95% grouped CI, worst-fold metrics, E2-minus-E0
paired CI, ablation delta, RTF, peak RSS/VRAM, selected family/model/threshold,
digests, test counts, and decision to the progress ledger.

Do not start E4 or custom pVAD work in the same decision run:

- `continue_ranker` queues the positive-only expected-CER/LambdaMART design;
- `consider_custom_pvad` queues a public-data compact three-class pVAD design;
- `falsified_firered` stops FireRed/custom-pVAD work and returns to target
  representation or candidate-generator design.
