# R11 E2 FireRedChat-pVAD Design

Date: 2026-08-10

## Objective and decision context

E0 proved that the current cached post-ASR gate cannot support the target:
CER `0.47645079899074855`, RR `0.9472573839662447`, Overall
`0.735403292487748`, and grouped-bootstrap 95% upper bound
`0.7567256721358552`. Therefore E2 must improve the enrollment-conditioned
target-speaker presence representation rather than tune the E0 classifier.

E2 integrates the open-weight FireRedChat personalized VAD as a zero-shot
representation. The wake audio is the target-speaker enrollment and the
command audio is processed frame by frame. E2 is an inference, caching, and
grouped-OOF falsification experiment; it does not train a neural network.

The experiment must answer two questions:

1. Do FireRed temporal target-speaker features add independent information over
   the cached E0 features?
2. Is the best group-disjoint gate sufficiently strong to justify implementing
   the positive-only ASR candidate ranker?

## External evidence and frozen upstream interface

The official model card is
`https://huggingface.co/FireRedTeam/FireRedChat-pvad`, Apache-2.0, Chinese and
English, and identifies `speechbrain/spkrec-ecapa-voxceleb` as its speaker
encoder. Model assets are pinned to Hugging Face revision
`74561b17a50fbe9d8f84dacc453f175cb97f567c`; the current package contains a
3.94 MB `pvad.onnx` and the ECAPA assets.

The official plugin is
`https://github.com/fireredchat-submodules/livekit-plugins-fireredchat-pvad`.
Its frozen runtime contract is:

- mono 16 kHz float32 audio;
- 160 command samples per inference step;
- speaker embedding shape `(1, 192)`;
- initial mel state `(1, 80, 15)` and GRU state `(2, 1, 256)`;
- ONNX inputs `input_audio`, `spkemb`, `mel_buffer`, and `gru_buffer`;
- raw target probability at output index 1 and recurrent states at outputs 2
  and 3;
- a fresh recurrent state for every command utterance;
- ECAPA enrollment embedding L2-normalized before pVAD inference.

The project will not import the LiveKit async stack. It will implement a small
offline adapter around the documented ONNX/ECAPA interface and retain upstream
URLs, revision, license, raw file SHA-256 values, required dependency versions,
and ONNX input/output metadata in the cache manifest.

## Options considered

### A. Offline official-model adapter — selected

Use SpeechBrain ECAPA for the wake embedding and ONNX Runtime for the official
pVAD recurrent model. This preserves the released model behavior without
adding LiveKit, exposes frame probabilities, and lets the repository measure
RTF and memory directly. The cost is one additional pinned dependency,
`speechbrain==1.0.3`, and an adapter that must be parity-tested against the
official shapes and state transitions.

### B. Vendor the complete LiveKit plugin — rejected

This minimizes translation from upstream code but introduces RTC frames,
async queues, event state, and a large dependency surface that the offline
competition pipeline does not need. It also makes deterministic per-file
caching and unit tests harder.

### C. Train a compact custom pVAD immediately — rejected for E2

This could better match Dataset-A but has the highest data and overfitting risk.
It is allowed only after E2 demonstrates independent FireRed temporal signal
and narrowly misses the promotion gates. Public AISHELL mixtures, not Dataset-A
audio or labels, would then supply neural training data.

## Architecture and file boundaries

### `xh202615/firered_pvad.py`

Owns offline neural inference only. It provides:

- `FireRedModelPaths`: resolved, pinned model paths;
- `PvadRuntimeConfig`: sample rate, frame size, provider, device, enrollment
  cap, and smoothing constant;
- `FireRedPvadRuntime`: ECAPA embedding extraction, recurrent-state reset, and
  frame inference;
- `PvadUtteranceFeatures`: the fixed label-free aggregate schema;
- `extract_pvad_features(wake_path, command_path)`: one deterministic record.

It must not import Dataset-A labels, candidate text, CER, RR, or evaluator code.
It accepts dependency injection for the ECAPA encoder and ONNX session so all
unit tests run without downloading model weights.

### `scripts/download_firered_pvad.py`

Downloads only the pinned Hugging Face revision into a gitignored local model
directory, verifies that `pvad.onnx` and the ECAPA assets exist, records raw
SHA-256 values, and fails closed if the requested revision or expected ONNX
interface disagrees. It never downloads code for execution and never stores a
credential in the repository.

### `scripts/cache_firered_pvad_features.py`

Loads the existing Dataset-A input JSONL only to obtain IDs and wake/command
audio paths. It rejects duplicate/missing IDs, missing audio, zero-length audio,
non-finite features, and output roots with foreign identity. It processes every
ID exactly once, supports restart through a validated per-record cache, and
publishes the final cache atomically.

The canonical cache uses ONNX `CPUExecutionProvider`, matching the released
plugin. ECAPA may use CUDA only after a fixed 32-item CPU/CUDA parity smoke has
maximum absolute feature difference at most `1e-4`; otherwise canonical ECAPA
also uses CPU. CUDA use, device name, driver, peak allocated VRAM, and parity
results are recorded.

### `xh202615/r11_pvad_oracle.py`

Joins the pVAD cache to the canonical R11 rows by exact ID and constructs three
predeclared gate families:

1. `firered_scalar`: direct frontiers over fixed pVAD aggregates, with no
   learned gate;
2. `firered_crossfit`: group-disjoint logistic regression and shallow
   histogram gradient boosting on pVAD features only;
3. `firered_fused_crossfit`: the same small grid on pVAD plus frozen E0 cached
   features.

All cross-fitting uses the same five `StratifiedGroupKFold` splits,
`shuffle=True`, `random_state=20260807`, and frozen `wake_component` groups as
E0. Model fitting may use presence labels only inside training folds. Feature
extraction and cached artifacts remain label-free. Accepted positive examples
receive the true best current candidate only for this diagnostic upper bound;
therefore the selected point remains non-deployable.

### `scripts/r11_pvad_oracle_oof.py`

Validates exact source/cache/manifest ID equality, runs the E2 oracle, performs
group bootstrap model-and-threshold reselection, cross-checks the official
evaluator, measures execution cost, and publishes an atomic evidence package.
It reuses the hardened E0 publication and validation primitives instead of
copying them.

## Audio preprocessing and fixed feature schema

Wake and command audio are decoded as mono float32, resampled deterministically
to 16 kHz, clipped to `[-1, 1]`, and rejected if shorter than 0.25 seconds. The
wake enrollment uses at most the first 5.0 seconds, matching the official
plugin's 80,000-sample cap. The full command is processed in consecutive
160-sample frames; the final incomplete frame is dropped and its sample count
is recorded. No label-dependent trimming is allowed.

For every command, recurrent buffers start at zero and the wake embedding is
fixed. The adapter stores raw pVAD probability and a transparent EMA defined by
`ema[0]=p[0]` and `ema[t]=0.8*ema[t-1]+0.2*p[t]`.

The fixed feature schema contains:

- frame count, analyzed duration, dropped-tail samples, and command duration;
- raw and EMA mean, standard deviation, min, max, and quantiles 10/25/50/75/90/95;
- raw and EMA fractions at thresholds 0.1/0.3/0.5/0.7/0.9;
- longest contiguous EMA run at 0.3/0.5/0.7, in frames and seconds;
- first and last EMA crossing at 0.3/0.5/0.7, active span, and transition count;
- enrollment duration, embedding norm before and after normalization;
- cold/warm elapsed seconds, audio seconds, RTF, peak RSS delta, and optional
  peak allocated VRAM.

Timing and memory fields are audit-only and excluded from gate fitting. Raw
frame arrays are not published in the final E2 evidence package; a temporary
compressed cache is permitted only under the gitignored
`tmp/r11_e2_firered_frames` path and is never placed inside either final output
root.

## Cache and evidence artifacts

The feature cache root `output/r11_e2_firered_cache` contains exactly:

- `pvad_features.jsonl`: one label-free aggregate record per ID;
- `pvad_manifest.json`: identity, schema, paths, revisions, model/audio/source
  digests, exact coverage, provider/device, environment, parity, timing, and
  memory summaries;
- `pvad_report.md`: compact operational report.

The final evaluation root `output/r11_e2_pvad_oracle` contains exactly:

- `e2_manifest.json`;
- `e2_oof_scores.jsonl`;
- `e2_frontier.jsonl`;
- `e2_summary.json`;
- `e2_report.md`.

Both writers use fixed sibling publication locks, unique staging and backup
paths, strict artifact identity/version markers, raw and joined-state digests,
and preserve all recoverable evidence on failure. Labels, reference text,
candidate CER, optimal actions, speaker embeddings, and frame-level arrays are
forbidden from published score/cache JSONL.

## Evaluation gates and branch decisions

The official objective remains
`Overall=((1-CER)+RR)/2`. The RR floor is frozen at `0.93` before E2. The full
CER/RR frontier is always published.

E2 returns one of three decisions:

- `continue_ranker`: pooled Overall at least `0.81`, pooled RR at least `0.93`,
  worst-fold Overall at least `0.77`, worst-fold RR at least `0.90`, every
  inner search feasible, and paired grouped-bootstrap lower 95% improvement
  bound versus E0 greater than zero. Proceed to the positive-only candidate
  ranker.
- `consider_custom_pvad`: `continue_ranker` fails, but pooled Overall is at
  least `0.78`, pooled RR at least `0.90`, paired improvement lower bound versus
  E0 is positive, and removing pVAD features reduces Overall by at least `0.01`.
  The representation transfers useful temporal evidence but needs public-data
  adaptation; design the compact three-class pVAD next.
- `falsified_firered`: all other outcomes, or the grouped-bootstrap upper 95%
  Overall bound is below `0.80`. Do not train a custom pVAD from this evidence;
  improve the target-presence representation or candidate generators first.

The default upstream activation threshold `0.5` is never transferred as the
competition decision threshold. It is only one fixed temporal aggregation
boundary among several label-free features.

## Failure handling and data boundary

- Dataset-A audio and labels are never copied into model downloads,
  checkpoints, public manifests, or Git.
- Dataset-A labels are allowed only in grouped-OOF target construction,
  fold-local fitting/calibration, oracle candidate assignment, and held-out
  evaluation. No full-data fit may be reported as OOF.
- Model/cache/source ID or digest disagreement fails before evaluation.
- Missing SpeechBrain/ONNX dependencies, provider fallback, model interface
  changes, non-finite probabilities, incomplete coverage, group overlap, or
  official metric disagreement fail closed.
- The model loader must not execute arbitrary remote code and must not use
  `trust_remote_code=True`.
- Generated models, audio, embeddings, caches, and evaluation outputs remain
  under existing gitignored paths.

## Testing and verification

Unit tests use fake ECAPA and fake ONNX sessions to cover audio normalization,
resampling, 160-sample framing, recurrent-state reset, embedding normalization,
EMA/quantile/run features, label independence, schema validation, and invalid
output preservation.

Integration tests use a small local generated-audio bundle and the real pinned
models to verify ONNX shapes, finite `[0,1]` probabilities, deterministic
repeat inference, CPU/CUDA parity policy, restart behavior, exact ID coverage,
and atomic publication. No Dataset-A label is needed for the cache integration
test.

Oracle tests cover exact joins, three model families, group-disjoint cross-fit,
no reference leakage in score artifacts, official evaluator parity, paired
group-bootstrap decisions, all three branch outcomes, and byte-identical
reproduction for a fixed seed.

The controlled execution order is:

1. dependency and model-interface preflight;
2. generated-audio integration smoke;
3. 32-item Dataset-A no-label cache smoke with CPU/CUDA parity and RTF/memory;
4. full 1,838-item cache;
5. E2 `n_boot=50` smoke;
6. E2 canonical `n_boot=2000` run;
7. independent digest, coverage, group leakage, official parity, deterministic
   rerun, latency, and memory audit;
8. record exactly one frozen branch decision before any E4 or custom-pVAD work.

## Out of scope

E2 does not train or fine-tune FireRed, ECAPA, TSE, ASR, or a custom pVAD. It
does not implement the positive-only candidate ranker, alter candidate ASR
generation, lower the frozen RR floor after observing results, or claim that an
OOF oracle point is deployable or a leaderboard score.
