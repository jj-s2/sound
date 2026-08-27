# R12 Repository Rebuild and Training Design

## Objective

Replace the repository contents with a minimal, reproducible R12 training repository, then publish that cleaned repository to the configured GitHub remote.

## Retained production surface

The rebuilt repository retains only:

- `xh202615/`: R12 pipeline and ASR training-support modules.
- `scripts/`: raw-derived data preparation, training, inference, and smoke command-line entry points.
- `tests/`: synthetic-fixture unit and focused regression tests.
- `configs/`: runtime/training configuration.
- Root `README.md`, runtime requirements, and concise R12 reproducibility documentation.

The implementation must add the formal Paraformer training entry point that consumes only a private Dataset-A train manifest and an outer-fold assignment. It must default to CPU-safe smoke/config validation. Actual GPU training is an explicit command and never reads validation or internal-test labels as training inputs.

## Removed surface

The rebuilt repository excludes legacy experiments, old handoffs, old output artifacts, model checkpoints, caches, temporary review folders, historical docs unrelated to the retained R12 workflow, and non-R12 code. Raw Dataset-A audio and labels remain local inputs and are not added to Git.

Deletion happens only in the isolated worktree after an exact allowlist is validated. Git history retains recoverability until the user explicitly requests history rewriting; this task replaces the repository tip, not Git history.

## Data and evaluation boundaries

Training consumes Dataset-A train-group supervision only. Validation is used only for checkpoint/model selection. Held-out internal test remains excluded from training, selection, threshold tuning, and smoke tests; it is evaluated once only after selection is frozen.

Private artifacts may contain text labels and hotwords below an explicitly named private output directory. Committed source, public provenance summaries, console output, and GitHub uploads must not contain Dataset-A text, audio paths, labels, caches, model weights, or generated outputs.

## Verification and publishing

The formal train CLI gets a synthetic CPU smoke test with an injected backend. It validates data/fold interfaces and creates no model checkpoint in smoke mode. The cleaned checkout must pass the focused R12 suite and CLI help/smoke checks before commit and push.

Publishing stages only the allowlisted repository surface. The GitHub remote is checked for authentication and reachability before pushing. If the remote default branch cannot be fast-forwarded, the branch is pushed and a draft pull request is opened; force-pushing or rewriting shared history requires a separate explicit user instruction.
