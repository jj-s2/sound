# R12 Raw Dataset-A Training Bootstrap Design

## Objective

Provide one local command that derives all R12 ASR training prerequisites from raw Dataset-A, private labels, and group assignments, without using held-out internal-test supervision.

## Inputs and output contract

The command accepts a Dataset-A root containing `pos.jsonl` and `neg.jsonl`, a private JSON label map, a private JSON group map, and a non-existent run root. It writes every generated artifact under that run root and fails if the root already exists. No source dataset file is changed.

## Pipeline

The command builds the immutable 70/15/15 parent-group split. Only train-role groups receive deterministic audio augmentation and lineage. From train-role parents only, it selects a deterministic group-disjoint 10% `inner_valid` subset using seed `20260814`; the remaining train parents form ASR fit supervision. It then produces private ASR train/valid manifests, a label-free three-fold OOF assignment, and train-only private hotword candidates.

Validation and held-out internal test never supply labels, commands, hotwords, or rows to the ASR train/inner-valid artifacts. Their IDs may exist in the split manifest but are excluded from augmented ASR supervision.

## Modes and testing

`--dry-run` validates raw IDs, labels, groups, output path, and deterministic role counts without audio augmentation or GPU use. Normal mode runs augmentation and all artifact builders. A synthetic-fixture end-to-end test verifies group boundaries, internal-test exclusion, complete artifacts, reproducibility, and fail-closed reruns.

## Publishing

Only the bootstrap source, its test, and public documentation are added to the GitHub snapshot allowlist. Dataset-A, labels, group maps, run roots, audio, hotwords, checkpoints, caches, and outputs are never committed.
