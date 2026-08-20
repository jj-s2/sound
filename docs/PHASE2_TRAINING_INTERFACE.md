# Phase 2 public/synthetic training interface

Phase 2 introduces a standard-library-only JSONL contract for future temporal-speaker and overlap
experiments. It defines data provenance and split isolation; it does not load audio or train a
model. Dataset-A remains evaluation-only and must not appear in a training manifest.

The implementation is `xh202615.training_data`. An illustrative, non-runnable manifest is provided
at `configs/phase2_training_manifest.example.jsonl`; its paths and labels are synthetic placeholders,
not Dataset-A records.

## JSONL fields

Each non-empty line is one JSON object with these fields:

| Field | Type | Meaning |
|---|---|---|
| `row_id` | string | Stable, manifest-wide unique record ID. |
| `split` | string | Exactly `train`, `val`, `test`, or `internal_test`. |
| `source` | string | Public corpus or synthetic-generation provenance. |
| `enrollment_audio` | path string | Enrollment/reference audio for the target speaker. |
| `target_audio` | path string | Clean target track, or an explicit non-Dataset-A placeholder track for a negative synthetic example. |
| `mixture_audio` | path string or null | Rendered mixture/noisy audio used as model input. |
| `target_speaker_id` | string | Identity represented by the enrollment audio. It remains populated for negative rows. |
| `interferer_speaker_id` | string or null | Interfering speaker identity, if present. |
| `target_present` | boolean | Whether the enrolled target speaker occurs in the example. |
| `overlap_ratio` | number | Fraction of the example containing overlap, in `[0, 1]`. |
| `snr_db` | number or null | Finite signal-to-noise ratio when noise was applied. |
| `sir_db` | number or null | Finite signal-to-interference ratio when an interferer was mixed. |
| `text` | string or null | Target transcript. It must be null when `target_present` is false. |
| `seed` | integer or null | Reproduction seed for synthetic generation. |

`TrainingManifestRow.to_dict()` emits all fields in the table order and converts `Path` values to
portable POSIX-style JSON strings. `TrainingManifestRow.from_dict()` rejects missing, unknown, or
incorrectly typed fields. When `base_dir` is supplied, relative audio paths are anchored there.
`read_training_manifest()` anchors them to the manifest's parent directory and reports malformed
JSON or malformed rows with the file and line number.

## Semantic invariants

`validate_training_manifest()` returns structured `ManifestIssue` values rather than raising for
semantic errors. It enforces:

- non-empty row IDs, source names, speaker IDs, and required path fields;
- the four allowed split names;
- unique `row_id` values;
- distinct target and interferer speaker IDs;
- finite `overlap_ratio` in `[0, 1]` and finite SNR/SIR values when supplied;
- null `text` for target-absent negatives;
- a `mixture_audio` path whenever overlap, SNR, or SIR metadata is present;
- speaker-disjoint splits across both roles: a speaker seen as either target or interferer in one
  split cannot appear in any role in another split; and
- exclusion of paths under caller-provided `forbidden_roots`.

Paths are normalized with `Path.resolve(strict=False)`. Root containment uses `Path` parent
relationships rather than string prefixes, so similarly named sibling directories are not confused.
The library deliberately does not hard-code a machine-specific Dataset-A path. Callers must pass all
evaluation-only and test-only roots appropriate to their environment, for example:

```python
from pathlib import Path

from xh202615.training_data import (
    assert_valid_training_manifest,
    read_training_manifest,
)

manifest_path = Path("configs/phase2_training_manifest.jsonl")
rows = read_training_manifest(manifest_path)
rows = assert_valid_training_manifest(
    rows,
    manifest_path=manifest_path,
    forbidden_roots=(
        Path("datasetA"),
        Path("private_test_audio"),
    ),
)
```

`assert_valid_training_manifest()` returns the materialized row tuple on success and raises
`ValueError` containing the structured issue codes and row IDs when any error remains.

## Public/synthetic split policy

1. **Declare provenance.** Every row names a public dataset release or a documented synthetic
   generator in `source`. Local cache location is not provenance.
2. **Assign speakers before examples.** Build a global speaker registry, assign each speaker to one
   split, then create utterances and mixtures. Never split individual utterances first.
3. **Treat both roles equally.** A speaker assigned to `train` may be a target or interferer only in
   `train`; the same identity cannot be reused as a validation/test interferer.
4. **Keep public corpus identities stable.** Namespace IDs by corpus when upstream identifiers can
   collide, such as `librispeech:1234` and `vctk:p225`.
5. **Give synthetic identities stable IDs.** Synthetic voices, voice configurations, or speaker
   simulations must receive stable identities and obey the same split rule. Changing the seed does
   not create a new speaker.
6. **Reserve evaluation splits.** Use `val` for model selection, `test` for a held-out public or
   synthetic benchmark, and `internal_test` for additional private-to-the-project checks. Do not tune
   on either test split.
7. **Exclude competition data by construction.** Pass Dataset-A, private test, and other
   evaluation-only roots through `forbidden_roots` on every validation gate. Do not copy, relabel,
   symlink, or derive training records from those roots.
8. **Persist the validated manifest.** Record the manifest, generator version, public dataset
   release, and seeds with experiment outputs so the exact split can be reproduced and audited.
