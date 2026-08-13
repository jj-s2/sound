# R12 strict evaluation

The staged `scripts/r12_strict_holdout.py` CLI has two phases:

1. `select` trains only on the train role and freezes the candidate-router, gate, blend, and finite threshold from validation labels.
2. `evaluate` refits the train-only artifacts, checks every provenance digest, generates fixed predictions, and only then loads scoring labels.

The previous Dataset-A held-out labels are contaminated and are not valid for promotion. A new blind partition or Dataset-B is required for a final independent Overall claim.

## Dataset-A augmented internal evaluation

`scripts/r12_dataa_internal_eval.py` is a separate protocol for the approved
70/15/15 Dataset-A wake-group split. Only train receives deterministic audio
augmentation; validation and internal test stay raw-only. It is useful as a
reproducible internal regression, but Dataset-A labels were historically
opened, so its `Overall` must be described as a **Dataset-A group-disjoint
internal test**, never independent blind-test evidence. See
`docs/r12/dataa-augmented-internal-runbook.md` for the source-audio rebuild and
one-time-evaluation procedure.

