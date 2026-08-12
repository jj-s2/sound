# R12 strict evaluation

The staged `scripts/r12_strict_holdout.py` CLI has two phases:

1. `select` trains only on the train role and freezes the candidate-router, gate, blend, and finite threshold from validation labels.
2. `evaluate` refits the train-only artifacts, checks every provenance digest, generates fixed predictions, and only then loads scoring labels.

The previous Dataset-A held-out labels are contaminated and are not valid for promotion. A new blind partition or Dataset-B is required for a final independent Overall claim.

