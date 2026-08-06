# Phase-2 v2 temporal-head training results

## Scope and data boundary

- Training manifest: `data/synthetic/aishell1_phase2_v2/manifest.jsonl`
- Rows: 2,160 (`train=1,200`, `val=480`, `test=480`)
- Speaker split: disjoint (`train=64`, `val=8`, `test=8`)
- Encoder: frozen WeSpeaker `chinese`, 256-dimensional embeddings
- Head: 3-window GRU, hidden size 128, 8 epochs, AdamW
- Dataset-A: `datasetA/datasetA`; used only for the final audit, never for training or threshold selection

## Training-domain result

At fixed threshold 0.5, the synthetic test split reached accuracy `0.8500` and presence F1 `0.8481`. The best validation loss was at epoch 7.

Artifacts:

- `output/phase2_temporal_v2_gru/frozen_features.pt`
- `output/phase2_temporal_v2_gru/best.pt`
- `output/phase2_temporal_v2_gru/summary.json`

## Dataset-A audit

The strongest existing ASR text source, `output/predictions/stage_dynamic_tse_fusion_full.jsonl`, was held fixed and passed through the temporal head with threshold 0.5.

| System | CER | RR | Overall |
|---|---:|---:|---:|
| Existing dynamic TSE fusion | 0.5079 | 0.9388 | 0.7155 |
| v2 temporal gate + same ASR | 0.6803 | 0.9620 | 0.6409 |

The gate improves rejection accuracy by `+0.0232`, but rejects too many positive commands (`false_reject_rate=0.5616`), increasing CER by `+0.1724`. It therefore fails the promotion gate and must not replace the existing route.

Audit artifact:

- `output/phase2_temporal_v2_gru/datasetA_audit_fusion.json`

## Decision

Do not start conditional enhancement or package a submission from this checkpoint. The next experiment should improve domain coverage and calibration using public/non-Dataset-A development data (noise/reverberation, SNR/SIR ranges, target-absent negatives, and a held-out calibration split), then re-run the same frozen Dataset-A audit once.

## Robust r2 follow-up (2026-08-05)

The prescribed follow-up was completed with an expanded public-only synthetic corpus and a fixed one-shot Dataset-A audit.

### Public-only data and selection protocol

- Training manifest: `data/synthetic/aishell1_phase2_robust_r2/manifest.jsonl`
- Rows: 2,640 (`train=1,152`, `val=768`, `test=720`), with a speaker-disjoint split (`train=129`, `val=16`, `test=16`)
- Augmentation: real OpenSLR RIR/noise assets, 50% reverberation, SNR `{0, 5, 10, 20}` dB, SIR `{-5, 0, 5}` dB, and partial/full overlap `{0.25, 0.5, 0.75, 1.0}`
- Dataset-A remained prohibited from manifest creation, training, checkpoint selection, and threshold calibration.
- Selection rule fixed before the audit: choose the lowest public validation loss. Presence thresholds are selected from the public validation split under a minimum 0.95 recall constraint.

| Head | Best epoch | Public val loss | Public val F1 | Public test F1 |
|---|---:|---:|---:|---:|
| fused (GRU + pooled summary) | 3 | 0.3159 | 0.9136 | 0.8764 |
| MLP | 8 | 0.2838 | 0.9136 | 0.8463 |
| GRU | 6 | **0.2772** | 0.8806 | 0.8386 |

The GRU was selected by the registered validation-loss rule. Its saved presence threshold is `0.2827059328556061`, with source `public_validation`; it was not adjusted after Dataset-A was read.

### Frozen Dataset-A rescue audit

The frozen dynamic-TSE fusion text was retained by default. Only a public-validation-confident positive was allowed to use the frozen raw-ASR text ("rescue" mode). No Dataset-A output was used to select the model, threshold, or policy.

| System | CER | RR | Overall |
|---|---:|---:|---:|
| Existing dynamic TSE fusion (frozen baseline) | 0.5079 | 0.9388 | 0.7155 |
| Robust-r2 GRU rescue audit | **0.4230** | 0.4030 | 0.4900 |

The router recovered CER but catastrophically increased false accepts (`false_accept_rate=0.5970`), reducing RR by `-0.5359` and Overall by `-0.2255`. It fails the promotion gate.

### Final decision for this optimization cycle

Keep `output/predictions/stage_dynamic_tse_fusion_full.jsonl` as the submission candidate. Do **not** use either temporal-head checkpoint or the generated rescue predictions for submission, and do not start conditional enhancement/TSE expansion from this failed gate. The r2 experiment is retained as a reproducible negative result and its Dataset-A outputs must not be reused for tuning.

Artifacts:

- `output/phase2_temporal_robust_r2/fused/summary.json`
- `output/phase2_temporal_robust_r2/mlp/summary.json`
- `output/phase2_temporal_robust_r2/gru/summary.json`
- `output/phase2_temporal_robust_r2/gru/datasetA_rescue_audit.json`
