# R12 Paraformer training and publishing

All Dataset-A paths below are private local inputs. They are not committed, and neither are labels, audio, checkpoints, caches, nor `output/` artifacts.

Prepare a private train/inner-validation manifest and the train-only hotword artifact:

```powershell
python scripts/r12_asr_prepare_manifest.py --lineage <LINEAGE> --train-labels <PRIVATE_TRAIN_LABELS> --output-root <RUN_ROOT>/asr
python scripts/r12_asr_prepare_folds.py --lineage <LINEAGE> --output <RUN_ROOT>/asr/folds.json
python scripts/r12_asr_prepare_hotwords.py --train-labels <PRIVATE_TRAIN_LABELS> --train-parent-ids <TRAIN_PARENT_IDS> --output-root <RUN_ROOT>/hotwords --capacities 10,25
python scripts/r12_asr_prepare_subphrase_hotwords.py --train-labels <PRIVATE_TRAIN_LABELS> --parent-ids <TRAIN_PARENT_IDS> --output-root <RUN_ROOT>/subphrase_hotwords --capacities 10,20,40
```

Or build all of those prerequisites from raw Dataset-A in one local command. The dry run validates IDs, private label/group maps, and the fixed group split without writing audio or using a GPU:

```powershell
python scripts/r12_bootstrap_training.py --dry-run --dataset-root <DATASET_A_ROOT> --labels <PRIVATE_LABELS_JSON> --groups <PRIVATE_GROUPS_JSON> --output-root <RUN_ROOT>
python scripts/r12_bootstrap_training.py --dataset-root <DATASET_A_ROOT> --labels <PRIVATE_LABELS_JSON> --groups <PRIVATE_GROUPS_JSON> --output-root <RUN_ROOT>
```

Validate the exact training recipe without importing FunASR or using a GPU:

```powershell
python scripts/r12_asr_train.py train --train-manifest <RUN_ROOT>/asr/private/asr_train.jsonl --valid-manifest <RUN_ROOT>/asr/private/asr_inner_valid.jsonl --output-dir <RUN_ROOT>/asr_train/fold0
```

Run GPU training only when memory is available and the output directory does not exist:

```powershell
python scripts/r12_asr_train.py train --execute --device cuda:0 --train-manifest <RUN_ROOT>/asr/private/asr_train.jsonl --valid-manifest <RUN_ROOT>/asr/private/asr_inner_valid.jsonl --output-dir <RUN_ROOT>/asr_train/fold0
```

For an 8GB Windows GPU, start with a smaller micro-batch and preserve the
effective batch through gradient accumulation:

```powershell
python scripts/r12_asr_train.py train --execute --device cuda:0 `
  --train-manifest <RUN_ROOT>/asr/private/asr_train.jsonl `
  --valid-manifest <RUN_ROOT>/asr/private/asr_inner_valid.jsonl `
  --output-dir <RUN_ROOT>/asr_train/fold0 `
  --batch-size 4 --accum-grad 32 --num-workers 0
```

The launcher resolves relative audio paths from the generated `augmented`
directory. `--num-workers 0` is the stable default for Windows CUDA runs;
increase it only after the first epoch is healthy.

8GB GPU 的起始配方是 FP16、encoder/decoder q-k-v-o LoRA、rank 8、alpha 16、
dropout 0.05、learning rate 1e-4、最多30 epoch、保留最佳10个并平均最佳5个。
`batch_size=4` 与 `accum_grad=32` 是 8GB Windows GPU 的保守起点；若显存持续充足，
只逐步增大 micro-batch，并保持 `accum_grad` 不超过每个 epoch 的 batch 数。

Personal VAD 需要单独运行，不与 Paraformer 同时占用 GPU：

```powershell
python scripts/r12_prepare_personal_vad_mixtures.py --source-manifest <TRAIN_MIXTURE_MANIFEST> --output <RUN_ROOT>/pvad/lineage.jsonl
python scripts/r12_train_personal_vad.py --features <RUN_ROOT>/pvad/features.npz --output <RUN_ROOT>/pvad/best.pt --device cuda:0
```

Validation is used only to select a checkpoint and frozen downstream choices. The held-out internal test is never a training, validation, threshold-tuning, or smoke input; it is evaluated only once after selection is frozen.

Personal VAD 是 speaker-conditioned gate：它输出非语音、目标人物语音和非目标人物语音三类帧概率，不能用 internal-test 音频或标签生成训练混合物。当前目标 Overall 0.85 仍必须通过新的盲测验证。

GitHub is built from a copy-only export snapshot. The source worktree is never deleted. The publish allowlist is explicit: the five `xh202615/r12_asr_*.py` modules, `r12_training_bootstrap.py`, and their three small lineage dependencies (`data.py`, `r12_dataa_augmented_split.py`, `r12_dataa_augmentation.py`); `scripts/r12_asr_*.py` plus `r12_bootstrap_training.py`; their tests; the two export-contract tests; the export script; this document; the example config; README; requirements; and `.gitignore`. Generated `output/`, private artifacts, Dataset-A data, model weights, caches, `.arena/`, `.superpowers/`, `.pytest_cache/`, legacy modules, and internal runbooks are not committed.
