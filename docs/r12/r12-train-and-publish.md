# R12 Paraformer training and publishing

All Dataset-A paths below are private local inputs. They are not committed, and neither are labels, audio, checkpoints, caches, nor `output/` artifacts.

Prepare a private train/inner-validation manifest and the train-only hotword artifact:

```powershell
python scripts/r12_asr_prepare_manifest.py --lineage <LINEAGE> --train-labels <PRIVATE_TRAIN_LABELS> --output-root <RUN_ROOT>/asr
python scripts/r12_asr_prepare_folds.py --lineage <LINEAGE> --output <RUN_ROOT>/asr/folds.json
python scripts/r12_asr_prepare_hotwords.py --train-labels <PRIVATE_TRAIN_LABELS> --train-parent-ids <TRAIN_PARENT_IDS> --output-root <RUN_ROOT>/hotwords --capacities 10,25
```

Validate the exact training recipe without importing FunASR or using a GPU:

```powershell
python scripts/r12_asr_train.py train --train-manifest <RUN_ROOT>/asr/private/train.jsonl --valid-manifest <RUN_ROOT>/asr/private/inner_valid.jsonl --output-dir <RUN_ROOT>/asr_train/fold0
```

Run GPU training only when memory is available and the output directory does not exist:

```powershell
python scripts/r12_asr_train.py train --execute --device cuda:0 --train-manifest <RUN_ROOT>/asr/private/train.jsonl --valid-manifest <RUN_ROOT>/asr/private/inner_valid.jsonl --output-dir <RUN_ROOT>/asr_train/fold0
```

Validation is used only to select a checkpoint and frozen downstream choices. The held-out internal test is never a training, validation, threshold-tuning, or smoke input; it is evaluated only once after selection is frozen.

GitHub is built from a copy-only export snapshot. The source worktree is never deleted. The publish allowlist is explicit: the five `xh202615/r12_asr_*.py` modules and their three small lineage dependencies (`data.py`, `r12_dataa_augmented_split.py`, `r12_dataa_augmentation.py`), `scripts/r12_asr_*.py`, the matching `test_r12_asr_*.py` tests, the two export-contract tests, the export script, this document, the example config, README, requirements, and `.gitignore`. Generated `output/`, private artifacts, Dataset-A data, model weights, caches, `.arena/`, `.superpowers/`, `.pytest_cache/`, legacy modules, and internal runbooks are not committed.
