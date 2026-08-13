# R12 Dataset-A 增强内部评测运行手册

本手册重建一条独立的 R12 训练与评测链路：从原始 Dataset-A 音频、冻结的 wake-group 清单和固定的公开基础模型开始，生成 `canonical_input.jsonl` 与 CPU pVAD cache。它不覆盖旧 R11/R12 输出。

结果的正确表述是 **Dataset-A group-disjoint internal test**。Dataset-A 标签已在历史实验中被使用过，所以这不是独立盲测证据，不能替代 Dataset-B 或新未打开标签的盲测集。

## 固定边界

- 原始数据：`F:\XH-202615\XH-202615\datasetA\datasetA`，只读。
- 冻结 wake-group：`F:\XH-202615\XH-202615\.superpowers\sdd\2026-08-07-r9-overall-08-arena\datasetA_group_manifest_v1.json`。
- split：`StratifiedGroupKFold(20, shuffle=True, random_state=20260812)`；fold `0..13` train、`14..16` validation、`17..19` internal_test。
- 仅 train 原始样本产生 `aug_a`、`aug_b`；validation/internal_test 永远只有原始音频。
- TSE、pVAD、两路 ASR、R3 都对 derived 音频重跑。旧 ASR、旧 pVAD cache、旧 canonical 不可复用。
- 除 `private/` 外，所有发布的 split、lineage、canonical、selection 和评测包都没有标签/参考文本。

## 变量

在 PowerShell 中设置：

```powershell
$Py = 'F:\XH-202615\XH-202615\.venv\Scripts\python.exe'
$Repo = 'F:\XH-202615\XH-202615'
$Code = 'F:\XH-202615\XH-202615\.worktrees\r12-dataa-augmented-internal'
$Raw = "$Repo\datasetA\datasetA"
$GroupManifest = "$Repo\.superpowers\sdd\2026-08-07-r9-overall-08-arena\datasetA_group_manifest_v1.json"
$Run = "$Repo\output\r12_dataa_augmented_internal_v1"
```

`$Run` 必须不存在。若已经存在，保留它作为已有实验记录；需要重跑时使用新的实验名，绝不覆盖。

## 1. 从原始 Dataset-A 构建私有映射、split 和派生音频

```powershell
& $Py "$Code\scripts\r12_dataa_prepare_private_maps.py" `
  --dataset-root $Raw --group-manifest $GroupManifest `
  --labels-output "$Run\private\labels.json" `
  --groups-output "$Run\private\groups.json"

& $Py "$Code\scripts\r12_dataa_prepare_split.py" `
  --dataset-root $Raw --labels "$Run\private\labels.json" `
  --groups "$Run\private\groups.json" --output "$Run\split_manifest.json"

& $Py "$Code\scripts\r12_dataa_augment_audio.py" `
  --dataset-root $Raw --split-manifest "$Run\split_manifest.json" `
  --output-root "$Run\derived_dataset"

foreach ($role in 'train','validation','internal_test') {
  & $Py "$Code\scripts\r12_dataa_export_role_labels.py" `
    --labels "$Run\private\labels.json" --split-manifest "$Run\split_manifest.json" `
    --role $role --output "$Run\private\$role`_labels.json"
}
```

停止条件：split 不是 `1286/275/277` 原始行，或某个 validation/internal_test ID 含 `__aug_`，或 `derived_dataset` 位于 `$Raw` 内部。任何一个条件满足时不得继续。

## 2. 恢复固定 FireRed 模型并重建 CPU pVAD cache

```powershell
$FireRed = "$Run\models\FireRedChat-pvad\74561b17a50fbe9d8f84dacc453f175cb97f567c"

& $Py "$Code\scripts\download_firered_pvad.py" --model-root $FireRed

& $Py "$Code\scripts\cache_firered_pvad_features.py" `
  --dataset-root "$Run\derived_dataset" --model-root $FireRed `
  --output-root "$Run\pvad_cache_cpu" --resume-root "$Run\pvad_resume" `
  --ecapa-device cpu
```

此命令只能产生新的 `pvad_cache_cpu/`。它的 manifest 必须声明 `CPUExecutionProvider` 与 `device=cpu`，并且 `source.per_id_audio_sha256` 必须覆盖所有 lineage ID。

## 3. 对派生音频重建四类候选特征

```powershell
$Derived = "$Run\derived_dataset"
$Sources = "$Run\sources"
$Tse = "$Repo\output\tse_r7_joint\best.pt"
$TseEmb = "$Repo\output\tse_r7_joint\enrollment_embeddings.pt"
$Temporal = "$Repo\output\training\r3_temporal_head\best.pt"

& $Py "$Code\scripts\run_funasr_asr.py" --dataset-root $Derived --splits pos,neg `
  --output "$Sources\primary_raw.jsonl" --device cpu --on-error raise

& $Py "$Code\scripts\run_funasr_asr.py" --dataset-root $Derived --splits pos,neg `
  --output "$Sources\energy_raw.jsonl" --device cpu --hotword-preset assistant --on-error raise

& $Py "$Code\scripts\select_best_asr_candidate.py" `
  --primary "$Sources\primary_raw.jsonl" --candidate "energy=$Sources\energy_raw.jsonl" `
  --source-priority primary,energy --output "$Sources\candidate_fusion_raw.jsonl"

& $Py "$Code\scripts\run_tse_inference.py" `
  --input-jsonl "$Derived\pos.jsonl" --input-jsonl "$Derived\neg.jsonl" `
  --checkpoint $Tse --output-root "$Run\tse_audio" --output-map "$Sources\audio_map_raw.jsonl" `
  --embedding-cache $TseEmb --device cpu --on-error raise

& $Py "$Code\scripts\run_funasr_asr.py" --dataset-root $Derived --splits pos,neg `
  --command-audio-map "$Sources\audio_map_raw.jsonl" --output "$Sources\tse_asr_raw.jsonl" `
  --device cpu --on-error raise

& $Py "$Code\scripts\run_temporal_head_inference.py" `
  --input-jsonl "$Derived\pos.jsonl" --input-jsonl "$Derived\neg.jsonl" `
  --candidate-asr "$Sources\candidate_fusion_raw.jsonl" --checkpoint $Temporal `
  --output "$Sources\r3_predictions.jsonl" --device cpu
```

`energy` 在本实验中表示第二个明确的 FunASR hotword 配置；它不是历史 SNR 文件。两路 ASR、TSE 与 temporal 都从这次 `$Derived` 中读取音频。

## 4. 签名候选来源并发布 canonical input

```powershell
$Lineage = "$Derived\augmentation_manifest.jsonl"

foreach ($name in 'candidate_fusion','tse_asr','audio_map') {
  & $Py "$Code\scripts\r12_dataa_rebuild_features.py" attest `
    --lineage $Lineage --source "$Sources\$name`_raw.jsonl" `
    --output "$Sources\$name`_attested.jsonl"
}

& $Py "$Code\scripts\r12_dataa_rebuild_features.py" canonical `
  --lineage $Lineage --candidate-fusion "$Sources\candidate_fusion_attested.jsonl" `
  --tse-asr "$Sources\tse_asr_attested.jsonl" --audio-map "$Sources\audio_map_attested.jsonl" `
  --r3-predictions "$Sources\r3_predictions.jsonl" `
  --pvad-manifest "$Run\pvad_cache_cpu\pvad_manifest.json" `
  --canonical-output "$Run\canonical_input.jsonl"
```

`attest` 会对 `original_command_audio`（TSE 优先）或 `command_audio` 实际算 SHA-256；任何一条与 lineage 不同、缺 ID 或重复 ID 都会中止。temporal 输出自己从输入音频计算 `command_audio_sha256`，由 canonical join 再次核对。

## 5. 选择、单次内部测试与归档

```powershell
& $Py "$Code\scripts\r12_dataa_internal_eval.py" select `
  --canonical-input-jsonl "$Run\canonical_input.jsonl" --lineage $Lineage `
  --split-manifest "$Run\split_manifest.json" --cache-root "$Run\pvad_cache_cpu" `
  --train-labels "$Run\private\train_labels.json" `
  --validation-labels "$Run\private\validation_labels.json" `
  --bootstrap-count 2000 --selection-output "$Run\selection_artifact.json"

Get-FileHash "$Run\selection_artifact.json" -Algorithm SHA256

& $Py "$Code\scripts\r12_dataa_internal_eval.py" evaluate `
  --canonical-input-jsonl "$Run\canonical_input.jsonl" --lineage $Lineage `
  --split-manifest "$Run\split_manifest.json" --cache-root "$Run\pvad_cache_cpu" `
  --train-labels "$Run\private\train_labels.json" `
  --validation-labels "$Run\private\validation_labels.json" `
  --internal-test-labels "$Run\private\internal_test_labels.json" `
  --bootstrap-count 2000 --selection-input "$Run\selection_artifact.json" `
  --evaluation-output "$Run\internal_test_result"
```

只有 `evaluate` 可以读取 `internal_test_labels.json`，而且输出目录只能不存在。成功时它原子发布四文件包：`r12_manifest.json`、`r12_internal_predictions.jsonl`、`r12_summary.json`、`r12_report.md`，并记录 `internal_test_label_read_count: 1`。

若 Overall 低于 0.8，禁止根据该测试包修改阈值、模型、候选或增强。应建立新的实验名，并只在 train/validation 上继续研究；对该 internal test 的结果保持冻结归档。
