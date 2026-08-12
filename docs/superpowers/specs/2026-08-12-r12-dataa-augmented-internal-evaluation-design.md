# R12 Dataset-A 增强训练与内部评估设计

## 目标与边界

本协议用于训练 R12 的门控与文本存在性模型，目标是在不把训练样本当测试样本的前提下，探索是否可将内部评估 Overall 提升至 `0.8` 以上。

数据来源仅为 Dataset-A。由于 Dataset-A 的标签已在历史实验中打开，本文的 `internal_test` 不是未知标签的盲测集，不能作为泛化性能的最终声明；但其标签、样本和阈值选择都必须在最后一次评估前保持不可访问，且绝不参与训练或验证选择。

该工作不训练或微调 ASR、TSE、FireRed pVAD 或 R3 模型；训练对象是现有 R12 的：

- `r12_calibrated_gate`（pVAD 特征上的拒绝门控）；
- `r12_candidate_router`（候选文本路由）；
- `r12_text_presence`（`r3_text + primary_text` 的文本存在性融合）。

## 冻结的数据划分

以 `datasetA_group_manifest_v1.json` 中的 `wake_component` 为不可拆分组。一个原始样本和其所有增强副本继承同一个 `wake_component`，不得进入不同角色。

使用 `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=20260812)`，按正/负样本分层。折 `0..13` 是 `train`，`14..16` 是 `validation`，`17..19` 是 `internal_test`。这比率是精确的 14/3/3 折，即目标 70/15/15。

在当前 1,838 条原始 Dataset-A 行上，预期原始行数为：

| 角色 | 行数 | 正例 | 负例 | wake groups |
| --- | ---: | ---: | ---: | ---: |
| train | 1,286 | 955 | 331 | 1,281 |
| validation | 275 | 204 | 71 | 274 |
| internal_test | 277 | 205 | 72 | 275 |

分割清单只序列化角色、ID、组和输入摘要，禁止写入标签、参考文本或识别文本。构建时可读取标签进行分层；写入后加载器必须拒绝含有这些私有字段的清单。

## 训练组音频增强

仅对 `train` 的 command audio 执行增强；原始训练音频始终保留。每个原始训练样本产生两个确定性副本，因此训练候选池为每个父样本的 `original + aug_a + aug_b`，预期最多 3,858 条。验证和内部测试严格只使用原始音频，且不生成副本。

增强变换以 `(parent_id, variant_name, seed=20260812)` 派生随机数，并在清单中写入参数和音频 SHA-256：

| 变体 | 变换 | 约束 |
| --- | --- | --- |
| `aug_a` | 速度扰动 0.95 或 1.05，随后增益 -3 至 +3 dB | 16 kHz、单声道、有限值；不可为零长度或削波失真 |
| `aug_b` | 增益 -3 至 +3 dB，加确定性白噪声，SNR 18 至 25 dB | 16 kHz、单声道、有限值；峰值归一化保留 1 dB 余量 |

增强只改变 command audio。wakeup/enrollment audio 保持原始文件，因其用于 TSE 的目标说话人条件；每条子样本必须记录 `parent_id`、`augmentation_id`、`wake_component`、输入/输出摘要和相对音频路径。所有派生 ID 使用 `parent_id + "__" + augmentation_id`，不得与原始 ID 冲突。

## 特征再生与样本谱系

禁止将增强音频与原始 ASR 文本、原始 pVAD 特征或原始 TSE 输出混用。对每个训练副本，必须以该副本的 command audio 重建下列全部输入特征：

1. CPU FireRed pVAD 缓存；
2. 主 ASR 与能量候选 ASR，并用现有标签无关规则生成 `candidate_fusion`；
3. TSE 推理音频、其 FunASR 文本及 TSE audio map；
4. R3 文本候选；
5. R12 canonical 行（`r3_text`、`primary_text`、`energy_text`、`tse_text`、音频特征）。

候选重生必须复用当前的无标签接口：FunASR 的 `--command-audio-map`、TSE 的 `--input-jsonl`、候选融合的 `select_best_asr_candidate.py` 与 CPU pVAD 缓存契约。R3 现有 Dataset-A 推理需抽为同一标签无关输入清单接口；在该接口存在前，流水线不得以原始 R3 文本替代增强副本的 R3 文本。

每个阶段写出输入清单摘要、模型/脚本版本、输出摘要和 `parent_id` 覆盖率。若任一副本在任一特征源失败，则整条副本从训练池排除，并在 `excluded_augmented_rows.jsonl` 中写明原因；不能静默回退到原始特征。

## 训练、选择与一次性内部测试

训练特征由 `train` 的原始行和完整增强行组成。训练标签仅来自该集合；同一父样本的多个副本允许同时训练，因为它们在同一训练角色。

验证仅使用原始 `validation` 行，并仅用于：

- 在预注册的小型网格中选择门控/文本模型超参数；
- 选择阈值与固定融合权重；
- 通过官方 evaluator 计算开发指标。

所有候选、网格、选择准则、验证结果以及选中配置的摘要必须在打开 `internal_test` 标签前保存为带摘要的 selection artifact。内部测试只执行一次：用选中配置在 `train + validation` 的原始及增强行上重拟合，再对原始 `internal_test` 特征推理；测试标签仅在该步骤读取一次，且不得据此再改阈值、特征、增强或模型。

报表中同时给出 `CER`、`RR`、`Overall = ((1 - CER) + RR) / 2`、混淆统计、分组 bootstrap 区间和覆盖率。`Overall >= 0.8` 是内部开发门槛，而非独立盲测结论。

## 产物、可复现性与保护措施

本协议新增的所有运行产物置于 `output/r12_dataa_augmented_internal_v1/`，包含：

- `split_manifest.json` 与其摘要；
- `augmentation_manifest.jsonl`、`excluded_augmented_rows.jsonl` 和音频目录；
- 每路候选、canonical input 和 CPU pVAD cache；
- `selection_artifact.json`（测试前封存）；
- `internal_test_result/`（唯一允许读取测试标签的阶段）；
- `run_manifest.json`（命令、版本、摘要、行数、父样本覆盖率）。

实现必须先有针对下列不变量的单元测试：分组不泄漏、只有训练组被增强、变换确定性、子样本谱系完整、增强行的所有候选特征来自同一音频摘要、验证/测试中不存在增强 ID、以及测试标签读取在 selection artifact 封存后仅发生一次。

## 不采用的捷径

- 不把训练集得分称为测试得分。
- 不将 validation 或 internal_test 的音频增强后并入训练。
- 不用 Dataset-A 标签训练 ASR/TSE/R3/pVAD 基模型。
- 不使用增强音频配原始文本或原始 pVAD 特征。
- 不根据 internal_test 结果调参并重复测试。
