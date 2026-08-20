# sound：R12 复杂场景语音命令识别

本项目是 XH-202615 语音命令识别系统的代码仓库。系统不是单一 ASR 模型，而是由
ASR 候选生成、目标人物语音判断、时序特征、候选路由和拒识门控组成的整体链路。

## 1. 总体架构

```text
原始 Dataset-A 音频
        │
        ├── 按 group 划分 train / validation / internal_test
        │       └── 只有 train 允许做增强
        │
        ├── 多路 ASR
        │       ├── primary ASR
        │       └── energy ASR
        │
        ├── 目标人物语音链路
        │       ├── pVAD：判断目标人物是否在说话
        │       └── TSE：提取目标人物语音并计算 speaker/presence 特征
        │
        ├── R3 temporal head
        │       └── 根据音频时序特征产生辅助预测
        │
        ├── canonical_input.jsonl
        │       └── 统一保存每条样本的多路文本和音频特征
        │
        ├── train：训练门控、路由和文本 presence 模型
        ├── validation：选择模型、候选组合和阈值
        └── internal_test：选择冻结后只评估一次
```

最终输出不是简单的 ASR 文本，而是：

```text
目标人物判断
    → 候选 ASR 选择
    → 文本 presence / 路由
    → CER 与拒识决策
    → Overall
```

项目中的 Overall 由识别质量和拒识质量共同决定，不能只看 ASR 的训练 loss。

## 2. 数据划分与防止泄漏

Dataset-A 必须先按 group 划分：

```text
Dataset-A train groups       → 训练 ASR/路由/门控
Dataset-A validation groups  → 选择模型和阈值
Dataset-A held-out test      → 选择冻结后只评估一次
```

数据增强只作用于 train-role 的原始样本。validation 和 internal_test 必须保持原始
音频和原始 ID，不能用于训练、阈值调整或增强策略选择。

仓库不包含 Dataset-A 音频、private labels、groups 文件和内部测试标签。它们必须由
数据拥有者在本地提供。

## 3. 代码目录

### `xh202615/`

核心 Python 模块：

- `data.py`：Dataset-A 样本读取和数据契约
- `r12_training_bootstrap.py`：生成 group split、train-only 增强规划和训练清单
- `r12_dataa_augmentation.py`：构建只含 train 增强样本的派生数据集
- `r12_dataa_canonical.py`：把多路 ASR 和音频特征合成为 canonical 输入
- `r12_calibrated_gate.py`：训练和冻结门控选择
- `r12_candidate_router.py`：在多路 ASR 候选之间做路由
- `r12_text_presence.py`：文本 presence 特征和预测
- `r11_pvad_oracle.py`：pVAD 特征和目标人物语音证据
- `temporal_head.py` / `temporal_training.py`：R3 temporal head
- `tse_presence.py` / `target_extractor.py`：TSE 和目标人物语音特征
- `evaluation.py` / `metrics.py`：CER、RR、Overall 和评估契约
- `text.py`：ASR 文本清洗和评估归一化

### `scripts/`

可执行流程脚本：

- `r12_bootstrap_training.py`：从原始 Dataset-A 生成训练准备包
- `r12_dataa_augment_audio.py`：执行 train-only 音频增强
- `r12_asr_train.py`：Paraformer 训练和 smoke 配置检查
- `run_funasr_asr.py`：运行 FunASR/Paraformer ASR
- `run_temporal_head_inference.py`：R3 temporal head 推理
- `run_tse_inference.py`：TSE 推理和增强音频生成
- `r12_dataa_rebuild_features.py`：校验并构建 canonical 输入
- `r12_dataa_internal_eval.py`：执行 validation selection 和一次性 internal evaluation
- `select_best_asr_candidate.py`：融合多路 ASR 候选
- `validate_competition_submission.py`：提交文件契约检查

### `tests/`

覆盖数据契约、group split、增强边界、pVAD、ASR smoke、TSE、路由、门控、评估和
提交验证。提交前建议运行：

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## 4. ASR 模型说明

默认 ASR 链路使用 FunASR Paraformer/SeacoParaformer 配置。运行本地训练 checkpoint
时，使用 `--init-param` 将 checkpoint 加载到对应的本地模型配置中：

```powershell
.venv\Scripts\python.exe scripts\run_funasr_asr.py `
  --dataset-root datasetA\datasetA `
  --splits pos,neg `
  --model <local-seaco-paraformer-directory> `
  --init-param <local-checkpoint.pt> `
  --device cpu `
  --vad-model none `
  --punc-model none
```

`--init-param` 只加载 checkpoint 参数，不包含模型配置和 tokenizer。模型目录中的
`config.yaml`、`tokens.json`、frontend 和训练时配置必须与 checkpoint 保持一致。

仓库不上传模型权重。当前验证过的 `model.pt.ep150` 并不是主链路推荐模型；是否使用
某个 checkpoint 必须由 validation CER/Overall 选择，不能因为 epoch 数最大就直接采用。

ASR 文本智能清洗是显式 opt-in：

```powershell
--smart-cleanup
```

默认关闭，以避免清洗规则改变路由器的拒识行为。任何清洗策略都必须在 validation
上验证后才能进入正式链路。

## 5. 从原始 Dataset-A 开始准备训练

准备本地目录：

```text
datasetA\datasetA\pos\*.wav
datasetA\datasetA\neg\*.wav
```

并提供不上传到 GitHub 的 private 文件：

```text
<run-root>\private\labels.json
<run-root>\private\groups.json
```

执行训练准备：

```powershell
.venv\Scripts\python.exe scripts\r12_bootstrap_training.py `
  --dataset-root datasetA\datasetA `
  --labels <run-root>\private\labels.json `
  --groups <run-root>\private\groups.json `
  --output-root output\r12_bootstrap_v1
```

该步骤会生成冻结 split、train-only augmentation 规划、ASR manifest、hotword 输入和
训练审计信息。它不会上传音频、标签或权重。

## 6. 完整评估原则

正式评估必须遵循以下顺序：

1. 从原始音频重新生成 ASR、pVAD、R3、TSE 和 canonical 特征。
2. 检查行数、ID 覆盖、source digest 和特征来源。
3. 只用 train labels 训练门控/路由模型。
4. 只用 validation labels 选择模型和阈值，并写入冻结 selection artifact。
5. 验证 selection provenance 和 digest。
6. 调用 `evaluate` 一次，读取 internal_test labels 一次。
7. 不根据 internal_test 结果调阈值、重训或重跑评估。

`Dataset-A internal_test` 是 group-disjoint 的内部结果，不等同于独立盲测证据。

## 7. 环境安装

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-runtime-windows.txt
.venv\Scripts\python -m pytest -q
```

模型、数据集、生成音频、缓存、评估输出和私有标签均由 `.gitignore` 排除。请不要把
数据集、权重、访问令牌或 internal-test 标签提交到 GitHub。
