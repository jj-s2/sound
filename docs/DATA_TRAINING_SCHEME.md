# XH-202615 数据训练方案

更新时间：2026-08-02

## 1. 目标

本项目的数据训练目标不是单独训练一个 ASR 或分离模型，而是服务最终评分：

- 降低正样本 CER；
- 提高负样本 RR；
- 控制正样本误拒；
- 只在困难样本触发高成本增强，控制推理时间和显存。

最终数据体系分为四类：

1. 中文指令 ASR 数据；
2. 目标说话人验证数据；
3. 重叠/噪声/混响增强数据；
4. 独立验证与消融评测数据。

Dataset A/B 只作为评测和格式核查数据，不进入训练、微调、阈值拟合、校准器训练或命令模板学习。

## 2. 数据源优先级

### P0：必须优先准备

| 数据集 | 用途 | 选择理由 |
|---|---|---|
| AISHELL-2 | 中文 ASR、中文命令语音、混音干净源 | 1000 小时普通话，文本包含唤醒词、语音控制词、智能家居等领域，最贴近赛题 |
| AISHELL-1 | 快速 ASR baseline、干净中文源 | 体量适中、16kHz、400 人、易跑通 |
| CN-Celeb | 声纹验证、同/异人 pair、hard negative | 中文真实场景、多说话人、多风格，适合训练/校准说话人验证 |
| HI-MIA | 唤醒音频声纹、远场文本相关验证 | 真实家庭环境、近/远场、唤醒词，与赛题 wakeup_audio 高度贴合 |
| DNS Challenge noise/RIR | 家庭噪声、混响、干扰说话人增强 | 可生成噪声、混响、干扰说话人条件 |

### P1：用于提升泛化

| 数据集 | 用途 | 选择理由 |
|---|---|---|
| WenetSpeech | 多领域中文 ASR 适应 | 覆盖新闻、访谈、综艺、戏剧等多场景，适合增强 ASR 鲁棒性 |
| MAGICDATA Mandarin | 中文 ASR、家居控制语料补充 | 755 小时，包含 home command and control 等领域 |
| AISHELL-4 | 重叠说话、说话人活动检测、会议场景鲁棒性 | 真实多人会议，含重叠、短停顿、说话人活动标注 |
| AISHELL-3 | 多说话人中文干净源、TSE 合成 | 218 人、85 小时、高质量普通话 |
| MobvoiHotwords | 唤醒词/非唤醒词区分、负样本扩充 | 智能音箱、1/3/5 米距离、家庭噪声、关键词和非关键词 |
| AISHELL-DMASH | 真实智能家居远场 ASR/声纹/唤醒补充 | 真实家庭、多点位、多设备、近讲与远讲，最贴近应用场景但体量大 |

### P2：只作为方法参考或补充

| 数据集 | 用途 | 注意 |
|---|---|---|
| LibriMix | 学习分离数据组织和混音脚本 | 英语，不作为中文主训练数据 |
| WHAM!/WHAMR! | 噪声/混响分离参考 | 英语，许可证含非商用限制 |
| MUSAN / SLR28 RIRS_NOISES | 噪声和混响补充 | 可用于数据增强，但需记录许可证 |

## 3. 推荐目录结构

```text
data/
├── raw_public/
│   ├── aishell1/
│   ├── aishell2/
│   ├── cnceleb/
│   ├── hi_mia/
│   ├── dns/
│   ├── magicdata/
│   ├── aishell3/
│   └── aishell4/
├── manifests/
│   ├── asr_train.jsonl
│   ├── asr_val.jsonl
│   ├── speaker_train_pairs.jsonl
│   ├── speaker_val_pairs.jsonl
│   ├── tse_train.jsonl
│   ├── tse_val.jsonl
│   └── internal_test.jsonl
├── synthetic/
│   ├── mixtures/
│   ├── enhanced/
│   └── metadata/
└── test_only/
    ├── datasetA/
    └── datasetB/
```

训练脚本必须禁止读取 `data/test_only`。

## 4. Manifest 统一字段

### ASR manifest

```json
{
  "utt_id": "aishell1_S0001_BAC009S0001W0123",
  "speaker_id": "S0001",
  "audio_path": "data/raw_public/aishell1/...",
  "text": "打开空调",
  "sample_rate": 16000,
  "duration": 2.35,
  "split": "train",
  "source": "AISHELL-1",
  "license": "Apache-2.0"
}
```

### Speaker pair manifest

```json
{
  "pair_id": "pair_000001",
  "enroll_audio": "data/raw_public/hi_mia/...",
  "test_audio": "data/raw_public/cnceleb/...",
  "same_speaker": true,
  "speaker_id_enroll": "spk001",
  "speaker_id_test": "spk001",
  "condition": "far_field_or_noisy",
  "split": "train"
}
```

### TSE/BSS synthetic manifest

```json
{
  "mix_id": "mix_000001",
  "enrollment": "data/raw_public/aishell3/spk001_ref.wav",
  "target_clean": "data/raw_public/aishell2/spk001_cmd.wav",
  "interferer": "data/raw_public/aishell2/spk209_cmd.wav",
  "noise": "data/raw_public/dns/noise_x.wav",
  "rir": "data/raw_public/dns/rir_x.wav",
  "mixture": "data/synthetic/mixtures/mix_000001.wav",
  "target": "data/synthetic/targets/mix_000001_target.wav",
  "snr_db": 0,
  "sir_db": -3,
  "overlap_ratio": 0.75,
  "target_speaker_id": "spk001",
  "interferer_speaker_id": "spk209",
  "split": "train",
  "seed": 20260802
}
```

## 5. 训练数据生成策略

### ASR 数据

第一阶段不从零训练 ASR，先用预训练 FunASR/Paraformer/Whisper 类模型得到 V0。

如需微调，优先顺序：

1. AISHELL-2：优先抽取智能家居、控制命令、唤醒词、短指令类文本；
2. AISHELL-1：作为稳定干净中文基础；
3. MAGICDATA：补充家居控制、问答、移动设备录音；
4. WenetSpeech：只取高置信标注数据，用于增强多场景泛化。

禁止使用 Dataset A/B 文本构造命令词典或纠错规则。

### 声纹验证数据

第一阶段使用预训练 WeSpeaker / ECAPA，不训练主干。

训练/校准数据构造：

- 同人 pair：同一说话人的不同音频；
- 异人 pair：不同说话人；
- hard negative：同性别、相似音高、相似口音、同一句唤醒词；
- 远近场 pair：HI-MIA 的近讲 enrollment 与远场 test；
- 噪声增强 pair：CN-Celeb/AISHELL 叠加 DNS/MUSAN 噪声。

第一版只训练轻量校准器：

```text
[global_similarity, topk_similarity, target_frame_ratio, audio_quality]
→ Logistic Regression / 小 MLP
→ P(target speaker present)
```

### TSE/BSS 数据

训练样本必须满足：

- enrollment 和 target_clean 同一人但不同录音；
- interferer 必须不同人；
- train/val/internal_test 说话人隔离；
- SNR 覆盖 -5、0、5 dB；
- SIR 覆盖 -5 到 5 dB；
- overlap 覆盖 0%、25%、50%、75%、100%；
- 加入空调、电视、风扇、厨房、机械背景等家庭噪声；
- 加入房间混响。

先做 20 条人工可听 demo，再批量生成 1k、10k、50k 三档。

## 6. 版本推进

### 第 1 阶段：V0/V1

目标：先拿到可提交版本。

1. 下载 AISHELL-1；
2. 跑预训练 ASR；
3. 下载 HI-MIA + CN-Celeb；
4. 生成声纹 pair；
5. 跑 WeSpeaker 相似度；
6. 得到 V1 Safe Baseline。

### 第 2 阶段：V2

目标：困难样本增强。

1. 下载 DNS noise/RIR；
2. 用 AISHELL-2/AISHELL-3 合成中文重叠语音；
3. 训练或接入 TSE；
4. 做 raw/enhanced 双路 ASR；
5. 只在困难样本触发。

### 第 3 阶段：V3

目标：创新消融。

1. 接入 SepReformer/BSS 作为可选 separator backend；
2. 分离后用 E_owner 选择主人声源；
3. 增强后声纹下降则回退；
4. 对 STFT、目标主导时频点、软掩码、残差抑制逐项消融。

## 7. 验收指标

每次实验必须记录：

- CER；
- RR；
- 正样本误拒率；
- 负样本误接受率；
- TSE/BSS 触发率；
- raw ASR vs enhanced ASR；
- 平均/P95 latency；
- 峰值显存；
- 回退率；
- 按 SNR、SIR、overlap、噪声类型分层指标。

合入主线条件：

- CER 降低；
- RR 不下降；
- 正样本误拒不增加；
- 推理时间和显存可接受；
- 能一键复现；
- 在独立验证集有效。
