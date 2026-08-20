# XH-202615 公开数据下载与整理指南

更新时间：2026-08-02

## 当前本地状态

已经创建目录：

```text
data/
├── raw_public/
│   ├── aishell1/
│   ├── cnceleb/
│   ├── dns/
│   ├── hi_mia/
│   └── mobvoi_hotwords/
├── manifests/
├── synthetic/
│   ├── mixtures/
│   └── metadata/
└── test_only/
```

已经尝试下载并成功放置：

```text
data/raw_public/aishell1/resource_aishell.tgz
```

这个只是 AISHELL-1 的小型资源包，不是主语音数据。

## 不建议由 Codex 直接下载的大包

以下数据体量大，部分还需要注册、同意协议或人工申请，不建议直接让 Codex 在后台下载：

- AISHELL-1 主数据：约十几 GB；
- AISHELL-2：约 1000 小时，需按官方要求获取；
- CN-Celeb：大规模声纹数据；
- WenetSpeech：万小时级；
- DNS Challenge：噪声/混响资源较大；
- AISHELL-DMASH：约 20,000 小时，需申请，体量极大。

## 第一批建议下载

### 1. AISHELL-1

用途：快速建立中文 ASR baseline 和干净中文混音源。

官方网站：

```text
https://www.openslr.org/33/
```

建议下载：

```text
data_aishell.tgz
resource_aishell.tgz
```

放置位置：

```text
data/raw_public/aishell1/
```

下载后目录应类似：

```text
data/raw_public/aishell1/
├── data_aishell.tgz
└── resource_aishell.tgz
```

### 2. HI-MIA

用途：远场唤醒音频、文本相关声纹验证，与赛题 wakeup_audio 最贴近。

官方网站：

```text
https://www.openslr.org/85/
```

放置位置：

```text
data/raw_public/hi_mia/
```

### 3. CN-Celeb

用途：中文声纹验证、同人/异人 pair、hard negative。

官方网站：

```text
https://www.openslr.org/82/
```

放置位置：

```text
data/raw_public/cnceleb/
```

### 4. DNS Challenge

用途：家庭噪声、混响、语音增强合成。

官方网站：

```text
https://github.com/microsoft/DNS-Challenge
```

放置位置：

```text
data/raw_public/dns/
```

### 5. MobvoiHotwords

用途：智能音箱、唤醒词、非关键词、家庭噪声负样本。

官方网站：

```text
https://www.openslr.org/87/
```

放置位置：

```text
data/raw_public/mobvoi_hotwords/
```

## 第二批建议下载

等 V0/V1 跑通后再考虑：

- AISHELL-2：中文 ASR 与智能家居语音控制更贴近；
- AISHELL-3：多说话人干净语音，适合 TSE 合成；
- AISHELL-4：多人重叠与说话人活动；
- MAGICDATA Mandarin：家居控制语料补充；
- WenetSpeech：中文 ASR 泛化；
- AISHELL-DMASH：真实智能家居远场数据，但体量和申请成本很高。

## 下载后下一步

下载数据后不要直接训练，先做三件事：

1. 解压到对应 `data/raw_public/<dataset>/`。
2. 生成 manifest：

```text
data/manifests/asr_train.jsonl
data/manifests/asr_val.jsonl
data/manifests/speaker_train_pairs.jsonl
data/manifests/speaker_val_pairs.jsonl
data/manifests/tse_train.jsonl
data/manifests/tse_val.jsonl
```

3. 确认训练脚本不读取：

```text
datasetA/
data/test_only/
```

## 手动下载建议

优先用浏览器或下载器手动下载大文件，原因：

- 大包容易中断；
- 有些数据需要注册或填写申请；
- 浏览器/网盘下载更稳定；
- 需要你自己确认许可证和使用范围。

下载完成后，把文件直接放到本指南指定目录即可。

