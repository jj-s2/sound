# R12 Overall 0.85 方案设计

## 目标

在 8GB 显存和严格 group-disjoint 数据边界下，构建可复现的 R12 语音命令识别链路，目标是在未参与模型选择的新盲测上达到 Overall 0.85；当前已经查看过的 Dataset-A internal test 只作为回归审计，不再用于调参。

当前已核实基线为 Overall=0.7899209、CER=0.3784916、RR=0.9583333。保持当前 RR 时，达到 0.85 需要最多 369 个字符错误；若达到 71/72 的 RR，则最多 409 个字符错误。因此方案必须同时降低门控产生的误拒绝删除和短命令 ASR 错误。

## 范围

本次改造包含四个有明确接口的子系统：

1. ASR 候选生成：no-VAD、padding、LoRA 和 train-only context hotword。
2. Personal VAD：使用 enrollment 条件输出目标人物/非目标人物/非语音帧概率。
3. 候选路由：利用候选一致性、声纹和 ASR 置信度预测候选 CER。
4. 校准门控：在 validation 上选择接受/拒绝阈值，内部测试不参与选择。

选择性 TSE 作为后续候选，不在第一批改动中与 ASR 同时训练，避免 8GB 显存下的不可控耦合。

## 总体架构

唤醒音频经冻结 speaker encoder 得到 enrollment embedding。命令音频同时进入 Personal VAD 和多个 ASR 候选分支。Personal VAD 产生目标说话人帧级统计量；ASR 候选产生文本、可用的置信度和候选一致性特征。候选 router 先选择预期 CER 最低的文本，calibrated gate 再根据目标人物概率决定输出文本或拒绝。疑似重叠语音才触发 TSE 分支。

## 数据边界

- Dataset-A 原始 train groups 负责 ASR、Personal VAD 校准和 router 训练。
- validation groups 只用于 checkpoint、候选、模型和阈值选择。
- internal-test groups 不进入训练、增强、hotword、模型选择或阈值选择。
- 增强样本继承 parent group，不能跨 group 混合出训练/验证泄漏。
- pVAD 合成帧标签来自 AISHELL-1 speaker-id 配对和本地 RIR/噪声；Dataset-A 只提供 train 级校准。

## 子系统设计

### ASR 候选

候选顺序固定为 `primary`、`no_vad`、`pad160_no_vad`、`lora`、`lora_hotword`，后续再加入 `selective_tse`。`run_funasr_asr.py` 显式支持 `--vad-model off`、`--punc-model off` 和前后静音 padding，避免把关闭模型误解释为模型名称。

### Paraformer LoRA

复现 FunASR 官方 Paraformer LoRA 参数：encoder 和 decoder 的 q/k/v/o，rank=8、alpha=16、dropout=0.05、lr=1e-4、max_epoch=30、保留最佳10个并平均最佳5个。8GB 以 800 token micro-batch、8 步梯度累积、FP16 和 2 个 worker 起步；OOM 时只降低 micro-batch。训练使用 Dataset-A train 正样本与 1:1 AISHELL replay，不进行全参数微调。

### Personal VAD

默认模型为 80 维 log-Mel、enrollment embedding 和帧级 cosine 拼接后进入两层 GRU（hidden=128）和三分类头。输出帧概率及 target ratio、最长连续区间、target/interferer ratio、overlap probability 等固定特征。起始损失为三分类加权 CE + 0.5 target-vs-rest BCE + 0.05 temporal smooth；权重只在 train 内部 OOF 选择。

### Router 与 Gate

router 预测每个候选的归一化 CER，保留当前 HistGradientBoosting 实现并增加候选一致性、长度/时长、ASR confidence、Personal VAD 和 overlap 特征。gate 只预测 target-present 概率。两者分离，避免为了提高 RR 直接牺牲 ASR 文本。

## 增强分布

训练增强包含速度 0.90/0.95/1.00/1.05/1.10、增益 -6 到 +6dB、RIR、真实噪声 SNR 0/5/10/20dB、非目标人重叠 SIR -5/0/5/10dB、编解码/带宽扰动和不超过200ms的端点 padding。validation 和 internal test 不生成增强副本。

## 验收

每个子系统先在 train 内做 grouped OOF，再在固定 validation 选择。推荐联合晋级条件为 validation Overall>=0.85、RR>=0.98、CER<=0.28、FRR<=0.10；同时要求不超过5字命令和最短时长四分位的 CER 都有下降。正式 0.85 结论只能来自全新盲测或赛事方未公开测试集。

## 资源与限制

训练设备为 8GB GPU；ASR、pVAD、TSE 分阶段运行并主动释放显存。Personal VAD 和 router 在 CPU 即可训练。模型权重、Dataset-A、标签、缓存和输出不提交 GitHub。当前 FunASR/CUDA 安装版本和梯度累积参数名在实现前通过 smoke test 锁定。

## 可追溯文献

- Personal VAD: https://arxiv.org/abs/1908.04284
- SeACo-Paraformer: https://arxiv.org/abs/2308.03266
- VoiceFilter-Lite: https://arxiv.org/abs/2009.04323
- TSE + ASR uncertainty: https://arxiv.org/abs/2011.13393
- FunASR official LoRA config: https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/conf/paraformer_lora.yaml

