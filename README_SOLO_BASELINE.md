# XH-202615 一个人推进 Baseline 工程说明

## 当前已经落地的内容

这个工程先把“可运行闭环”搭起来，不绑定任何大模型：

- `xh202615/data.py`：读取 Dataset-A 风格 `pos.jsonl` / `neg.jsonl`。
- `xh202615/metrics.py`：中文 CER、RR、误拒率、误接受率。
- `xh202615/run_inference.py`：统一推理入口。
- `xh202615/evaluate_predictions.py`：统一评测入口。
- `configs/`：V0/V1/V2/V3 四套版本配置。
- `experiments/experiment_log.csv`：实验记录表。

默认 ASR 是 `noop`，只用于验证流程。真正冲榜时，需要接入 FunASR 或导入外部 ASR 预测。

## 快速 smoke test

```powershell
python -m unittest tests.test_metrics
python -m xh202615.run_inference --dataset-root datasetA --splits pos,neg --config configs/v0_asr_only.json --output output/predictions/v0_noop.jsonl --limit 20
python -m xh202615.evaluate_predictions --dataset-root datasetA --splits pos,neg --predictions output/predictions/v0_noop.jsonl --output output/metrics/v0_noop_metrics.json --missing-policy skip
```

## 真实 V0：FunASR 预训练模型接入

先安装依赖：

```powershell
python -m pip install -U pip
pip install -U funasr modelscope huggingface_hub
```

GPU 小批量生成 ASR map：

```powershell
python scripts/run_funasr_asr.py --dataset-root datasetA/datasetA --splits pos --output output/asr/funasr_pos_100.jsonl --limit 100 --device cuda:0
```

接入 V0 pipeline 并评测：

```powershell
python -m xh202615.run_inference --dataset-root datasetA/datasetA --splits pos --config configs/v0_asr_only.json --asr-map output/asr/funasr_pos_100.jsonl --output output/predictions/v0_funasr_pos_100.jsonl --limit 100
python -m xh202615.evaluate_predictions --dataset-root datasetA/datasetA --splits pos --predictions output/predictions/v0_funasr_pos_100.jsonl --output output/metrics/v0_funasr_pos_100_metrics.json --missing-policy skip
```

生成错误分析报告和精简提交文件：

```powershell
python scripts/analyze_predictions.py --dataset-root datasetA/datasetA --splits pos --predictions output/predictions/v0_funasr_pos_100.jsonl --output output/reports/v0_funasr_pos_100_errors.csv --missing-policy skip
python scripts/make_submission.py --predictions output/predictions/v0_funasr_pos_100.jsonl --output output/submissions/v0_funasr_pos_100_submission.jsonl
```

## 接入真实 ASR 的最小方式

先用 FunASR 单独生成一个 JSONL：

```json
{"id": "0", "text": "空调打开"}
{"id": "1", "text": "灯光亮度调到百分之三十"}
```

然后运行：

```powershell
python -m xh202615.run_inference --dataset-root datasetA --splits pos,neg --config configs/v0_asr_only.json --asr-map output/asr/funasr_predictions.jsonl --output output/predictions/v0_funasr.jsonl
python -m xh202615.evaluate_predictions --dataset-root datasetA --splits pos,neg --predictions output/predictions/v0_funasr.jsonl --output output/metrics/v0_funasr_metrics.json
```

## 接入声纹分数的最小方式

先生成 CSV：

```csv
id,target_probability,global_similarity,topk_similarity,target_frame_ratio,noise_score,overlap_probability,audio_quality
0,0.92,0.71,0.82,0.76,0.12,0.08,1.0
1000,0.04,0.10,0.18,0.00,0.18,0.05,1.0
```

然后运行 V1/V2：

```powershell
python -m xh202615.run_inference --dataset-root datasetA --splits pos,neg --config configs/v1_safe_baseline.json --asr-map output/asr/funasr_predictions.jsonl --speaker-scores output/speaker/wespeaker_scores.csv --output output/predictions/v1_safe.jsonl
python -m xh202615.evaluate_predictions --dataset-root datasetA --splits pos,neg --predictions output/predictions/v1_safe.jsonl --output output/metrics/v1_safe_metrics.json
```

### 使用 WeSpeaker 生成 V1 声纹分数

安装 WeSpeaker：

```powershell
pip install git+https://github.com/wenet-e2e/wespeaker.git
```

先小批量生成声纹分数：

```powershell
python scripts/run_funasr_asr.py --dataset-root datasetA/datasetA --splits pos,neg --output output/asr/funasr_balanced_200.jsonl --per-split-limit 100 --device cuda:0
python scripts/run_wespeaker_scores.py --dataset-root datasetA/datasetA --splits pos,neg --output output/speaker/wespeaker_scores_200.csv --per-split-limit 100 --gpu 0
```

用已有 FunASR ASR map 跑 V1：

```powershell
python -m xh202615.run_inference --dataset-root datasetA/datasetA --splits pos,neg --config configs/v1_safe_baseline.json --asr-map output/asr/funasr_balanced_200.jsonl --speaker-scores output/speaker/wespeaker_scores_200.csv --output output/predictions/v1_safe_200.jsonl --limit 200
python -m xh202615.evaluate_predictions --dataset-root datasetA/datasetA --splits pos,neg --predictions output/predictions/v1_safe_200.jsonl --output output/metrics/v1_safe_200_metrics.json --missing-policy skip
python scripts/analyze_predictions.py --dataset-root datasetA/datasetA --splits pos,neg --predictions output/predictions/v1_safe_200.jsonl --output output/reports/v1_safe_200_errors.csv --missing-policy skip
```

## 当前路线

1. V0：ASR-only，先拿到真实 CER 和耗时。
2. V1：声纹验证 + 保守拒识 + 原始 ASR，作为保底提交。
3. V2：动态路由 + 困难样本 TSE + 双路 ASR。
4. V3：BSS/SepReformer 先验，只在消融有效后进入主提交。

合入标准：CER 降、RR 不降、误拒不升、效率可接受、可一键复现。

## 数据训练方案

数据源优先级和训练构造方案见：

- `docs/DATA_TRAINING_SCHEME.md`
- `configs/data_sources.json`

快速查看：

```powershell
python scripts/check_data_sources.py
```

下载与整理指南：

```powershell
python scripts/list_local_data.py
```

详细说明见 `docs/DATA_DOWNLOAD_GUIDE.md`。
# V4 Current Stable Plan

Use `docs/V4_ROBUSTNESS_WORKFLOW.md` as the current execution guide.

Core rules:

- Read `XH-202615_比赛注意文档.md` before code, training, submission, PPT, or report work.
- Main submission is the stable Dynamic Enhancement line, not the equal-weight high-RR risk line.
- Do not use Dataset-A labels to build correction templates or rejection rules.
- Keep broad smart-assistant hotwords and intent protection enabled for home, media, life, QA, and reminder/tool commands.
- BSS/TSE is a hard-sample ablation branch first; it enters the main path only after CER/RR/false-reject/runtime checks pass.
- `competition_json` requires explicit real `--duration`.

Useful local checks:

```powershell
python -m unittest discover -s tests
python scripts\scan_router_thresholds.py --dataset-root datasetA\datasetA --splits pos,neg --config configs\v4_balanced_058.json --asr-map output\asr\v4_auto_energy_robust_merged.jsonl --speaker-scores output\speaker\wespeaker_scores_full.csv --output output\reports\router_threshold_scan.csv
python scripts\summarize_experiments.py --entry Stable=output\metrics\v4_auto_energy_robust_full_metrics.json^|output\predictions\v4_auto_energy_robust_full.jsonl --output output\reports\experiment_summary.csv
```
