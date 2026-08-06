# V4 Robustness Workflow

This workflow is the deployable V4 path. It does not use labels to decide
which samples need enhancement or rejection.

Before any run, read `XH-202615_比赛注意文档.md` and keep these constraints:

- Dataset-A is diagnostic only; do not tune rejection templates from labels.
- Final objective is CER 40%, RR 40%, efficiency 20%.
- Positive false rejection counts as deletion errors in CER.
- Submission JSON must follow the official structure; do not add extra fields.
- `duration` must come from a real batch=1 inference run, not a lookup table.

Current stable-line target:

- CER below 38%.
- RR above 65%.
- false reject rate below 10%.
- enhancement candidate ratio between 10% and 35%.

## 1. Select Enhancement Candidates

Default profile:

```powershell
python scripts\select_enhancement_candidates.py `
  --dataset-root datasetA\datasetA `
  --splits pos,neg `
  --asr-map output\asr\funasr_full_hotword_safe.jsonl `
  --speaker-scores output\speaker\wespeaker_scores_full.csv `
  --output output\reports\v4_auto_candidates_v2_full.csv `
  --ids-output output\reports\v4_auto_candidates_v2_full.ids.txt `
  --min-selected-ratio 0.10 `
  --max-selected-ratio 0.35 `
  --selected-only
```

Conservative profile:

```powershell
python scripts\select_enhancement_candidates.py `
  --dataset-root datasetA\datasetA `
  --splits pos,neg `
  --asr-map output\asr\funasr_full_hotword_safe.jsonl `
  --speaker-scores output\speaker\wespeaker_scores_full.csv `
  --output output\reports\v4_auto_candidates_conservative_full.csv `
  --ids-output output\reports\v4_auto_candidates_conservative_full.ids.txt `
  --min-text-length 10 `
  --incomplete-text-length 14 `
  --very-long-text-length 20 `
  --min-selected-ratio 0.10 `
  --max-selected-ratio 0.35 `
  --selected-only
```

## 2. Enhance Selected Audio

```powershell
python scripts\enhance_target_speaker_audio.py `
  --dataset-root datasetA\datasetA `
  --splits pos,neg `
  --ids-file output\reports\v4_auto_candidates_v2_full.ids.txt `
  --output-root output\enhanced\v4_auto_energy `
  --manifest output\enhanced\v4_auto_energy_manifest.csv `
  --method energy `
  --resume
```

## 3. Run ASR On Enhanced Audio

```powershell
python scripts\run_funasr_asr.py `
  --dataset-root datasetA\datasetA `
  --splits pos,neg `
  --ids-file output\reports\v4_auto_candidates_v2_full.ids.txt `
  --command-audio-map output\enhanced\v4_auto_energy_manifest.csv `
  --output output\asr\v4_auto_energy_asr.jsonl `
  --device cuda:0 `
  --hotword-preset assistant `
  --resume
```

## 4. Merge Primary And Enhanced ASR

```powershell
python scripts\merge_asr_fallback.py `
  --primary output\asr\funasr_full_hotword_safe.jsonl `
  --fallback output\asr\v4_auto_energy_asr.jsonl `
  --output output\asr\v4_auto_energy_robust_merged.jsonl `
  --use-robustness-trigger `
  --min-primary-length 8 `
  --max-primary-domain-score 0 `
  --min-length-reduction-ratio 0.00 `
  --require-fallback-nonempty `
  --prefer-higher-domain-score
```

## 5. Run Pipeline And Evaluate

```powershell
python -m xh202615.run_inference `
  --dataset-root datasetA\datasetA `
  --splits pos,neg `
  --config configs\v4_balanced_058.json `
  --asr-map output\asr\v4_auto_energy_robust_merged.jsonl `
  --speaker-scores output\speaker\wespeaker_scores_full.csv `
  --output output\predictions\v4_auto_energy_robust_full.jsonl

python -m xh202615.evaluate_predictions `
  --dataset-root datasetA\datasetA `
  --splits pos,neg `
  --predictions output\predictions\v4_auto_energy_robust_full.jsonl `
  --output output\metrics\v4_auto_energy_robust_full_metrics.json
```

## 6. Sweep 0.58-0.64 Thresholds

Use this as a diagnostic scan. Labels are used only after inference to measure
CER/RR tradeoffs; they are not used by the routing decision.

```powershell
python scripts\scan_router_thresholds.py `
  --dataset-root datasetA\datasetA `
  --splits pos,neg `
  --config configs\v4_balanced_058.json `
  --asr-map output\asr\v4_auto_energy_robust_merged.jsonl `
  --speaker-scores output\speaker\wespeaker_scores_full.csv `
  --thresholds 0.58,0.59,0.60,0.61,0.62,0.63,0.64 `
  --max-false-reject 0.10 `
  --output output\reports\router_threshold_scan.csv `
  --pred-dir output\predictions\threshold_scan
```

Pick the best eligible row first by `false_reject_rate < 0.10`, then by
`official_80_score`.

## 7. Multi-Candidate Fusion

Candidate sources can be FunASR hotword, energy-enhanced ASR, SenseVoice, and
later BSS/TSE ASR. This script does not read labels.

```powershell
python scripts\select_best_asr_candidate.py `
  --primary output\asr\funasr_full_hotword_safe.jsonl `
  --candidate energy=output\asr\v4_auto_energy_asr.jsonl `
  --candidate sensevoice=output\asr\sensevoice_full.jsonl `
  --speaker-scores output\speaker\wespeaker_scores_full.csv `
  --output output\asr\v4_candidate_fusion.jsonl
```

Then run the normal pipeline/evaluation with `--asr-map output\asr\v4_candidate_fusion.jsonl`.

## 8. BSS/TSE Ablation Branch

BSS is a hard-sample enhancement branch, not a full-dataset preprocessor.
Start with 20-100 difficult sample ids.

```powershell
python scripts\prepare_bss_ablation.py `
  --dataset-root datasetA\datasetA `
  --splits pos `
  --ids-file output\reports\v4_hard_100.ids.txt `
  --raw output\asr\funasr_full_hotword_safe.jsonl `
  --energy output\asr\v4_auto_energy_asr.jsonl `
  --bss output\asr\bss_hard_asr.jsonl `
  --fusion output\asr\v4_candidate_fusion.jsonl `
  --output output\reports\bss_ablation.csv
```

BSS enters the main path only if it lowers hard-sample CER, does not reduce RR,
does not increase owner false rejection, and has acceptable runtime with raw
fallback.

## 9. Unified Experiment Table

```powershell
python scripts\summarize_experiments.py `
  --entry "ASR-Only=output\metrics\stage_asr_only_full_metrics.json|output\predictions\stage_asr_only_full.jsonl" `
  --entry "SV+ASR=output\metrics\stage_sv_asr_full_metrics.json|output\predictions\stage_sv_asr_full.jsonl" `
  --entry "Stable=output\metrics\v4_auto_energy_robust_full_metrics.json|output\predictions\v4_auto_energy_robust_full.jsonl" `
  --output output\reports\experiment_summary.csv
```

The table includes CER, RR, false reject, false accept, mean latency, P95,
candidate ratio, fallback rate, and notes.

## 10. Competition JSON

Formal submission requires explicit real duration:

```powershell
python scripts\make_submission.py `
  --predictions output\predictions\v4_auto_energy_robust_full.jsonl `
  --dataset-root datasetA\datasetA `
  --splits pos,neg `
  --format competition_json `
  --id-source command_audio_name `
  --duration 338.0 `
  --output output\submissions\v4_stable_submission.json
```

Use `--allow-latency-duration` only for local diagnostics.
