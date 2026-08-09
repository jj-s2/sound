# R11 Overall router: literature decision and experiment route

Date: 2026-08-09

## Decision

Do not continue the R10 five-action classifier. Build a factorized R11 cascade:

1. **Enrollment-conditioned target-presence gate**: use the wake audio as the
   target-speaker enrollment. Start with a zero-shot FireRedChat-pVAD benchmark
   and cached-feature fusion; train a compact custom three-class Personal VAD
   only if temporal enrollment-conditioned evidence is useful but transfer is
   insufficient.
2. **Positive-only candidate risk ranker**: one utterance is one group and each
   ASR source is one item. Compare expected-CER regression with shallow
   LightGBM LambdaMART. Reject is not an item in this ranker.
3. **Calibrated official-cost controller**: select the gate threshold and switch
   margin using grouped inner folds. Directly maximize
   `Overall=((1-CER)+RR)/2` subject to pooled `RR >= 0.93`, worst-fold
   `RR >= 0.90`, and robustness gates. Use R3 fallback only for uncertain
   candidate selection, not as the universal action for uncertain presence.

This route is not a guarantee of Overall above 0.8. The current all-candidate
oracle Overall of 0.843934 leaves only about 0.0439 absolute routing-regret
budget, so the first job is to falsify feasibility before additional training.

## Why the current route cannot work

- R3 is `CER=0.508410`, `RR=0.938819`, `Overall=0.715204`.
- Perfect current-candidate selection is `CER=0.312132`, `RR=1`,
  `Overall=0.843934`, so candidate complementarity is real.
- Keeping the R3 rejection gate and making candidate selection perfect still
  reaches only about `0.72372`. R3 falsely rejects about 32.2% of positives.
- The best current scalar-cosine gate plus perfect candidate selection reaches
  only about `0.717`.
- R10 repeats one optimal action label over all five action rows and then
  averages predicted class probabilities over those rows. This is neither
  candidate-risk estimation nor listwise ranking.

Therefore more epochs on R10 cannot fix the objective mismatch.

## Literature-to-module mapping

| Evidence | Relevant result | R11 use | Limitation |
|---|---|---|---|
| [Personal VAD (Ding et al., 2020)](https://www.isca-archive.org/odyssey_2020/ding20_odyssey.html) | Speaker-conditioned frame model with non-speech, target-speech and non-target-speech outputs; the paper reports a 130K-parameter model outperforming a VAD+SV cascade. | Defines the gate target and compact architecture family. | Paper results do not establish XH transfer. |
| [Personal VAD 2.0 (Ding et al., 2022)](https://www.isca-archive.org/interspeech_2022/ding22_interspeech.html) | Enrollment modulation plus streaming/runtime optimization. | Wake audio becomes enrollment; preserve a streaming-friendly small gate for final efficiency scoring. | Production data and exact model are not public drop-in assets. |
| [Robust pVAD under domain mismatch (Lin et al., 2025)](https://www.isca-archive.org/interspeech_2025/lin25_interspeech.html) | Embedding update and on-the-fly hard-sample simulation reduce false acceptance under mismatch. | Use impostor speakers, inactive-target cases and difficult target/non-target mixtures in custom training. | Apply only after a zero-shot/fused gate demonstrates temporal signal. |
| [Learning to rank ASR hypotheses (Wu et al., 2022)](https://www.isca-archive.org/interspeech_2022/wu22_interspeech.html) | List context and acoustic/text confidence improve ASR rescoring; LambdaMART is a suitable lightweight ranker. | One utterance/list, one candidate/item; learn CER preference/regret. | Published WERR values are task-specific and are not an XH forecast. |
| [ASR confidence with deletion prediction (Qiu et al., 2021)](https://www.isca-archive.org/interspeech_2021/qiu21b_interspeech.html) | Word/utterance confidence alone misses deletion behavior; joint deletion modeling improves confidence. | Export Paraformer token/sequence scores plus CIF/emitted-length residual and VAD coverage. | R11 first uses compact features, not this paper's full neural confidence head. |
| [Quality-estimated ROVER (Jalalvand et al., 2017)](https://arxiv.org/abs/1706.07238) | Segment-level QE can rank hypotheses before ROVER without decoder confidence. | Later baseline/additional candidate when ranking works but selection regret remains high. | Fusion can add latency and is not the primary gate/ranker. |
| [Learning to defer to multiple experts (Verma et al., 2023)](https://proceedings.mlr.press/v206/verma23a.html) | Multiple-expert correctness estimates require calibration-aware losses. | Supports separating expert quality estimates and calibrated controller decisions. | R11 has no true external fallback; use the principle, not the terminology as a claim. |
| [FireRedChat-pVAD model card](https://huggingface.co/FireRedTeam/FireRedChat-pvad) | Open-weight Chinese/English ONNX pVAD, Apache-2.0, with speaker embedding updates and ECAPA-VoxCeleb dependency. | First no-training neural gate benchmark and frame-score feature source. | No published Dataset-A metrics, latency, memory, RR or transfer guarantee. |

## R11 data flow

```text
wake audio -> frozen speaker embedding ----------------------+
                                                            v
command audio -> cached features + frame pVAD -> p(target present)

R3 / primary / energy / TSE hypotheses
  -> source identity + pairwise text disagreement + duration/coverage
  -> Paraformer score total/mean + token entropy/margin + length residual
  -> positive-only expected-CER/LambdaMART ranker

p(target present) + predicted candidate risk
  -> calibrated policy maximizing official Overall under RR constraints
  -> reject OR selected hypothesis
```

## Required confidence adapter

Preserve exact decoded-text parity while exporting only compact values:

- `score_kind`, hypothesis score total, decoded non-special token count and
  length-normalized score;
- selected-token log-posterior mean/minimum/lower quantile/std;
- token-distribution entropy and top-1/top-2 margin summaries;
- CIF predicted length, emitted length and their residual;
- FSMN-VAD speech duration, audio coverage, interval count and timestamps;
- decoder/source/configuration identity and explicit missingness.

Do not invent a Paraformer `no_speech_probability`, compare uncalibrated raw
scores across sources, or persist full time-by-vocabulary tensors.

## Ordered experiments and hard stops

### E0 — cached gate-oracle falsification

Cross-fit a small predeclared cached-feature gate grid. For accepted positives,
assign the true best current candidate only for this diagnostic upper bound.

- Continue cached cascade only if gate-oracle `Overall >= 0.81`, `RR >= 0.93`
  and worst-fold `Overall >= 0.77`.
- If the upper 95% grouped-bootstrap bound of best gate-oracle Overall is below
  0.80, stop cached-gate tuning and go directly to E2.

### E1 — complete cached negative control

If E0 passes, compare positive-only expected-CER regression and LambdaMART.
Promote only if pooled `Overall >= 0.80`, pooled `RR >= 0.93`, worst-fold
`Overall >= 0.77`, worst-fold `RR >= 0.90`, every inner search is feasible and
the paired group-bootstrap lower 95% bound versus R3 is positive.

### E2/E3 — FireRed zero-shot and fused gate-oracle

Run 16-kHz wake enrollment plus command-frame inference, cache aggregates, and
measure CPU latency, memory and determinism. Never transfer the default 0.5
threshold. Test FireRed-only and FireRed+cached gate-oracle frontiers before
training a new ranker; require the same 0.81 preferred headroom.

### E4/E5 — confidence adapter and candidate ranking

First establish a black-box ranker baseline; then add the compact FunASR values
and measure incremental outer-fold CER-regret reduction. Use utterance groups,
positive rows only, and fold-local source calibration.

### E6 — composed R11

Tune only gate threshold and switch margin in inner OOF. Promotion requires all
quality gates above plus evaluator parity, complete coverage, manifest/digest
checks, no group leakage, and measured end-to-end latency/memory.

### E7 — custom compact Personal VAD, conditional

Train it only if FireRed/fused features show independent temporal
enrollment-conditioned signal or narrowly miss due to calibration/domain
mismatch. Use a frozen ECAPA/WeSpeaker encoder, a small causal GRU/Conformer or
depthwise-convolution backbone, three frame classes and hard-negative mixtures.

## RR decision

The exact feasibility identity is `RR >= 0.6 + CER` for `Overall >= 0.8`.
At the current oracle CER 0.312132, the mathematical boundary is RR 0.912132.
The chosen fixed floor 0.93 leaves about 1.8 points of safety above that boundary
and remains close to R3's 0.938819, while avoiding R10's unjustified 0.95 hard
constraint. Overall still forces a higher realized RR when CER is worse: at CER
0.34, RR must be at least 0.94.

The 0.93 floor must be declared before outer-fold evaluation and must not be
lowered after seeing results. Also publish the whole CER/RR Pareto frontier.

## Alternatives and flip conditions

- **QE-ranked ROVER**: add only if the gate is viable and ranker selection
  regret remains high. Prefer it over a large text-only quality model initially.
- **NoRefER/BERT confidence**: defer; data and latency risk are higher than
  compact decoder features plus shallow ranking.
- **Joint USEF-TP/audio-text MoE**: strongest long-term alternative if current
  candidates/gates have no feasible oracle frontier, but too costly as the first
  diagnostic.
- **Custom pVAD**: do not start merely because FireRed zero-shot fails; start
  only when temporal target-speaker features show conditional value.
- **Early conditional candidate execution**: switch to it if running all ASR/TSE
  candidates erases the quality gain under final latency/memory scoring.

The entire routing branch is falsified with the current candidates if no strictly
outer-OOF gate family has a gate-oracle upper bound above 0.8. In that case,
improve the target-presence representation or candidate generators rather than
tuning the selector again.

## Arena resolution

Codex and Claude agree on the final factorization and experiment order after two
rounds. Claude revised its initial RR>=0.95 position to RR>=0.93 and revised the
cached gate from presumptive first architecture to mandatory cheap negative
control. Remaining caution: Claude would not automatically train a custom pVAD
after FireRed failure; independent evidence of useful enrollment-conditioned
temporal signal is required. Codex accepts that safeguard.
