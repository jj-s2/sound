# Phase 2 AISHELL synthetic overlap builder

`scripts/prepare_phase2_synthetic.py` creates a small, deterministic public/synthetic manifest for
temporal-speaker and overlap experiments. It reads an extracted AISHELL-1 tree, assigns speakers to
disjoint train/validation/test partitions, renders 16 kHz mono enrollment, target, and mixture WAVs,
and validates every row with the existing `xh202615.training_data` contract.

The builder does not read or derive records from Dataset-A. Dataset-A and every other evaluation-only
location must be supplied as a forbidden root. The builder rejects an AISHELL input root or output
root located at or below any forbidden root before discovering audio or writing files.

## Expected AISHELL files

The normal extracted layout is supported directly:

```text
<AISHELL_ROOT>/
  transcript/aishell_transcript_v0.8.txt
  wav/train/S0001/*.wav
  wav/dev/S0002/*.wav
  wav/test/S0003/*.wav
```

The audio search is recursive, so alternate extraction wrappers are also accepted when each audio
file has an ancestor speaker directory whose name begins with `S`. The transcript file contains one
utterance ID and transcript per line, separated by a tab (whitespace separation is also accepted).
Audio without a matching transcript is skipped rather than assigned invented text. The command fails
with an actionable error when no transcribed WAVs can be discovered.

Input audio is decoded by `soundfile`. Multichannel clips are averaged to mono and non-16-kHz clips
are deterministically resampled with NumPy linear interpolation. No Torch, ffmpeg, network access, or
Dataset-A content is required.

## Command line

```bash
python scripts/prepare_phase2_synthetic.py \
  --aishell-root data/public/AISHELL-1 \
  --output-root output/phase2_aishell_synthetic \
  --manifest output/phase2_aishell_synthetic/manifest.jsonl \
  --max-speakers-per-split 4 \
  --utterances-per-speaker 2 \
  --seed 20260804 \
  --snr-db 5 \
  --sir-db 0 \
  --forbidden-root datasetA \
  --forbidden-root private_test_audio
```

`--manifest` is optional. Without it, the manifest is written to
`<output-root>/manifest.jsonl`. When a different path is supplied, the output-root manifest is kept
and an identical validated JSONL file is also written to the requested location. Repeat
`--forbidden-root` for every evaluation-only tree. Use `none`, `null`, or `off` for `--snr-db` or
`--sir-db` to disable that component.

`--snr-db`, `--sir-db`, and `--overlap-ratio` also accept comma-separated deterministic schedules.
For example, `--snr-db 0,5,10,20 --sir-db -5,0,5 --overlap-ratio 0.25,0.5,0.75,1.0` selects one
value per rendered row using its seed and stable row ID. `--rir-root` and `--noise-root` accept
public WAV trees; their files are assigned to train/validation/test before rendering, so an asset
cannot cross a split boundary. `--reverb-probability` independently reverberates target and
interferer tracks with split-local RIRs. When no noise root is supplied and SNR is enabled, the
builder records and uses a deterministic colored-noise fallback.

The Python API exposes:

```python
from scripts.prepare_phase2_synthetic import (
    build_synthetic_manifest,
    discover_aishell_utterances,
    split_speakers,
)
```

`build_synthetic_manifest(...)` returns `tuple[TrainingManifestRow, ...]` after rendering and
validation.

## Construction policy

1. Unique speaker IDs are sorted, seeded-shuffled, and assigned once by `split_speakers`. The same
   identity cannot occur across splits in either target or interferer role.
2. At most `max_speakers_per_split` eligible speakers are selected per split. A speaker must have at
   least `utterances_per_speaker` transcribed clips.
3. Each selected target clip uses another selected clip from the same speaker for enrollment and a
   different same-split speaker for interference.
4. The interferer is deterministically cropped, repeated when needed, and placed to realize the
   scheduled positive-row overlap ratio. Its gain is adjusted to the requested finite SIR. Optional
   split-local noise is adjusted to the requested finite SNR; target and interferer can be
   independently reverberated before mixing.
5. Each positive example has a corresponding target-absent negative. Positive rows contain the
   target transcript; target-absent rows always have `text: null`, a silent
   `target_audio` safety placeholder, and no invented label.
6. WAVs are written below `enrollment/<split>`, `target/<split>`, and `mixture/<split>` as mono
   16-kHz PCM. IDs and selection order are stable for the same source corpus, parameters, and seed.

Small corpora may not produce rows in all three partitions. `split_speakers` reserves at least one
speaker for each non-zero validation/test fraction when possible, while retaining a training
speaker. If that leaves every split with fewer than the two speakers needed for mixing, the builder
deterministically falls back to a train-only fixture so two/three-speaker fake corpora remain usable.
The metadata records the final assignment count for every split.

## Outputs

`<output-root>/manifest.jsonl` is UTF-8 JSONL using the Phase 2 training fields. Audio paths are
absolute in the returned rows and JSONL, which allows an external `--manifest` copy to resolve to the
same rendered files.

`<output-root>/metadata.json` records:

- `seed`;
- resolved public source root;
- generator version;
- train/validation/test speaker assignment counts; and
- validated row count.
- selected SNR/SIR/overlap schedules, RIR/noise roots, reverb probability, and fallback status;
- split-local RIR/noise asset IDs and counts.

Before either manifest is accepted, `assert_valid_training_manifest` checks row semantics,
speaker-disjointness, and all caller-provided forbidden roots.

## Offline test fixture

The unit tests create a temporary fake AISHELL tree with short sine-wave files, including stereo and
8-kHz inputs to exercise conversion. They do not download or inspect the real corpus:

```bash
python -m unittest tests.test_phase2_synthetic_builder -v
python -m unittest discover -s tests -v
python -m compileall scripts/prepare_phase2_synthetic.py tests/test_phase2_synthetic_builder.py
```
