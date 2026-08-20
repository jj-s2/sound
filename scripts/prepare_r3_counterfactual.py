"""Prepare deterministic matched-counterfactual R3 mixtures.

Discovers public AISHELL utterances, partitions speakers/RIR/noise/renderer
families into disjoint train/val/test splits, then renders matched
counterfactual pairs. Dataset-A is never read or used; a fail-closed preflight
runs before any output directory creation or audio loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_phase2_synthetic import (  # noqa: E402
    discover_aishell_utterances,
    split_speakers,
)
from xh202615.acoustic_profile import (  # noqa: E402
    AudioStats,
    assert_aggregate_profile,
    read_aggregate_profile,
)
from xh202615.r3_data import (  # noqa: E402
    assert_r3_manifest_safe,
    write_r3_manifest,
)
from xh202615.r3_mixing import (  # noqa: E402
    RendererConfig,
    load_mono_16k,
    render_counterfactual_pair,
)

SPLITS = ("train", "val", "test")
SNR_GRID = (-5.0, 0.0, 5.0, 10.0, 20.0)
SIR_GRID = (-10.0, -5.0, 0.0, 5.0, 10.0)
OVERLAP_GRID = (0.25, 0.5, 0.75, 1.0)
INTERFERER_COUNT_GRID = (1, 2, 3)
GENERATOR_VERSION = "r3-counterfactual-v1"
_VAL_FRACTION = 0.2
_TEST_FRACTION = 0.2


def _is_under(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _stable_int(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _audio_assets(root: Path, kind: str) -> tuple[Path, ...]:
    if not root.is_dir():
        raise ValueError(f"{kind} root does not exist or is not a directory: {root}")
    candidates = tuple(sorted(path.resolve(strict=False) for path in root.rglob("*.wav")))
    if not candidates:
        raise ValueError(f"No WAV files found below {kind} root: {root}")
    return candidates


def _split_assets(paths: tuple[Path, ...], *, seed: int, kind: str) -> dict[str, tuple[Path, ...]]:
    if not paths:
        return {split: () for split in SPLITS}
    assignments = split_speakers(
        (str(path) for path in paths), seed=_stable_int(seed, kind),
        val_fraction=_VAL_FRACTION, test_fraction=_TEST_FRACTION,
    )
    return {
        split: tuple(path for path in paths if assignments[str(path)] == split)
        for split in SPLITS
    }


def _select_asset(assets: dict[str, tuple[Path, ...]], split: str, seed: int, pair_id: str, role: str) -> Path | None:
    choices = assets.get(split, ())
    if not choices:
        return None
    return choices[_stable_int(seed, pair_id, role) % len(choices)]


def _load_acoustic_profile(path: str | Path) -> AudioStats:
    """Read and validate an aggregate acoustic profile; reject non-aggregate content."""
    profile_path = Path(path).expanduser().resolve(strict=False)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert_aggregate_profile(payload)
    return read_aggregate_profile(profile_path)


def _sample_duration_target(profile: AudioStats, seed: int, pair_id: str) -> float:
    """Deterministically sample a target duration from the aggregate duration quantiles."""
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantiles = profile.metrics.get("duration", {}).get("quantiles", {})
    vals = [float(quantiles.get(f"{q:g}", 0.0)) for q in qs]
    if max(vals) <= 0.0:
        return 0.0
    frac = _stable_int(seed, pair_id, "duration-target") / float(2 ** 64)
    pct = 0.05 + frac * (0.95 - 0.05)
    return float(np.interp(pct, qs, vals))


def _wav_duration(path: str) -> float:
    """Header-only duration in seconds.

    Uses the stdlib ``wave`` parser (a 44-byte header read, no libsndfile
    overhead) for PCM WAVs, falling back to ``soundfile.info`` for non-PCM or
    float subtypes. This keeps bounded discovery fast over large public corpora.
    """
    try:
        with wave.open(path, "rb") as handle:
            sample_rate = handle.getframerate()
            return (handle.getnframes() / sample_rate) if sample_rate else 0.0
    except (wave.Error, OSError, EOFError):
        info = sf.info(path)
        return (info.frames / info.samplerate) if info.samplerate else 0.0


def _discover_durations(utterances) -> dict[str, float]:
    """Header-only duration per public utterance path; cached per path."""
    durations: dict[str, float] = {}
    for utterance in utterances:
        key = str(utterance["audio_path"])
        if key not in durations:
            durations[key] = _wav_duration(key)
    return durations


def _plan_pairs(
    utterances,
    speaker_splits,
    *,
    pairs: int,
    seed: int,
    acoustic_profile: AudioStats | None = None,
    durations: dict[str, float] | None = None,
):
    by_speaker: dict[str, list] = {}
    for utterance in utterances:
        by_speaker.setdefault(utterance["speaker_id"], []).append(utterance)
    plan = []
    for split in SPLITS:
        speakers = sorted(
            speaker_id
            for speaker_id, assigned in speaker_splits.items()
            if assigned == split and len(by_speaker.get(speaker_id, ())) >= 2
        )
        if len(speakers) < 2:
            continue
        for index in range(pairs):
            target_speaker = speakers[index % len(speakers)]
            target_utts = sorted(by_speaker[target_speaker], key=lambda u: u["utterance_id"])
            if acoustic_profile is not None and durations:
                target_seconds = _sample_duration_target(acoustic_profile, seed, f"{split}-{index}")
                ranked = sorted(
                    target_utts,
                    key=lambda u: (
                        abs(durations.get(str(u["audio_path"]), 0.0) - target_seconds),
                        _stable_int(seed, f"{split}-{index}", "target", u["utterance_id"]),
                        u["utterance_id"],
                    ),
                )
                target_info = ranked[0]
                enroll_candidates = [
                    u for u in ranked if u["utterance_id"] != target_info["utterance_id"]
                ]
                enroll_info = enroll_candidates[0] if enroll_candidates else target_utts[1]
            else:
                target_info = target_utts[index % len(target_utts)]
                enroll_info = target_utts[(index + 1) % len(target_utts)]
            pair_id = f"r3-{split}-{index + 1:04d}"
            # Select N DISTINCT interferer speakers from the same split,
            # excluding the target speaker (one utterance each).  The plan
            # grid and final_decision.md specify 1..3 interfering *speakers*,
            # not multiple utterances of a single speaker.
            other_speakers = [s for s in speakers if s != target_speaker]
            raw_count = INTERFERER_COUNT_GRID[_stable_int(seed, pair_id, "icount") % len(INTERFERER_COUNT_GRID)]
            interferer_count = max(1, min(raw_count, len(other_speakers)))
            # Deterministic seeded selection without replacement.
            ranked_others = sorted(
                other_speakers,
                key=lambda s: (_stable_int(seed, pair_id, "interferer-speaker", s), s),
            )
            interferer_speakers = ranked_others[:interferer_count]
            interferer_infos = []
            for k, ispk in enumerate(interferer_speakers):
                ispk_utts = sorted(by_speaker[ispk], key=lambda u: u["utterance_id"])
                interferer_infos.append(ispk_utts[(index + k) % len(ispk_utts)])
            plan.append(
                {
                    "pair_id": pair_id,
                    "split": split,
                    "target_info": target_info,
                    "enroll_info": enroll_info,
                    "interferer_infos": tuple(interferer_infos),
                    "snr": SNR_GRID[_stable_int(seed, pair_id, "snr") % len(SNR_GRID)],
                    "sir": SIR_GRID[_stable_int(seed, pair_id, "sir") % len(SIR_GRID)],
                    "overlap": OVERLAP_GRID[_stable_int(seed, pair_id, "overlap") % len(OVERLAP_GRID)],
                    "renderer_family": f"r3-{split}",
                }
            )
    return plan


def build_r3_manifest(
    aishell_root,
    output_root,
    *,
    dataset_a_root,
    pairs=2,
    seed=20260805,
    rir_root=None,
    noise_root=None,
    dry_run=False,
    acoustic_profile=None,
):
    """Discover, partition, plan, and (unless dry-run) render R3 pairs.

    When ``acoustic_profile`` (a path to an aggregate profile JSON) is supplied,
    each pair is assigned a target duration sampled from the profile and the
    public target utterance is chosen to minimize ``abs(duration-target)`` while
    preserving same-speaker enrollment, distinct same-split interferers, and the
    existing SNR/SIR/overlap grids. Without it, behavior and seeded output are
    unchanged.
    """
    aishell = Path(aishell_root).expanduser().resolve(strict=False)
    output = Path(output_root).expanduser().resolve(strict=False)
    dataset_a = Path(dataset_a_root).expanduser().resolve(strict=False)
    rir = Path(rir_root).expanduser().resolve(strict=False) if rir_root else None
    noise = Path(noise_root).expanduser().resolve(strict=False) if noise_root else None

    profile = _load_acoustic_profile(acoustic_profile) if acoustic_profile else None

    # Fail-closed preflight before output creation or audio/model imports.
    for label, path in (
        ("AISHELL source", aishell),
        ("output", output),
        ("RIR", rir),
        ("noise", noise),
    ):
        if path is not None and _is_under(path, dataset_a):
            raise ValueError(
                f"Dataset-A containment violation: {label} path {path} resolves "
                f"under Dataset-A root {dataset_a}"
            )

    utterances = discover_aishell_utterances(aishell)
    speaker_splits = split_speakers(
        (u["speaker_id"] for u in utterances),
        seed=seed,
        val_fraction=_VAL_FRACTION,
        test_fraction=_TEST_FRACTION,
    )
    rir_assets = (
        _split_assets(_audio_assets(rir, "RIR"), seed=seed, kind="rir") if rir else {s: () for s in SPLITS}
    )
    noise_assets = (
        _split_assets(_audio_assets(noise, "noise"), seed=seed, kind="noise")
        if noise
        else {s: () for s in SPLITS}
    )

    durations = _discover_durations(utterances) if profile is not None else None
    plan = _plan_pairs(
        utterances, speaker_splits, pairs=pairs, seed=seed,
        acoustic_profile=profile, durations=durations,
    )

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "pair_count": len(plan),
        "splits": {split: sum(1 for p in plan if p["split"] == split) for split in SPLITS},
        "snr_grid": list(SNR_GRID),
        "sir_grid": list(SIR_GRID),
        "overlap_grid": list(OVERLAP_GRID),
        "interferer_count_grid": list(INTERFERER_COUNT_GRID),
        "dataset_a_used_for_training": False,
        "asset_split_counts": {
            split: {"rir": len(rir_assets[split]), "noise": len(noise_assets[split])}
            for split in SPLITS
        },
        "source_root": str(aishell),
    }
    if profile is not None:
        metadata["acoustic_profile_hash"] = profile.hash
        metadata["duration_target_seconds"] = float(
            profile.metrics.get("duration", {}).get("quantiles", {}).get("0.5", 0.0)
        )
        metadata["profiled_source_duration_seconds"] = (
            float(sum(durations.values())) if durations else 0.0
        )

    if dry_run:
        return {"dry_run": True, "pair_count": len(plan), "metadata": metadata, "rows": ()}

    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    audio_cache: dict[Path, np.ndarray] = {}

    def load(path: Path) -> np.ndarray:
        if path not in audio_cache:
            audio_cache[path] = load_mono_16k(path)
        return audio_cache[path]

    rows = []
    for spec in plan:
        split = spec["split"]
        pair_id = spec["pair_id"]
        target_audio = load(spec["target_info"]["audio_path"])
        enroll_audio = load(spec["enroll_info"]["audio_path"])
        interferer_audios = tuple(load(ii["audio_path"]) for ii in spec["interferer_infos"])
        interferer_source_ids = tuple(ii["utterance_id"] for ii in spec["interferer_infos"])

        use_rir = bool(rir_assets.get(split))
        target_rir_path = _select_asset(rir_assets, split, seed, pair_id, "target-rir") if use_rir else None
        interferer_rir_paths = [
            _select_asset(rir_assets, split, seed, pair_id, f"interferer-rir-{k}") if use_rir else None
            for k in range(len(interferer_audios))
        ]
        target_rir = load(target_rir_path) if target_rir_path else None
        target_rir_id = str(target_rir_path) if target_rir_path else None
        interferer_rirs = tuple(load(p) if p else None for p in interferer_rir_paths)
        interferer_rir_ids = tuple(str(p) for p in interferer_rir_paths) if use_rir else ()

        noise_path = _select_asset(noise_assets, split, seed, pair_id, "noise")
        noise_audio = load(noise_path) if noise_path else None
        noise_source_id = str(noise_path) if noise_path else None

        config = RendererConfig(
            snr_db=spec["snr"], sir_db=spec["sir"], overlap_ratio=spec["overlap"]
        )
        enrollment_path = output / "enrollment" / split / f"{pair_id}.wav"
        mixture_paths = (
            output / "mixture" / split / f"{pair_id}-pos.wav",
            output / "mixture" / split / f"{pair_id}-neg.wav",
        )
        clean_paths = (
            output / "clean_target" / split / f"{pair_id}-pos.wav",
            output / "clean_target" / split / f"{pair_id}-neg.wav",
        )
        positive, negative = render_counterfactual_pair(
            pair_id=pair_id,
            split=split,
            config=config,
            rng=rng,
            enrollment=enroll_audio,
            target=target_audio,
            interferers=interferer_audios,
            noise=noise_audio,
            target_source_id=spec["target_info"]["utterance_id"],
            interferer_source_ids=interferer_source_ids,
            noise_source_id=noise_source_id,
            target_rir=target_rir,
            target_rir_id=target_rir_id,
            interferer_rirs=interferer_rirs,
            interferer_rir_ids=interferer_rir_ids,
            renderer_family=spec["renderer_family"],
            enrollment_path=enrollment_path,
            mixture_paths=mixture_paths,
            clean_target_paths=clean_paths,
        )
        rows.append(positive)
        rows.append(negative)

    rows = assert_r3_manifest_safe(rows, dataset_a_root=dataset_a)
    manifest_path = output / "manifest.jsonl"
    write_r3_manifest(manifest_path, rows)
    metadata["row_count"] = len(rows)
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"dry_run": False, "pair_count": len(rows) // 2, "rows": tuple(rows), "metadata": metadata}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aishell-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-a-root", required=True)
    parser.add_argument("--rir-root", default=None)
    parser.add_argument("--noise-root", default=None)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--acoustic-profile", default=None,
        help="aggregate acoustic profile JSON guiding duration matching (optional)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_r3_manifest(
        args.aishell_root,
        args.output_root,
        dataset_a_root=args.dataset_a_root,
        pairs=args.pairs,
        seed=args.seed,
        rir_root=args.rir_root,
        noise_root=args.noise_root,
        dry_run=args.dry_run,
        acoustic_profile=args.acoustic_profile,
    )
    if args.dry_run:
        print(f"Planned {result['pair_count']} counterfactual pairs (dry run)")
    else:
        print(f"Wrote {len(result['rows'])} rows to {Path(args.output_root) / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
