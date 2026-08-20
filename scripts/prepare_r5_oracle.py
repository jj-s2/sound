"""Build and validate the bounded public R5 ASR oracle set.

Generates two deterministic renderer seeds (A, B), each with 600 target-present
examples split 50/50 into frozen public val/test (20 examples per (snr, overlap)
bucket per split). Each scene has at most one interferer speaker (distinct from
the target), a valid overlap=0.0 no-speech-overlap scene, SNR drawn from
{-5, 0, 5} dB, and natural source levels preserved by a shared anti-clipping
limiter (no forced 0.98 peak). Dataset-A is never read; a fail-closed preflight
runs before any output directory creation or audio loading.

Outputs (under ``--output-root``)::

    seedA/manifest.jsonl seedA/metadata.json
    seedA/{mixture,clean_target}/{val,test}/*.wav
    seedB/...
    index.json   (both seeds' counts + manifest digests)
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
from xh202615.r5_oracle import (  # noqa: E402
    GENERATOR_VERSION,
    OVERLAP_GRID,
    R5OracleConfig,
    R5OracleRow,
    SNR_GRID,
    SIR_GRID,
    FULL_PROFILE,
    assert_r5_complete_design,
    assert_r5_manifest_safe,
    sha256_file,
    smoke_profile,
    write_r5_manifest,
)
from xh202615.r3_mixing import load_mono_16k  # noqa: E402

SEEDS = ("A", "B")
SEED_INT = {"A": 20260806, "B": 20260807}
PARTITION_SEED = 20260806
PER_BUCKET_PER_SPLIT = 20  # 15 buckets * 20 * 2 splits = 600 per seed
VAL_FRACTION = 0.5
TEST_FRACTION = 0.49  # val+test < 1; remainder (2 speakers) unused
DEFAULT_AISHELL = "data/external/aishell1/extracted"
DEFAULT_RIR = "data/external/sim_rir_16k/extracted"
DEFAULT_NOISE = "data/external/rirs_noises/extracted/RIRS_NOISES/pointsource_noises"
DEFAULT_DATASETA = "datasetA/datasetA"


def _stable_int(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(p) for p in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _is_under(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _audio_assets(root: Path, kind: str) -> tuple[Path, ...]:
    if not root.is_dir():
        raise ValueError(f"{kind} root does not exist or is not a directory: {root}")
    candidates = tuple(sorted(p.resolve(strict=False) for p in root.rglob("*.wav")))
    if not candidates:
        raise ValueError(f"No WAV files found below {kind} root: {root}")
    return candidates


def _split_assets(paths: tuple[Path, ...], *, seed: int, kind: str) -> dict[str, tuple[Path, ...]]:
    """Disjoint val/test partition of assets (no train needed for R5)."""
    if not paths:
        return {"val": (), "test": ()}
    assignments = split_speakers(
        (str(p) for p in paths), seed=_stable_int(seed, kind),
        val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION,
    )
    return {
        split: tuple(p for p in paths if assignments[str(p)] == split)
        for split in ("val", "test")
    }


def _select(assets: dict[str, tuple[Path, ...]], split: str, seed: int, *parts: object) -> Path | None:
    choices = assets.get(split, ())
    if not choices:
        return None
    return choices[_stable_int(seed, *parts) % len(choices)]


def _wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as handle:
            sr = handle.getframerate()
            return (handle.getnframes() / sr) if sr else 0.0
    except (wave.Error, OSError, EOFError):
        info = sf.info(path)
        return (info.frames / info.samplerate) if info.samplerate else 0.0


def _eligible_utterances(utterances, speaker_splits) -> dict[str, list[dict]]:
    """Per-split sorted lists of utterances whose speaker has >=2 utterances."""
    by_speaker: dict[str, list[dict]] = {}
    for u in utterances:
        by_speaker.setdefault(u["speaker_id"], []).append(u)
    eligible_speakers = {
        spk for spk, us in by_speaker.items() if len(us) >= 2
    }
    out: dict[str, list[dict]] = {"val": [], "test": []}
    for u in utterances:
        spk = u["speaker_id"]
        if spk not in eligible_speakers:
            continue
        split = speaker_splits.get(spk)
        if split in ("val", "test"):
            out[split].append(u)
    for split in out:
        out[split].sort(key=lambda u: u["utterance_id"])
    return out


def _plan_seed(
    seed_label: str,
    eligible: dict[str, list[dict]],
    *,
    rir_assets: dict[str, tuple[Path, ...]],
    noise_assets: dict[str, tuple[Path, ...]],
    per_bucket_per_split: int = PER_BUCKET_PER_SPLIT,
) -> list[dict]:
    """Plan scenes for one renderer seed across val/test x 15 buckets."""
    seed = SEED_INT[seed_label]
    plan: list[dict] = []
    for split in ("val", "test"):
        utts = eligible[split]
        if len(utts) < 2:
            raise ValueError(f"split {split!r} has too few eligible utterances ({len(utts)})")
        # Group utterances by speaker for enrollment + interferer selection.
        by_speaker: dict[str, list[dict]] = {}
        for u in utts:
            by_speaker.setdefault(u["speaker_id"], []).append(u)
        speakers = sorted(by_speaker)
        seed_offset = _stable_int(seed, split, "target-offset") % len(utts)
        global_k = 0
        for snr in SNR_GRID:
            for overlap in OVERLAP_GRID:
                for _ in range(per_bucket_per_split):
                    target = utts[(seed_offset + global_k) % len(utts)]
                    global_k += 1
                    t_spk = target["speaker_id"]
                    t_utts = by_speaker[t_spk]
                    # enrollment: a different utterance from the same speaker.
                    enroll = next(
                        (u for u in t_utts if u["utterance_id"] != target["utterance_id"]),
                        t_utts[0],
                    )
                    sir = None
                    interferer = None
                    if overlap > 0.0:
                        other_speakers = [s for s in speakers if s != t_spk]
                        if not other_speakers:
                            raise ValueError(
                                f"split {split!r} has only one speaker; cannot pick interferer"
                            )
                        i_spk = other_speakers[
                            _stable_int(seed, split, snr, overlap, global_k, "intf-spk")
                            % len(other_speakers)
                        ]
                        i_utts = by_speaker[i_spk]
                        interferer = i_utts[
                            _stable_int(seed, split, snr, overlap, global_k, "intf-utt")
                            % len(i_utts)
                        ]
                        sir = SIR_GRID[
                            _stable_int(seed, split, snr, overlap, global_k, "sir")
                            % len(SIR_GRID)
                        ]
                    rir = _select(rir_assets, split, seed, split, snr, overlap, global_k, "rir")
                    noise = _select(
                        noise_assets, split, seed, split, snr, overlap, global_k, "noise"
                    )
                    plan.append({
                        "seed": seed_label,
                        "split": split,
                        "snr": float(snr),
                        "overlap": float(overlap),
                        "sir": (float(sir) if sir is not None else None),
                        "target": target,
                        "enrollment": enroll,
                        "interferer": interferer,
                        "rir": rir,
                        "noise": noise,
                    })
    return plan


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    # 32-bit float WAV: standard, FunASR/soundfile-compatible, half the size of
    # 64-bit DOUBLE. The additive invariant (mixture = clean_target + base) holds
    # exactly in-memory (float64) and within float32 tolerance (~1e-7, far below
    # any ASR-perceptual threshold) on the stored files. Hashes are computed on
    # these actual stored float32 files.
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float64), sample_rate, subtype="FLOAT")


def build_r5_oracle(
    *,
    aishell_root,
    output_root,
    dataset_a_root,
    rir_root,
    noise_root,
    seeds=SEEDS,
    dry_run=False,
    per_bucket_per_split=PER_BUCKET_PER_SPLIT,
):
    """Discover, partition, plan, and (unless dry-run) render the R5 oracle set."""
    aishell = Path(aishell_root).expanduser().resolve(strict=False)
    output = Path(output_root).expanduser().resolve(strict=False)
    dataset_a = Path(dataset_a_root).expanduser().resolve(strict=False)
    rir = Path(rir_root).expanduser().resolve(strict=False) if rir_root else None
    noise = Path(noise_root).expanduser().resolve(strict=False) if noise_root else None

    # Fail-closed preflight before output creation or audio/model imports.
    for label, path in (("AISHELL source", aishell), ("output", output),
                        ("RIR", rir), ("noise", noise)):
        if path is not None and _is_under(path, dataset_a):
            raise ValueError(
                f"Dataset-A containment violation: {label} path {path} resolves "
                f"under Dataset-A root {dataset_a}"
            )
    if _is_under(output, aishell):
        raise ValueError("output root must not be under the AISHELL source root")

    utterances = discover_aishell_utterances(aishell)
    speaker_splits = split_speakers(
        (u["speaker_id"] for u in utterances),
        seed=PARTITION_SEED,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
    )
    eligible = _eligible_utterances(utterances, speaker_splits)
    rir_assets = _split_assets(_audio_assets(rir, "RIR"), seed=PARTITION_SEED, kind="rir") if rir else {"val": (), "test": ()}
    noise_assets = _split_assets(_audio_assets(noise, "noise"), seed=PARTITION_SEED, kind="noise") if noise else {"val": (), "test": ()}

    plans = {s: _plan_seed(s, eligible, rir_assets=rir_assets, noise_assets=noise_assets,
                            per_bucket_per_split=per_bucket_per_split) for s in seeds}

    metadata_common = {
        "generator_version": GENERATOR_VERSION,
        "seeds": list(seeds),
        "snr_grid": list(SNR_GRID),
        "overlap_grid": list(OVERLAP_GRID),
        "sir_grid": list(SIR_GRID),
        "per_bucket_per_split": per_bucket_per_split,
        "bucket_weights": {f"{s}|{o}": 1.0 / (len(SNR_GRID) * len(OVERLAP_GRID))
                           for s in SNR_GRID for o in OVERLAP_GRID},
        "dataset_a_used": False,
        "aishell_root": str(aishell),
        "rir_root": str(rir) if rir else None,
        "noise_root": str(noise) if noise else None,
        "split_speakers": {"val": sum(1 for v in speaker_splits.values() if v == "val"),
                           "test": sum(1 for v in speaker_splits.values() if v == "test")},
        "asset_split_counts": {
            split: {"rir": len(rir_assets[split]), "noise": len(noise_assets[split])}
            for split in ("val", "test")
        },
    }

    if dry_run:
        return {"dry_run": True, "plans": plans, "metadata": metadata_common}

    output.mkdir(parents=True, exist_ok=True)
    index = {"generator_version": GENERATOR_VERSION, "seeds": {}}
    all_rows_by_seed: dict[str, list[R5OracleRow]] = {}
    audio_cache: dict[Path, np.ndarray] = {}

    def load(path: Path) -> np.ndarray:
        if path not in audio_cache:
            audio_cache[path] = load_mono_16k(path)
        return audio_cache[path]

    for seed_label in seeds:
        plan = plans[seed_label]
        seed_dir = output / f"seed{seed_label}"
        rows: list[R5OracleRow] = []
        rng = np.random.default_rng(SEED_INT[seed_label])
        for idx, spec in enumerate(plan):
            split = spec["split"]
            target_info = spec["target"]
            target_audio = load(target_info["audio_path"])
            interferer_audio = (
                load(spec["interferer"]["audio_path"]) if spec["interferer"] else None
            )
            noise_audio = load(spec["noise"]) if spec["noise"] else None
            rir_path = spec["rir"]
            target_rir = load(rir_path) if rir_path else None
            # One shared RIR per scene for the interferer (same room) when present.
            interferer_rir = target_rir if (spec["interferer"] and rir_path) else None

            config = R5OracleConfig(
                snr_db=spec["snr"],
                sir_db=spec["sir"],
                overlap_ratio=spec["overlap"],
            )
            from xh202615.r5_oracle import render_oracle_audio
            result = render_oracle_audio(
                target=target_audio,
                interferer=interferer_audio,
                noise=noise_audio,
                target_rir=target_rir,
                interferer_rir=interferer_rir,
                config=config,
                rng=rng,
            )
            stem = (f"r5-seed{seed_label}-{split}-snr{int(spec['snr'])}"
                    f"-ov{str(spec['overlap']).replace('.', 'p')}-{idx:04d}")
            mix_path = seed_dir / "mixture" / split / f"{stem}.wav"
            clean_path = seed_dir / "clean_target" / split / f"{stem}.wav"
            _write_wav(mix_path, result.mixture, config.sample_rate)
            _write_wav(clean_path, result.clean_target, config.sample_rate)

            interferer_info = spec["interferer"]
            row = R5OracleRow(
                row_id=stem,
                seed=seed_label,
                split=split,
                target_speaker_id=target_info["utterance_id"],
                target_speaker=target_info["speaker_id"],
                interferer_speaker_id=(
                    interferer_info["utterance_id"] if interferer_info else None
                ),
                interferer_speaker=(
                    interferer_info["speaker_id"] if interferer_info else None
                ),
                enrollment_audio=Path(spec["enrollment"]["audio_path"]).resolve(strict=False),
                mixture_audio=mix_path.resolve(strict=False),
                clean_target_audio=clean_path.resolve(strict=False),
                snr_db=spec["snr"],
                sir_db=spec["sir"],
                overlap_ratio=spec["overlap"],
                rir_id=str(rir_path.resolve(strict=False)) if rir_path else None,
                noise_source_id=str(spec["noise"].resolve(strict=False)) if spec["noise"] else None,
                transcript=target_info["text"],
                mixture_digest=sha256_file(mix_path),
                clean_digest=sha256_file(clean_path),
                level_stats=result.level_stats,
                generator_version=GENERATOR_VERSION,
            )
            rows.append(row)
            if (idx + 1) % 50 == 0:
                print(f"[seed{seed_label}] rendered {idx + 1}/{len(plan)}", flush=True)

        rows = list(assert_r5_manifest_safe(rows, dataset_a_root=dataset_a))
        manifest_path = seed_dir / "manifest.jsonl"
        write_r5_manifest(manifest_path, rows)
        meta = dict(metadata_common)
        meta.update({
            "seed": seed_label,
            "seed_int": SEED_INT[seed_label],
            "row_count": len(rows),
            "split_rows": {
                split: sum(1 for r in rows if r.split == split) for split in ("val", "test")
            },
            "bucket_counts": _bucket_counts(rows),
            "manifest_path": str(manifest_path.resolve(strict=False)),
            "manifest_digest": sha256_file(manifest_path),
        })
        (seed_dir / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        all_rows_by_seed[seed_label] = rows
        index["seeds"][seed_label] = {
            "row_count": len(rows),
            "split_rows": meta["split_rows"],
            "bucket_counts": meta["bucket_counts"],
            "manifest_path": str(manifest_path.resolve(strict=False)),
            "manifest_digest": meta["manifest_digest"],
        }
        print(f"[seed{seed_label}] wrote {len(rows)} rows to {manifest_path}", flush=True)

    index_path = output / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Complete-design validation (full or smoke profile) on the actual files.
    all_rows = [r for s in seeds for r in all_rows_by_seed[s]]
    if per_bucket_per_split == PER_BUCKET_PER_SPLIT and tuple(sorted(seeds)) == SEEDS:
        profile = FULL_PROFILE
    else:
        profile = smoke_profile(seeds=seeds, per_bucket_per_split=per_bucket_per_split)
    assert_r5_complete_design(
        all_rows, profile=profile, dataset_a_root=dataset_a,
        check_files=True, index=index,
    )
    print(f"complete-design validation passed ({profile.name})", flush=True)
    return {"dry_run": False, "rows_by_seed": all_rows_by_seed, "index": index, "metadata": metadata_common}


def _bucket_counts(rows: Iterable[R5OracleRow]) -> dict:
    counts: dict[str, int] = {}
    for r in rows:
        key = f"{r.split}|snr{int(r.snr_db)}|ov{r.overlap_ratio}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aishell-root", default=DEFAULT_AISHELL)
    parser.add_argument("--output-root", default="data/synthetic/r5_oracle_v1")
    parser.add_argument("--dataset-a-root", default=DEFAULT_DATASETA)
    parser.add_argument("--rir-root", default=DEFAULT_RIR)
    parser.add_argument("--noise-root", default=DEFAULT_NOISE)
    parser.add_argument("--seeds", default="A,B", help="Comma-separated subset of A,B")
    parser.add_argument("--per-bucket-per-split", type=int, default=PER_BUCKET_PER_SPLIT,
                        help="Examples per (split, snr, overlap) bucket (20 -> 600/seed)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = tuple(s.strip() for s in args.seeds.split(",") if s.strip())
    build_r5_oracle(
        aishell_root=args.aishell_root,
        output_root=args.output_root,
        dataset_a_root=args.dataset_a_root,
        rir_root=args.rir_root,
        noise_root=args.noise_root,
        seeds=seeds,
        dry_run=args.dry_run,
        per_bucket_per_split=args.per_bucket_per_split,
    )
    if args.dry_run:
        print("Dry run complete (no files written)")


if __name__ == "__main__":
    main()
