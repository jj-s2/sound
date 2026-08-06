"""Acoustic profiler CLI and public comparator (Task 2).

Profiles a WAV root into an aggregate acoustic profile, or compares a Dataset-A
aggregate profile against a public aggregate profile. Inputs are WAV roots
only: the CLI rejects roots that resolve to files (including ``.json``/``.jsonl``)
and never opens non-WAV files. Output is aggregate-only -- no input paths,
filenames, IDs, labels, or per-file vectors.

Profiling command::

    python scripts/profile_r3_acoustics.py --audio-root ROOT \
        --output PROFILE.json --kind {dataseta_audio,public_speech,public_noise,public_rir} \
        [--max-files N] [--seed N]

Comparator command::

    python scripts/profile_r3_acoustics.py --reference-profile DATASETA.json \
        --candidate-profile PUBLIC.json --output REPORT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.acoustic_profile import (  # noqa: E402
    AudioStats,
    compare_profiles,
    profile_audio_paths,
    read_aggregate_profile,
)

_KINDS = ("dataseta_audio", "public_speech", "public_noise", "public_rir")


def _stable_hash(seed: int, *parts: object) -> str:
    payload = "\0".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def discover_wavs(
    root: str | Path, *, max_files: int | None = None, seed: int = 0
) -> tuple[tuple[Path, ...], int]:
    """Deterministically discover WAV files below ``root``.

    Only ``.wav`` (case-insensitive) files are returned; no other file is ever
    opened. When ``max_files`` is set and the root contains more WAVs, a
    deterministic hash-ranking selects a bounded subset so dry-run/profiling
    latency stays bounded regardless of corpus size. Returns ``(selected,
    total_discovered)``.
    """
    audio_root = Path(root).expanduser().resolve(strict=False)
    if not audio_root.is_dir():
        raise ValueError(
            f"audio root is not a directory: {audio_root} (WAV roots only; "
            "JSON/JSONL/file inputs are rejected)"
        )
    discovered: list[Path] = []
    for pattern in ("*.wav", "*.WAV"):
        discovered.extend(audio_root.rglob(pattern))
    # Deduplicate while preserving deterministic (path) order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in sorted(discovered):
        resolved = path.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    total = len(unique)
    if max_files is not None and total > max_files:
        ranked = sorted(unique, key=lambda p: (_stable_hash(seed, str(p)), str(p)))
        unique = ranked[:max_files]
    return tuple(unique), total


def write_profile_with_meta(
    path: str | Path,
    stats: AudioStats,
    *,
    kind: str,
    discovered_count: int,
    profiled_count: int,
    seed: int,
    max_files: int | None,
) -> None:
    """Write an aggregate profile JSON with auxiliary profiling metadata.

    The acoustic profile hash covers only the aggregate acoustic statistics; the
    ``profiling`` block records discovery counts and config (no paths).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = stats.to_json_dict()
    payload["profiling"] = {
        "kind": kind,
        "discovered_count": int(discovered_count),
        "profiled_count": int(profiled_count),
        "seed": int(seed),
        "max_files": max_files,
    }
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def profile_root(
    audio_root: str | Path,
    output: str | Path,
    *,
    kind: str,
    max_files: int | None = None,
    seed: int = 0,
) -> dict:
    """Profile a WAV root and write an aggregate profile JSON."""
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {_KINDS}")
    wavs, discovered_count = discover_wavs(audio_root, max_files=max_files, seed=seed)
    if not wavs:
        raise ValueError(f"No WAV files found below audio root: {audio_root}")
    stats = profile_audio_paths(wavs)
    write_profile_with_meta(
        output,
        stats,
        kind=kind,
        discovered_count=discovered_count,
        profiled_count=stats.file_count,
        seed=seed,
        max_files=max_files,
    )
    return {
        "kind": kind,
        "discovered_count": discovered_count,
        "profiled_count": stats.file_count,
        "hash": stats.hash,
        "output": str(output),
    }


def compare(
    reference_profile: str | Path,
    candidate_profile: str | Path,
    output: str | Path,
) -> dict:
    """Compare a Dataset-A aggregate profile against a public aggregate profile."""
    reference = read_aggregate_profile(reference_profile)
    candidate = read_aggregate_profile(candidate_profile)
    report = compare_profiles(reference, candidate)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", default=None, help="WAV root to profile")
    parser.add_argument("--output", required=True, help="output profile/report JSON path")
    parser.add_argument(
        "--kind", choices=_KINDS, default=None, help="asset kind (profiling mode)"
    )
    parser.add_argument("--max-files", type=int, default=None, help="bounded deterministic subset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-profile", default=None, help="Dataset-A profile (compare mode)")
    parser.add_argument("--candidate-profile", default=None, help="public profile (compare mode)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.reference_profile and args.candidate_profile:
        compare(args.reference_profile, args.candidate_profile, args.output)
        print(f"Wrote comparison report to {args.output}")
        return
    if args.reference_profile or args.candidate_profile:
        raise ValueError("--reference-profile and --candidate-profile must be supplied together")
    if not args.audio_root:
        raise ValueError("--audio-root is required in profiling mode")
    if not args.kind:
        raise ValueError("--kind is required in profiling mode")
    result = profile_root(
        args.audio_root,
        args.output,
        kind=args.kind,
        max_files=args.max_files,
        seed=args.seed,
    )
    print(
        f"Profiled {result['profiled_count']} of {result['discovered_count']} WAV(s) "
        f"({result['kind']}) -> {result['output']} hash={result['hash'][:12]}"
    )


if __name__ == "__main__":
    main()
