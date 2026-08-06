"""R5 Overall-first ASR oracle: public clean-target vs raw-mixture renderer.

This module renders *target-present* oracle scenes from public AISHELL-1 audio.
For each scene it produces a **mixture** (target + at most one interferer +
noise) and the **clean target** - the exact target track as it appears in the
mixture (after resampling, RIR, timing, channel response, codec, and the shared
scene gain) but before the interferer/noise is added. Transcribing both with the
same frozen FunASR gives a CER gap that is an Overall ceiling for CER-only
recovery (see ``r5-oracle-decision.md``).

Key differences from the R3 counterfactual renderer
(``xh202615/r3_mixing.render_pair_audio``), all mandated by the R5 brief:

* **Natural levels.** No forced ``target_peak=0.98``. A shared anti-clipping
  limiter ``scale = min(1.0, clip_threshold / peak)`` only *attenuates* and only
  when clipping would occur, so natural public source levels are preserved. The
  limiter guarantees no sample reaches the clip threshold, so the safety clip is
  a no-op and ``mixture == clean_target + scaled_base`` exactly.
* **At most one interferer.** Target + interferer <= 2 simultaneous speakers.
* **Valid overlap=0.0.** ``overlap_samples = round(T * overlap)`` with no
  ``max(1, ...)`` coercion; ``overlap=0.0`` yields zero interferer samples - an
  explicitly documented no-speech-overlap scene (target + noise only).
* **Pre/post peak and RMS** are recorded for both paths.

Dataset-A is never read. Reusable acoustic primitives (RMS-preserving RIR,
codec, channel response, crop/repeat, clip, mono 16k load) are imported from
``xh202615.r3_mixing``; only the gain model, placement, and level bookkeeping
are new.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from xh202615.r3_mixing import (
    _apply_rir,
    _channel_response,
    _clip,
    _codec_simulate,
    _crop_or_repeat,
    _rms,
    load_mono_16k,
)

GENERATOR_VERSION = "r5-oracle-v1"
ALLOWED_SPLITS = ("val", "test")
SNR_GRID = (-5.0, 0.0, 5.0)
OVERLAP_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SIR_GRID = (-5.0, 0.0, 5.0)  # nuisance (not a bucket); averaged within buckets
# Predeclared public bucket weights: uniform over the 15 (snr, overlap) buckets.
BUCKET_WEIGHTS = {
    (float(snr), float(ov)): 1.0 / (len(SNR_GRID) * len(OVERLAP_GRID))
    for snr in SNR_GRID
    for ov in OVERLAP_GRID
}


@dataclass(frozen=True)
class R5OracleConfig:
    """Acoustic scene parameters for one R5 oracle scene."""

    sample_rate: int = 16_000
    snr_db: float | None = 5.0
    sir_db: float | None = 0.0
    overlap_ratio: float = 0.5
    codec: str = "pcm16"
    clip_threshold: float = 1.0
    channel_response: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.snr_db is not None and not math.isfinite(self.snr_db):
            raise ValueError("snr_db must be finite when provided")
        if self.sir_db is not None and not math.isfinite(self.sir_db):
            raise ValueError("sir_db must be finite when provided")
        if not math.isfinite(self.overlap_ratio) or not 0.0 <= self.overlap_ratio <= 1.0:
            raise ValueError("overlap_ratio must be finite and in [0, 1]")
        if not math.isfinite(self.clip_threshold) or not 0.0 < self.clip_threshold <= 1.0:
            raise ValueError("clip_threshold must be finite and in (0, 1]")
        if not self.codec.strip():
            raise ValueError("codec must be non-empty")


@dataclass(frozen=True)
class LevelRecord:
    """Pre/post peak and RMS for one audio path (clean or mixture)."""

    pre_peak: float
    pre_rms: float
    post_peak: float
    post_rms: float

    def to_dict(self) -> dict:
        return {
            "pre_peak": self.pre_peak,
            "pre_rms": self.pre_rms,
            "post_peak": self.post_peak,
            "post_rms": self.post_rms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LevelRecord":
        return cls(
            pre_peak=float(value["pre_peak"]),
            pre_rms=float(value["pre_rms"]),
            post_peak=float(value["post_peak"]),
            post_rms=float(value["post_rms"]),
        )


@dataclass(frozen=True)
class LevelStats:
    clean: LevelRecord
    mixture: LevelRecord

    def to_dict(self) -> dict:
        return {"clean": self.clean.to_dict(), "mixture": self.mixture.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LevelStats":
        return cls(
            clean=LevelRecord.from_dict(value["clean"]),
            mixture=LevelRecord.from_dict(value["mixture"]),
        )


@dataclass(frozen=True)
class RenderResult:
    """Output of :func:`render_oracle_audio`."""

    mixture: np.ndarray
    clean_target: np.ndarray
    base_component: np.ndarray  # interferer + noise after the shared scale
    level_stats: LevelStats
    scale: float


def _max_abs(audio: np.ndarray) -> float:
    a = np.asarray(audio, dtype=np.float64)
    return float(np.max(np.abs(a))) if a.size else 0.0


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _valid_digest(value: object) -> bool:
    """True iff value is a lowercase 64-char hex SHA-256 digest."""
    return isinstance(value, str) and bool(_DIGEST_RE.fullmatch(value))


def _overlap_samples(length: int, overlap_ratio: float) -> int:
    """Interferer active-segment length in samples (0 when overlap is 0.0).

    No ``max(1, ...)`` coercion: ``overlap=0.0`` returns exactly 0 so the
    interferer is absent (a documented no-speech-overlap scene).
    """
    if overlap_ratio <= 0.0:
        return 0
    return min(length, int(round(length * float(overlap_ratio))))


def _place_active(active: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Zero-pad an active interferer segment to ``length`` at a random start.

    The segment is already SIR-scaled over its active region (active-segment
    SIR, so interferer loudness is independent of overlap).
    """
    placed = np.zeros(length, dtype=np.float64)
    seg_len = len(active)
    if seg_len <= 0:
        return placed
    if seg_len >= length:
        placed[:] = active[:length]
        return placed
    max_start = length - seg_len
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    placed[start : start + seg_len] = active
    return placed


def render_oracle_audio(
    *,
    target: np.ndarray,
    interferer: np.ndarray | None,
    noise: np.ndarray | None,
    target_rir: np.ndarray | None,
    interferer_rir: np.ndarray | None,
    config: R5OracleConfig,
    rng: np.random.Generator,
) -> RenderResult:
    """Render (mixture, clean_target) with natural levels and a shared limiter.

    The clean target is the exact target track in the mixture after the shared
    scene gain, before interferer/noise. Because the limiter guarantees no sample
    reaches ``clip_threshold``, the safety clips are no-ops and
    ``mixture == clean_target + base_component`` exactly.
    """
    length = len(target)
    if length == 0:
        raise ValueError("target audio must be non-empty")

    target_component = _apply_rir(target, target_rir)
    target_rms = _rms(target_component)
    if target_rms == 0.0:
        raise ValueError("target audio is silent after rendering")

    # Interferer: RIR, active-segment SIR scaling, placement. Absent at 0 overlap.
    seg_len = _overlap_samples(length, config.overlap_ratio)
    if interferer is not None and seg_len > 0:
        intf_rendered = _apply_rir(interferer, interferer_rir)
        intf_active = _crop_or_repeat(intf_rendered, seg_len, rng)
        intf_rms = _rms(intf_active)
        if config.sir_db is not None and intf_rms > 0.0:
            sir_ratio = 10.0 ** (config.sir_db / 20.0)
            intf_active = intf_active * (target_rms / (sir_ratio * intf_rms))
        interferer_component = _place_active(intf_active, length, rng)
    else:
        interferer_component = np.zeros(length, dtype=np.float64)

    # Noise: SNR relative to target + interferer (window RMS), matching R3.
    if noise is not None and config.snr_db is not None:
        noise_component = _crop_or_repeat(noise, length, rng)
        signal_rms = _rms(target_component + interferer_component)
        noise_rms = _rms(noise_component)
        if noise_rms > 0.0 and signal_rms > 0.0:
            noise_component = noise_component * (
                signal_rms / (10.0 ** (config.snr_db / 20.0) * noise_rms)
            )
        else:
            noise_component = np.zeros(length, dtype=np.float64)
    else:
        noise_component = np.zeros(length, dtype=np.float64)

    base = interferer_component + noise_component
    target_channel = _channel_response(target_component, config.channel_response)
    base_channel = _channel_response(base, config.channel_response)

    clean_raw = _codec_simulate(target_channel, config.codec)
    base_raw = _codec_simulate(base_channel, config.codec)
    mixture_raw = base_raw + clean_raw  # pre-sum, used for peak + pre-gain stats

    # Shared anti-clipping limiter: only attenuates, never amplifies. Codec is
    # applied SEPARATELY to target and base BEFORE summation (documented in
    # r5-oracle-decision.md), so the mixture is built as the sum of the scaled
    # components and the clean target is exactly the target component of the
    # mixture: mixture = clip(base_component + clean_target). The limiter
    # guarantees no sample reaches clip_threshold, so the safety clip is a
    # numerical no-op (verified within 1e-9 in tests).
    peak = max(_max_abs(mixture_raw), _max_abs(clean_raw))
    scale = min(1.0, float(config.clip_threshold) / peak) if peak > 0.0 else 1.0
    clean_target = _clip(clean_raw * scale, config.clip_threshold)
    base_component = base_raw * scale
    mixture = _clip(base_component + clean_target, config.clip_threshold)

    level_stats = LevelStats(
        clean=LevelRecord(
            pre_peak=_max_abs(clean_raw),
            pre_rms=_rms(clean_raw),
            post_peak=_max_abs(clean_target),
            post_rms=_rms(clean_target),
        ),
        mixture=LevelRecord(
            pre_peak=_max_abs(mixture_raw),
            pre_rms=_rms(mixture_raw),
            post_peak=_max_abs(mixture),
            post_rms=_rms(mixture),
        ),
    )
    return RenderResult(
        mixture=mixture,
        clean_target=clean_target,
        base_component=base_component,
        level_stats=level_stats,
        scale=scale,
    )


@dataclass(frozen=True)
class R5OracleRow:
    """One R5 oracle manifest row with full provenance for leakage auditing."""

    row_id: str
    seed: str
    split: str
    target_speaker_id: str
    target_speaker: str
    interferer_speaker_id: str | None
    interferer_speaker: str | None
    enrollment_audio: Path
    mixture_audio: Path
    clean_target_audio: Path
    snr_db: float
    sir_db: float | None
    overlap_ratio: float
    rir_id: str | None
    noise_source_id: str | None
    transcript: str
    mixture_digest: str
    clean_digest: str
    level_stats: LevelStats
    generator_version: str

    def bucket(self) -> tuple[float, float]:
        return (float(self.snr_db), float(self.overlap_ratio))

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "seed": self.seed,
            "split": self.split,
            "target_speaker_id": self.target_speaker_id,
            "target_speaker": self.target_speaker,
            "interferer_speaker_id": self.interferer_speaker_id,
            "interferer_speaker": self.interferer_speaker,
            "enrollment_audio": Path(self.enrollment_audio).as_posix(),
            "mixture_audio": Path(self.mixture_audio).as_posix(),
            "clean_target_audio": Path(self.clean_target_audio).as_posix(),
            "snr_db": self.snr_db,
            "sir_db": self.sir_db,
            "overlap_ratio": self.overlap_ratio,
            "rir_id": self.rir_id,
            "noise_source_id": self.noise_source_id,
            "transcript": self.transcript,
            "mixture_digest": self.mixture_digest,
            "clean_digest": self.clean_digest,
            "level_stats": self.level_stats.to_dict(),
            "generator_version": self.generator_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object], *, base_dir: Path | None = None) -> "R5OracleRow":
        if not isinstance(value, Mapping):
            raise ValueError("R5OracleRow must be a dict")
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"R5OracleRow has unrecognized field(s): {unknown}")

        def _path_field(name: str) -> Path:
            raw = value[name]
            if not isinstance(raw, str):
                raise ValueError(f"{name} must be a path string")
            p = Path(raw)
            if base_dir is not None and not p.is_absolute():
                p = base_dir.resolve(strict=False) / p
            return p

        return cls(
            row_id=str(value["row_id"]),
            seed=str(value["seed"]),
            split=str(value["split"]),
            target_speaker_id=str(value["target_speaker_id"]),
            target_speaker=str(value["target_speaker"]),
            interferer_speaker_id=(
                None if value.get("interferer_speaker_id") is None
                else str(value["interferer_speaker_id"])
            ),
            interferer_speaker=(
                None if value.get("interferer_speaker") is None
                else str(value["interferer_speaker"])
            ),
            enrollment_audio=_path_field("enrollment_audio"),
            mixture_audio=_path_field("mixture_audio"),
            clean_target_audio=_path_field("clean_target_audio"),
            snr_db=float(value["snr_db"]),
            sir_db=(None if value.get("sir_db") is None else float(value["sir_db"])),
            overlap_ratio=float(value["overlap_ratio"]),
            rir_id=(None if value.get("rir_id") is None else str(value["rir_id"])),
            noise_source_id=(
                None if value.get("noise_source_id") is None
                else str(value["noise_source_id"])
            ),
            transcript=str(value["transcript"]),
            mixture_digest=str(value["mixture_digest"]),
            clean_digest=str(value["clean_digest"]),
            level_stats=LevelStats.from_dict(value["level_stats"]),
            generator_version=str(value["generator_version"]),
        )


@dataclass(frozen=True)
class R5ManifestIssue:
    code: str
    row_id: str | None
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "row_id": self.row_id, "message": self.message}


def _is_under(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _row_speakers(row: R5OracleRow) -> list[str]:
    speakers = [row.target_speaker]
    if row.interferer_speaker is not None:
        speakers.append(row.interferer_speaker)
    return speakers


def validate_r5_manifest(rows: Iterable[R5OracleRow]) -> tuple[R5ManifestIssue, ...]:
    """Return all validation issues for the R5 manifest without raising."""
    materialized = tuple(rows)
    issues: list[R5ManifestIssue] = []
    seen: dict[str, int] = {}
    speaker_splits: dict[str, str] = {}

    def add(code: str, row: object, message: str) -> None:
        issues.append(R5ManifestIssue(code, getattr(row, "row_id", None), message))

    for index, row in enumerate(materialized):
        if not isinstance(row, R5OracleRow):
            raise ValueError(f"R5 manifest row {index + 1} must be an R5OracleRow")
        if not row.row_id.strip():
            add("empty_row_id", row, "row_id must be non-empty")
        elif row.row_id in seen:
            add("duplicate_row_id", row, f"row_id {row.row_id!r} duplicates row {seen[row.row_id] + 1}")
        else:
            seen[row.row_id] = index

        if row.split not in ALLOWED_SPLITS:
            add("invalid_split", row, f"split must be one of {list(ALLOWED_SPLITS)}")
        if row.seed not in ("A", "B"):
            add("invalid_seed", row, "seed must be 'A' or 'B'")
        if not row.target_speaker.strip() or not row.target_speaker_id.strip():
            add("empty_target_speaker", row, "target speaker/utterance must be non-empty")

        # overlap=0.0 <=> no interferer (documented no-speech-overlap scene).
        if row.overlap_ratio == 0.0:
            if row.interferer_speaker_id is not None or row.interferer_speaker is not None:
                add("interferer_at_zero_overlap", row,
                    "overlap=0.0 must have no interferer (no-speech-overlap scene)")
            if row.sir_db is not None:
                add("sir_at_zero_overlap", row, "overlap=0.0 must have null sir_db")
        else:
            if row.interferer_speaker_id is None or row.interferer_speaker is None:
                add("missing_interferer", row,
                    "overlap>0 requires an interferer speaker/utterance")
            elif row.interferer_speaker == row.target_speaker:
                add("interferer_equals_target", row,
                    "target and interferer speakers must differ")
            if row.sir_db is None:
                add("missing_sir", row, "overlap>0 requires sir_db")

        if row.snr_db not in (float(s) for s in SNR_GRID):
            add("invalid_snr", row, f"snr_db must be one of {list(SNR_GRID)}")
        if row.overlap_ratio not in (float(o) for o in OVERLAP_GRID):
            add("invalid_overlap", row, f"overlap_ratio must be one of {list(OVERLAP_GRID)}")
        if not row.transcript:
            add("empty_transcript", row, "transcript must be non-empty")
        if not _valid_digest(row.mixture_digest) or not _valid_digest(row.clean_digest):
            add("invalid_digest", row, "digests must be lowercase [0-9a-f]{64}")
        if row.generator_version != GENERATOR_VERSION:
            add("generator_version", row, f"generator_version must be {GENERATOR_VERSION!r}")

        for fname in ("enrollment_audio", "mixture_audio", "clean_target_audio"):
            if str(getattr(row, fname)).strip() in {"", "."}:
                add("empty_path", row, f"{fname} must be a non-empty path")

        # Speaker split leakage: a speaker may appear in only one split.
        for spk in _row_speakers(row):
            prev = speaker_splits.get(spk)
            if prev is None:
                speaker_splits[spk] = row.split
            elif prev != row.split:
                add("speaker_split_leakage", row,
                    f"speaker {spk!r} appears in split {prev!r} and {row.split!r}")

    return tuple(issues)


def assert_r5_manifest_safe(
    rows: Iterable[R5OracleRow], dataset_a_root: str | Path
) -> tuple[R5OracleRow, ...]:
    """Containment + validity guard run before output creation or ASR."""
    materialized = tuple(rows)
    root = Path(dataset_a_root).resolve(strict=False)
    for row in materialized:
        if not isinstance(row, R5OracleRow):
            raise ValueError("R5 manifest row must be an R5OracleRow")
        for fname in ("enrollment_audio", "mixture_audio", "clean_target_audio"):
            resolved = Path(getattr(row, fname)).resolve(strict=False)
            if _is_under(resolved, root):
                raise ValueError(
                    f"Dataset-A containment violation: {fname} for row "
                    f"{row.row_id!r} resolves under Dataset-A root {root}"
                )
    issues = validate_r5_manifest(materialized)
    if issues:
        details = "; ".join(f"{i.code} ({i.row_id or '<unknown>'}): {i.message}" for i in issues)
        raise ValueError(f"R5 manifest validation failed: {details}")
    return materialized


def write_r5_manifest(path: str | Path, rows: Iterable[R5OracleRow]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True) for r in rows]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_r5_manifest(path: str | Path) -> tuple[R5OracleRow, ...]:
    manifest_path = Path(path)
    rows: list[R5OracleRow] = []
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"R5 manifest {manifest_path} line {line_no} malformed: {exc.msg}"
                ) from exc
            try:
                rows.append(
                    R5OracleRow.from_dict(
                        value, base_dir=manifest_path.resolve(strict=False).parent
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"R5 manifest {manifest_path} line {line_no} is malformed: {exc}"
                ) from exc
    return tuple(rows)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Design profiles and complete-design validation.
# --------------------------------------------------------------------------- #

ALL_BUCKETS: tuple[tuple[float, float], ...] = tuple(
    (float(s), float(o)) for s in SNR_GRID for o in OVERLAP_GRID
)


@dataclass(frozen=True)
class DesignProfile:
    """Frozen expected shape of an R5 oracle set (full or smoke)."""

    name: str
    seeds: tuple[str, ...]
    per_bucket_per_split: int
    expected_splits: tuple[str, ...] = ("val", "test")
    expected_buckets: tuple[tuple[float, float], ...] = ALL_BUCKETS

    @property
    def rows_per_split(self) -> int:
        return len(self.expected_buckets) * self.per_bucket_per_split

    @property
    def rows_per_seed(self) -> int:
        return self.rows_per_split * len(self.expected_splits)


FULL_PROFILE = DesignProfile(name="full", seeds=("A", "B"), per_bucket_per_split=20)


def smoke_profile(
    *, seeds: Sequence[str] = ("A",), per_bucket_per_split: int = 2
) -> DesignProfile:
    """A smoke profile with its own expected count; never masquerades as full."""
    return DesignProfile(
        name="smoke", seeds=tuple(seeds), per_bucket_per_split=int(per_bucket_per_split)
    )


def validate_r5_complete_design(
    rows: Iterable[R5OracleRow],
    *,
    profile: DesignProfile,
    dataset_a_root: str | Path | None = None,
    check_files: bool = True,
    index: Mapping[str, object] | None = None,
) -> tuple[R5ManifestIssue, ...]:
    """Validate the frozen full/smoke design: counts, buckets, speaker hygiene,
    file existence, recomputed audio digests, and manifest/index agreement."""
    materialized = tuple(rows)
    issues: list[R5ManifestIssue] = []

    def add(code: str, row_id: str | None, message: str) -> None:
        issues.append(R5ManifestIssue(code, row_id, message))

    # Structural validity first (allowed values, digests format, etc.).
    issues.extend(validate_r5_manifest(materialized))

    # Dataset-A containment.
    if dataset_a_root is not None:
        root = Path(dataset_a_root).resolve(strict=False)
        for row in materialized:
            for fname in ("enrollment_audio", "mixture_audio", "clean_target_audio"):
                resolved = Path(getattr(row, fname)).resolve(strict=False)
                if _is_under(resolved, root):
                    add("dataset_a_containment", row.row_id,
                        f"{fname} resolves under Dataset-A root {root}")

    # Seeds present.
    seeds_present = tuple(sorted({r.seed for r in materialized}))
    if seeds_present != tuple(sorted(profile.seeds)):
        add("seeds", None,
            f"expected seeds {tuple(sorted(profile.seeds))}, got {seeds_present}")

    # Per-seed, per-split, per-bucket counts.
    for seed in profile.seeds:
        srows = [r for r in materialized if r.seed == seed]
        if len(srows) != profile.rows_per_seed:
            add("seed_row_count", None,
                f"seed {seed}: expected {profile.rows_per_seed} rows, got {len(srows)}")
        for split in profile.expected_splits:
            sprows = [r for r in srows if r.split == split]
            if len(sprows) != profile.rows_per_split:
                add("split_row_count", None,
                    f"seed {seed} split {split}: expected {profile.rows_per_split}, "
                    f"got {len(sprows)}")
            for b in profile.expected_buckets:
                cnt = sum(1 for r in sprows if (r.snr_db, r.overlap_ratio) == b)
                if cnt != profile.per_bucket_per_split:
                    add("bucket_count", None,
                        f"seed {seed} split {split} bucket {b}: expected "
                        f"{profile.per_bucket_per_split}, got {cnt}")

    # Speaker hygiene: val/test disjoint across target + interferer roles.
    val_speakers: set[str] = set()
    test_speakers: set[str] = set()
    for r in materialized:
        bucket = val_speakers if r.split == "val" else test_speakers
        bucket.add(r.target_speaker)
        if r.interferer_speaker is not None:
            bucket.add(r.interferer_speaker)
    leak = val_speakers & test_speakers
    if leak:
        add("speaker_split_leakage", None,
            f"speakers in both val and test: {sorted(leak)[:5]}")

    # Target utterance must not appear as a target in both val and test.
    val_targets = {r.target_speaker_id for r in materialized if r.split == "val"}
    test_targets = {r.target_speaker_id for r in materialized if r.split == "test"}
    target_leak = val_targets & test_targets
    if target_leak:
        add("target_utterance_split_leakage", None,
            f"target utterances in both val and test: {sorted(target_leak)[:5]}")

    # Target utterance distinct from enrollment utterance (same speaker allowed).
    for r in materialized:
        enroll_stem = Path(r.enrollment_audio).stem
        if r.target_speaker_id == enroll_stem:
            add("target_equals_enrollment", r.row_id,
                "target utterance must differ from enrollment utterance")

    # File existence + recomputed audio digests.
    if check_files:
        for r in materialized:
            for fname, dfield in (
                ("mixture_audio", "mixture_digest"),
                ("clean_target_audio", "clean_digest"),
            ):
                p = Path(getattr(r, fname))
                if not p.is_file():
                    add("missing_file", r.row_id, f"{fname} not found: {p}")
                    continue
                actual = sha256_file(p)
                if actual != getattr(r, dfield):
                    add("file_digest_mismatch", r.row_id,
                        f"{fname} digest mismatch: file={actual[:12]} "
                        f"manifest={getattr(r, dfield)[:12]}")

    # Manifest/index agreement: recompute manifest file digest vs index.
    if index is not None:
        for seed, info in index.get("seeds", {}).items():
            mp = Path(str(info["manifest_path"]))
            if not mp.is_file():
                add("index_manifest_missing", None, f"seed {seed}: {mp}")
                continue
            actual = sha256_file(mp)
            expected = info.get("manifest_digest")
            if actual != expected:
                add("index_manifest_digest_mismatch", None,
                    f"seed {seed}: manifest digest file={actual[:12]} index={str(expected)[:12]}")
            n = sum(1 for r in materialized if r.seed == seed)
            if n != info.get("row_count"):
                add("index_row_count_mismatch", None,
                    f"seed {seed}: rows={n} index={info.get('row_count')}")

    return tuple(issues)


def assert_r5_complete_design(
    rows: Iterable[R5OracleRow],
    *,
    profile: DesignProfile,
    dataset_a_root: str | Path | None = None,
    check_files: bool = True,
    index: Mapping[str, object] | None = None,
) -> tuple[R5OracleRow, ...]:
    materialized = tuple(rows)
    issues = validate_r5_complete_design(
        materialized, profile=profile, dataset_a_root=dataset_a_root,
        check_files=check_files, index=index,
    )
    if issues:
        details = "; ".join(f"{i.code} ({i.row_id or '<unknown>'}): {i.message}" for i in issues)
        raise ValueError(f"R5 complete-design validation failed ({profile.name}): {details}")
    return materialized


def asr_config_digest(
    *,
    model: str,
    vad_model: str | None,
    punc_model: str | None,
    hotword: str | None,
    hotword_preset: str,
    language: str | None,
    use_itn: str | None,
    device: str,
    batch_size_s: int,
    trust_remote_code: bool,
) -> str:
    """Canonical SHA-256 of the frozen ASR inference configuration.

    Stored in every ASR record so resume can reject transcripts produced under a
    different configuration (model/VAD/punc/hotword/language/ITN/device-relevant
    options). Text normalization is fixed in code (not part of this digest).
    """
    cfg = {
        "model": model,
        "vad_model": vad_model or "",
        "punc_model": punc_model or "",
        "hotword": hotword or "",
        "hotword_preset": hotword_preset,
        "language": language or "",
        "use_itn": use_itn or "",
        "device": device,
        "batch_size_s": int(batch_size_s),
        "trust_remote_code": bool(trust_remote_code),
    }
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Analysis: CER, G_b, H_ASR, stratified bootstrap CI. Fail-closed: missing
# evidence or absent buckets raise rather than default to CER 0 / empty text.
# --------------------------------------------------------------------------- #


def _edit_totals(ref: str, hyp: str) -> tuple[int, int]:
    """Return (errors, ref_chars) for one (ref, hyp) pair using repo CER."""
    from xh202615.metrics import cer_stats
    st = cer_stats(ref, hyp)
    return st.errors, st.ref_chars


def _pooled_cer_totals(rows: list[dict], asr_by_row: Mapping[str, Mapping[str, str]],
                       path: str) -> tuple[int, int]:
    """Pooled (errors, ref_chars) over rows for one path; fails on missing evidence."""
    total_err = 0
    total_ref = 0
    for r in rows:
        rid = r["row_id"]
        evidence = asr_by_row.get(rid)
        if evidence is None or path not in evidence:
            raise ValueError(
                f"missing ASR evidence: row {rid!r} has no {path!r} transcript"
            )
        err, ref = _edit_totals(r["transcript"], evidence[path])
        total_err += err
        total_ref += ref
    return total_err, total_ref


def pooled_cer(rows: list[dict], asr_by_row: Mapping[str, Mapping[str, str]],
               path: str) -> float:
    err, ref = _pooled_cer_totals(rows, asr_by_row, path)
    return (err / ref) if ref else 0.0


def validate_weights(weights: Mapping[tuple[float, float], float],
                     *, expected_buckets: Sequence[tuple[float, float]]) -> None:
    """Weights must have exactly the expected bucket keys, be finite/nonnegative,
    and sum to one."""
    expected = set(expected_buckets)
    given = set(weights)
    if given != expected:
        raise ValueError(
            f"weight keys must match buckets exactly; missing={sorted(expected - given)} "
            f"extra={sorted(given - expected)}"
        )
    total = 0.0
    for key in expected:
        v = float(weights[key])
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"weight {key} must be finite and nonnegative, got {v}")
        total += v
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"weights must sum to 1.0, got {total}")


def _group_by_bucket(rows: list[dict]) -> dict[tuple[float, float], list[dict]]:
    groups: dict[tuple[float, float], list[dict]] = {}
    for r in rows:
        key = (float(r["snr_db"]), float(r["overlap_ratio"]))
        groups.setdefault(key, []).append(r)
    return groups


def compute_h_asr(
    rows: list[dict],
    asr_by_row: Mapping[str, Mapping[str, str]],
    *,
    weights: Mapping[tuple[float, float], float] | None = None,
    expected_buckets: Sequence[tuple[float, float]] = ALL_BUCKETS,
) -> dict:
    """Compute bucket gaps G_b = CER_mix,b - CER_clean,b (signed) and
    H_ASR = 0.5 * sum_b(w_b * G_b).

    Fail-closed: every weighted bucket must have >=1 row, every row must have
    both mixture and clean evidence, and weights must be valid. Missing evidence
    or absent buckets raise rather than fabricate a score.
    """
    w = dict(weights) if weights is not None else dict(BUCKET_WEIGHTS)
    validate_weights(w, expected_buckets=expected_buckets)
    groups = _group_by_bucket(rows)
    buckets: dict[tuple[float, float], dict] = {}
    h_sum = 0.0
    for key in sorted(w):
        if key not in groups or not groups[key]:
            raise ValueError(f"absent weighted bucket {key}: no rows")
        brows = groups[key]
        cer_mix = pooled_cer(brows, asr_by_row, "mixture")
        cer_clean = pooled_cer(brows, asr_by_row, "clean")
        gap = cer_mix - cer_clean  # signed; negative gaps preserved
        weight = w[key]
        h_sum += weight * gap
        buckets[key] = {
            "snr_db": key[0],
            "overlap_ratio": key[1],
            "cer_mix": cer_mix,
            "cer_clean": cer_clean,
            "gap": gap,
            "weight": weight,
            "count": len(brows),
        }
    return {
        "buckets": buckets,
        "h_asr": 0.5 * h_sum,
        "weighted_gap_sum": h_sum,
        "n_rows": len(rows),
    }


def bootstrap_h_asr_ci(
    rows: list[dict],
    asr_by_row: Mapping[str, Mapping[str, str]],
    *,
    n_boot: int = 2000,
    weights: Mapping[tuple[float, float], float] | None = None,
    expected_buckets: Sequence[tuple[float, float]] = ALL_BUCKETS,
    rng: np.random.Generator | None = None,
) -> dict:
    """Stratified bootstrap 95% CI for H_ASR.

    Resamples rows WITH REPLACEMENT inside each (snr, overlap) bucket
    independently, so all weighted buckets remain represented in every
    replicate. Per-bucket pooled (errors, ref_chars) totals are recomputed on
    each replicate. Returns {point, mean, ci_low, ci_high, n_boot, n_rows,
    bucket_sizes}.
    """
    w = dict(weights) if weights is not None else dict(BUCKET_WEIGHTS)
    validate_weights(w, expected_buckets=expected_buckets)
    groups = _group_by_bucket(rows)
    for key in w:
        if key not in groups or not groups[key]:
            raise ValueError(f"absent weighted bucket {key}: no rows")
    point = compute_h_asr(rows, asr_by_row, weights=w,
                          expected_buckets=expected_buckets)["h_asr"]
    rng = rng or np.random.default_rng(20260806)
    bucket_keys = sorted(w)
    bucket_sizes = {k: len(groups[k]) for k in bucket_keys}
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        h_sum = 0.0
        for key in bucket_keys:
            brows = groups[key]
            idx = rng.integers(0, len(brows), size=len(brows))
            sampled = [brows[i] for i in idx]
            err_m, ref_m = _pooled_cer_totals(sampled, asr_by_row, "mixture")
            err_c, ref_c = _pooled_cer_totals(sampled, asr_by_row, "clean")
            cer_mix = (err_m / ref_m) if ref_m else 0.0
            cer_clean = (err_c / ref_c) if ref_c else 0.0
            h_sum += w[key] * (cer_mix - cer_clean)
        boot[b] = 0.5 * h_sum
    return {
        "point": float(point),
        "mean": float(np.mean(boot)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n_boot": int(n_boot),
        "n_rows": int(len(rows)),
        "bucket_sizes": bucket_sizes,
    }


def validate_asr_evidence(
    manifest_rows: Sequence[R5OracleRow],
    asr_records: Sequence[Mapping[str, object]],
    *,
    recheck_files: bool = True,
) -> tuple[dict[str, dict[str, str]], str]:
    """Validate ASR records against the verified manifest.

    Fail-closed: every manifest row must have exactly one successful ``mixture``
    and one successful ``clean_target`` record with matching metadata
    (seed/split/snr/overlap/transcript), matching current manifest digest,
    matching ASR-time audio digest, a consistent ASR config digest across all
    records, and (unless ``recheck_files`` is False) a recomputed current-file
    digest. No duplicates, no unexpected records, no ASR errors. Returns
    ``(asr_by_row, config_digest)`` where ``asr_by_row`` maps
    ``row_id -> {"mixture": text, "clean": text}`` (the ``clean_target`` role is
    mapped to ``clean`` at this validated boundary).
    """
    manifest_by_row = {r.row_id: r for r in manifest_rows}
    seen: dict[tuple[str, str], dict] = {}
    configs: set[str] = set()
    for rec in asr_records:
        rid = str(rec.get("row_id", ""))
        role = str(rec.get("path_role", ""))
        if role not in ("mixture", "clean_target"):
            raise ValueError(f"unexpected path_role {role!r} for row {rid!r}")
        key = (rid, role)
        if key in seen:
            raise ValueError(f"duplicate ASR record for {rid!r}/{role}")
        seen[key] = dict(rec)
        if rid not in manifest_by_row:
            raise ValueError(f"unexpected ASR record for unknown row {rid!r}/{role}")
        if rec.get("error"):
            raise ValueError(f"ASR error for {rid!r}/{role}: {rec['error']}")
        if not rec.get("digest_ok"):
            raise ValueError(f"digest_ok is false for {rid!r}/{role}")
        cd = rec.get("config_digest")
        if not _valid_digest(cd):
            raise ValueError(f"missing/invalid config_digest for {rid!r}/{role}")
        configs.add(str(cd))
    if len(configs) > 1:
        raise ValueError(f"inconsistent ASR config digests across records: {sorted(configs)}")
    config_digest = next(iter(configs)) if configs else ""

    asr_by_row: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        rid = row.row_id
        for role, dfield, apath in (
            ("mixture", "mixture_digest", row.mixture_audio),
            ("clean_target", "clean_digest", row.clean_target_audio),
        ):
            rec = seen.get((rid, role))
            if rec is None:
                raise ValueError(f"missing ASR evidence: {rid!r}/{role}")
            manifest_digest = getattr(row, dfield)
            if rec.get("manifest_digest") != manifest_digest:
                raise ValueError(
                    f"manifest_digest mismatch for {rid!r}/{role}: "
                    f"record={rec.get('manifest_digest')} manifest={manifest_digest}"
                )
            if rec.get("digest") != manifest_digest:
                raise ValueError(
                    f"ASR-time audio digest mismatch for {rid!r}/{role}: "
                    f"record={rec.get('digest')} manifest={manifest_digest}"
                )
            # metadata consistency
            for fname, expected in (
                ("seed", row.seed), ("split", row.split),
                ("transcript", row.transcript),
            ):
                if str(rec.get(fname)) != str(expected):
                    raise ValueError(
                        f"{fname} mismatch for {rid!r}/{role}: "
                        f"{rec.get(fname)!r} != {expected!r}"
                    )
            for fname, expected in (("snr_db", row.snr_db), ("overlap_ratio", row.overlap_ratio)):
                if float(rec.get(fname)) != float(expected):
                    raise ValueError(
                        f"{fname} mismatch for {rid!r}/{role}: "
                        f"{rec.get(fname)} != {expected}"
                    )
            if recheck_files:
                p = Path(apath)
                if not p.is_file():
                    raise ValueError(f"missing audio file for {rid!r}/{role}: {p}")
                if sha256_file(p) != manifest_digest:
                    raise ValueError(f"current file digest mismatch for {rid!r}/{role}")
            asr_text = rec.get("asr_text")
            if not isinstance(asr_text, str):
                raise ValueError(
                    f"missing/non-string asr_text for {rid!r}/{role}: "
                    f"{type(asr_text).__name__}"
                )
            role_key = "clean" if role == "clean_target" else "mixture"
            asr_by_row.setdefault(rid, {})[role_key] = asr_text
    return asr_by_row, config_digest


# Decision gates (Overall ceiling for CER-only recovery).
GATE_BOUNDARIES = (0.020, 0.040)
GATE_LABELS = {
    "close": "recommend closing custom TSE; switch to ASR/LM/domain work",
    "mini": "permit only a pretrained or very short TSE mini-pilot",
    "eligible": "scratch TSE becomes eligible (not automatically approved)",
}


def gate_band(h_asr: float) -> str:
    if h_asr < GATE_BOUNDARIES[0]:
        return "close"
    if h_asr < GATE_BOUNDARIES[1]:
        return "mini"
    return "eligible"


def branch_recommendation(per_seed: list[dict]) -> dict:
    """Branch recommendation under the gates.

    Inconclusive if ANY of: a seed's 95% CI crosses a gate boundary, the seeds
    occupy different gate bands, or the seeds' CIs do not overlap. Does not rely
    only on point-estimate crossing or CI overlap.
    """
    bands = [gate_band(r["h_asr_test"]) for r in per_seed]
    ci_crosses = False
    crossed = []
    for r in per_seed:
        ci_low = r["ci_test"]["ci_low"]
        ci_high = r["ci_test"]["ci_high"]
        for b in GATE_BOUNDARIES:
            if ci_low < b < ci_high:
                ci_crosses = True
                crossed.append((r["seed"], b))
    different_bands = len(set(bands)) > 1
    ci_overlap = True
    if len(per_seed) >= 2:
        a, b = per_seed[0]["ci_test"], per_seed[1]["ci_test"]
        ci_overlap = not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"])
    inconclusive = bool(ci_crosses or different_bands or not ci_overlap)
    if inconclusive or not bands:
        return {
            "recommendation": "INCONCLUSIVE at the gate",
            "inconclusive": True,
            "reasons": {
                "ci_crosses_boundary": ci_crosses, "crossed": crossed,
                "different_bands": different_bands, "bands": bands,
                "ci_overlap": ci_overlap,
            },
        }
    return {
        "recommendation": GATE_LABELS[bands[0]],
        "inconclusive": False,
        "reasons": {
            "ci_crosses_boundary": False, "crossed": [],
            "different_bands": False, "bands": bands, "ci_overlap": ci_overlap,
        },
    }
