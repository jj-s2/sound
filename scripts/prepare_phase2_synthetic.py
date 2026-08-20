"""Build a deterministic speaker-disjoint synthetic AISHELL-1 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.training_data import (
    TrainingManifestRow,
    assert_valid_training_manifest,
)


GENERATOR_VERSION = "aishell-synthetic-overlap-v2"
TARGET_SAMPLE_RATE = 16_000
_SPEAKER_PATTERN = re.compile(r"^S.+", re.IGNORECASE)


def _is_under(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _transcript_path(root: Path) -> Path | None:
    preferred = root / "transcript" / "aishell_transcript_v0.8.txt"
    if preferred.is_file():
        return preferred
    matches = sorted(root.rglob("aishell_transcript_v0.8.txt"))
    return matches[0] if matches else None


def _read_transcripts(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    transcripts = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t", maxsplit=1)
            if len(parts) != 2:
                parts = stripped.split(maxsplit=1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError(
                    f"Malformed AISHELL transcript line {line_number} in {path}; "
                    "expected utterance ID followed by text"
                )
            transcripts[parts[0].strip()] = parts[1].strip()
    return transcripts


def _speaker_for_audio(path: Path, root: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    for parent in relative.parents:
        if parent.name and _SPEAKER_PATTERN.fullmatch(parent.name):
            return parent.name
    return None


def discover_aishell_utterances(root: str | Path) -> tuple[dict, ...]:
    """Discover transcribed AISHELL utterances beneath an extracted corpus root."""

    corpus_root = Path(root).expanduser().resolve(strict=False)
    if not corpus_root.is_dir():
        raise ValueError(f"AISHELL root does not exist or is not a directory: {corpus_root}")

    transcript_file = _transcript_path(corpus_root)
    transcripts = _read_transcripts(transcript_file)
    utterances = []
    for audio_path in sorted(corpus_root.rglob("*.wav")):
        speaker_id = _speaker_for_audio(audio_path, corpus_root)
        utterance_id = audio_path.stem
        text = transcripts.get(utterance_id)
        if speaker_id is None or text is None:
            continue
        relative_parts = audio_path.relative_to(corpus_root).parts
        split = "unknown"
        for candidate in ("train", "dev", "test", "val"):
            if candidate in {part.lower() for part in relative_parts}:
                split = candidate
                break
        utterances.append(
            {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "split": split,
                "audio_path": audio_path.resolve(strict=False),
                "text": text,
            }
        )

    if not utterances:
        transcript_hint = (
            f"Transcript file used: {transcript_file}."
            if transcript_file is not None
            else "No transcript/aishell_transcript_v0.8.txt file was found."
        )
        raise ValueError(
            f"No transcribed AISHELL WAV utterances were found under {corpus_root}. "
            "Expected WAV files below a speaker directory named S* and matching transcript IDs. "
            + transcript_hint
        )
    return tuple(utterances)


def split_speakers(
    speaker_ids: Iterable[str],
    *,
    seed: int = 20260804,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, str]:
    """Assign each unique speaker deterministically to exactly one split."""

    if not math.isfinite(val_fraction) or not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be finite and in [0, 1)")
    if not math.isfinite(test_fraction) or not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be finite and in [0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    speakers = sorted({str(speaker_id).strip() for speaker_id in speaker_ids})
    if any(not speaker_id for speaker_id in speakers):
        raise ValueError("speaker_ids must not contain empty IDs")
    random.Random(seed).shuffle(speakers)

    def fraction_count(fraction: float) -> int:
        if not speakers or fraction == 0.0:
            return 0
        return max(1, int(len(speakers) * fraction))

    val_count = fraction_count(val_fraction)
    test_count = fraction_count(test_fraction)
    while val_count + test_count > max(0, len(speakers) - 1):
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1

    assignments = {}
    for index, speaker_id in enumerate(speakers):
        if index < val_count:
            split = "val"
        elif index < val_count + test_count:
            split = "test"
        else:
            split = "train"
        assignments[speaker_id] = split
    return assignments


def _stable_int(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _ranking_identity(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("utterance_id", value))
    return str(value)


def _ranked(values: Iterable, seed: int, *parts: object) -> list:
    return sorted(
        values,
        key=lambda value: (
            _stable_int(seed, *parts, _ranking_identity(value)),
            _ranking_identity(value),
        ),
    )


def _load_mono_16k(path: Path) -> np.ndarray:
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"Could not read AISHELL audio {path}: {exc}") from exc
    if sample_rate <= 0 or audio.shape[0] == 0:
        raise ValueError(f"AISHELL audio is empty or has an invalid sample rate: {path}")
    mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"AISHELL audio contains non-finite samples: {path}")
    if sample_rate != TARGET_SAMPLE_RATE:
        output_length = max(1, int(round(len(mono) * TARGET_SAMPLE_RATE / sample_rate)))
        source_positions = np.arange(output_length, dtype=np.float64) * (
            sample_rate / TARGET_SAMPLE_RATE
        )
        mono = np.interp(
            source_positions,
            np.arange(len(mono), dtype=np.float64),
            mono.astype(np.float64),
            left=float(mono[0]),
            right=float(mono[-1]),
        ).astype(np.float32)
    return mono


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _normalize_schedule(
    values: Iterable[float | None],
    *,
    name: str,
    allow_none: bool,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float | None, ...]:
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{name} must contain at least one value")
    for value in normalized:
        if value is None:
            if not allow_none:
                raise ValueError(f"{name} values must be finite")
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(
            float(value)
        ):
            raise ValueError(f"{name} values must be finite")
        numeric = float(value)
        if minimum is not None and numeric < minimum:
            raise ValueError(f"{name} values must be at least {minimum}")
        if maximum is not None and numeric > maximum:
            raise ValueError(f"{name} values must be at most {maximum}")
    return tuple(None if value is None else float(value) for value in normalized)


def _scheduled(
    values: tuple[float | None, ...], seed: int, row_id: str, name: str
) -> float | None:
    return values[_stable_int(seed, row_id, name) % len(values)]


def _repeat_or_crop(audio: np.ndarray, length: int, token: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("requested audio length must be positive")
    if len(audio) == 0:
        raise ValueError("augmentation audio must not be empty")
    if len(audio) >= length:
        start = token % (len(audio) - length + 1)
        return audio[start : start + length].astype(np.float32, copy=False)
    offset = token % len(audio)
    repeated = np.resize(audio, length + offset)
    return repeated[offset : offset + length].astype(np.float32, copy=False)


def _align(
    interferer: np.ndarray,
    length: int,
    token: int,
    *,
    requested_overlap: float,
) -> tuple[np.ndarray, float]:
    if not 0.0 < requested_overlap <= 1.0:
        raise ValueError("requested_overlap must be in (0, 1]")
    overlap_samples = max(1, int(round(length * requested_overlap)))
    overlap_samples = min(length, overlap_samples)
    crop = _repeat_or_crop(interferer, overlap_samples, token)
    aligned = np.zeros(length, dtype=np.float32)
    start = _stable_int(token, "placement") % (length - overlap_samples + 1)
    aligned[start : start + overlap_samples] = crop
    return aligned, overlap_samples / length


def _apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if _rms(audio) == 0.0:
        return audio.astype(np.float32, copy=False)
    if _rms(rir) == 0.0:
        raise ValueError("RIR audio is silent")
    normalized_rir = rir.astype(np.float64) / max(float(np.max(np.abs(rir))), 1e-8)
    full_length = len(audio) + len(normalized_rir) - 1
    fft_length = 1 << (full_length - 1).bit_length()
    convolved = np.fft.irfft(
        np.fft.rfft(audio.astype(np.float64), fft_length)
        * np.fft.rfft(normalized_rir, fft_length),
        fft_length,
    )[:full_length]
    direct_index = int(np.argmax(np.abs(normalized_rir)))
    segment = convolved[direct_index : direct_index + len(audio)]
    if len(segment) < len(audio):
        segment = np.pad(segment, (0, len(audio) - len(segment)))
    target_rms = _rms(audio)
    segment_rms = _rms(segment)
    if segment_rms == 0.0:
        raise ValueError("RIR convolution produced silence")
    return (segment * (target_rms / segment_rms)).astype(np.float32)


def _colored_noise(length: int, seed: int, row_id: str) -> np.ndarray:
    rng = np.random.default_rng(_stable_int(seed, row_id, "colored-noise"))
    white = rng.standard_normal(length).astype(np.float64)
    colored = np.convolve(white, np.array([1.0, 0.85, 0.6], dtype=np.float64), mode="same")
    return colored.astype(np.float32)


def _audio_assets(root: Path, *, kind: str) -> tuple[Path, ...]:
    if not root.is_dir():
        raise ValueError(f"{kind} root does not exist or is not a directory: {root}")
    candidates = tuple(sorted(path.resolve(strict=False) for path in root.rglob("*.wav")))
    if not candidates:
        raise ValueError(f"No WAV files found below {kind} root: {root}")
    if kind == "rir":
        filtered = tuple(
            path
            for path in candidates
            if any("rir" in part.lower() for part in path.relative_to(root).parts)
        )
    elif kind == "noise":
        filtered = tuple(
            path
            for path in candidates
            if any("noise" in part.lower() for part in path.relative_to(root).parts)
        )
    else:
        raise ValueError(f"unknown asset kind: {kind}")
    return filtered or candidates


def _split_assets(paths: Iterable[Path], *, seed: int, kind: str) -> dict[str, tuple[Path, ...]]:
    ordered = tuple(_ranked(paths, seed, "augmentation-asset", kind))
    assignments = split_speakers((str(path) for path in ordered), seed=_stable_int(seed, kind))
    return {
        split: tuple(path for path in ordered if assignments[str(path)] == split)
        for split in ("train", "val", "test")
    }


def _select_asset(
    split_assets: dict[str, tuple[Path, ...]],
    *,
    split: str,
    seed: int,
    row_id: str,
    role: str,
) -> Path | None:
    choices = split_assets.get(split, ())
    if not choices:
        return None
    return choices[_stable_int(seed, row_id, role) % len(choices)]


def _mix(
    target: np.ndarray,
    interferer: np.ndarray,
    *,
    target_present: bool,
    seed: int,
    row_id: str,
    snr_db: float | None,
    sir_db: float | None,
    requested_overlap: float,
    noise_audio: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    aligned, overlap_ratio = _align(
        interferer,
        len(target),
        _stable_int(seed, row_id, "alignment"),
        requested_overlap=requested_overlap if target_present else 1.0,
    )
    if _rms(aligned) == 0.0:
        raise ValueError(f"Interferer audio for row {row_id} is silent")

    if target_present:
        if _rms(target) == 0.0:
            raise ValueError(f"Target audio for row {row_id} is silent")
        if sir_db is not None:
            desired_ratio = 10.0 ** (sir_db / 20.0)
            aligned = aligned * (_rms(target) / (desired_ratio * _rms(aligned)))
        speech = target.astype(np.float64) + aligned.astype(np.float64)
    else:
        speech = aligned.astype(np.float64)
        overlap_ratio = 0.0

    mixture = speech
    if snr_db is not None:
        noise = (
            _repeat_or_crop(noise_audio, len(target), _stable_int(seed, row_id, "noise"))
            if noise_audio is not None
            else _colored_noise(len(target), seed, row_id)
        ).astype(np.float64)
        noise_rms = _rms(noise)
        signal_rms = _rms(speech)
        if signal_rms == 0.0:
            raise ValueError(f"Speech mixture for row {row_id} is silent")
        noise *= signal_rms / (10.0 ** (snr_db / 20.0) * noise_rms)
        mixture = speech + noise

    peak = float(np.max(np.abs(mixture)))
    if peak > 0.98:
        mixture = mixture * (0.98 / peak)
    return mixture.astype(np.float32), float(overlap_ratio)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, TARGET_SAMPLE_RATE, subtype="PCM_16")


def _write_manifest(path: Path, rows: Iterable[TrainingManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def _finite_optional(value: float | None, name: str) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{name} must be finite or None")


def build_synthetic_manifest(
    aishell_root: str | Path,
    output_root: str | Path,
    *,
    max_speakers_per_split: int = 4,
    utterances_per_speaker: int = 2,
    seed: int = 20260804,
    snr_db: float | None = 5.0,
    sir_db: float | None = 0.0,
    snr_values: Iterable[float | None] | None = None,
    sir_values: Iterable[float | None] | None = None,
    overlap_values: Iterable[float] = (1.0,),
    rir_root: str | Path | None = None,
    noise_root: str | Path | None = None,
    reverb_probability: float = 0.0,
    forbidden_roots: Iterable[str | Path] = (),
) -> tuple[TrainingManifestRow, ...]:
    """Render positive and target-absent synthetic rows and persist their manifest."""

    if max_speakers_per_split < 2:
        raise ValueError("max_speakers_per_split must be at least 2")
    if utterances_per_speaker < 2:
        raise ValueError("utterances_per_speaker must be at least 2")
    _finite_optional(snr_db, "snr_db")
    _finite_optional(sir_db, "sir_db")
    if not math.isfinite(float(reverb_probability)) or not 0.0 <= reverb_probability <= 1.0:
        raise ValueError("reverb_probability must be finite and in [0, 1]")
    normalized_snr = _normalize_schedule(
        (snr_db,) if snr_values is None else snr_values,
        name="snr_values",
        allow_none=True,
    )
    normalized_sir = _normalize_schedule(
        (sir_db,) if sir_values is None else sir_values,
        name="sir_values",
        allow_none=True,
    )
    normalized_overlap = _normalize_schedule(
        overlap_values,
        name="overlap_values",
        allow_none=False,
        minimum=0.0,
        maximum=1.0,
    )
    if any(value is None or float(value) <= 0.0 for value in normalized_overlap):
        raise ValueError("overlap_values must be in (0, 1]")

    source_root = Path(aishell_root).expanduser().resolve(strict=False)
    resolved_forbidden = tuple(
        Path(root).expanduser().resolve(strict=False) for root in forbidden_roots
    )
    for forbidden_root in resolved_forbidden:
        if _is_under(source_root, forbidden_root):
            raise ValueError(
                f"AISHELL input root {source_root} is under forbidden root {forbidden_root}; "
                "refusing to read evaluation-only data"
            )

    def resolve_asset_root(value: str | Path | None, kind: str) -> Path | None:
        if value is None:
            return None
        asset_root = Path(value).expanduser().resolve(strict=False)
        for forbidden_root in resolved_forbidden:
            if _is_under(asset_root, forbidden_root):
                raise ValueError(
                    f"{kind} root {asset_root} is under forbidden root {forbidden_root}; "
                    "refusing to read evaluation-only data"
                )
        return asset_root

    resolved_rir_root = resolve_asset_root(rir_root, "RIR")
    resolved_noise_root = resolve_asset_root(noise_root, "noise")
    rir_paths = (
        _audio_assets(resolved_rir_root, kind="rir") if resolved_rir_root is not None else ()
    )
    noise_paths = (
        _audio_assets(resolved_noise_root, kind="noise")
        if resolved_noise_root is not None
        else ()
    )
    global_assets = _split_assets(
        tuple(sorted(set(rir_paths) | set(noise_paths))),
        seed=seed,
        kind="augmentation",
    ) if (rir_paths or noise_paths) else {split: () for split in ("train", "val", "test")}
    rir_path_set = set(rir_paths)
    noise_path_set = set(noise_paths)
    rir_assets = {
        split: tuple(path for path in global_assets[split] if path in rir_path_set)
        for split in ("train", "val", "test")
    }
    noise_assets = {
        split: tuple(path for path in global_assets[split] if path in noise_path_set)
        for split in ("train", "val", "test")
    }

    utterances = discover_aishell_utterances(source_root)
    assignments = split_speakers(
        (utterance["speaker_id"] for utterance in utterances), seed=seed
    )
    source_counts = Counter(assignments.values())
    if all(source_counts.get(split, 0) < 2 for split in ("train", "val", "test")):
        assignments = split_speakers(
            (utterance["speaker_id"] for utterance in utterances),
            seed=seed,
            val_fraction=0.0,
            test_fraction=0.0,
        )
    by_speaker = defaultdict(list)
    for utterance in utterances:
        by_speaker[utterance["speaker_id"]].append(utterance)

    output = Path(output_root).expanduser().resolve(strict=False)
    for forbidden_root in resolved_forbidden:
        if _is_under(output, forbidden_root):
            raise ValueError(
                f"Synthetic output root {output} is under forbidden root {forbidden_root}; "
                "refusing to write evaluation-only data"
            )
    rows = []
    audio_cache: dict[Path, np.ndarray] = {}

    def load(path: Path) -> np.ndarray:
        if path not in audio_cache:
            audio_cache[path] = _load_mono_16k(path)
        return audio_cache[path]

    for split in ("train", "val", "test"):
        eligible = [
            speaker_id
            for speaker_id, assigned_split in assignments.items()
            if assigned_split == split
            and len(by_speaker[speaker_id]) >= utterances_per_speaker
        ]
        selected_speakers = _ranked(eligible, seed, split, "speaker")[
            :max_speakers_per_split
        ]
        if len(selected_speakers) < 2:
            continue

        selected_utterances = {
            speaker_id: _ranked(
                by_speaker[speaker_id], seed, split, speaker_id, "utterance"
            )[:utterances_per_speaker]
            for speaker_id in selected_speakers
        }
        for speaker_index, target_speaker in enumerate(selected_speakers):
            interferer_speaker = selected_speakers[
                (speaker_index + 1) % len(selected_speakers)
            ]
            targets = selected_utterances[target_speaker]
            interferers = selected_utterances[interferer_speaker]
            for utterance_index, target_info in enumerate(targets):
                enrollment_info = targets[(utterance_index + 1) % len(targets)]
                interferer_info = interferers[utterance_index % len(interferers)]
                target_audio = load(target_info["audio_path"])
                enrollment_audio = load(enrollment_info["audio_path"])
                interferer_audio = load(interferer_info["audio_path"])

                for target_present, kind in ((True, "positive"), (False, "target-absent")):
                    row_id = (
                        f"aishell-synthetic-{split}-{target_speaker.lower()}-"
                        f"{target_info['utterance_id'].lower()}-{kind}"
                    )
                    enrollment_path = output / "enrollment" / split / f"{row_id}.wav"
                    target_path = output / "target" / split / f"{row_id}.wav"
                    mixture_path = output / "mixture" / split / f"{row_id}.wav"
                    selected_snr = _scheduled(normalized_snr, seed, row_id, "snr")
                    selected_sir = _scheduled(normalized_sir, seed, row_id, "sir")
                    selected_overlap = float(
                        _scheduled(normalized_overlap, seed, row_id, "overlap")
                    )
                    should_reverb = (
                        _stable_int(seed, row_id, "reverb") / float(2**64)
                    ) < reverb_probability
                    rendered_target = target_audio
                    rendered_interferer = interferer_audio
                    if should_reverb:
                        target_rir_path = _select_asset(
                            rir_assets,
                            split=split,
                            seed=seed,
                            row_id=row_id,
                            role="target-rir",
                        )
                        interferer_rir_path = _select_asset(
                            rir_assets,
                            split=split,
                            seed=seed,
                            row_id=row_id,
                            role="interferer-rir",
                        )
                        if target_rir_path is not None:
                            rendered_target = _apply_rir(target_audio, load(target_rir_path))
                        if interferer_rir_path is not None:
                            rendered_interferer = _apply_rir(
                                interferer_audio, load(interferer_rir_path)
                            )
                    noise_path = _select_asset(
                        noise_assets,
                        split=split,
                        seed=seed,
                        row_id=row_id,
                        role="background-noise",
                    )
                    mixture, overlap_ratio = _mix(
                        rendered_target,
                        rendered_interferer,
                        target_present=target_present,
                        seed=seed,
                        row_id=row_id,
                        snr_db=selected_snr,
                        sir_db=selected_sir,
                        requested_overlap=selected_overlap,
                        noise_audio=load(noise_path) if noise_path is not None else None,
                    )
                    _write_wav(enrollment_path, enrollment_audio)
                    # Keep the required target_audio field safe for target-absent rows: a
                    # consumer must not be able to recover the enrolled target from a negative
                    # example and accidentally train on a leaked positive track.
                    target_track = rendered_target if target_present else np.zeros_like(target_audio)
                    _write_wav(target_path, target_track)
                    _write_wav(mixture_path, mixture)
                    rows.append(
                        TrainingManifestRow(
                            row_id=row_id,
                            split=split,
                            source=GENERATOR_VERSION,
                            enrollment_audio=enrollment_path,
                            target_audio=target_path,
                            mixture_audio=mixture_path,
                            target_speaker_id=f"aishell1:{target_speaker}",
                            interferer_speaker_id=f"aishell1:{interferer_speaker}",
                            target_present=target_present,
                            overlap_ratio=overlap_ratio,
                            snr_db=selected_snr,
                            sir_db=selected_sir if target_present else None,
                            text=target_info["text"] if target_present else None,
                            seed=seed,
                        )
                    )

    if not rows:
        raise ValueError(
            "No synthetic rows could be built. Each generated split needs at least two "
            f"speakers with at least {utterances_per_speaker} transcribed utterances each."
        )

    manifest_path = output / "manifest.jsonl"
    validated = assert_valid_training_manifest(
        rows,
        manifest_path=manifest_path,
        forbidden_roots=resolved_forbidden,
    )
    _write_manifest(manifest_path, validated)
    split_counts = Counter(assignments.values())
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "row_count": len(validated),
        "seed": seed,
        "source_root": str(source_root),
        "speaker_split_counts": {
            split: split_counts.get(split, 0) for split in ("train", "val", "test")
        },
        "augmentation": {
            "snr_values": list(normalized_snr),
            "sir_values": list(normalized_sir),
            "overlap_values": list(normalized_overlap),
            "reverb_probability": float(reverb_probability),
            "rir_root": str(resolved_rir_root) if resolved_rir_root is not None else None,
            "noise_root": str(resolved_noise_root) if resolved_noise_root is not None else None,
            "colored_noise_fallback": resolved_noise_root is None,
        },
        "asset_split_counts": {
            split: {"rir": len(rir_assets[split]), "noise": len(noise_assets[split])}
            for split in ("train", "val", "test")
        },
        "asset_split_ids": {
            split: [
                *(f"rir:{path}" for path in rir_assets[split]),
                *(f"noise:{path}" for path in noise_assets[split]),
            ]
            for split in ("train", "val", "test")
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validated


def _optional_float(value: str) -> float | None:
    if value.strip().lower() in {"none", "null", "off"}:
        return None
    return float(value)


def _optional_float_values(value: str) -> tuple[float | None, ...]:
    return tuple(_optional_float(item.strip()) for item in value.split(",") if item.strip())


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a public AISHELL-1 synthetic overlap training manifest"
    )
    parser.add_argument("--aishell-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-speakers-per-split", type=int, default=4)
    parser.add_argument("--utterances-per-speaker", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--snr-db", type=_optional_float_values, default=(5.0,))
    parser.add_argument("--sir-db", type=_optional_float_values, default=(0.0,))
    parser.add_argument("--overlap-ratio", type=_optional_float_values, default=(1.0,))
    parser.add_argument("--rir-root", default=None)
    parser.add_argument("--noise-root", default=None)
    parser.add_argument("--reverb-probability", type=float, default=0.0)
    parser.add_argument("--forbidden-root", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    rows = build_synthetic_manifest(
        args.aishell_root,
        args.output_root,
        max_speakers_per_split=args.max_speakers_per_split,
        utterances_per_speaker=args.utterances_per_speaker,
        seed=args.seed,
        snr_values=args.snr_db,
        sir_values=args.sir_db,
        overlap_values=args.overlap_ratio,
        rir_root=args.rir_root,
        noise_root=args.noise_root,
        reverb_probability=args.reverb_probability,
        forbidden_roots=args.forbidden_root,
    )
    default_manifest = Path(args.output_root).expanduser().resolve(strict=False) / "manifest.jsonl"
    manifest_path = (
        Path(args.manifest).expanduser().resolve(strict=False)
        if args.manifest is not None
        else default_manifest
    )
    if manifest_path != default_manifest:
        assert_valid_training_manifest(
            rows,
            manifest_path=manifest_path,
            forbidden_roots=args.forbidden_root,
        )
        _write_manifest(manifest_path, rows)
    print(f"Wrote {len(rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
