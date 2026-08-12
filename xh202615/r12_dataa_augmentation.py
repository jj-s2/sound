"""Deterministic, train-only command-audio augmentation for R12."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

from .r12_dataa_augmented_split import AugmentedInternalSplitManifest


_SEED = 20260812
_RATE = 16_000
_HEADROOM = float(10 ** (-1 / 20))


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class LineageRow:
    id: str
    parent_id: str
    augmentation_id: str
    role: str
    group: str
    source_split: str
    command_audio_sha256: str
    wake_audio_sha256: str
    wakeup_audio: str
    command_audio: str
    parameters: Mapping[str, float]


@dataclass(frozen=True)
class AugmentationSummary:
    dataset_root: Path
    lineage_path: Path
    exclusions_path: Path
    lineage_digest: str
    row_count: int
    exclusion_count: int


def augmentation_rng(parent_id: str, augmentation_id: str, seed: int = _SEED) -> np.random.Generator:
    raw = hashlib.sha256(f"{seed}\0{parent_id}\0{augmentation_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(raw[:8], "big"))


def _read_audio(path: Path) -> np.ndarray:
    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != _RATE:
        raise ValueError(f"expected {_RATE} Hz, got {rate}")
    mono = np.mean(waveform, axis=1, dtype=np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError("audio is empty or non-finite")
    return np.ascontiguousarray(mono)


def _headroom(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if not np.isfinite(audio).all() or peak == 0.0:
        if peak == 0.0 and audio.size:
            return audio.astype(np.float32, copy=False)
        raise ValueError("augmentation is non-finite or empty")
    if peak > _HEADROOM:
        audio = audio * (_HEADROOM / peak)
    return np.asarray(audio, dtype=np.float32)


def _gain(audio: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    db = float(rng.uniform(-3.0, 3.0))
    return audio * float(10 ** (db / 20.0)), db


def augment_a(audio: np.ndarray, rate: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    """Apply deterministic speed perturbation then amplitude gain."""
    if rate != _RATE:
        raise ValueError(f"expected {_RATE} Hz")
    speed = float(rng.choice(np.asarray([0.95, 1.05])))
    size = max(1, int(round(audio.size / speed)))
    source = np.linspace(0.0, 1.0, num=audio.size, endpoint=True)
    target = np.linspace(0.0, 1.0, num=size, endpoint=True)
    shifted = np.interp(target, source, audio).astype(np.float32)
    shifted, gain_db = _gain(shifted, rng)
    return _headroom(shifted), {"speed": speed, "gain_db": gain_db}


def augment_b(audio: np.ndarray, rate: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    """Apply deterministic gain and white noise at a bounded SNR."""
    if rate != _RATE:
        raise ValueError(f"expected {_RATE} Hz")
    gained, gain_db = _gain(audio, rng)
    snr_db = float(rng.uniform(18.0, 25.0))
    rms = float(np.sqrt(np.mean(np.square(gained, dtype=np.float64))))
    noise = rng.standard_normal(gained.size).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
    if rms > 0.0 and noise_rms > 0.0:
        noise *= float((rms / (10 ** (snr_db / 20.0))) / noise_rms)
    else:
        noise.fill(0.0)
    return _headroom(gained + noise), {"gain_db": gain_db, "snr_db": snr_db}


def _raw_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_split in ("pos", "neg"):
        for number, line in enumerate((root / f"{source_split}.jsonl").read_text(encoding="utf-8-sig").splitlines(), 1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{source_split}.jsonl:{number} must be an object")
            sample_id, wake, command = raw.get("id"), raw.get("wakeup_audio"), raw.get("command_audio")
            if not all(isinstance(value, str) and value for value in (sample_id, wake, command)):
                raise ValueError(f"{source_split}.jsonl:{number} has invalid input fields")
            rows.append({"id": sample_id, "source_split": source_split, "wakeup_audio": wake, "command_audio": command})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("raw Dataset-A has duplicate IDs")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_lineage(rows: Mapping[str, LineageRow]) -> None:
    seen: set[tuple[str, str]] = set()
    for sample_id, row in rows.items():
        if sample_id != row.id or not row.parent_id or row.augmentation_id not in {"original", "aug_a", "aug_b"}:
            raise ValueError("invalid lineage identity")
        if row.augmentation_id != "original" and row.role != "train":
            raise ValueError("augmentation children must belong to train")
        if (row.parent_id, row.augmentation_id) in seen:
            raise ValueError("duplicate lineage parent/augmentation")
        seen.add((row.parent_id, row.augmentation_id))
        if len(row.command_audio_sha256) != 64 or len(row.wake_audio_sha256) != 64:
            raise ValueError("lineage digest is invalid")


def load_lineage(path: Path) -> dict[str, LineageRow]:
    result: dict[str, LineageRow] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        raw = json.loads(line)
        try:
            row = LineageRow(**raw)
        except TypeError as exc:
            raise ValueError(f"invalid lineage row {number}") from exc
        if row.id in result:
            raise ValueError("duplicate lineage ID")
        result[row.id] = row
    validate_lineage(result)
    return result


def build_augmented_dataset(
    dataset_root: Path, split: AugmentedInternalSplitManifest, output_root: Path
) -> AugmentationSummary:
    """Build a Dataset-A-shaped root using raw rows plus train-only children."""
    dataset_root, output_root = Path(dataset_root).resolve(strict=True), Path(output_root).resolve(strict=False)
    if output_root == dataset_root or dataset_root in output_root.parents:
        raise ValueError("output root must not be inside raw Dataset-A")
    raw_rows = _raw_rows(dataset_root)
    ids = {row["id"] for row in raw_rows}
    if ids != set(split.roles_by_id) or ids != set(split.groups_by_id):
        raise ValueError("raw rows do not exactly match split IDs")
    output_root.mkdir(parents=True, exist_ok=False)
    for source_split in ("pos", "neg"):
        (output_root / source_split).mkdir()
    lineage: dict[str, LineageRow] = {}
    exclusions: list[dict[str, str]] = []
    output_rows: dict[str, list[dict[str, str]]] = {"pos": [], "neg": []}
    for raw in raw_rows:
        sample_id, source_split = raw["id"], raw["source_split"]
        role, group = split.roles_by_id[sample_id], split.groups_by_id[sample_id]
        wake_path, command_path = dataset_root / raw["wakeup_audio"], dataset_root / raw["command_audio"]
        wake_digest, command_digest = _sha_file(wake_path), _sha_file(command_path)
        original = LineageRow(sample_id, sample_id, "original", role, group, source_split, command_digest, wake_digest, raw["wakeup_audio"], raw["command_audio"], {})
        lineage[sample_id] = original
        output_rows[source_split].append({"id": sample_id, "split": source_split, "wakeup_audio": raw["wakeup_audio"], "command_audio": raw["command_audio"]})
        if role != "train":
            continue
        try:
            source_audio = _read_audio(command_path)
            for variant, transform in (("aug_a", augment_a), ("aug_b", augment_b)):
                child_id = f"{sample_id}__{variant}"
                augmented, parameters = transform(source_audio, _RATE, augmentation_rng(sample_id, variant))
                relative = Path(source_split) / f"cmd-{child_id}.wav"
                destination = output_root / relative
                sf.write(destination, augmented, _RATE, subtype="FLOAT")
                child = LineageRow(child_id, sample_id, variant, role, group, source_split, _sha_file(destination), wake_digest, raw["wakeup_audio"], str(relative), parameters)
                lineage[child_id] = child
                output_rows[source_split].append({"id": child_id, "split": source_split, "wakeup_audio": raw["wakeup_audio"], "command_audio": str(relative)})
        except (OSError, RuntimeError, ValueError) as exc:
            for variant in ("aug_a", "aug_b"):
                exclusions.append({"parent_id": sample_id, "augmentation_id": variant, "reason": str(exc)})
    validate_lineage(lineage)
    for source_split, values in output_rows.items():
        _write_jsonl(output_root / f"{source_split}.jsonl", values)
    lineage_path, exclusions_path = output_root / "augmentation_manifest.jsonl", output_root / "excluded_augmented_rows.jsonl"
    with lineage_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in lineage.values():
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    with exclusions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in exclusions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest_rows = [asdict(row) for row in lineage.values()]
    return AugmentationSummary(output_root, lineage_path, exclusions_path, hashlib.sha256(_canonical(digest_rows)).hexdigest(), len(lineage), len(exclusions))
