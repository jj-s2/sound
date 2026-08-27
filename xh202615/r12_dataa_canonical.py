"""Digest-safe label-free canonical rows for augmented R12 training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .r12_dataa_augmentation import LineageRow, load_lineage
from .r12_personal_vad import PERSONAL_VAD_FEATURE_SCHEMA


@dataclass(frozen=True)
class CanonicalBuildSummary:
    output: Path
    row_count: int
    output_sha256: str
    rows: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class SourceAttestationSummary:
    output: Path
    row_count: int
    output_sha256: str


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_audio_path(row: Mapping[str, object]) -> Path:
    """Return the original derived command source, never TSE-enhanced output."""
    for key in ("original_command_audio", "command_audio", "audio"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return Path(value)
    raise ValueError("source row has no original_command_audio or command_audio")


def attest_source_command_audio(
    lineage_path: Path, source_path: Path, output: Path
) -> SourceAttestationSummary:
    """Bind an inference JSONL to the exact command bytes in the lineage.

    TSE ASR records preserve ``original_command_audio`` alongside their enhanced
    waveform.  That field has priority, so the attestation proves candidate
    provenance from the derived source audio rather than confusing it with an
    intermediate enhanced file.
    """
    lineage = load_lineage(lineage_path)
    source = _read_unique_without_digest(source_path)
    if set(source) != set(lineage):
        raise ValueError("source IDs must exactly cover lineage IDs")
    rendered_rows: list[dict[str, object]] = []
    for sample_id, lineage_row in lineage.items():
        row = dict(source[sample_id])
        audio_path = _source_audio_path(row)
        if not audio_path.is_file():
            raise ValueError(f"source audio is missing for {sample_id}")
        if _sha_file(audio_path) != lineage_row.command_audio_sha256:
            raise ValueError(f"source command audio does not match lineage for {sample_id}")
        row["command_audio_sha256"] = lineage_row.command_audio_sha256
        rendered_rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rendered_rows
    )
    output.write_text(rendered, encoding="utf-8", newline="\n")
    return SourceAttestationSummary(output, len(rendered_rows), _sha_file(output))


def _read_unique_without_digest(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        raw = json.loads(line)
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), (str, int)):
            raise ValueError(f"{path}:{number} has invalid ID")
        sample_id = str(raw["id"])
        if sample_id in result:
            raise ValueError(f"duplicate ID {sample_id!r} in {path}")
        result[sample_id] = raw
    return result


def _read_unique(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        raw = json.loads(line)
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), (str, int)):
            raise ValueError(f"{path}:{number} has invalid ID")
        sample_id = str(raw["id"])
        if sample_id in result:
            raise ValueError(f"duplicate ID {sample_id!r} in {path}")
        digest = raw.get("command_audio_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{path}:{number} has invalid command_audio_sha256")
        result[sample_id] = raw
    return result


def _text(row: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _digest(values: object) -> str:
    encoded = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_augmented_canonical(
    lineage_path: Path,
    fusion_path: Path,
    tse_path: Path,
    audio_path: Path,
    r3_path: Path,
    pvad_manifest_path: Path,
    output: Path,
) -> CanonicalBuildSummary:
    """Join four regenerated sources only when all match lineage audio bytes."""
    lineage = load_lineage(lineage_path)
    source_maps = {
        "fusion": _read_unique(fusion_path), "tse": _read_unique(tse_path),
        "audio": _read_unique(audio_path), "r3": _read_unique(r3_path),
    }
    expected = set(lineage)
    for name, rows in source_maps.items():
        if set(rows) != expected:
            raise ValueError(f"{name} IDs must exactly cover lineage IDs")
    try:
        pvad_audio = json.loads(Path(pvad_manifest_path).read_text(encoding="utf-8-sig"))["source"]["per_id_audio_sha256"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("pVAD manifest has no per-ID audio provenance") from exc
    if not isinstance(pvad_audio, Mapping) or set(pvad_audio) != expected:
        raise ValueError("pVAD manifest IDs must exactly cover lineage IDs")
    output_rows: list[dict[str, object]] = []
    for sample_id, source in lineage.items():
        for name, rows in source_maps.items():
            if rows[sample_id]["command_audio_sha256"] != source.command_audio_sha256:
                raise ValueError(f"{name} command_audio_sha256 does not match lineage for {sample_id}")
        pvad_entry = pvad_audio[sample_id]
        if not isinstance(pvad_entry, Mapping) or pvad_entry.get("command_sha256") != source.command_audio_sha256:
            raise ValueError(f"pVAD command_audio_sha256 does not match lineage for {sample_id}")
        if source.role != "train" and source.augmentation_id != "original":
            raise ValueError("validation/internal_test rows must be original")
        fusion, tse, audio, r3 = (source_maps[name][sample_id] for name in ("fusion", "tse", "audio", "r3"))
        candidates = fusion.get("candidate_texts")
        if not isinstance(candidates, Mapping):
            raise ValueError("fusion candidate_texts must be an object")
        primary, energy = _text(candidates, "primary"), _text(candidates, "energy")
        r3_text, tse_text = _text(r3, "recognition_text", "text"), _text(tse, "text", "recognition_text")
        features = {
            "presence_score": _finite(audio.get("presence_score")),
            "enhanced_cosine": _finite(audio.get("enhanced_cosine")),
            "mixture_cosine": _finite(audio.get("mixture_cosine")),
            "max_cosine": _finite(audio.get("max_cosine")),
            "latency_ms": _finite(audio.get("latency_ms")),
            "cmd_duration_sec": _finite(audio.get("cmd_duration_sec")),
            "cmd_rms": _finite(audio.get("cmd_rms")),
        }
        features.update({name: _finite(audio.get(name)) for name in PERSONAL_VAD_FEATURE_SCHEMA})
        digest = _digest([sample_id, source.command_audio_sha256, r3_text, primary, energy, tse_text])
        output_rows.append({
            "id": sample_id, "split": source.role, "r3_text": r3_text,
            "primary_text": primary, "energy_text": energy, "tse_text": tse_text,
            "audio_features": features, "source_digest": digest,
        })
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=True) + "\n" for row in output_rows)
    output.write_text(rendered, encoding="utf-8", newline="\n")
    return CanonicalBuildSummary(output, len(output_rows), hashlib.sha256(output.read_bytes()).hexdigest(), tuple(output_rows))
