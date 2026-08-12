"""Build the frozen R12 60/20/20 stratified group split manifest.

This CLI reuses the R11 canonical label-free JSONL loader
(``scripts.r11_pvad_oracle_oof.load_canonical_rows``) to read the ordered sample
IDs, then loads the private label and group mappings from separate JSON files.
It validates exact ID-set equality and value types across the three inputs before
creating the manifest, so the split is never constructed from mismatched sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.r11_pvad_oracle_oof import load_canonical_rows
from xh202615.r10_selector import _finite_or_nan, _load_text_by_id, _read_jsonl, _wav_duration_sec
from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_ids(path: Path) -> list[str]:
    """Return ordered unique IDs from a canonical label-free JSONL file."""
    rows = load_canonical_rows(path)
    return [row.id for row in rows]


def _reject_duplicate_keys(path: Path, pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r} in {path}")
        seen.add(key)
        result[key] = value
    return result


def _load_json_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON object mapping and reject duplicate keys."""
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(
            handle,
            object_pairs_hook=lambda pairs: _reject_duplicate_keys(path, pairs),
        )
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _validate_label_value(sid: str, value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"label for id {sid!r} must be str or None")


def _validate_group_value(sid: str, value: object) -> str:
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"group for id {sid!r} must be a nonempty string")


def prepare_r12_split(
    canonical_input_jsonl: str | Path,
    labels_path: str | Path,
    groups_path: str | Path,
    output: str | Path,
) -> dict:
    """Build and write the R12 split manifest from the three source files.

    Returns a summary dict with the source file paths, row counts, role counts,
    source digests, and the manifest SHA-256.
    """

    canonical_input_jsonl = Path(canonical_input_jsonl).expanduser().resolve(strict=True)
    labels_path = Path(labels_path).expanduser().resolve(strict=True)
    groups_path = Path(groups_path).expanduser().resolve(strict=True)
    output_path = Path(output).expanduser().resolve(strict=False)

    ids_in_order = _load_ids(canonical_input_jsonl)
    ids_set = set(ids_in_order)
    labels_raw = _load_json_mapping(labels_path)
    groups_raw = _load_json_mapping(groups_path)

    if set(labels_raw) != ids_set:
        raise ValueError(
            f"labels IDs do not match canonical JSONL IDs: "
            f"jsonl={len(ids_in_order)} labels={len(labels_raw)}"
        )
    if set(groups_raw) != ids_set:
        raise ValueError(
            f"groups IDs do not match canonical JSONL IDs: "
            f"jsonl={len(ids_in_order)} groups={len(groups_raw)}"
        )

    labels = {}
    groups = {}
    for sid in ids_in_order:
        labels[sid] = _validate_label_value(sid, labels_raw[sid])
        groups[sid] = _validate_group_value(sid, groups_raw[sid])

    manifest = build_r12_split(ids_in_order, labels, groups)
    write_r12_split(output_path, manifest)

    loaded = load_r12_split(output_path, expected_ids=ids_in_order)
    role_counts: dict[str, int] = {}
    for role in loaded.roles_by_id.values():
        role_counts[role] = role_counts.get(role, 0) + 1

    source_digests = {
        "canonical_input_jsonl": _sha256_hex(canonical_input_jsonl.read_bytes()),
        "labels": _sha256_hex(labels_path.read_bytes()),
        "groups": _sha256_hex(groups_path.read_bytes()),
    }

    return {
        "canonical_input_jsonl": str(canonical_input_jsonl),
        "labels": str(labels_path),
        "groups": str(groups_path),
        "output": str(output_path),
        "row_count": len(ids_in_order),
        "role_counts": role_counts,
        "source_digests": source_digests,
        "manifest_sha256": loaded.manifest_sha256,
    }


def build_canonical_input(
    candidate_fusion: str | Path,
    tse_asr: str | Path,
    audio_map: str | Path,
    r3_predictions: str | Path,
    group_manifest: str | Path,
    output: str | Path,
) -> dict:
    """Project the four existing R10 candidate sources into label-free JSONL.

    The group manifest is consumed only to establish the exact ID order and
    source coverage; labels are never written to the canonical output.
    """

    source_paths = {
        "candidate_fusion": Path(candidate_fusion).expanduser().resolve(strict=True),
        "tse_asr": Path(tse_asr).expanduser().resolve(strict=True),
        "audio_map": Path(audio_map).expanduser().resolve(strict=True),
        "r3_predictions": Path(r3_predictions).expanduser().resolve(strict=True),
        "group_manifest": Path(group_manifest).expanduser().resolve(strict=True),
    }
    manifest = json.loads(source_paths["group_manifest"].read_text(encoding="utf-8-sig"))
    manifest_rows = manifest.get("rows") if isinstance(manifest, dict) else None
    if not isinstance(manifest_rows, list) or not manifest_rows:
        raise ValueError("group manifest must contain nonempty rows")
    ids: list[str] = []
    groups: dict[str, str] = {}
    for item in manifest_rows:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("group manifest row has invalid id")
        sid = item["id"]
        if sid in groups:
            raise ValueError(f"duplicate group manifest id {sid}")
        group = item.get("wake_component")
        if not isinstance(group, str) or not group:
            raise ValueError(f"group manifest row has invalid group for {sid}")
        ids.append(sid)
        groups[sid] = group
    ids = sorted(ids, key=lambda value: int(value) if value.isdigit() else value)

    fusion_by_id = {str(item.get("id")): item for item in _read_jsonl(source_paths["candidate_fusion"])}
    audio_by_id = {str(item.get("id")): item for item in _read_jsonl(source_paths["audio_map"])}
    tse_by_id = _load_text_by_id(source_paths["tse_asr"], "text", "recognition_text")
    r3_by_id = _load_text_by_id(source_paths["r3_predictions"], "recognition_text", "text")
    if set(fusion_by_id) != set(ids) or set(audio_by_id) != set(ids) or set(tse_by_id) != set(ids) or set(r3_by_id) != set(ids):
        raise ValueError("R10 candidate sources do not exactly cover group manifest IDs")
    rows: list[dict[str, object]] = []
    for sid in ids:
        fusion = fusion_by_id[sid]
        candidates = fusion.get("candidate_texts", {})
        r3_text = r3_by_id[sid]
        primary_text = str(candidates.get("primary", ""))
        energy_text = str(candidates.get("energy", ""))
        tse_text = tse_by_id[sid]
        audio = audio_by_id[sid]
        original_audio = audio.get("original_command_audio")
        duration = _wav_duration_sec(original_audio) if original_audio else math.nan
        audio_features = {
            "presence_score": _finite_or_nan(audio.get("presence_score")),
            "enhanced_cosine": _finite_or_nan(audio.get("enhanced_cosine")),
            "mixture_cosine": _finite_or_nan(audio.get("mixture_cosine")),
            "max_cosine": _finite_or_nan(audio.get("max_cosine")),
            "latency_ms": _finite_or_nan(audio.get("latency_ms")),
            "cmd_duration_sec": duration,
            "cmd_rms": math.nan,
        }
        source_digest = hashlib.sha256(json.dumps(
            [sid, r3_text, primary_text, energy_text, tse_text],
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest()
        rows.append({
            "id": sid,
            "split": "unknown",
            "r3_text": r3_text,
            "primary_text": primary_text,
            "energy_text": energy_text,
            "tse_text": tse_text,
            "audio_features": audio_features,
            "source_digest": source_digest,
        })
    output_path = Path(output).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True, separators=(",", ":")) + "\n")
    return {
        "output": str(output_path),
        "row_count": len(rows),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_digests": {name: _sha256_hex(path.read_bytes()) for name, path in source_paths.items()},
        "output_sha256": _sha256_hex(output_path.read_bytes()),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-input-jsonl", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--groups", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> dict:
    args = parse_args(argv)
    summary = prepare_r12_split(
        args.canonical_input_jsonl,
        args.labels,
        args.groups,
        args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()

