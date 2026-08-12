"""Resumable, label-free FireRed pVAD aggregate feature cache."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import secrets
import stat
import sys
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .artifact_publish import ArtifactContract, _create_unique_staging, _directory_identity, _lexists, _rename_no_replace, _rename_no_replace_native, publish_text_package
from .data import FIELD_ALIASES, load_split
from .firered_model_assets import _EXPECTED_INPUTS, _EXPECTED_POSITIONAL_OUTPUTS, _RAW_SNAPSHOT_FILES, _REQUIRED_DEPENDENCY_VERSIONS, _UPSTREAM_IDENTITY, _ALLOWED_SYMBOLIC_DIMENSIONS, FireRedModelPaths, download_and_verify_model
from .firered_pvad import PVAD_GATE_FEATURE_SCHEMA, FireRedPvadRuntime, PvadRuntimeConfig

_ARTIFACT_KIND = "r11_e2_firered_cache"
_SCHEMA_VERSION = "v1"
_FEATURES, _MANIFEST, _REPORT = "pvad_features.jsonl", "pvad_manifest.json", "pvad_report.md"
_CONTRACT = ArtifactContract(_ARTIFACT_KIND, _SCHEMA_VERSION, (_FEATURES, _MANIFEST, _REPORT), (_MANIFEST,))
_SCHEMA_SHA256 = "610c7e711fda490405a66a01e5ca6e7b01bf230c00333d891ebbaf20140e270f"
_RESUME_PREFIX = "context-"
_CONTEXT_IDENTITY = "context_identity.json"
_AUDIT_COMMON = {"elapsed_seconds", "audio_seconds", "rtf", "peak_rss_delta_bytes", "dropped_tail_samples", "extraction_phase", "onnx_provider", "ecapa_device"}
# Frozen cross-device gate features; audit-only tail padding remains schema-validated evidence.
_PVAD_PARITY_FEATURE_ALLOWLIST: tuple[str, ...] = (
    "frame_count",
    "analyzed_duration_sec",
    "command_duration_sec",
    "raw_mean",
    "raw_std",
    "raw_min",
    "raw_max",
    "raw_q10",
    "raw_q25",
    "raw_q50",
    "raw_q75",
    "raw_q90",
    "raw_q95",
    "raw_fraction_ge_0_1",
    "raw_fraction_ge_0_3",
    "raw_fraction_ge_0_5",
    "raw_fraction_ge_0_7",
    "raw_fraction_ge_0_9",
    "ema_mean",
    "ema_std",
    "ema_min",
    "ema_max",
    "ema_q10",
    "ema_q25",
    "ema_q50",
    "ema_q75",
    "ema_q90",
    "ema_q95",
    "ema_fraction_ge_0_1",
    "ema_fraction_ge_0_3",
    "ema_fraction_ge_0_5",
    "ema_fraction_ge_0_7",
    "ema_fraction_ge_0_9",
    "ema_longest_run_ge_0_3_frames",
    "ema_longest_run_ge_0_3_seconds",
    "ema_first_crossing_ge_0_3_frame",
    "ema_last_crossing_ge_0_3_frame",
    "ema_active_span_ge_0_3_frames",
    "ema_transitions_ge_0_3",
    "ema_longest_run_ge_0_5_frames",
    "ema_longest_run_ge_0_5_seconds",
    "ema_first_crossing_ge_0_5_frame",
    "ema_last_crossing_ge_0_5_frame",
    "ema_active_span_ge_0_5_frames",
    "ema_transitions_ge_0_5",
    "ema_longest_run_ge_0_7_frames",
    "ema_longest_run_ge_0_7_seconds",
    "ema_first_crossing_ge_0_7_frame",
    "ema_last_crossing_ge_0_7_frame",
    "ema_active_span_ge_0_7_frames",
    "ema_transitions_ge_0_7",
    "enrollment_duration_sec",
    "embedding_norm_before",
    "embedding_norm_after",
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _ordered(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object, label: str) -> float | int:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a native finite JSON number")
    return value


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _load_object(text: str, label: str) -> dict[str, object]:
    try:
        value = json.loads(text, object_pairs_hook=_object_pairs, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _id_key(value: str) -> tuple[int, int | str, str]:
    return (0, int(value), value) if value.isdigit() else (1, value, value)


def _valid_id(value: object) -> str:
    if type(value) is int:
        value = str(value)
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value or "\0" in value:
        raise ValueError("Dataset-A id must be a nonempty traversal-safe string or integer")
    return value


def _first(row: Mapping[str, object], field: str) -> object | None:
    return next((row[key] for key in FIELD_ALIASES[field] if key in row), None)


def _raw_rows(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    sources: dict[str, str] = {}
    for split in ("pos", "neg"):
        path = root / f"{split}.jsonl"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Dataset-A split must be a regular non-symlink file: {path}")
        sources[split] = _file_sha256(path)
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                raise ValueError(f"Dataset-A JSONL {path}:{number} must not contain an empty row")
            row = _load_object(line, f"Dataset-A JSONL {path}:{number}")
            sample_id = _valid_id(_first(row, "id"))
            wake, command = _first(row, "wakeup_audio"), _first(row, "command_audio")
            if sample_id in rows or not isinstance(wake, str) or not wake or not isinstance(command, str) or not command:
                raise ValueError(f"Dataset-A id {sample_id!r} is duplicate or missing wake/command audio")
            rows[sample_id] = {"wakeup_audio": wake, "command_audio": command}
    if not rows:
        raise ValueError("Dataset-A contains no rows")
    return rows, sources


def _safe_audio(root: Path, relative: str, label: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"{label} audio path must be relative to Dataset-A root")
    path = root / relative
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"{label} audio path escapes Dataset-A root") from exc
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} audio is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} audio must be a regular non-symlink file: {path}")
    return path


def _schema_digest() -> str:
    digest = _sha256((r"\n".join(PVAD_GATE_FEATURE_SCHEMA) + r"\n").encode("utf-8"))
    if digest != _SCHEMA_SHA256:
        raise ValueError("fixed feature schema digest disagrees with the frozen Task 3 constant")
    return digest


def _config(device: str) -> PvadRuntimeConfig:
    return PvadRuntimeConfig(onnx_provider="CPUExecutionProvider", ecapa_device=device)


def verify_existing_model(paths: FireRedModelPaths) -> FireRedModelPaths:
    if not Path(paths.root).exists():
        raise ValueError(f"model root does not exist: {paths.root}")
    return download_and_verify_model(Path(paths.root))


def _model_identity(paths: FireRedModelPaths) -> dict[str, object]:
    parsed = _load_object(paths.manifest.read_text(encoding="utf-8"), "model manifest")
    required = {"aggregate_sha256", "raw_sha256", "upstream", "onnx", "required_dependency_versions"}
    if not required <= set(parsed) or any(parsed[key] in ({}, None, "") for key in required):
        raise ValueError("verified model manifest is missing required identity fields")
    dependencies = parsed["required_dependency_versions"]
    if not isinstance(dependencies, Mapping) or not dependencies or any(not isinstance(key, str) or not key or not isinstance(value, str) or not value for key, value in dependencies.items()):
        raise ValueError("verified model manifest has invalid required_dependency_versions")
    return {"manifest_sha256": _file_sha256(paths.manifest), "aggregate_sha256": parsed["aggregate_sha256"], "raw_sha256": parsed["raw_sha256"], "upstream": parsed["upstream"], "onnx": parsed["onnx"], "required_dependency_versions": dict(dependencies)}


def _digest(value: object) -> str:
    return _sha256(_canonical(value).encode("utf-8"))


def _sha256_value(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("digest must be a lowercase SHA-256 hexadecimal string")
    return value


def _aggregate_digest(raw_sha256: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(raw_sha256):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_value(raw_sha256[relative_path])))
    return digest.hexdigest()


def _json_value(value: object) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) in (int, float):
        _finite(value, "provenance value")
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, Mapping) and all(isinstance(key, str) and key for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise ValueError("provenance value is outside the JSON domain")


def _runtime_config(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(asdict(_config("cpu"))):
        raise ValueError("runtime config has an invalid exact key contract")
    try:
        config = asdict(PvadRuntimeConfig(**dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime config has invalid values") from exc
    if config["onnx_provider"] != "CPUExecutionProvider":
        raise ValueError("runtime config must use the canonical CPU ONNX provider")
    return config


def _validate_model_identity(model: object) -> dict[str, object]:
    if not isinstance(model, Mapping) or set(model) != {"manifest_sha256", "aggregate_sha256", "raw_sha256", "upstream", "onnx", "required_dependency_versions", "identity_sha256"}:
        raise ValueError("model identity has an invalid exact key contract")
    _sha256_value(model["manifest_sha256"])
    _sha256_value(model["aggregate_sha256"])
    raw = model["raw_sha256"]
    onnx = model["onnx"]
    if not isinstance(raw, Mapping) or set(raw) != _RAW_SNAPSHOT_FILES or any(not isinstance(key, str) or not key for key in raw):
        raise ValueError("model raw digest domain is invalid")
    for value in raw.values():
        _sha256_value(value)
    if model["aggregate_sha256"] != _aggregate_digest(raw) or model["upstream"] != _UPSTREAM_IDENTITY or model["required_dependency_versions"] != _REQUIRED_DEPENDENCY_VERSIONS:
        raise ValueError("model identity disagrees with the pinned Task 2 contract")
    if not isinstance(onnx, Mapping) or set(onnx) != {"sample_rate_hz", "frame_samples", "probability_output_index", "mel_state_output_index", "gru_state_output_index", "inputs", "outputs"} or {key: onnx[key] for key in ("sample_rate_hz", "frame_samples", "probability_output_index", "mel_state_output_index", "gru_state_output_index")} != {"sample_rate_hz": 16000, "frame_samples": 160, "probability_output_index": 1, "mel_state_output_index": 2, "gru_state_output_index": 3}:
        raise ValueError("model ONNX contract is invalid")
    inputs = [{"name": name, "type": type_, "shape": list(shape)} for name, type_, shape in _EXPECTED_INPUTS]
    if onnx["inputs"] != inputs or not isinstance(onnx["outputs"], list) or len(onnx["outputs"]) < 4:
        raise ValueError("model ONNX contract is invalid")
    if len({input_['name'] for input_ in onnx["inputs"]}) != len(onnx["inputs"]):
        raise ValueError("model ONNX input names must be unique")
    for output in onnx["outputs"]:
        if not isinstance(output, Mapping) or set(output) != {"name", "type", "shape"} or not isinstance(output["name"], str) or not output["name"] or not isinstance(output["type"], str) or not output["type"] or not isinstance(output["shape"], list) or any((type(item) is not int and item not in _ALLOWED_SYMBOLIC_DIMENSIONS) for item in output["shape"]):
            raise ValueError("model ONNX output domain is invalid")
    if len({output["name"] for output in onnx["outputs"]}) != len(onnx["outputs"]):
        raise ValueError("model ONNX output names must be unique")
    if not all((onnx["outputs"][index]["type"], onnx["outputs"][index]["shape"]) == (expected_type, list(expected_shape)) for index, (_, expected_type, expected_shape) in _EXPECTED_POSITIONAL_OUTPUTS.items()):
        raise ValueError("model ONNX output contract is invalid")
    result = dict(model)
    if result["identity_sha256"] != _digest({key: result[key] for key in result if key != "identity_sha256"}):
        raise ValueError("model identity digest disagrees")
    return result


def _provenance(model: Mapping[str, object], source: Mapping[str, object], coverage: Mapping[str, object], config: Mapping[str, object]) -> dict[str, object]:
    return {"model": {key: value for key, value in model.items() if key != "identity_sha256"}, "source": dict(source), "coverage": dict(coverage), "feature_schema": list(PVAD_GATE_FEATURE_SCHEMA), "feature_schema_sha256": _schema_digest(), "runtime_config": dict(config)}


def _cpu_config(config: Mapping[str, object]) -> dict[str, object]:
    result = dict(config)
    result["ecapa_device"] = "cpu"
    return result


def _parity_provenance(manifest: Mapping[str, object], current: Mapping[str, object], *, cuda: bool) -> tuple[dict[str, object], dict[str, object]]:
    reference_config = manifest["runtime_config"]
    current_config = current["runtime_config"]
    if cuda:
        reference_config = _cpu_config(reference_config)
        current_config = _cpu_config(current_config)
    return _provenance(manifest["model"], manifest["source"], manifest["coverage"], reference_config), _provenance(current["model"], current["source"], current["coverage"], current_config)


def _overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(strict=False), right.resolve(strict=False)
    return os.path.commonpath((str(left), str(right))) in {str(left), str(right)}


def _regular_directory(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISDIR(mode) and not path.is_symlink()


def _validate_package(root: Path, selected: list[str] | None = None, *, cpu_only: bool = False) -> tuple[dict[str, object], dict[str, OrderedDict[str, float | int]], str]:
    if not _regular_directory(root):
        raise ValueError("parity/output root is not a recognizable cache package")
    children = list(root.iterdir())
    if {child.name for child in children} != {_FEATURES, _MANIFEST, _REPORT} or any(child.is_symlink() or not child.is_file() for child in children):
        raise ValueError("parity/output root is not a recognizable cache package")
    manifest_bytes = (root / _MANIFEST).read_bytes()
    manifest = _load_object(manifest_bytes.decode("utf-8"), "cache manifest")
    required = {"artifact_kind", "schema_version", "feature_schema", "feature_schema_sha256", "digest_algorithms", "coverage", "source", "model", "runtime_config", "runtime_config_sha256", "provider", "device", "environment", "reuse", "timing", "parity", "limit", "records_sha256", "per_id_record_sha256", "joined_state_sha256"}
    allowed = required | {"cuda"}
    if set(manifest) not in (required, allowed) or manifest_bytes.replace(b"\r\n", b"\n") != (_canonical(manifest) + "\n").encode("utf-8") or manifest.get("artifact_kind") != _ARTIFACT_KIND or manifest.get("schema_version") != _SCHEMA_VERSION or manifest.get("feature_schema") != list(PVAD_GATE_FEATURE_SCHEMA) or manifest.get("feature_schema_sha256") != _schema_digest():
        raise ValueError("parity/output root is not a recognizable cache package")
    if cpu_only and (manifest.get("provider") != "CPUExecutionProvider" or manifest.get("device") != "cpu"):
        raise ValueError("parity reference must be a CPU cache package")
    text = (root / _FEATURES).read_text(encoding="utf-8")
    rows: dict[str, OrderedDict[str, float | int]] = {}
    lines = text.splitlines()
    if text != "\n".join(lines) + "\n":
        raise ValueError("parity reference feature bytes are not canonical")
    for number, line in enumerate(lines, 1):
        record = _load_object(line, f"parity record {number}")
        if tuple(record) != ("id", "features") or not isinstance(record.get("id"), str) or record["id"] in rows or not isinstance(record.get("features"), Mapping):
            raise ValueError("malformed parity reference")
        rows[record["id"]] = _validate_values(record["features"])
    ids = list(rows)
    coverage = manifest.get("coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != {"selected", "source"} or not isinstance(coverage.get("selected"), Mapping) or not isinstance(coverage.get("source"), Mapping) or set(coverage["selected"]) != {"count", "ids", "id_sha256"} or set(coverage["source"]) != {"count", "ids", "id_sha256"} or coverage["selected"].get("ids") != ids or coverage["selected"].get("count") != len(ids) or coverage["selected"].get("id_sha256") != _digest(ids) or not isinstance(coverage["source"].get("ids"), list) or coverage["source"].get("count") != len(coverage["source"]["ids"]) or coverage["source"].get("id_sha256") != _digest(coverage["source"]["ids"]) or manifest.get("records_sha256") != _sha256(text.encode("utf-8")) or manifest.get("per_id_record_sha256") != {sample_id: _sha256(line.encode("utf-8")) for sample_id, line in zip(ids, lines)}:
        raise ValueError("parity reference coverage or digest disagrees")
    try:
        if ids != sorted(ids, key=_id_key) or len(set(ids)) != len(ids) or any(_valid_id(sample_id) != sample_id for sample_id in ids) or coverage["source"]["ids"] != sorted(coverage["source"]["ids"], key=_id_key) or len(set(coverage["source"]["ids"])) != len(coverage["source"]["ids"]) or any(_valid_id(sample_id) != sample_id for sample_id in coverage["source"]["ids"]) or not set(ids) <= set(coverage["source"]["ids"]):
            raise ValueError("coverage IDs are invalid")
    except (TypeError, ValueError):
        raise ValueError("parity reference coverage or digest disagrees") from None
    if selected is not None and ids != selected:
        raise ValueError("parity reference IDs disagree")
    source = manifest.get("source")
    model = manifest.get("model")
    environment = manifest.get("environment")
    if not isinstance(source, Mapping) or set(source) != {"jsonl_sha256", "per_id_audio_sha256", "projection_sha256"} or not isinstance(source["jsonl_sha256"], Mapping) or set(source["jsonl_sha256"]) != {"pos", "neg"} or not isinstance(source["per_id_audio_sha256"], Mapping) or set(source["per_id_audio_sha256"]) != set(ids) or source["projection_sha256"] != _digest({"jsonl_sha256": source["jsonl_sha256"], "per_id_audio_sha256": source["per_id_audio_sha256"]}) or not isinstance(model, Mapping) or set(model) != {"manifest_sha256", "aggregate_sha256", "raw_sha256", "upstream", "onnx", "required_dependency_versions", "identity_sha256"} or not isinstance(model["required_dependency_versions"], Mapping) or not model["required_dependency_versions"] or model["identity_sha256"] != _digest({key: model[key] for key in model if key != "identity_sha256"}) or not isinstance(environment, Mapping) or set(environment) != {"python", "platform", "observed_dependencies"} or not isinstance(environment["observed_dependencies"], Mapping):
        raise ValueError("parity/output root provenance is incomplete")
    try:
        for value in source["jsonl_sha256"].values():
            _sha256_value(value)
        _sha256_value(model["manifest_sha256"])
        _sha256_value(model["aggregate_sha256"])
        model = _validate_model_identity(model)
        if not isinstance(environment["python"], str) or not environment["python"] or not isinstance(environment["platform"], str) or not environment["platform"] or set(environment["observed_dependencies"]) != set(_observed_dependencies()) or any(value is not None and not isinstance(value, str) for value in environment["observed_dependencies"].values()):
            raise ValueError("environment domain is invalid")
        config = _runtime_config(manifest.get("runtime_config"))
    except (AttributeError, ValueError):
        raise ValueError("parity/output root provenance is invalid") from None
    if manifest.get("runtime_config_sha256") != _digest(config) or manifest.get("provider") != config["onnx_provider"] or manifest.get("device") != config["ecapa_device"]:
        raise ValueError("parity/output root runtime config disagrees")
    for sample_id, audio in source["per_id_audio_sha256"].items():
        if not isinstance(audio, Mapping) or set(audio) != {"wake_sha256", "command_sha256"} or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in audio.values()):
            raise ValueError("parity/output root source audio provenance is invalid")
    algorithms = manifest.get("digest_algorithms")
    expected_algorithms = {"feature_schema_sha256": "sha256(UTF-8 ordered schema names joined and terminated by literal backslash-n bytes)", "records_sha256": "sha256(UTF-8 canonical JSONL bytes)", "per_id_record_sha256": "sha256(UTF-8 canonical JSON record bytes)", "joined_state_sha256": "sha256(UTF-8 canonical JSON)", "source_audio_sha256": "sha256(raw wake/command audio bytes)", "source_projection_sha256": "sha256(UTF-8 canonical JSON source projection)", "model_sha256": "sha256(UTF-8 canonical JSON verified model identity)"}
    if algorithms != expected_algorithms:
        raise ValueError("parity/output root digest algorithms are incomplete")
    if manifest.get("joined_state_sha256") != _digest({sample_id: _resume_expected(sample_id, source["per_id_audio_sha256"][sample_id], model, config) for sample_id in ids}):
        raise ValueError("parity/output root joined state disagrees")
    _validate_manifest_domains(manifest, ids)
    if (root / _REPORT).read_text(encoding="utf-8") != _report(manifest):
        raise ValueError("parity/output root report disagrees with manifest")
    return manifest, rows, text


def _validate_manifest_domains(manifest: Mapping[str, object], ids: list[str]) -> None:
    for value in (manifest["records_sha256"], manifest["joined_state_sha256"], manifest["runtime_config_sha256"]):
        _sha256_value(value)
    if not isinstance(manifest["per_id_record_sha256"], Mapping) or set(manifest["per_id_record_sha256"]) != set(ids):
        raise ValueError("parity/output root record digest domain is invalid")
    for value in manifest["per_id_record_sha256"].values():
        _sha256_value(value)
    reuse, parity, limit, timing = manifest["reuse"], manifest["parity"], manifest["limit"], manifest["timing"]
    config = _runtime_config(manifest["runtime_config"])
    cuda = config["ecapa_device"].startswith("cuda")
    canonical_config = asdict(_config(config["ecapa_device"]))
    valid_parity = isinstance(parity, Mapping) and (parity == {"status": "not-run", "passed": None, "max_abs_feature_delta": None} or (parity.get("status") == "passed" and parity.get("passed") is True and type(parity.get("max_abs_feature_delta")) in (int, float) and math.isfinite(parity["max_abs_feature_delta"]) and 0 <= parity["max_abs_feature_delta"] <= 1e-4))
    coverage = manifest["coverage"]
    source_ids = coverage["source"]["ids"]
    expected_ids = source_ids if isinstance(limit, Mapping) and limit.get("value") is None else source_ids[: min(limit["value"], len(source_ids))] if isinstance(limit, Mapping) and type(limit.get("value")) is int else None
    valid_limit = isinstance(limit, Mapping) and set(limit) == {"value", "canonical", "reason"} and ((limit["value"] is None and limit["canonical"] is True and limit["reason"] is None and ids == source_ids) or (type(limit["value"]) is int and limit["value"] > 0 and limit["canonical"] is False and limit["reason"] == "explicit noncanonical partial cache" and ids == expected_ids))
    if config != canonical_config or not isinstance(reuse, Mapping) or set(reuse) != {"reused", "new"} or any(type(value) is not int or value < 0 for value in reuse.values()) or sum(reuse.values()) != len(ids) or not isinstance(parity, Mapping) or set(parity) != {"status", "passed", "max_abs_feature_delta"} or not valid_parity or (cuda and parity["status"] != "passed") or not valid_limit or not isinstance(timing, Mapping) or set(timing) != {"cold_elapsed_seconds", "warm_elapsed_seconds", "rtf", "peak_rss_delta_bytes", "cuda_peak_bytes"}:
        raise ValueError("parity/output root manifest domain is invalid")
    for name, percentile in timing.items():
        expected_count = len(ids) if name in {"rtf", "peak_rss_delta_bytes"} or (name == "cuda_peak_bytes" and cuda) else None
        if not isinstance(percentile, Mapping) or set(percentile) != {"count", "p50", "p95", "max"} or type(percentile["count"]) is not int or percentile["count"] < 0 or (expected_count is not None and percentile["count"] != expected_count) or (name == "cuda_peak_bytes" and not cuda and percentile["count"] != 0) or (percentile["count"] == 0 and any(percentile[key] is not None for key in ("p50", "p95", "max"))) or (percentile["count"] > 0 and any(type(percentile[key]) not in (int, float) or not math.isfinite(percentile[key]) or percentile[key] < 0 for key in ("p50", "p95", "max"))) or (percentile["count"] > 0 and not percentile["p50"] <= percentile["p95"] <= percentile["max"]):
            raise ValueError("parity/output root timing domain is invalid")
    if timing["cold_elapsed_seconds"]["count"] + timing["warm_elapsed_seconds"]["count"] != len(ids):
        raise ValueError("parity/output root timing coverage disagrees")
    if ("cuda" in manifest) != cuda:
        raise ValueError("parity/output root CUDA evidence is invalid")
    if cuda:
        evidence = manifest["cuda"]
        if not isinstance(evidence, Mapping) or set(evidence) != {"cuda_device_name", "cuda_driver", "cuda_runtime_version", "peak_bytes"} or not isinstance(evidence["cuda_device_name"], str) or not evidence["cuda_device_name"] or not isinstance(evidence["cuda_runtime_version"], str) or not evidence["cuda_runtime_version"] or not isinstance(evidence["cuda_driver"], Mapping) or set(evidence["cuda_driver"]) != {"status", "value"} or evidence["cuda_driver"].get("status") not in {"available", "unavailable"} or (evidence["cuda_driver"]["status"] == "available" and (not isinstance(evidence["cuda_driver"]["value"], str) or not evidence["cuda_driver"]["value"])) or (evidence["cuda_driver"]["status"] == "unavailable" and evidence["cuda_driver"]["value"] is not None) or evidence["peak_bytes"] != timing["cuda_peak_bytes"]:
            raise ValueError("parity/output root CUDA evidence is invalid")


def _preflight_output(path: Path) -> None:
    if os.path.lexists(path) and _validate_package(path) is None:
        raise AssertionError("unreachable")


def _lock(root: Path) -> tuple[Path, tuple[int, int]]:
    root.mkdir(parents=True, exist_ok=True)
    if not _regular_directory(root):
        raise ValueError("resume root must be a regular non-symlink directory")
    lock = root / ".resume.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"resume lock already exists: {lock}") from exc
    item = lock.lstat()
    return lock, (item.st_dev, item.st_ino)


def _unlock(lock: Path, identity: tuple[int, int]) -> None:
    try:
        item = lock.lstat()
        if (item.st_dev, item.st_ino) == identity and not lock.is_symlink():
            lock.rmdir()
    except OSError:
        pass


def _resume_name(sample_id: str) -> str:
    return "record-" + _sha256(sample_id.encode("utf-8"))[:32] + ".json"


def _context_name(model: Mapping[str, object], config: Mapping[str, object]) -> str:
    return _RESUME_PREFIX + _digest(_context_identity(model, config))


def _context_identity(model: Mapping[str, object], config: Mapping[str, object]) -> dict[str, object]:
    config = _runtime_config(config)
    return {"feature_schema": list(PVAD_GATE_FEATURE_SCHEMA), "feature_schema_sha256": _schema_digest(), "model": dict(model), "runtime_config": config, "runtime_config_sha256": _digest(config)}


def _validate_context(path: Path, model: Mapping[str, object] | None = None, config: Mapping[str, object] | None = None) -> None:
    identity_path = path / _CONTEXT_IDENTITY
    if not identity_path.is_file() or identity_path.is_symlink():
        raise ValueError(f"resume namespace identity is missing or unsafe: {path}")
    identity_bytes = identity_path.read_bytes()
    identity = _load_object(identity_bytes.decode("utf-8"), "resume namespace identity")
    try:
        _validate_model_identity(identity.get("model"))
    except ValueError:
        raise ValueError(f"resume namespace identity disagrees with its directory: {path}") from None
    if identity_bytes != (_canonical(identity) + "\n").encode("utf-8") or set(identity) != {"feature_schema", "feature_schema_sha256", "model", "runtime_config", "runtime_config_sha256"} or identity != _context_identity(identity.get("model", {}), identity.get("runtime_config", {})) or _context_name(identity["model"], identity["runtime_config"]) != path.name:
        raise ValueError(f"resume namespace identity disagrees with its directory: {path}")
    if model is not None and identity != _context_identity(model, config or {}):
        raise ValueError(f"resume namespace identity disagrees with current context: {path}")
    for child in path.iterdir():
        if child.name == _CONTEXT_IDENTITY:
            continue
        if re.fullmatch(r"\.record-[0-9a-f]{32}\.json\.[0-9a-f]{16}\.tmp", child.name):
            _validate_context_record(child, identity, child.name.removeprefix(".").rsplit(".", 2)[0])
            continue
        if child.is_symlink() or not child.is_file() or not re.fullmatch(r"record-[0-9a-f]{32}\.json", child.name):
            raise ValueError(f"foreign resume namespace state is preserved: {child}")
        _validate_context_record(child, identity, child.name)


def _create_namespace(root: Path, name: str, identity: Mapping[str, object]) -> Path:
    """Atomically publish a complete resume context under the caller-held lock."""
    namespace = root / name
    if os.path.lexists(namespace):
        if not _regular_directory(namespace):
            raise ValueError("resume namespace must be a regular non-symlink directory")
        _validate_context(namespace, identity["model"], identity["runtime_config"])
        return namespace
    staging, staging_identity = _create_unique_staging(root, f".{name}.staging")
    staging_bytes = (_canonical(identity) + "\n").encode("utf-8")
    published = False
    try:
        with (staging / _CONTEXT_IDENTITY).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(staging_bytes.decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        _rename_no_replace(staging, namespace)
        published = True
    except OSError:
        if not os.path.lexists(namespace) or not _regular_directory(namespace):
            raise ValueError("resume namespace publication collided with unsafe state") from None
        _validate_context(namespace, identity["model"], identity["runtime_config"])
    finally:
        if published:
            cleanup_error = "published staging path reappeared" if _lexists(staging) else None
        else:
            cleanup_error = _cleanup_namespace_staging(staging, staging_identity, staging_bytes)
        if cleanup_error is not None:
            raise RuntimeError("resume namespace staging cleanup failed: " + cleanup_error)
    _validate_context(namespace, identity["model"], identity["runtime_config"])
    return namespace


def _cleanup_namespace_staging(path: Path, identity: tuple[int, int], expected_bytes: bytes) -> str | None:
    """Quarantine then remove only the exact identity file created by this call."""
    if not _lexists(path):
        return f"owned directory disappeared before cleanup: {path}"
    quarantine = path.parent / f"{path.name}.cleanup.{secrets.token_hex(8)}"
    try:
        _rename_no_replace_native(path, quarantine)
    except OSError as exc:
        return f"could not quarantine owned directory {path}: {exc}"
    identity_file = quarantine / _CONTEXT_IDENTITY
    exact = _directory_identity(quarantine) == identity
    try:
        exact = exact and {child.name for child in quarantine.iterdir()} == {_CONTEXT_IDENTITY} and identity_file.is_file() and not identity_file.is_symlink() and identity_file.read_bytes() == expected_bytes
    except OSError:
        exact = False
    if not exact:
        if not _lexists(path):
            try:
                _rename_no_replace(quarantine, path)
                return f"unexpected directory was preserved at {path}"
            except OSError:
                pass
        return f"unexpected directory was preserved at {quarantine}"
    try:
        identity_file.unlink()
        quarantine.rmdir()
    except OSError as exc:
        return f"could not remove owned directory {quarantine}: {exc}"
    return None


def _validate_context_record(path: Path, identity: Mapping[str, object], expected_name: str) -> None:
    record_bytes = path.read_bytes()
    record = _load_object(record_bytes.decode("utf-8"), f"resume record {path}")
    sample_id = record.get("id")
    required = {"id", "input", "model", "runtime_config", "runtime_config_sha256", "feature_schema", "feature_schema_sha256", "values", "audit"}
    if record_bytes != (_ordered(record) + "\n").encode("utf-8") or set(record) != required or not isinstance(sample_id, str) or expected_name != _resume_name(sample_id) or record.get("model") != identity["model"] or record.get("runtime_config") != identity["runtime_config"] or record.get("runtime_config_sha256") != identity["runtime_config_sha256"] or record.get("feature_schema") != identity["feature_schema"] or record.get("feature_schema_sha256") != identity["feature_schema_sha256"] or not isinstance(record.get("input"), Mapping) or set(record["input"]) != {"wake_sha256", "command_sha256"} or not isinstance(record.get("values"), Mapping) or not isinstance(record.get("audit"), Mapping):
        raise ValueError(f"resume namespace contains a forged record: {path}")
    for value in record["input"].values():
        _sha256_value(value)
    _validate_values(record["values"])
    _validate_audit(record["audit"], PvadRuntimeConfig(**identity["runtime_config"]))


def _resume_expected(sample_id: str, input_state: Mapping[str, object], model: Mapping[str, object], config: Mapping[str, object]) -> dict[str, object]:
    config = _runtime_config(config)
    return {"id": sample_id, "input": {"wake_sha256": input_state["wake_sha256"], "command_sha256": input_state["command_sha256"]}, "model": dict(model), "runtime_config": config, "runtime_config_sha256": _digest(config), "feature_schema": list(PVAD_GATE_FEATURE_SCHEMA), "feature_schema_sha256": _schema_digest()}


def _validate_values(values: Mapping[str, object]) -> OrderedDict[str, float | int]:
    if tuple(values) != PVAD_GATE_FEATURE_SCHEMA:
        raise ValueError("features disagree with the fixed feature schema")
    return OrderedDict((name, _finite(values[name], f"feature {name}")) for name in PVAD_GATE_FEATURE_SCHEMA)


def _max_abs_feature_delta(reference: Mapping[str, object], current: Mapping[str, object]) -> float:
    return max(
        (abs(float(reference[name]) - float(current[name])) for name in _PVAD_PARITY_FEATURE_ALLOWLIST),
        default=0.0,
    )


def _validate_audit(audit: Mapping[str, object], config: PvadRuntimeConfig) -> dict[str, object]:
    expected = _AUDIT_COMMON | ({"cuda_peak_bytes"} if config.ecapa_device.startswith("cuda") else set())
    if not isinstance(audit, Mapping) or set(audit) != expected:
        raise ValueError("runtime audit has an invalid exact key contract")
    result: dict[str, object] = {}
    for key, value in audit.items():
        if key in {"extraction_phase", "onnx_provider", "ecapa_device"}:
            if not isinstance(value, str):
                raise ValueError(f"audit {key} must be a string")
            result[key] = value
        elif key in {"dropped_tail_samples", "peak_rss_delta_bytes", "cuda_peak_bytes"}:
            if type(value) is not int or value < 0:
                raise ValueError(f"audit {key} must be a nonnegative integer")
            result[key] = value
        else:
            numeric = _finite(value, f"audit {key}")
            if numeric < 0:
                raise ValueError(f"audit {key} must be nonnegative")
            result[key] = numeric
    if result["extraction_phase"] not in {"cold", "warm"} or result["onnx_provider"] != config.onnx_provider or result["ecapa_device"] != config.ecapa_device:
        raise ValueError("runtime audit provider/device/phase disagrees with runtime config")
    if result["audio_seconds"] <= 0 or not math.isclose(float(result["rtf"]), float(result["elapsed_seconds"]) / float(result["audio_seconds"]), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("runtime audit duration and RTF disagree")
    return result


def _validate_resume_record(path: Path, expected: Mapping[str, Mapping[str, object]], config: PvadRuntimeConfig, *, expected_name: str | None = None) -> tuple[str, OrderedDict[str, float | int], Mapping[str, object]]:
    record = _load_object(path.read_text(encoding="utf-8"), f"resume record {path}")
    sample_id = record.get("id")
    required = set(expected.get(sample_id, {})) | {"values", "audit"} if isinstance(sample_id, str) else set()
    if set(record) != required or not isinstance(sample_id, str) or (expected_name or path.name) != _resume_name(sample_id) or sample_id not in expected:
        raise ValueError(f"foreign resume record is preserved: {path}")
    if {key: record[key] for key in expected[sample_id]} != expected[sample_id] or not isinstance(record["values"], Mapping) or not isinstance(record["audit"], Mapping):
        raise ValueError(f"resume record identity mismatch is preserved: {path}")
    return sample_id, _validate_values(record["values"]), _validate_audit(record["audit"], config)


def _resume_records(root: Path, expected: Mapping[str, Mapping[str, object]], config: PvadRuntimeConfig) -> tuple[dict[str, OrderedDict[str, float | int]], list[Mapping[str, object]]]:
    result: dict[str, OrderedDict[str, float | int]] = {}
    audits: list[Mapping[str, object]] = []
    temp_pattern = re.compile(r"^\.record-[0-9a-f]{32}\.json\.[0-9a-f]{16}\.tmp$")
    for path in root.iterdir():
        if path.name == _CONTEXT_IDENTITY:
            continue
        if temp_pattern.fullmatch(path.name) and path.is_file() and not path.is_symlink():
            destination = root / path.name.removeprefix(".").rsplit(".", 2)[0]
            try:
                sample_id, _, _ = _validate_resume_record(path, expected, config, expected_name=destination.name)
            except ValueError as exc:
                # Filename similarity does not establish ownership; preserve unknown bytes.
                raise ValueError(f"unverified interrupted resume temp is preserved: {path}") from exc
            if _resume_name(sample_id) != destination.name:
                raise ValueError(f"unverified interrupted resume temp is preserved: {path}")
            if os.path.lexists(destination):
                if destination.read_bytes() != path.read_bytes():
                    raise ValueError(f"competing interrupted resume temp is preserved: {path}")
                path.unlink()
                if sample_id in result:
                    continue
            else:
                _rename_no_replace(path, destination)
            if sample_id in result:
                raise ValueError(f"duplicate interrupted resume temp is preserved: {path}")
            _, values, audit = _validate_resume_record(destination, expected, config)
            result[sample_id] = values
            audits.append(audit)
            continue
        if temp_pattern.fullmatch(path.name.removesuffix(".quarantined")) and path.name.endswith(".quarantined") and path.is_file() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file() or not path.name.endswith(".json"):
            raise ValueError(f"foreign resume state is preserved: {path}")
        sample_id, values, audit = _validate_resume_record(path, expected, config)
        if sample_id in result:
            continue
        result[sample_id] = values
        audits.append(audit)
    return result, audits


def _write_resume(root: Path, record: Mapping[str, object]) -> None:
    destination = root / _resume_name(str(record["id"]))
    if os.path.lexists(destination):
        raise ValueError(f"resume destination already exists and is preserved: {destination}")
    temporary = root / f".{destination.name}.{secrets.token_hex(8)}.tmp"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_ordered(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if temporary.is_symlink() or os.path.lexists(destination):
        raise ValueError("resume write encountered a symlink")
    _rename_no_replace(temporary, destination)


def _percentiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {"count": len(ordered), "p50": ordered[round((len(ordered) - 1) * .5)], "p95": ordered[round((len(ordered) - 1) * .95)], "max": ordered[-1]}


def _cuda_audit_adapter(device: str) -> object:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for CUDA audit evidence") from exc
    index = 0 if device == "cuda" else int(device.split(":", 1)[1])
    class Adapter:
        def reset_peak(self) -> None:
            with torch.cuda.device(index):
                torch.cuda.reset_peak_memory_stats()
        def peak_bytes(self) -> int:
            with torch.cuda.device(index):
                return int(torch.cuda.max_memory_allocated())
        def evidence(self) -> dict[str, object]:
            driver: str | None = None
            try:
                import pynvml
                pynvml.nvmlInit()
                driver = str(pynvml.nvmlSystemGetDriverVersion())
            except Exception:
                pass
            with torch.cuda.device(index):
                name = str(torch.cuda.get_device_name(index))
            return {"cuda_device_name": name, "cuda_driver": {"status": "available", "value": driver} if driver else {"status": "unavailable", "value": None}, "cuda_runtime_version": str(getattr(torch.version, "cuda", "unknown"))}
    return Adapter()


def _observed_dependencies() -> dict[str, str | None]:
    observed: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "soundfile", "onnxruntime", "onnxruntime-gpu", "speechbrain", "torch", "torchaudio", "huggingface-hub", "hyperpyyaml", "joblib", "packaging", "PyYAML", "ruamel.yaml", "ruamel.yaml.clib", "sentencepiece", "tqdm"):
        try:
            observed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed[name] = None
    return observed


def _report(manifest: Mapping[str, object]) -> str:
    coverage = manifest["coverage"]
    reuse = manifest["reuse"]
    parity = manifest["parity"]
    assert isinstance(coverage, Mapping) and isinstance(reuse, Mapping) and isinstance(parity, Mapping)
    selected = coverage["selected"]
    assert isinstance(selected, Mapping)
    return f"# FireRed pVAD Cache\n\n- Selected IDs: {selected['count']}\n- New records: {reuse['new']}\n- Reused records: {reuse['reused']}\n- Parity: {parity['status']}\n"


def build_pvad_cache(dataset_root: Path, model_paths: FireRedModelPaths, output_root: Path, *, resume_root: Path, ecapa_device: str, limit: int | None = None, parity_reference: Path | None = None) -> dict[str, Path]:
    """Build and atomically publish one complete label-free pVAD cache."""
    dataset_root, output_root, resume_root = Path(dataset_root), Path(output_root), Path(resume_root)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if any(_overlap(a, b) for a, b in ((output_root, resume_root), (output_root, dataset_root), (output_root, model_paths.root), (resume_root, dataset_root), (resume_root, model_paths.root), (dataset_root, model_paths.root))):
        raise ValueError("output, resume, model, and Dataset-A roots must not overlap")
    if os.path.lexists(output_root):
        _validate_package(output_root)
    raw, source_digests = _raw_rows(dataset_root)
    loaded = {sample.id for split in ("pos", "neg") for sample in load_split(dataset_root, split)}
    if loaded != set(raw):
        raise ValueError("Dataset-A loader disagrees with independently parsed raw IDs")
    selected = sorted(raw, key=_id_key)[:limit] if limit is not None else sorted(raw, key=_id_key)
    config = _config(ecapa_device)
    reference_root = Path(parity_reference).parent if parity_reference else None
    if config.ecapa_device.startswith("cuda") and parity_reference is None:
        raise ValueError("CUDA cache requires a complete CPU parity reference before inference")
    if reference_root is not None:
        if Path(parity_reference).name != _FEATURES or any(_overlap(reference_root, root) for root in (output_root, resume_root, dataset_root, model_paths.root)):
            raise ValueError("parity reference overlaps an unsafe root")
        _validate_package(reference_root, selected, cpu_only=config.ecapa_device.startswith("cuda"))
    verified = verify_existing_model(model_paths)
    model = _model_identity(verified)
    model = {**model, "identity_sha256": _digest(model)}
    config_state = asdict(config)
    config_digest = _digest(config_state)
    state: dict[str, dict[str, object]] = {}
    for sample_id in selected:
        wake = _safe_audio(dataset_root, raw[sample_id]["wakeup_audio"], "wake")
        command = _safe_audio(dataset_root, raw[sample_id]["command_audio"], "command")
        state[sample_id] = {"wake_sha256": _file_sha256(wake), "command_sha256": _file_sha256(command), "wake_path": wake, "command_path": command}
    expected = {sample_id: _resume_expected(sample_id, state[sample_id], model, config_state) for sample_id in selected}
    per_id_audio = {sample_id: {"wake_sha256": state[sample_id]["wake_sha256"], "command_sha256": state[sample_id]["command_sha256"]} for sample_id in selected}
    source = {"jsonl_sha256": source_digests, "per_id_audio_sha256": per_id_audio, "projection_sha256": _digest({"jsonl_sha256": source_digests, "per_id_audio_sha256": per_id_audio})}
    coverage = {"selected": {"count": len(selected), "ids": selected, "id_sha256": _digest(selected)}, "source": {"count": len(raw), "ids": sorted(raw, key=_id_key), "id_sha256": _digest(sorted(raw, key=_id_key))}}
    current_provenance = _provenance(model, source, coverage, config_state)
    reference_manifest: dict[str, object] | None = None
    if reference_root is not None:
        reference_manifest, _, _ = _validate_package(reference_root, selected, cpu_only=config.ecapa_device.startswith("cuda"))
        reference_provenance, current_parity_provenance = _parity_provenance(reference_manifest, current_provenance, cuda=config.ecapa_device.startswith("cuda"))
        if reference_provenance != current_parity_provenance:
            raise ValueError("parity reference provenance disagrees with current CPU identity")
    lock, lock_identity = _lock(resume_root)
    try:
        namespace_name = _context_name(model, config_state)
        for child in resume_root.iterdir():
            if child.name == ".resume.lock":
                continue
            if not child.name.startswith(_RESUME_PREFIX) or not re.fullmatch(r"context-[0-9a-f]{64}", child.name) or not _regular_directory(child):
                raise ValueError(f"foreign resume namespace is preserved: {child}")
            _validate_context(child)
        namespace = _create_namespace(resume_root, namespace_name, _context_identity(model, config_state))
        existing, audits = _resume_records(namespace, expected, config)
        cuda = _cuda_audit_adapter(config.ecapa_device) if config.ecapa_device.startswith("cuda") else None
        runtime = FireRedPvadRuntime(verified, config=config, cuda_peak_bytes=cuda.peak_bytes if cuda else None)
        features = dict(existing)
        new_count = 0
        for sample_id in selected:
            if sample_id in features:
                continue
            if cuda:
                cuda.reset_peak()
            result = runtime.extract(sample_id, state[sample_id]["wake_path"], state[sample_id]["command_path"])
            if result.sample_id != sample_id:
                raise ValueError("runtime returned a foreign sample ID")
            values, audit = _validate_values(result.values), _validate_audit(result.audit, config)
            record = {**expected[sample_id], "values": values, "audit": audit}
            _write_resume(namespace, record)
            features[sample_id] = values
            audits.append(audit)
            new_count += 1
        if set(features) != set(selected):
            raise ValueError("incomplete resume coverage")
        parity = {"status": "not-run", "passed": None, "max_abs_feature_delta": None}
        if reference_root is not None:
            checked_manifest, reference, _ = _validate_package(reference_root, selected, cpu_only=config.ecapa_device.startswith("cuda"))
            checked_provenance, current_parity_provenance = _parity_provenance(checked_manifest, current_provenance, cuda=config.ecapa_device.startswith("cuda"))
            if reference_manifest != checked_manifest or checked_provenance != current_parity_provenance:
                raise ValueError("parity reference provenance changed before publish")
            maximum = max((_max_abs_feature_delta(reference[sample_id], features[sample_id]) for sample_id in selected), default=0.0)
            if maximum > 1e-4:
                raise ValueError(f"parity failed: maximum feature delta {maximum}")
            parity = {"status": "passed", "passed": True, "max_abs_feature_delta": maximum}
        lines = [_ordered(OrderedDict((("id", sample_id), ("features", features[sample_id])))) for sample_id in selected]
        feature_text = "\n".join(lines) + "\n"
        cold = [float(a["elapsed_seconds"]) for a in audits if a["extraction_phase"] == "cold"]
        warm = [float(a["elapsed_seconds"]) for a in audits if a["extraction_phase"] == "warm"]
        cuda_peaks = [float(a["cuda_peak_bytes"]) for a in audits if "cuda_peak_bytes" in a]
        source = current_provenance["source"]
        manifest: dict[str, Any] = {"artifact_kind": _ARTIFACT_KIND, "schema_version": _SCHEMA_VERSION, "feature_schema": list(PVAD_GATE_FEATURE_SCHEMA), "feature_schema_sha256": _schema_digest(), "digest_algorithms": {"feature_schema_sha256": "sha256(UTF-8 ordered schema names joined and terminated by literal backslash-n bytes)", "records_sha256": "sha256(UTF-8 canonical JSONL bytes)", "per_id_record_sha256": "sha256(UTF-8 canonical JSON record bytes)", "joined_state_sha256": "sha256(UTF-8 canonical JSON)", "source_audio_sha256": "sha256(raw wake/command audio bytes)", "source_projection_sha256": "sha256(UTF-8 canonical JSON source projection)", "model_sha256": "sha256(UTF-8 canonical JSON verified model identity)"}, "coverage": current_provenance["coverage"], "source": source, "model": model, "runtime_config": config_state, "runtime_config_sha256": config_digest, "provider": config.onnx_provider, "device": config.ecapa_device, "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "observed_dependencies": _observed_dependencies()}, "reuse": {"reused": len(existing), "new": new_count}, "timing": {"cold_elapsed_seconds": _percentiles(cold), "warm_elapsed_seconds": _percentiles(warm), "rtf": _percentiles([float(a["rtf"]) for a in audits]), "peak_rss_delta_bytes": _percentiles([float(a["peak_rss_delta_bytes"]) for a in audits]), "cuda_peak_bytes": _percentiles(cuda_peaks)}, "parity": parity, "limit": {"value": limit, "canonical": limit is None, "reason": None if limit is None else "explicit noncanonical partial cache"}, "records_sha256": _sha256(feature_text.encode()), "per_id_record_sha256": {sample_id: _sha256(line.encode()) for sample_id, line in zip(selected, lines)}, "joined_state_sha256": _digest({sample_id: {**expected[sample_id], "model": model} for sample_id in selected})}
        if cuda:
            manifest["cuda"] = {**cuda.evidence(), "peak_bytes": _percentiles(cuda_peaks)}
        _validate_manifest_domains(manifest, selected)
        report = _report(manifest)
        return {"features": path for name, path in publish_text_package(output_root, _CONTRACT, {_FEATURES: feature_text, _MANIFEST: _canonical(manifest) + "\n", _REPORT: report}).items() if name == _FEATURES} | {"manifest": output_root / _MANIFEST, "report": output_root / _REPORT}
    finally:
        _unlock(lock, lock_identity)
