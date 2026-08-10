"""Resumable, label-free FireRed pVAD aggregate feature cache."""

from __future__ import annotations

import ctypes
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

from .artifact_publish import ArtifactContract, publish_text_package
from .data import FIELD_ALIASES, load_split
from .firered_model_assets import FireRedModelPaths, download_and_verify_model
from .firered_pvad import PVAD_GATE_FEATURE_SCHEMA, FireRedPvadRuntime, PvadRuntimeConfig

_ARTIFACT_KIND = "r11_e2_firered_cache"
_SCHEMA_VERSION = "v1"
_FEATURES, _MANIFEST, _REPORT = "pvad_features.jsonl", "pvad_manifest.json", "pvad_report.md"
_CONTRACT = ArtifactContract(_ARTIFACT_KIND, _SCHEMA_VERSION, (_FEATURES, _MANIFEST, _REPORT), (_MANIFEST,))
_SCHEMA_SHA256 = "5ca5d2c0f2bbbec0c5c1251518ca3fa8aac4c04144211de7cb6448bd7087830a"
_RESUME_PREFIX = "context-"
_CONTEXT_IDENTITY = "context_identity.json"
_AUDIT_COMMON = {"elapsed_seconds", "audio_seconds", "rtf", "peak_rss_delta_bytes", "dropped_tail_samples", "extraction_phase", "onnx_provider", "ecapa_device"}


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
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value or "\0" in value:
        raise ValueError("Dataset-A id must be a nonempty traversal-safe string")
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
    digest = _sha256("\n".join(PVAD_GATE_FEATURE_SCHEMA).encode("utf-8"))
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
    if not isinstance(coverage, Mapping) or not isinstance(coverage.get("selected"), Mapping) or coverage["selected"].get("ids") != ids or coverage["selected"].get("count") != len(ids) or manifest.get("records_sha256") != _sha256(text.encode("utf-8")) or manifest.get("per_id_record_sha256") != {sample_id: _sha256(line.encode("utf-8")) for sample_id, line in zip(ids, lines)}:
        raise ValueError("parity reference coverage or digest disagrees")
    if selected is not None and ids != selected:
        raise ValueError("parity reference IDs disagree")
    source = manifest.get("source")
    model = manifest.get("model")
    environment = manifest.get("environment")
    if not isinstance(source, Mapping) or set(source) != {"jsonl_sha256", "per_id_audio_sha256"} or not isinstance(source["jsonl_sha256"], Mapping) or not isinstance(source["per_id_audio_sha256"], Mapping) or set(source["per_id_audio_sha256"]) != set(ids) or not isinstance(model, Mapping) or set(model) != {"manifest_sha256", "aggregate_sha256", "raw_sha256", "upstream", "onnx", "required_dependency_versions"} or not isinstance(model["required_dependency_versions"], Mapping) or not model["required_dependency_versions"] or not isinstance(environment, Mapping) or set(environment) != {"python", "platform", "observed_dependencies"} or not isinstance(environment["observed_dependencies"], Mapping):
        raise ValueError("parity/output root provenance is incomplete")
    for sample_id, audio in source["per_id_audio_sha256"].items():
        if not isinstance(audio, Mapping) or set(audio) != {"wake_sha256", "command_sha256"} or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in audio.values()):
            raise ValueError("parity/output root source audio provenance is invalid")
    algorithms = manifest.get("digest_algorithms")
    if not isinstance(algorithms, Mapping) or {"feature_schema_sha256", "records_sha256", "per_id_record_sha256", "joined_state_sha256", "source_audio_sha256", "model_sha256"} - set(algorithms):
        raise ValueError("parity/output root digest algorithms are incomplete")
    if (root / _REPORT).read_text(encoding="utf-8") != _report(manifest):
        raise ValueError("parity/output root report disagrees with manifest")
    return manifest, rows, text


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


def _context_name(model: Mapping[str, object], config_digest: str) -> str:
    return _RESUME_PREFIX + _sha256(_canonical({"model": model, "runtime_config_sha256": config_digest, "feature_schema_sha256": _schema_digest()}).encode("utf-8"))


def _context_identity(model: Mapping[str, object], config_digest: str) -> dict[str, object]:
    return {"feature_schema_sha256": _schema_digest(), "model": dict(model), "runtime_config_sha256": config_digest}


def _validate_context(path: Path, model: Mapping[str, object] | None = None, config_digest: str | None = None) -> None:
    identity_path = path / _CONTEXT_IDENTITY
    if not identity_path.is_file() or identity_path.is_symlink():
        raise ValueError(f"resume namespace identity is missing or unsafe: {path}")
    identity_bytes = identity_path.read_bytes()
    identity = _load_object(identity_bytes.decode("utf-8"), "resume namespace identity")
    if identity_bytes != (_canonical(identity) + "\n").encode("utf-8") or set(identity) != {"feature_schema_sha256", "model", "runtime_config_sha256"} or _RESUME_PREFIX + _sha256(_canonical(identity).encode("utf-8")) != path.name:
        raise ValueError(f"resume namespace identity disagrees with its directory: {path}")
    if model is not None and identity != _context_identity(model, config_digest or ""):
        raise ValueError(f"resume namespace identity disagrees with current context: {path}")
    for child in path.iterdir():
        if child.name == _CONTEXT_IDENTITY:
            continue
        if re.fullmatch(r"\.record-[0-9a-f]{32}\.json\.[0-9a-f]{16}\.tmp", child.name):
            continue
        if child.is_symlink() or not child.is_file() or not re.fullmatch(r"record-[0-9a-f]{32}\.json", child.name):
            raise ValueError(f"foreign resume namespace state is preserved: {child}")
        record = _load_object(child.read_text(encoding="utf-8"), f"resume record {child}")
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or child.name != _resume_name(sample_id) or record.get("model") != identity["model"] or record.get("runtime_config_sha256") != identity["runtime_config_sha256"] or record.get("feature_schema_sha256") != identity["feature_schema_sha256"] or record.get("feature_schema") != list(PVAD_GATE_FEATURE_SCHEMA):
            raise ValueError(f"resume namespace contains a forged record: {child}")


def _resume_expected(sample_id: str, input_state: Mapping[str, object], model: Mapping[str, object], config_digest: str) -> dict[str, object]:
    return {"id": sample_id, "input": {"wake_sha256": input_state["wake_sha256"], "command_sha256": input_state["command_sha256"]}, "model": dict(model), "runtime_config_sha256": config_digest, "feature_schema": list(PVAD_GATE_FEATURE_SCHEMA), "feature_schema_sha256": _schema_digest()}


def _validate_values(values: Mapping[str, object]) -> OrderedDict[str, float | int]:
    if tuple(values) != PVAD_GATE_FEATURE_SCHEMA:
        raise ValueError("features disagree with the fixed feature schema")
    return OrderedDict((name, _finite(values[name], f"feature {name}")) for name in PVAD_GATE_FEATURE_SCHEMA)


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
        else:
            numeric = _finite(value, f"audit {key}")
            if numeric < 0:
                raise ValueError(f"audit {key} must be nonnegative")
            result[key] = numeric
    if result["extraction_phase"] not in {"cold", "warm"} or result["onnx_provider"] != config.onnx_provider or result["ecapa_device"] != config.ecapa_device:
        raise ValueError("runtime audit provider/device/phase disagrees with runtime config")
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
                os.rename(path, destination)
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
    if os.name == "nt":
        if not ctypes.windll.kernel32.MoveFileW(str(temporary), str(destination)):
            raise OSError(ctypes.get_last_error(), "could not publish resume record")
    else:
        os.link(temporary, destination)
        temporary.unlink()


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
            torch.cuda.reset_peak_memory_stats(index)
        def peak_bytes(self) -> int:
            return int(torch.cuda.max_memory_allocated(index))
        def evidence(self) -> dict[str, object]:
            driver: str | None = None
            try:
                import pynvml
                pynvml.nvmlInit()
                driver = str(pynvml.nvmlSystemGetDriverVersion())
            except Exception:
                pass
            return {"cuda_device_name": str(torch.cuda.get_device_name(index)), "cuda_driver": {"status": "available", "value": driver} if driver else {"status": "unavailable", "value": None}, "cuda_runtime_version": str(getattr(torch.version, "cuda", "unknown"))}
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
    config_digest = _sha256(_canonical(asdict(config)).encode("utf-8"))
    state: dict[str, dict[str, object]] = {}
    for sample_id in selected:
        wake = _safe_audio(dataset_root, raw[sample_id]["wakeup_audio"], "wake")
        command = _safe_audio(dataset_root, raw[sample_id]["command_audio"], "command")
        state[sample_id] = {"wake_sha256": _file_sha256(wake), "command_sha256": _file_sha256(command), "wake_path": wake, "command_path": command}
    expected = {sample_id: _resume_expected(sample_id, state[sample_id], model, config_digest) for sample_id in selected}
    lock, lock_identity = _lock(resume_root)
    try:
        namespace_name = _context_name(model, config_digest)
        for child in resume_root.iterdir():
            if child.name == ".resume.lock":
                continue
            if not child.name.startswith(_RESUME_PREFIX) or not re.fullmatch(r"context-[0-9a-f]{64}", child.name) or not _regular_directory(child):
                raise ValueError(f"foreign resume namespace is preserved: {child}")
            _validate_context(child)
        namespace = resume_root / namespace_name
        try:
            namespace.mkdir()
            (namespace / _CONTEXT_IDENTITY).write_bytes((_canonical(_context_identity(model, config_digest)) + "\n").encode("utf-8"))
        except FileExistsError:
            pass
        if not _regular_directory(namespace):
            raise ValueError("resume namespace must be a regular non-symlink directory")
        _validate_context(namespace, model, config_digest)
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
            _, reference, _ = _validate_package(reference_root, selected, cpu_only=config.ecapa_device.startswith("cuda"))
            maximum = max((abs(float(reference[sample_id][name]) - float(features[sample_id][name])) for sample_id in selected for name in PVAD_GATE_FEATURE_SCHEMA), default=0.0)
            if maximum > 1e-4:
                raise ValueError(f"parity failed: maximum feature delta {maximum}")
            parity = {"status": "passed", "passed": True, "max_abs_feature_delta": maximum}
        lines = [_ordered(OrderedDict((("id", sample_id), ("features", features[sample_id])))) for sample_id in selected]
        feature_text = "\n".join(lines) + "\n"
        cold = [float(a["elapsed_seconds"]) for a in audits if a["extraction_phase"] == "cold"]
        warm = [float(a["elapsed_seconds"]) for a in audits if a["extraction_phase"] == "warm"]
        cuda_peaks = [float(a["cuda_peak_bytes"]) for a in audits if "cuda_peak_bytes" in a]
        manifest: dict[str, Any] = {"artifact_kind": _ARTIFACT_KIND, "schema_version": _SCHEMA_VERSION, "feature_schema": list(PVAD_GATE_FEATURE_SCHEMA), "feature_schema_sha256": _schema_digest(), "digest_algorithms": {"feature_schema_sha256": "sha256(newline-joined ordered schema names with trailing newline)", "records_sha256": "sha256(UTF-8 canonical JSONL bytes)", "per_id_record_sha256": "sha256(UTF-8 canonical JSON record bytes)", "joined_state_sha256": "sha256(UTF-8 canonical JSON)", "source_audio_sha256": "sha256(raw wake/command audio bytes)", "model_sha256": "sha256(verified Task 2 model manifest/assets)"}, "coverage": {"selected": {"count": len(selected), "ids": selected, "id_sha256": _sha256(_canonical(selected).encode())}, "source": {"count": len(raw), "ids": sorted(raw, key=_id_key), "id_sha256": _sha256(_canonical(sorted(raw, key=_id_key)).encode())}}, "source": {"jsonl_sha256": source_digests, "per_id_audio_sha256": {sample_id: {"wake_sha256": state[sample_id]["wake_sha256"], "command_sha256": state[sample_id]["command_sha256"]} for sample_id in selected}}, "model": model, "runtime_config": asdict(config), "runtime_config_sha256": config_digest, "provider": config.onnx_provider, "device": config.ecapa_device, "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "observed_dependencies": _observed_dependencies()}, "reuse": {"reused": len(existing), "new": new_count}, "timing": {"cold_elapsed_seconds": _percentiles(cold), "warm_elapsed_seconds": _percentiles(warm), "rtf": _percentiles([float(a["rtf"]) for a in audits]), "peak_rss_delta_bytes": _percentiles([float(a["peak_rss_delta_bytes"]) for a in audits]), "cuda_peak_bytes": _percentiles(cuda_peaks)}, "parity": parity, "limit": {"value": limit, "canonical": limit is None, "reason": None if limit is None else "explicit noncanonical partial cache"}, "records_sha256": _sha256(feature_text.encode()), "per_id_record_sha256": {sample_id: _sha256(line.encode()) for sample_id, line in zip(selected, lines)}, "joined_state_sha256": _sha256(_canonical({sample_id: expected[sample_id] for sample_id in selected}).encode())}
        if cuda:
            manifest["cuda"] = {**cuda.evidence(), "peak_bytes": _percentiles(cuda_peaks)}
        report = _report(manifest)
        return {"features": path for name, path in publish_text_package(output_root, _CONTRACT, {_FEATURES: feature_text, _MANIFEST: _canonical(manifest) + "\n", _REPORT: report}).items() if name == _FEATURES} | {"manifest": output_root / _MANIFEST, "report": output_root / _REPORT}
    finally:
        _unlock(lock, lock_identity)
