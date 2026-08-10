"""Resumable, label-free FireRed pVAD aggregate feature cache."""

from __future__ import annotations

import hashlib
import ctypes
import json
import math
import os
import platform
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
_FEATURES = "pvad_features.jsonl"
_MANIFEST = "pvad_manifest.json"
_REPORT = "pvad_report.md"
_CONTRACT = ArtifactContract(_ARTIFACT_KIND, _SCHEMA_VERSION, (_FEATURES, _MANIFEST, _REPORT), (_MANIFEST,))


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _ordered_json(value: object) -> str:
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
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a native finite JSON number")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value!r} is forbidden")


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _load_object(text: str, label: str) -> dict[str, object]:
    try:
        value = json.loads(text, object_pairs_hook=_object_pairs, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _id_key(sample_id: str) -> tuple[int, int | str, str]:
    # The tag makes all-numeric and textual IDs comparable while retaining a total order.
    return (0, int(sample_id), sample_id) if sample_id.isdigit() else (1, sample_id, sample_id)


def _valid_id(value: object) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError("Dataset-A id must be a nonempty traversal-safe string")
    return value


def _first(row: Mapping[str, object], field: str) -> object | None:
    return next((row[name] for name in FIELD_ALIASES[field] if name in row), None)


def _raw_rows(dataset_root: Path) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    rows: dict[str, dict[str, object]] = {}
    sources: dict[str, str] = {}
    for split in ("pos", "neg"):
        path = dataset_root / f"{split}.jsonl"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Dataset-A split must be a regular non-symlink file: {path}")
        sources[split] = _file_sha256(path)
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                raise ValueError(f"Dataset-A JSONL {path}:{number} must not contain an empty row")
            row = _load_object(line, f"Dataset-A JSONL {path}:{number}")
            sample_id = _valid_id(_first(row, "id"))
            if sample_id in rows:
                raise ValueError(f"duplicate Dataset-A id {sample_id!r} within or across splits")
            wake, command = _first(row, "wakeup_audio"), _first(row, "command_audio")
            if not isinstance(wake, str) or not wake or not isinstance(command, str) or not command:
                raise ValueError(f"Dataset-A id {sample_id!r} is missing wake/command audio")
            rows[sample_id] = {"id": sample_id, "split": split, "wakeup_audio": wake, "command_audio": command}
    if not rows:
        raise ValueError("Dataset-A contains no rows")
    return rows, sources


def _regular_audio(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} audio is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} audio must be a regular non-symlink file: {path}")
    return path


def _safe_path(root: Path, relative: str, label: str) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"{label} audio path must be relative to Dataset-A root")
    candidate = root / relative
    # Dataset data is local, but reject lexical traversal before any filesystem access.
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"{label} audio path escapes Dataset-A root") from exc
    return _regular_audio(candidate, label)


def _schema_digest() -> str:
    return _sha256(_canonical(list(PVAD_GATE_FEATURE_SCHEMA)).encode("utf-8"))


def _config(ecapa_device: str) -> PvadRuntimeConfig:
    return PvadRuntimeConfig(onnx_provider="CPUExecutionProvider", ecapa_device=ecapa_device)


def verify_existing_model(paths: FireRedModelPaths) -> FireRedModelPaths:
    if not Path(paths.root).exists():
        raise ValueError(f"model root does not exist: {paths.root}")
    return download_and_verify_model(Path(paths.root))


def _model_identity(paths: FireRedModelPaths) -> dict[str, object]:
    parsed = _load_object(paths.manifest.read_text(encoding="utf-8"), "model manifest")
    return {
        "manifest_sha256": _file_sha256(paths.manifest),
        "aggregate_sha256": parsed.get("aggregate_sha256"),
        "raw_sha256": parsed.get("raw_sha256"),
        "upstream": parsed.get("upstream"),
        "onnx": parsed.get("onnx"),
    }


def _lock(root: Path) -> tuple[Path, tuple[int, int]]:
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
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
    except OSError:
        return
    if (item.st_dev, item.st_ino) != identity or lock.is_symlink():
        return
    owner = lock / "owner.json"
    if owner.exists():
        owner.unlink()
    lock.rmdir()


def _resume_name(sample_id: str) -> str:
    return "record-" + _sha256(sample_id.encode("utf-8")) + ".json"


def _resume_expected(sample_id: str, input_state: Mapping[str, object], model: Mapping[str, object], config_digest: str) -> dict[str, object]:
    public_input = {key: value for key, value in input_state.items() if key not in {"wake_path", "command_path"}}
    return {"id": sample_id, "input": public_input, "model": dict(model), "config_sha256": config_digest, "schema": list(PVAD_GATE_FEATURE_SCHEMA), "schema_sha256": _schema_digest()}


def _validate_values(values: Mapping[str, object]) -> OrderedDict[str, float | int]:
    if tuple(values) != PVAD_GATE_FEATURE_SCHEMA:
        raise ValueError("features disagree with the fixed feature schema")
    return OrderedDict((name, _finite(values[name], f"feature {name}")) for name in PVAD_GATE_FEATURE_SCHEMA)


def _validate_audit(audit: Mapping[str, object], config: PvadRuntimeConfig | None = None) -> dict[str, object]:
    if not isinstance(audit, Mapping):
        raise ValueError("runtime audit must be an object")
    result: dict[str, object] = {}
    for name, value in audit.items():
        if type(name) is not str or not name:
            raise ValueError("runtime audit names must be nonempty strings")
        if isinstance(value, str):
            result[name] = value
        else:
            result[name] = _finite(value, f"audit {name}")
    required = {"elapsed_seconds", "audio_seconds", "rtf", "peak_rss_delta_bytes", "extraction_phase", "onnx_provider", "ecapa_device"}
    if not required.issubset(result) or result["extraction_phase"] not in {"cold", "warm"}:
        raise ValueError("runtime audit is incomplete or has an invalid extraction phase")
    if config is not None and (result["onnx_provider"] != config.onnx_provider or result["ecapa_device"] != config.ecapa_device):
        raise ValueError("runtime audit provider/device disagrees with runtime config")
    return result


def _resume_records(root: Path, expected: Mapping[str, Mapping[str, object]]) -> tuple[dict[str, OrderedDict[str, float | int]], list[Mapping[str, object]]]:
    result: dict[str, OrderedDict[str, float | int]] = {}
    audits: list[Mapping[str, object]] = []
    for path in root.iterdir():
        if path.name == ".resume.lock":
            continue
        if path.is_symlink() or not path.is_file() or not path.name.endswith(".json"):
            raise ValueError(f"foreign resume state is preserved: {path}")
        record = _load_object(path.read_text(encoding="utf-8"), f"resume record {path}")
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or sample_id not in expected or path.name != _resume_name(sample_id):
            raise ValueError(f"foreign resume record is preserved: {path}")
        if sample_id in result or {key: record.get(key) for key in expected[sample_id]} != expected[sample_id]:
            raise ValueError(f"resume record identity mismatch is preserved: {path}")
        input_state = expected[sample_id]["input"]
        if record.get("wake_sha256") != input_state["wake_sha256"] or record.get("command_sha256") != input_state["command_sha256"]:
            raise ValueError(f"resume record audio digest mismatch is preserved: {path}")
        values = record.get("values")
        audit = record.get("audit")
        if not isinstance(values, Mapping) or not isinstance(audit, Mapping):
            raise ValueError(f"malformed resume record is preserved: {path}")
        audits.append(_validate_audit(audit))
        result[sample_id] = _validate_values(values)
    return result, audits


def _write_resume(root: Path, record: Mapping[str, object]) -> None:
    sample_id = str(record["id"])
    destination = root / _resume_name(sample_id)
    if os.path.lexists(destination):
        raise ValueError(f"resume destination already exists and is preserved: {destination}")
    temporary = root / f".{destination.name}.{secrets.token_hex(8)}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(_ordered_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.is_symlink() or os.path.lexists(destination):
            raise ValueError("resume write encountered a symlink")
        if os.name == "nt":
            move = ctypes.windll.kernel32.MoveFileW
            move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
            move.restype = ctypes.c_int
            if not move(str(temporary), str(destination)):
                error = ctypes.get_last_error()
                raise OSError(error, ctypes.FormatError(error), str(destination))
        else:
            os.link(temporary, destination)
            temporary.unlink()
    except Exception:
        # A temporary can be evidence of an interrupted write, so it is never deleted here.
        raise


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]
    return {"count": len(ordered), "p50": at(0.5), "p95": at(0.95), "max": ordered[-1]}


def _parity(reference: Path | None, selected: list[str], features: Mapping[str, Mapping[str, float | int]]) -> dict[str, object]:
    if reference is None:
        return {"status": "not-run", "passed": None, "max_abs_feature_delta": None}
    parent = reference.parent
    manifest = parent / _MANIFEST
    if reference.name != _FEATURES or not manifest.is_file():
        raise ValueError("parity reference parent is not a recognized cache package")
    identity = _load_object(manifest.read_text(encoding="utf-8"), "parity manifest")
    if identity.get("artifact_kind") != _ARTIFACT_KIND or identity.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("parity reference parent is not a recognized cache package")
    output: dict[str, Mapping[str, object]] = {}
    for number, line in enumerate(reference.read_text(encoding="utf-8").splitlines(), 1):
        record = _load_object(line, f"parity record {number}")
        sample_id, values = record.get("id"), record.get("features")
        if not isinstance(sample_id, str) or sample_id in output or not isinstance(values, Mapping):
            raise ValueError("malformed parity reference")
        output[sample_id] = values
    if list(output) != selected:
        raise ValueError("parity reference IDs disagree")
    maximum = 0.0
    for sample_id in selected:
        reference_values = _validate_values(output[sample_id])
        for name in PVAD_GATE_FEATURE_SCHEMA:
            maximum = max(maximum, abs(float(reference_values[name]) - float(features[sample_id][name])))
    if maximum > 1e-4:
        raise ValueError(f"parity failed: maximum feature delta {maximum}")
    return {"status": "passed", "passed": True, "max_abs_feature_delta": maximum}


def _overlap(left: Path, right: Path) -> bool:
    a, b = str(left.resolve(strict=False)), str(right.resolve(strict=False))
    return os.path.commonpath((a, b)) in {a, b}


def build_pvad_cache(dataset_root: Path, model_paths: FireRedModelPaths, output_root: Path, *, resume_root: Path, ecapa_device: str, limit: int | None = None, parity_reference: Path | None = None) -> dict[str, Path]:
    """Build and atomically publish one complete label-free pVAD cache."""
    dataset_root, output_root, resume_root = Path(dataset_root), Path(output_root), Path(resume_root)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if any(_overlap(left, right) for left, right in ((output_root, resume_root), (output_root, dataset_root), (output_root, model_paths.root), (resume_root, dataset_root), (resume_root, model_paths.root), (dataset_root, model_paths.root))):
        raise ValueError("output, resume, model, and Dataset-A roots must not overlap")
    raw, source_digests = _raw_rows(dataset_root)
    # Reuse the project loader only after raw IDs were independently validated.
    loaded = {sample.id: sample for split in ("pos", "neg") for sample in load_split(dataset_root, split)}
    if set(loaded) != set(raw):
        raise ValueError("Dataset-A loader disagrees with independently parsed raw IDs")
    selected = sorted(raw, key=_id_key)
    if limit is not None:
        selected = selected[:limit]
    verified = verify_existing_model(model_paths)
    model = _model_identity(verified)
    config = _config(ecapa_device)
    config_digest = _sha256(_canonical(asdict(config)).encode("utf-8"))
    state: dict[str, dict[str, object]] = {}
    for sample_id in selected:
        wake = _safe_path(dataset_root, str(raw[sample_id]["wakeup_audio"]), "wake")
        command = _safe_path(dataset_root, str(raw[sample_id]["command_audio"]), "command")
        state[sample_id] = {"wake_sha256": _file_sha256(wake), "command_sha256": _file_sha256(command), "projection": raw[sample_id], "wake_path": wake, "command_path": command}
    expected = {sample_id: _resume_expected(sample_id, state[sample_id], model, config_digest) for sample_id in selected}
    lock, lock_identity = _lock(resume_root)
    try:
        existing, prior_audits = _resume_records(resume_root, expected)
        runtime = FireRedPvadRuntime(verified, config=config)
        features: dict[str, OrderedDict[str, float | int]] = dict(existing)
        audits: list[Mapping[str, object]] = list(prior_audits)
        new_audits = 0
        for sample_id in selected:
            if sample_id in features:
                continue
            result = runtime.extract(sample_id, state[sample_id]["wake_path"], state[sample_id]["command_path"])
            if result.sample_id != sample_id:
                raise ValueError("runtime returned a foreign sample ID")
            values = _validate_values(result.values)
            audit = _validate_audit(result.audit, config)
            record = {**expected[sample_id], "wake_sha256": state[sample_id]["wake_sha256"], "command_sha256": state[sample_id]["command_sha256"], "values": values, "audit": audit}
            _write_resume(resume_root, record)
            features[sample_id] = values
            audits.append(audit)
            new_audits += 1
        if set(features) != set(selected):
            raise ValueError("incomplete resume coverage")
        parity = _parity(Path(parity_reference) if parity_reference else None, selected, features)
        lines = [_ordered_json(OrderedDict((("id", sample_id), ("features", features[sample_id])))) for sample_id in selected]
        feature_text = "\n".join(lines) + "\n"
        per_id = {sample_id: _sha256(lines[index].encode("utf-8")) for index, sample_id in enumerate(selected)}
        elapsed = [float(a["elapsed_seconds"]) for a in audits if isinstance(a.get("elapsed_seconds"), (int, float))]
        rtf = [float(a["rtf"]) for a in audits if isinstance(a.get("rtf"), (int, float))]
        public_projection = {sample_id: {key: value for key, value in state[sample_id].items() if key not in {"wake_path", "command_path"}} for sample_id in selected}
        manifest = {"artifact_kind": _ARTIFACT_KIND, "schema_version": _SCHEMA_VERSION, "feature_schema": list(PVAD_GATE_FEATURE_SCHEMA), "feature_schema_sha256": _schema_digest(), "coverage": {"selected": {"count": len(selected), "ids": selected, "id_sha256": _sha256(_canonical(selected).encode("utf-8"))}, "source": {"count": len(raw), "ids": sorted(raw, key=_id_key), "id_sha256": _sha256(_canonical(sorted(raw, key=_id_key)).encode("utf-8"))}}, "source": {"jsonl_sha256": source_digests, "input_projection": public_projection}, "model": model, "runtime_config": asdict(config), "runtime_config_sha256": config_digest, "provider": "CPUExecutionProvider", "device": ecapa_device, "environment": {"python": sys.version.split()[0], "platform": platform.platform()}, "reuse": {"reused": len(existing), "new": new_audits}, "timing": {"cold_warm_elapsed_seconds": _percentiles(elapsed), "rtf": _percentiles(rtf)}, "parity": parity, "limit": {"value": limit, "canonical": limit is None, "reason": None if limit is None else "explicit noncanonical partial cache"}, "records_sha256": _sha256(feature_text.encode("utf-8")), "per_id_record_sha256": per_id, "joined_state_sha256": _sha256(_canonical({sample_id: expected[sample_id] for sample_id in selected}).encode("utf-8"))}
        manifest_text = _canonical(manifest) + "\n"
        report = f"# FireRed pVAD Cache\n\n- Selected IDs: {len(selected)}\n- New records: {new_audits}\n- Reused records: {len(existing)}\n- Parity: {parity['status']}\n"
        published = publish_text_package(output_root, _CONTRACT, {_FEATURES: feature_text, _MANIFEST: manifest_text, _REPORT: report})
        return {"features": published[_FEATURES], "manifest": published[_MANIFEST], "report": published[_REPORT]}
    finally:
        _unlock(lock, lock_identity)
