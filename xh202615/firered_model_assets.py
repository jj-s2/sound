"""Pinned acquisition and fail-closed preflight for FireRedChat pVAD assets."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from xh202615.artifact_publish import (
    _cleanup_owned_directory,
    _create_unique_staging,
    _directory_identity,
    _lexists,
    _rename_no_replace,
)


FIRERED_REPO_ID = "FireRedTeam/FireRedChat-pvad"
FIRERED_REVISION = "74561b17a50fbe9d8f84dacc453f175cb97f567c"
FIRERED_ALLOW_PATTERNS = (
    "pvad.onnx",
    "NOTICE",
    "README.md",
    "spkrec-ecapa-voxceleb/**",
)

_MODEL_MANIFEST_NAME = "model_manifest.json"
_ECAPA_DIRECTORY = "spkrec-ecapa-voxceleb"
_RAW_SNAPSHOT_FILES = frozenset(
    {
        "NOTICE",
        "README.md",
        "pvad.onnx",
        "spkrec-ecapa-voxceleb/README.md",
        "spkrec-ecapa-voxceleb/classifier.ckpt",
        "spkrec-ecapa-voxceleb/config.json",
        "spkrec-ecapa-voxceleb/embedding_model.ckpt",
        "spkrec-ecapa-voxceleb/hyperparams.yaml",
        "spkrec-ecapa-voxceleb/label_encoder.ckpt",
        "spkrec-ecapa-voxceleb/label_encoder.txt",
        "spkrec-ecapa-voxceleb/mean_var_norm_emb.ckpt",
    }
)
_AGGREGATE_DIGEST_ALGORITHM = "sha256(sorted(path_utf8 + NUL + sha256_bytes))"
_ARTIFACT_KIND = "firered_model_assets"
_SCHEMA_VERSION = "v1"

_UPSTREAM_IDENTITY = {
    "license": "Apache-2.0",
    "model_url": "https://huggingface.co/FireRedTeam/FireRedChat-pvad",
    "repo_id": FIRERED_REPO_ID,
    "revision": FIRERED_REVISION,
    "runtime_contract_url": (
        "https://github.com/fireredchat-submodules/"
        "livekit-plugins-fireredchat-pvad/blob/"
        "3163734bf27878c2a76eba8849e973c5288a6b16/"
        "livekit/plugins/fireredchat_pvad/vad.py"
    ),
    "speaker_encoder_id": "speechbrain/spkrec-ecapa-voxceleb",
    "speaker_encoder_url": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
}

# These frozen requirements describe the runtime needed for this model. They are
# not observations from the active environment.
_REQUIRED_DEPENDENCY_VERSIONS = {
    "huggingface-hub": "0.36.2",
    "hyperpyyaml": "1.2.3",
    "joblib": "1.5.3",
    "numpy": "1.26.4",
    "onnxruntime-gpu": "1.28.0",
    "packaging": "24.2",
    "PyYAML": "6.0.3",
    "ruamel.yaml": "0.18.16",
    "ruamel.yaml.clib": "0.2.15",
    "sentencepiece": "0.2.2",
    "speechbrain": "1.0.3",
    "torch": "2.4.1+cu121",
    "torchaudio": "2.4.1+cu121",
    "tqdm": "4.65.2",
}

_EXPECTED_INPUTS = (
    ("input_audio", "tensor(float)", (1, 160)),
    ("spkemb", "tensor(float)", (1, 192)),
    ("mel_buffer", "tensor(float)", (1, 80, 15)),
    ("gru_buffer", "tensor(float)", (2, 1, 256)),
)
_PROBABILITY_OUTPUT_INDEX = 1
_MEL_STATE_OUTPUT_INDEX = 2
_GRU_STATE_OUTPUT_INDEX = 3
_EXPECTED_POSITIONAL_OUTPUTS = {
    _PROBABILITY_OUTPUT_INDEX: ("probability output", "tensor(float)", (1, 1)),
    _MEL_STATE_OUTPUT_INDEX: ("mel state output", "tensor(float)", (1, 80, 15)),
    _GRU_STATE_OUTPUT_INDEX: ("GRU state output", "tensor(float)", (2, 1, 256)),
}


@dataclass(frozen=True)
class FireRedModelPaths:
    """Resolved paths for one verified pinned FireRed model root."""

    root: Path
    pvad_onnx: Path
    ecapa_root: Path
    manifest: Path


@dataclass(frozen=True)
class _MetadataRecord:
    name: str
    type: str
    shape: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "type": self.type, "shape": list(self.shape)}


def snapshot_download(**kwargs: object) -> str:
    """Load Hugging Face lazily so importing the CLI for ``--help`` stays cheap."""

    try:
        from huggingface_hub import snapshot_download as huggingface_snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to download the pinned FireRed model"
        ) from exc
    return str(huggingface_snapshot_download(**kwargs))


def _load_onnx_session(model_path: Path) -> object:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime-gpu is required to verify the FireRed ONNX interface"
        ) from exc
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _metadata_records(values: Iterable[object], *, kind: str) -> list[_MetadataRecord]:
    records: list[_MetadataRecord] = []
    for index, value in enumerate(values):
        name = getattr(value, "name", None)
        type_name = getattr(value, "type", None)
        shape = getattr(value, "shape", None)
        if not isinstance(name, str) or not name:
            raise ValueError(f"ONNX {kind} metadata at index {index} has no valid name")
        if not isinstance(type_name, str) or not type_name:
            raise ValueError(
                f"ONNX {kind} metadata {name!r} has no valid type"
            )
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            raise ValueError(
                f"ONNX {kind} metadata {name!r} has no valid shape"
            )
        dimensions: list[int] = []
        for dimension in shape:
            if not isinstance(dimension, int) or isinstance(dimension, bool):
                raise ValueError(
                    f"ONNX {kind} metadata {name!r} must have a concrete integer shape"
                )
            dimensions.append(dimension)
        records.append(_MetadataRecord(name, type_name, tuple(dimensions)))

    names = [record.name for record in records]
    if len(set(names)) != len(names):
        raise ValueError(f"ONNX {kind} names must be unique")
    return records


def verify_onnx_contract(session: object) -> dict[str, object]:
    """Validate and serialize the frozen official FireRed ONNX interface."""

    try:
        inputs = _metadata_records(session.get_inputs(), kind="input")
        outputs = _metadata_records(session.get_outputs(), kind="output")
    except AttributeError as exc:
        raise ValueError("ONNX session does not expose input/output metadata") from exc

    expected_input_names = [name for name, _, _ in _EXPECTED_INPUTS]
    actual_input_names = [metadata.name for metadata in inputs]
    if set(actual_input_names) != set(expected_input_names) or len(inputs) != len(
        expected_input_names
    ):
        raise ValueError(
            "ONNX input names disagree with the official contract: "
            f"expected {expected_input_names}, got {actual_input_names}"
        )
    inputs_by_name = {metadata.name: metadata for metadata in inputs}
    ordered_inputs = [inputs_by_name[name] for name in expected_input_names]
    for metadata, (_, expected_type, expected_shape) in zip(
        ordered_inputs, _EXPECTED_INPUTS, strict=True
    ):
        if metadata.type != expected_type:
            raise ValueError(
                f"ONNX input {metadata.name!r} type must be {expected_type}, "
                f"got {metadata.type}"
            )
        if metadata.shape != expected_shape:
            raise ValueError(
                f"ONNX input {metadata.name!r} shape must be {expected_shape}, "
                f"got {metadata.shape}"
            )

    if len(outputs) < 4:
        raise ValueError(
            f"ONNX contract requires at least four outputs, got {len(outputs)}"
        )
    for index, (label, expected_type, expected_shape) in _EXPECTED_POSITIONAL_OUTPUTS.items():
        metadata = outputs[index]
        if metadata.type != expected_type:
            raise ValueError(
                f"ONNX {label} at index {index} type must be {expected_type}, "
                f"got {metadata.type}"
            )
        if metadata.shape != expected_shape:
            raise ValueError(
                f"ONNX {label} at index {index} shape must be {expected_shape}, "
                f"got {metadata.shape}"
            )

    return {
        "frame_samples": 160,
        "gru_state_output_index": _GRU_STATE_OUTPUT_INDEX,
        "inputs": [metadata.as_dict() for metadata in ordered_inputs],
        "mel_state_output_index": _MEL_STATE_OUTPUT_INDEX,
        "outputs": [metadata.as_dict() for metadata in outputs],
        "probability_output_index": _PROBABILITY_OUTPUT_INDEX,
        "sample_rate_hz": 16000,
    }


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_digest(raw_sha256: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(raw_sha256):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(raw_sha256[relative_path]))
    return digest.hexdigest()


def _remove_huggingface_metadata(root: Path) -> None:
    metadata_root = root / ".cache"
    if not _lexists(metadata_root):
        return
    metadata = metadata_root.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError("downloaded .cache metadata path must be a regular directory")
    import shutil

    shutil.rmtree(metadata_root)


def _scan_model_tree(root: Path, *, require_manifest: bool) -> dict[str, Path]:
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise ValueError("model root must be a regular non-symlink directory")

    files: dict[str, Path] = {}
    empty_directories: list[str] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        entries = list(os.scandir(directory))
        if directory != root and not entries:
            empty_directories.append(directory.relative_to(root).as_posix())
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(
                    f"model asset must be a regular non-symlink file or directory: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                if relative != _ECAPA_DIRECTORY and not relative.startswith(
                    _ECAPA_DIRECTORY + "/"
                ):
                    raise ValueError(f"unexpected model asset directory: {relative}")
                stack.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    f"model asset must be a regular non-symlink file: {relative}"
                )
            if relative in _RAW_SNAPSHOT_FILES:
                files[relative] = path
                continue
            if relative == _MODEL_MANIFEST_NAME:
                if not require_manifest:
                    raise ValueError(f"unexpected model asset file: {relative}")
                files[relative] = path
                continue
            raise ValueError(f"unexpected model asset file: {relative}")

    if empty_directories:
        raise ValueError(
            f"unexpected empty model asset directories: {sorted(empty_directories)}"
        )
    raw_files = set(files) - {_MODEL_MANIFEST_NAME}
    missing = sorted(_RAW_SNAPSHOT_FILES - raw_files)
    if missing:
        raise ValueError(f"missing required model asset files: {missing}")
    extras = sorted(raw_files - _RAW_SNAPSHOT_FILES)
    if extras:
        raise ValueError(f"unexpected model asset files: {extras}")
    if require_manifest and _MODEL_MANIFEST_NAME not in files:
        raise ValueError(f"missing required model asset file: {_MODEL_MANIFEST_NAME}")
    return files


def _raw_file_digests(files: Mapping[str, Path]) -> dict[str, str]:
    return {
        name: _sha256_file(files[name])
        for name in sorted(files)
        if name != _MODEL_MANIFEST_NAME
    }


def _manifest_payload(
    raw_sha256: Mapping[str, str], onnx_contract: Mapping[str, object]
) -> dict[str, object]:
    return {
        "aggregate_sha256": _aggregate_digest(raw_sha256),
        "aggregate_sha256_algorithm": _AGGREGATE_DIGEST_ALGORITHM,
        "artifact_kind": _ARTIFACT_KIND,
        "required_dependency_versions": dict(_REQUIRED_DEPENDENCY_VERSIONS),
        "onnx": dict(onnx_contract),
        "raw_sha256": dict(raw_sha256),
        "schema_version": _SCHEMA_VERSION,
        "upstream": dict(_UPSTREAM_IDENTITY),
    }


def _load_json_object(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid model manifest JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("model manifest must be a JSON object")
    return parsed


def _model_paths(root: Path) -> FireRedModelPaths:
    return FireRedModelPaths(
        root=root,
        pvad_onnx=root / "pvad.onnx",
        ecapa_root=root / _ECAPA_DIRECTORY,
        manifest=root / _MODEL_MANIFEST_NAME,
    )


def _verify_complete_root(
    root: Path, session_factory: Callable[[Path], object]
) -> FireRedModelPaths:
    files = _scan_model_tree(root, require_manifest=True)
    manifest_path = files[_MODEL_MANIFEST_NAME]
    parsed = _load_json_object(manifest_path)
    if manifest_path.read_bytes() != _canonical_json_bytes(parsed):
        raise ValueError("model manifest is not canonical JSON")

    raw_sha256 = _raw_file_digests(files)
    try:
        session = session_factory(files["pvad.onnx"])
        onnx_contract = verify_onnx_contract(session)
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"could not inspect FireRed ONNX metadata: {exc}") from exc
    expected = _manifest_payload(raw_sha256, onnx_contract)
    if parsed != expected:
        raise ValueError("model manifest disagrees with pinned identity, metadata, or digests")
    return _model_paths(root)


def download_and_verify_model(
    root: Path,
    *,
    downloader: Callable[..., str] = snapshot_download,
    session_factory: Callable[[Path], object] = _load_onnx_session,
) -> FireRedModelPaths:
    """Reuse or acquire the exact pinned model root without overwriting anything."""

    root = Path(root)
    if not root.name:
        raise ValueError("model root must name a directory")
    if _lexists(root):
        try:
            return _verify_complete_root(root, session_factory)
        except Exception as exc:
            raise ValueError(
                f"existing model root {root} is not a recognized pinned FireRed model root"
            ) from exc

    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging, staging_identity = _create_unique_staging(
        parent, root.name + ".staging"
    )
    active_staging: Path | None = staging
    active_identity = staging_identity
    try:
        downloaded = Path(
            downloader(
                repo_id=FIRERED_REPO_ID,
                revision=FIRERED_REVISION,
                local_dir=str(staging),
                allow_patterns=list(FIRERED_ALLOW_PATTERNS),
            )
        )
        if downloaded.resolve(strict=True) != staging.resolve(strict=True):
            raise ValueError(
                f"downloader returned unexpected model directory {downloaded}"
            )
        if _directory_identity(staging) != staging_identity:
            raise RuntimeError("owned model staging directory changed during download")

        _remove_huggingface_metadata(staging)
        raw_files = _scan_model_tree(staging, require_manifest=False)
        raw_sha256 = _raw_file_digests(raw_files)
        try:
            onnx_contract = verify_onnx_contract(
                session_factory(raw_files["pvad.onnx"])
            )
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"could not inspect FireRed ONNX metadata: {exc}") from exc
        payload = _manifest_payload(raw_sha256, onnx_contract)
        (staging / _MODEL_MANIFEST_NAME).write_bytes(_canonical_json_bytes(payload))
        _verify_complete_root(staging, session_factory)

        try:
            _rename_no_replace(staging, root)
        except OSError as publish_exc:
            if _lexists(root):
                try:
                    winner = _verify_complete_root(root, session_factory)
                except Exception as winner_exc:
                    raise ValueError(
                        f"competing model root {root} was preserved but is not a "
                        "recognized pinned FireRed model root"
                    ) from winner_exc
                if _load_json_object(winner.manifest) != payload:
                    raise ValueError(
                        f"competing model root {root} was preserved but disagrees "
                        "with the verified staged snapshot"
                    )
                return winner
            raise publish_exc
        active_staging = None
        if _directory_identity(root) != staging_identity:
            raise RuntimeError(
                f"published model root changed unexpectedly and was preserved at {root}"
            )
        try:
            return _verify_complete_root(root, session_factory)
        except Exception as exc:
            raise RuntimeError(
                f"published model root failed verification and was preserved at {root}"
            ) from exc
    finally:
        if active_staging is not None:
            cleanup_error = _cleanup_owned_directory(active_staging, active_identity)
            if cleanup_error is not None:
                message = "model staging cleanup failed: " + cleanup_error
                active_exception = sys.exc_info()[1]
                if active_exception is None:
                    raise RuntimeError(message)
                active_exception.add_note(message)
