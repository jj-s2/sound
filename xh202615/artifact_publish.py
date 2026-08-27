"""Fail-closed atomic publication for exact text artifact packages."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import secrets
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_PathIdentity = tuple[int, int]


@dataclass(frozen=True)
class ArtifactContract:
    """Identity and exact filename contract for one artifact package kind."""

    artifact_kind: str
    schema_version: str
    required_names: tuple[str, ...]
    identity_json_names: tuple[str, ...]


def _validate_contract(contract: ArtifactContract) -> None:
    required = contract.required_names
    identity = contract.identity_json_names
    if not contract.artifact_kind or not contract.schema_version:
        raise ValueError("artifact contract identity values must be non-empty")
    if not required or len(set(required)) != len(required):
        raise ValueError("artifact contract required_names must be non-empty and unique")
    if not identity or len(set(identity)) != len(identity):
        raise ValueError("artifact contract identity_json_names must be non-empty and unique")
    if not set(identity).issubset(required):
        raise ValueError("artifact contract identity JSON names must be required artifact names")
    for name in required:
        if (
            not name
            or "/" in name
            or "\\" in name
            or Path(name).name != name
            or name in {".", ".."}
        ):
            raise ValueError(f"artifact names must be safe basenames: {name!r}")


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _directory_identity(path: Path) -> _PathIdentity | None:
    try:
        metadata = path.lstat()
    except OSError:
        return None
    if not stat.S_ISDIR(metadata.st_mode):
        return None
    return metadata.st_dev, metadata.st_ino


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically move a directory without replacing an existing destination."""
    if os.name == "nt":
        os.rename(source, destination)
        return

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            at_fdcwd = -100
            rename_noreplace = 1
            result = renameat2(
                at_fdcwd,
                os.fsencode(source),
                at_fdcwd,
                os.fsencode(destination),
                rename_noreplace,
            )
            if result == 0:
                return
            error = ctypes.get_errno()
            if error not in {errno.ENOSYS, errno.EINVAL}:
                raise OSError(error, os.strerror(error), str(destination))

    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is not None:
            renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            renamex_np.restype = ctypes.c_int
            rename_excl = 0x00000004
            result = renamex_np(
                os.fsencode(source),
                os.fsencode(destination),
                rename_excl,
            )
            if result == 0:
                return
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))

    raise RuntimeError(
        f"atomic no-replace directory rename is unsupported on {sys.platform}"
    )


def _rename_no_replace_native(source: Path, destination: Path) -> None:
    """No-replace rename used by cleanup, independent of injected publish moves."""
    if os.name != "nt":
        _rename_no_replace(source, destination)
        return

    move_file = ctypes.windll.kernel32.MoveFileW
    move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    move_file.restype = ctypes.c_int
    if move_file(str(source), str(destination)):
        return
    error = ctypes.get_last_error()
    raise OSError(error, ctypes.FormatError(error), str(destination))


def _package_bytes(path: Path, contract: ArtifactContract) -> dict[str, bytes] | None:
    if _directory_identity(path) is None:
        return None
    try:
        children = list(path.iterdir())
    except OSError:
        return None
    if {child.name for child in children} != set(contract.required_names):
        return None
    if any(not child.is_file() or child.is_symlink() for child in children):
        return None

    package: dict[str, bytes] = {}
    try:
        for name in contract.required_names:
            package[name] = (path / name).read_bytes()
        for name in contract.identity_json_names:
            identity = json.loads(package[name].decode("utf-8"))
            if not isinstance(identity, dict):
                return None
            if identity.get("artifact_kind") != contract.artifact_kind:
                return None
            if identity.get("schema_version") != contract.schema_version:
                return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return package


def _create_unique_staging(parent: Path, prefix: str) -> tuple[Path, _PathIdentity]:
    while True:
        candidate = parent / f"{prefix}.{secrets.token_hex(8)}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        identity = _directory_identity(candidate)
        if identity is None:
            raise RuntimeError(f"could not identify owned staging directory {candidate}")
        return candidate, identity


def _move_to_unique_backup(
    output_root: Path,
    parent: Path,
    prefix: str,
) -> tuple[Path, _PathIdentity]:
    while True:
        candidate = parent / f"{prefix}.{secrets.token_hex(8)}"
        if _lexists(candidate):
            continue
        try:
            _rename_no_replace(output_root, candidate)
        except FileExistsError:
            if _lexists(output_root) and _lexists(candidate):
                continue
            raise
        identity = _directory_identity(candidate)
        if identity is None:
            raise RuntimeError(f"could not identify recovery backup {candidate}")
        return candidate, identity


def _restore_quarantined_directory(quarantine: Path, original: Path) -> Path:
    if not _lexists(original):
        try:
            _rename_no_replace_native(quarantine, original)
            return original
        except OSError:
            pass
    return quarantine


def _remove_owned_lock(
    lock: Path,
    lock_identity: _PathIdentity,
    owner_metadata: bytes,
) -> None:
    if not _lexists(lock):
        return
    quarantine = lock.parent / f"{lock.name}.cleanup.{secrets.token_hex(8)}"
    _rename_no_replace_native(lock, quarantine)
    try:
        moved_identity = _directory_identity(quarantine)
        owner = quarantine / "owner.json"
        owner_matches = (
            owner.is_file()
            and not owner.is_symlink()
            and owner.read_bytes() == owner_metadata
        )
    except OSError:
        _restore_quarantined_directory(quarantine, lock)
        raise
    if moved_identity != lock_identity or not owner_matches:
        _restore_quarantined_directory(quarantine, lock)
        return
    owner.unlink()
    quarantine.rmdir()


def _cleanup_owned_directory(
    path: Path,
    identity: _PathIdentity,
    *,
    expected_package: dict[str, bytes] | None = None,
    contract: ArtifactContract | None = None,
) -> str | None:
    if not _lexists(path):
        return f"owned directory disappeared before cleanup: {path}"
    quarantine = path.parent / f"{path.name}.cleanup.{secrets.token_hex(8)}"
    try:
        _rename_no_replace_native(path, quarantine)
    except OSError as exc:
        return f"could not quarantine owned directory {path}: {exc}"
    moved_identity = _directory_identity(quarantine)

    moved_package_matches = True
    if expected_package is not None:
        assert contract is not None
        moved_package_matches = _package_bytes(quarantine, contract) == expected_package
    if moved_identity != identity or not moved_package_matches:
        preserved = quarantine
        if not _lexists(path):
            try:
                _rename_no_replace(quarantine, path)
                preserved = path
            except OSError:
                pass
        return f"unexpected directory was preserved at {preserved}"

    try:
        shutil.rmtree(quarantine)
    except OSError as exc:
        return f"could not remove owned directory {quarantine}: {exc}"
    return None


def publish_text_package(
    output_root: Path,
    contract: ArtifactContract,
    contents: Mapping[str, str],
) -> dict[str, Path]:
    """Publish prepared UTF-8 text under an exact, identity-checked contract."""

    output_root = Path(output_root)
    _validate_contract(contract)
    if set(contents) != set(contract.required_names):
        raise ValueError(
            "contents must contain the exact artifact names required by the contract"
        )
    prepared = {name: contents[name] for name in contract.required_names}
    if any(not isinstance(text, str) for text in prepared.values()):
        raise TypeError("artifact contents must be text strings")

    final_paths = {name: output_root / name for name in contract.required_names}
    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock = output_root.with_name(output_root.name + ".publish.lock")
    owner_metadata = json.dumps(
        {"pid": os.getpid(), "token": secrets.token_hex(16)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"publication lock already exists for {output_root}: {lock}; refusing to modify it"
        ) from exc
    lock_identity = _directory_identity(lock)
    if lock_identity is None:
        raise RuntimeError(f"could not identify owned publication lock {lock}")

    owner_written = False
    staging: Path | None = None
    staging_identity: _PathIdentity | None = None
    staged_package: dict[str, bytes] | None = None
    backup: Path | None = None
    backup_identity: _PathIdentity | None = None
    old_package: dict[str, bytes] | None = None
    backup_verified = False
    publish_confirmed = False
    try:
        (lock / "owner.json").write_bytes(owner_metadata)
        owner_written = True
        staging, staging_identity = _create_unique_staging(
            parent, output_root.name + ".staging"
        )
        for name, text in prepared.items():
            # Keep package bytes identical to the UTF-8 text used by callers
            # for content digests.  ``write_text`` translates LF to CRLF on
            # Windows, which would otherwise make manifest digests unverifiable.
            (staging / name).write_bytes(text.encode("utf-8"))
        staged_package = _package_bytes(staging, contract)
        if staged_package is None:
            raise ValueError("staged package does not satisfy its artifact contract")

        if _lexists(output_root):
            old_identity = _directory_identity(output_root)
            old_package = _package_bytes(output_root, contract)
            if old_identity is None or old_package is None:
                raise ValueError(
                    f"existing output root {output_root} is not a recognizable artifact directory"
                )
            backup, backup_identity = _move_to_unique_backup(
                output_root,
                parent,
                output_root.name + ".backup",
            )
            backup_verified = (
                backup_identity == old_identity
                and _package_bytes(backup, contract) == old_package
            )
            if not backup_verified:
                raise RuntimeError(
                    f"unexpected output root changed during replacement; "
                    f"recovery backup preserved at {backup}"
                )

        staging_source = staging
        _rename_no_replace(staging_source, output_root)
        staging = None
        staging_identity = None
        if _package_bytes(output_root, contract) != staged_package:
            raise RuntimeError(
                f"publish failed and unexpected output root was preserved at {output_root}; "
                f"recovery backup preserved at {backup}"
            )
        publish_confirmed = True
    except Exception as publish_exc:
        if backup is not None and _lexists(backup):
            if not backup_verified:
                raise RuntimeError(
                    f"publish failed after the output root changed; "
                    f"recovery backup preserved at {backup}"
                ) from publish_exc
            if _lexists(output_root):
                raise RuntimeError(
                    f"publish failed and unexpected output root was preserved at {output_root}; "
                    f"recovery backup preserved at {backup}"
                ) from publish_exc
            restore_source = backup
            try:
                _rename_no_replace(restore_source, output_root)
            except Exception as restore_exc:
                if _lexists(output_root):
                    raise RuntimeError(
                        f"publish failed and unexpected output root was preserved at {output_root}; "
                        f"recovery backup preserved at {backup}"
                    ) from publish_exc
                raise RuntimeError(
                    f"publish failed and restore failed; recovery backup at {backup}: {restore_exc}"
                ) from publish_exc
            backup = None
            backup_identity = None
            if _package_bytes(output_root, contract) != old_package:
                raise RuntimeError(
                    f"publish failed and restore failed; recovery package preserved at {output_root}"
                ) from publish_exc
        raise
    finally:
        active_exception = sys.exc_info()[1]
        cleanup_errors: list[str] = []

        if staging is not None and staging_identity is not None:
            error = _cleanup_owned_directory(staging, staging_identity)
            if error is not None:
                cleanup_errors.append(error)

        if (
            publish_confirmed
            and backup is not None
            and backup_identity is not None
            and old_package is not None
        ):
            error = _cleanup_owned_directory(
                backup,
                backup_identity,
                expected_package=old_package,
                contract=contract,
            )
            if error is not None:
                cleanup_errors.append(error)

        try:
            if owner_written:
                _remove_owned_lock(lock, lock_identity, owner_metadata)
            elif _lexists(lock) and _directory_identity(lock) == lock_identity:
                lock.rmdir()
        except OSError as exc:
            cleanup_errors.append(f"could not remove owned publication lock {lock}: {exc}")

        if cleanup_errors:
            message = "artifact publication cleanup failed: " + "; ".join(cleanup_errors)
            if active_exception is None:
                raise RuntimeError(message)
            active_exception.add_note(message)

    return final_paths
