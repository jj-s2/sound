"""Tests for the shared fail-closed artifact publisher."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from xh202615.artifact_publish import ArtifactContract, publish_text_package


CONTRACT = ArtifactContract(
    artifact_kind="test_artifact",
    schema_version="v1",
    required_names=("manifest.json", "summary.json", "payload.txt"),
    identity_json_names=("manifest.json", "summary.json"),
)


def package(marker: str) -> dict[str, str]:
    identity = json.dumps(
        {
            "artifact_kind": CONTRACT.artifact_kind,
            "schema_version": CONTRACT.schema_version,
            "marker": marker,
        },
        sort_keys=True,
    )
    return {
        "manifest.json": identity,
        "summary.json": identity,
        "payload.txt": f"payload:{marker}\n",
    }


def write_package(root: Path, contents: dict[str, str]) -> None:
    root.mkdir()
    for name, text in contents.items():
        (root / name).write_text(text, encoding="utf-8")


def read_marker(root: Path) -> str:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))["marker"]


def tree_snapshot(root: Path) -> dict[str, tuple[str, bytes | None]]:
    return {
        path.name: ("directory", None) if path.is_dir() else ("file", path.read_bytes())
        for path in root.iterdir()
    }


def matching_siblings(parent: Path, prefix: str) -> list[Path]:
    return sorted(path for path in parent.iterdir() if path.name.startswith(prefix))


def single_backup(parent: Path) -> Path:
    backups = matching_siblings(parent, "out.backup.")
    assert len(backups) == 1
    return backups[0]


@pytest.mark.parametrize("mismatch", ["extra", "missing"])
def test_exact_name_rejection_happens_before_any_write(tmp_path: Path, mismatch: str) -> None:
    output_root = tmp_path / "missing-parent" / "out"
    contents = package("bad")
    if mismatch == "extra":
        contents["extra.txt"] = "unexpected"
    else:
        del contents["payload.txt"]

    with pytest.raises(ValueError, match="exact artifact names"):
        publish_text_package(output_root, CONTRACT, contents)

    assert not output_root.parent.exists()
    assert not output_root.with_name("out.publish.lock").exists()


def test_fresh_publication_writes_exact_names_and_contents(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    contents = package("fresh")

    paths = publish_text_package(output_root, CONTRACT, contents)

    assert paths == {name: output_root / name for name in CONTRACT.required_names}
    assert {path.name for path in output_root.iterdir()} == set(CONTRACT.required_names)
    assert {name: path.read_text(encoding="utf-8") for name, path in paths.items()} == contents


def test_publication_preserves_utf8_lf_bytes(tmp_path: Path) -> None:
    """Package digests must be stable on Windows as well as POSIX."""
    output_root = tmp_path / "out"
    contents = package("line-endings")

    paths = publish_text_package(output_root, CONTRACT, contents)

    for name, text in contents.items():
        assert paths[name].read_bytes() == text.encode("utf-8")


def test_foreign_root_is_preserved(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    foreign = output_root / "foreign.txt"
    foreign.write_bytes(b"foreign bytes\x00")

    with pytest.raises(ValueError, match="not a recognizable artifact directory"):
        publish_text_package(output_root, CONTRACT, package("new"))

    assert foreign.read_bytes() == b"foreign bytes\x00"
    assert {path.name for path in output_root.iterdir()} == {"foreign.txt"}
    assert not matching_siblings(tmp_path, "out.staging.")
    assert not (tmp_path / "out.publish.lock").exists()


@pytest.mark.parametrize(
    "foreign_contents,directory_name",
    [
        ({"manifest.json": "not json", "summary.json": package("old")["summary.json"], "payload.txt": "old"}, None),
        ({**package("old"), "manifest.json": json.dumps({"artifact_kind": "other", "schema_version": "v1"})}, None),
        ({**package("old"), "summary.json": json.dumps({"artifact_kind": CONTRACT.artifact_kind, "schema_version": "v2"})}, None),
        (package("old"), "payload.txt"),
    ],
)
def test_exact_name_foreign_roots_are_preserved(
    tmp_path: Path,
    foreign_contents: dict[str, str],
    directory_name: str | None,
) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    for name, text in foreign_contents.items():
        path = output_root / name
        if name == directory_name:
            path.mkdir()
        else:
            path.write_text(text, encoding="utf-8")
    before = tree_snapshot(output_root)

    with pytest.raises(ValueError, match="not a recognizable artifact directory"):
        publish_text_package(output_root, CONTRACT, package("new"))

    assert tree_snapshot(output_root) == before
    assert not matching_siblings(tmp_path, "out.staging.")
    assert not (tmp_path / "out.publish.lock").exists()


def test_recognized_root_is_replaced(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    publish_text_package(output_root, CONTRACT, package("old"))

    new = package("new")
    publish_text_package(output_root, CONTRACT, new)

    assert {name: (output_root / name).read_text(encoding="utf-8") for name in CONTRACT.required_names} == new
    assert not matching_siblings(tmp_path, "out.staging.")
    assert not matching_siblings(tmp_path, "out.backup.")
    assert not (tmp_path / "out.publish.lock").exists()


def test_existing_publication_lock_is_preserved(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    lock = tmp_path / "out.publish.lock"
    lock.mkdir()
    owner = lock / "owner.json"
    owner.write_bytes(b'{"writer":"other"}')

    with pytest.raises(RuntimeError, match="publication lock"):
        publish_text_package(output_root, CONTRACT, package("new"))

    assert owner.read_bytes() == b'{"writer":"other"}'
    assert not output_root.exists()
    assert not matching_siblings(tmp_path, "out.staging.")


def test_changed_lock_owner_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "out"
    lock = tmp_path / "out.publish.lock"
    foreign_owner = b'{"writer":"other"}'

    from xh202615 import artifact_publish

    original_rename = artifact_publish.os.rename

    def changing_owner_rename(src: os.PathLike[str], dst: os.PathLike[str]) -> None:
        (lock / "owner.json").write_bytes(foreign_owner)
        original_rename(src, dst)

    monkeypatch.setattr(artifact_publish.os, "rename", changing_owner_rename)
    publish_text_package(output_root, CONTRACT, package("new"))

    assert read_marker(output_root) == "new"
    assert lock.is_dir()
    assert (lock / "owner.json").read_bytes() == foreign_owner


def test_no_replace_rename_preserves_existing_empty_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "marker").write_bytes(b"source")
    destination.mkdir()

    from xh202615 import artifact_publish

    with pytest.raises(OSError):
        artifact_publish._rename_no_replace(source, destination)

    assert (source / "marker").read_bytes() == b"source"
    assert destination.is_dir()
    assert not list(destination.iterdir())


def test_lock_quarantine_does_not_delete_recreated_foreign_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    lock = tmp_path / "out.publish.lock"
    foreign_owner = b'{"writer":"foreign"}'

    from xh202615 import artifact_publish

    original_move = artifact_publish._rename_no_replace_native
    lock_cleanup_calls = 0

    def swapping_move(source: Path, destination: Path) -> None:
        nonlocal lock_cleanup_calls
        original_move(source, destination)
        if source == lock:
            lock_cleanup_calls += 1
            lock.mkdir()
            (lock / "owner.json").write_bytes(foreign_owner)

    monkeypatch.setattr(artifact_publish, "_rename_no_replace_native", swapping_move)
    publish_text_package(output_root, CONTRACT, package("new"))

    assert lock_cleanup_calls == 1
    assert (lock / "owner.json").read_bytes() == foreign_owner
    assert read_marker(output_root) == "new"


def test_staging_and_backup_paths_are_unique_siblings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "out"
    stale_staging = tmp_path / "out.staging"
    stale_backup = tmp_path / "out.backup"
    stale_staging.mkdir()
    stale_backup.mkdir()
    publish_text_package(output_root, CONTRACT, package("old"))

    from xh202615 import artifact_publish

    original_rename = artifact_publish.os.rename
    renames: list[tuple[Path, Path]] = []

    def recording_rename(src: os.PathLike[str], dst: os.PathLike[str]) -> None:
        renames.append((Path(src), Path(dst)))
        original_rename(src, dst)

    monkeypatch.setattr(artifact_publish.os, "rename", recording_rename)
    publish_text_package(output_root, CONTRACT, package("new"))

    assert renames[0][0] == output_root
    assert renames[0][1].name.startswith("out.backup.")
    assert renames[1][0].name.startswith("out.staging.")
    assert renames[1][1] == output_root
    assert stale_staging.is_dir()
    assert stale_backup.is_dir()


def test_single_publish_rename_failure_restores_old_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "out"
    old = package("old")
    publish_text_package(output_root, CONTRACT, old)
    old_bytes = {name: (output_root / name).read_bytes() for name in CONTRACT.required_names}

    from xh202615 import artifact_publish

    original_rename = artifact_publish.os.rename
    calls = 0

    def failing_rename(src: os.PathLike[str], dst: os.PathLike[str]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced publish rename failure")
        original_rename(src, dst)

    monkeypatch.setattr(artifact_publish.os, "rename", failing_rename)
    with pytest.raises(OSError, match="forced publish rename failure"):
        publish_text_package(output_root, CONTRACT, package("new"))

    assert {name: (output_root / name).read_bytes() for name in CONTRACT.required_names} == old_bytes
    assert not matching_siblings(tmp_path, "out.staging.")
    assert not matching_siblings(tmp_path, "out.backup.")
    assert not (tmp_path / "out.publish.lock").exists()


def test_double_rename_failure_retains_recovery_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "out"
    old = package("old")
    publish_text_package(output_root, CONTRACT, old)
    old_bytes = {name: (output_root / name).read_bytes() for name in CONTRACT.required_names}

    from xh202615 import artifact_publish

    original_rename = artifact_publish.os.rename
    calls = 0

    def failing_rename(src: os.PathLike[str], dst: os.PathLike[str]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_rename(src, dst)
            return
        raise OSError("forced rename failure")

    monkeypatch.setattr(artifact_publish.os, "rename", failing_rename)
    with pytest.raises(RuntimeError, match="recovery backup"):
        publish_text_package(output_root, CONTRACT, package("new"))

    assert not output_root.exists()
    backup = single_backup(tmp_path)
    assert {name: (backup / name).read_bytes() for name in CONTRACT.required_names} == old_bytes
    assert not matching_siblings(tmp_path, "out.staging.")
    assert not (tmp_path / "out.publish.lock").exists()


def test_competing_output_and_old_backup_both_survive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "out"
    publish_text_package(output_root, CONTRACT, package("old"))
    competing = package("competing")

    from xh202615 import artifact_publish

    original_rename = artifact_publish.os.rename
    calls = 0

    def competing_writer_rename(src: os.PathLike[str], dst: os.PathLike[str]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            write_package(output_root, competing)
            raise OSError("simulated competing publication")
        original_rename(src, dst)

    monkeypatch.setattr(artifact_publish.os, "rename", competing_writer_rename)
    with pytest.raises(RuntimeError, match="unexpected output root"):
        publish_text_package(output_root, CONTRACT, package("new"))

    assert read_marker(output_root) == "competing"
    assert read_marker(single_backup(tmp_path)) == "old"
    assert not matching_siblings(tmp_path, "out.staging.")
    assert not (tmp_path / "out.publish.lock").exists()


def test_foreign_directory_at_consumed_staging_path_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    marker = b"foreign staging"

    from xh202615 import artifact_publish

    original_rename = artifact_publish.os.rename
    injected: Path | None = None

    def injecting_rename(src: os.PathLike[str], dst: os.PathLike[str]) -> None:
        nonlocal injected
        original_rename(src, dst)
        if Path(dst) == output_root:
            injected = Path(src)
            injected.mkdir()
            (injected / "marker").write_bytes(marker)

    monkeypatch.setattr(artifact_publish.os, "rename", injecting_rename)
    publish_text_package(output_root, CONTRACT, package("new"))

    assert injected is not None
    assert (injected / "marker").read_bytes() == marker


def test_foreign_directory_at_consumed_backup_path_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    publish_text_package(output_root, CONTRACT, package("old"))
    marker = b"foreign backup"

    from xh202615 import artifact_publish

    original_rename = artifact_publish.os.rename
    calls = 0
    injected: Path | None = None

    def injecting_rename(src: os.PathLike[str], dst: os.PathLike[str]) -> None:
        nonlocal calls, injected
        calls += 1
        if calls == 2:
            raise OSError("forced publish rename failure")
        original_rename(src, dst)
        if calls == 3:
            injected = Path(src)
            injected.mkdir()
            (injected / "marker").write_bytes(marker)

    monkeypatch.setattr(artifact_publish.os, "rename", injecting_rename)
    with pytest.raises(OSError, match="forced publish rename failure"):
        publish_text_package(output_root, CONTRACT, package("new"))

    assert injected is not None
    assert (injected / "marker").read_bytes() == marker
    assert read_marker(output_root) == "old"


def test_replaced_backup_before_cleanup_is_preserved_and_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    publish_text_package(output_root, CONTRACT, package("old"))
    marker = b"foreign backup replacement"

    from xh202615 import artifact_publish

    original_package_bytes = artifact_publish._package_bytes
    calls = 0
    injected: Path | None = None

    def replacing_package_bytes(path: Path, contract: ArtifactContract):
        nonlocal calls, injected
        result = original_package_bytes(path, contract)
        if path == output_root:
            calls += 1
            if calls == 2:
                backup = single_backup(tmp_path)
                for child in backup.iterdir():
                    child.unlink()
                backup.rmdir()
                backup.mkdir()
                (backup / "marker").write_bytes(marker)
                injected = backup
        return result

    monkeypatch.setattr(artifact_publish, "_package_bytes", replacing_package_bytes)
    with pytest.raises(RuntimeError, match="unexpected directory was preserved"):
        publish_text_package(output_root, CONTRACT, package("new"))

    assert injected is not None
    assert (injected / "marker").read_bytes() == marker
    assert read_marker(output_root) == "new"


def test_competing_output_created_during_restore_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    publish_text_package(output_root, CONTRACT, package("old"))
    competing = package("competing")

    from xh202615 import artifact_publish

    original_rename = artifact_publish.os.rename
    calls = 0

    def interleaving_rename(src: os.PathLike[str], dst: os.PathLike[str]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced publish rename failure")
        if calls == 3:
            write_package(output_root, competing)
            raise FileExistsError("competing restore destination")
        original_rename(src, dst)

    monkeypatch.setattr(artifact_publish.os, "rename", interleaving_rename)
    with pytest.raises(RuntimeError, match="unexpected output root"):
        publish_text_package(output_root, CONTRACT, package("new"))

    assert read_marker(output_root) == "competing"
    assert read_marker(single_backup(tmp_path)) == "old"


def test_cleanup_quarantine_does_not_delete_swapped_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    publish_text_package(output_root, CONTRACT, package("old"))
    marker = b"foreign at original backup path"

    from xh202615 import artifact_publish

    original_move = artifact_publish._rename_no_replace_native
    cleanup_calls = 0
    foreign_path: Path | None = None

    def swapping_move(output: Path, quarantine: Path) -> None:
        nonlocal cleanup_calls, foreign_path
        original_move(output, quarantine)
        if ".backup." in output.name:
            cleanup_calls += 1
            output.mkdir()
            (output / "marker").write_bytes(marker)
            foreign_path = output

    monkeypatch.setattr(artifact_publish, "_rename_no_replace_native", swapping_move)
    publish_text_package(output_root, CONTRACT, package("new"))

    assert cleanup_calls == 1
    assert foreign_path is not None
    assert (foreign_path / "marker").read_bytes() == marker
    assert read_marker(output_root) == "new"


def test_cleanup_failure_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "out"

    from xh202615 import artifact_publish

    original_rmdir = Path.rmdir

    def failing_rmdir(path: Path) -> None:
        if path.name.startswith("out.publish.lock.cleanup."):
            raise OSError("forced lock cleanup failure")
        original_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", failing_rmdir)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        publish_text_package(output_root, CONTRACT, package("new"))

    assert read_marker(output_root) == "new"
    quarantined_locks = matching_siblings(tmp_path, "out.publish.lock.cleanup.")
    assert len(quarantined_locks) == 1


def test_byte_identical_replacement(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    contents = package("same")
    publish_text_package(output_root, CONTRACT, contents)
    before = {name: (output_root / name).read_bytes() for name in CONTRACT.required_names}

    publish_text_package(output_root, CONTRACT, contents)

    after = {name: (output_root / name).read_bytes() for name in CONTRACT.required_names}
    assert after == before
    assert not matching_siblings(tmp_path, "out.staging.")
    assert not matching_siblings(tmp_path, "out.backup.")
    assert not (tmp_path / "out.publish.lock").exists()
