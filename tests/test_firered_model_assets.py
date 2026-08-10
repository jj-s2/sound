"""Tests for pinned, fail-closed FireRed model acquisition."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from xh202615.firered_model_assets import (
    FIRERED_ALLOW_PATTERNS,
    FIRERED_REPO_ID,
    FIRERED_REVISION,
    download_and_verify_model,
    verify_onnx_contract,
)


INPUTS = (
    ("input_audio", "tensor(float)", (1, 160)),
    ("spkemb", "tensor(float)", (1, 192)),
    ("mel_buffer", "tensor(float)", (1, 80, 15)),
    ("gru_buffer", "tensor(float)", (2, 1, 256)),
)
OUTPUTS = (
    ("output", "tensor(float)", (1, 1)),
    ("prob", "tensor(float)", (1, 1)),
    ("mel_buffer_out", "tensor(float)", (1, 80, 15)),
    ("gru_buffer_out", "tensor(float)", (2, 1, 256)),
)
RAW_FILES = {
    "NOTICE": b"Apache notice\n",
    "README.md": b"FireRed model card\n",
    "pvad.onnx": b"fake onnx bytes\x00\x01",
    "spkrec-ecapa-voxceleb/hyperparams.yaml": b"sample_rate: 16000\n",
    "spkrec-ecapa-voxceleb/embedding_model.ckpt": b"fake ecapa weights\x02",
}


class FakeMetadata:
    def __init__(self, name: str, type_: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.type = type_
        self.shape = list(shape)


class FakeSession:
    def __init__(
        self,
        *,
        inputs: tuple[tuple[str, str, tuple[int, ...]], ...] = INPUTS,
        outputs: tuple[tuple[str, str, tuple[int, ...]], ...] = OUTPUTS,
    ) -> None:
        self._inputs = [FakeMetadata(*metadata) for metadata in inputs]
        self._outputs = [FakeMetadata(*metadata) for metadata in outputs]

    def get_inputs(self) -> list[FakeMetadata]:
        return self._inputs

    def get_outputs(self) -> list[FakeMetadata]:
        return self._outputs

    def run(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("Task 2 preflight must not run model inference")


def fake_session_factory(_model_path: Path) -> FakeSession:
    return FakeSession()


def fake_downloader(
    calls: list[dict[str, object]],
    *,
    files: dict[str, bytes] | None = None,
    extra: tuple[str, bytes] | None = None,
    symlink: str | None = None,
):
    prepared = RAW_FILES if files is None else files

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        for relative, payload in prepared.items():
            path = local_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if extra is not None:
            path = local_dir / extra[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(extra[1])
        if symlink is not None:
            path = local_dir / symlink
            path.unlink()
            try:
                os.symlink(local_dir / "NOTICE", path)
            except OSError:
                pytest.skip("symlink creation is unavailable")
        return str(local_dir)

    return download


def manifest(root: Path) -> dict[str, object]:
    return json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))


def snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_download_is_revision_pinned(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    paths = download_and_verify_model(
        tmp_path / "model",
        downloader=fake_downloader(calls),
        session_factory=fake_session_factory,
    )

    assert calls == [
        {
            "repo_id": FIRERED_REPO_ID,
            "revision": FIRERED_REVISION,
            "local_dir": calls[0]["local_dir"],
            "allow_patterns": list(FIRERED_ALLOW_PATTERNS),
        }
    ]
    assert Path(calls[0]["local_dir"]).name.startswith("model.staging.")
    assert paths.pvad_onnx.is_file()
    assert paths.ecapa_root.is_dir()
    assert paths.manifest.is_file()


def test_manifest_records_raw_file_and_aggregate_digests(tmp_path: Path) -> None:
    root = tmp_path / "model"
    download_and_verify_model(
        root,
        downloader=fake_downloader([]),
        session_factory=fake_session_factory,
    )

    data = manifest(root)
    expected_files = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in sorted(RAW_FILES.items())
    }
    aggregate = hashlib.sha256()
    for name, digest in expected_files.items():
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))

    assert data["artifact_kind"] == "firered_model_assets"
    assert data["schema_version"] == "v1"
    assert data["upstream"]["repo_id"] == FIRERED_REPO_ID
    assert data["upstream"]["revision"] == FIRERED_REVISION
    assert data["raw_sha256"] == expected_files
    assert data["aggregate_sha256"] == aggregate.hexdigest()
    assert data["aggregate_sha256_algorithm"] == (
        "sha256(sorted(path_utf8 + NUL + sha256_bytes))"
    )
    assert data["onnx"] == verify_onnx_contract(FakeSession())


def test_manifest_records_frozen_dependency_versions(tmp_path: Path) -> None:
    root = tmp_path / "model"
    download_and_verify_model(
        root,
        downloader=fake_downloader([]),
        session_factory=fake_session_factory,
    )

    assert manifest(root)["dependency_versions"] == {
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


def test_onnx_contract_records_official_inputs_and_observed_outputs() -> None:
    contract = verify_onnx_contract(FakeSession())

    assert contract == {
        "sample_rate_hz": 16000,
        "frame_samples": 160,
        "probability_output_index": 1,
        "mel_state_output_index": 2,
        "gru_state_output_index": 3,
        "inputs": [
            {"name": name, "type": type_, "shape": list(shape)}
            for name, type_, shape in INPUTS
        ],
        "outputs": [
            {"name": name, "type": type_, "shape": list(shape)}
            for name, type_, shape in OUTPUTS
        ],
    }


@pytest.mark.parametrize(
    "inputs,match",
    [
        (INPUTS[1:], "input names"),
        ((*INPUTS, ("extra", "tensor(float)", (1,))), "input names"),
        ((INPUTS[0], ("speaker", "tensor(float)", (1, 192)), *INPUTS[2:]), "input names"),
        ((INPUTS[0], ("spkemb", "tensor(float16)", (1, 192)), *INPUTS[2:]), "spkemb.*type"),
        ((INPUTS[0], ("spkemb", "tensor(float)", (192,)), *INPUTS[2:]), "spkemb.*shape"),
        ((INPUTS[0], INPUTS[1], ("mel_buffer", "tensor(float)", (1, 80, 14)), INPUTS[3]), "mel_buffer.*shape"),
        ((*INPUTS[:3], ("gru_buffer", "tensor(float)", (1, 2, 256))), "gru_buffer.*shape"),
    ],
)
def test_onnx_contract_rejects_input_disagreement(
    inputs: tuple[tuple[str, str, tuple[int, ...]], ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        verify_onnx_contract(FakeSession(inputs=inputs))


def test_onnx_contract_accepts_input_metadata_in_any_listing_order() -> None:
    permuted = (INPUTS[2], INPUTS[0], INPUTS[3], INPUTS[1])

    contract = verify_onnx_contract(FakeSession(inputs=permuted))

    assert [metadata["name"] for metadata in contract["inputs"]] == [
        name for name, _, _ in INPUTS
    ]


@pytest.mark.parametrize(
    "outputs,match",
    [
        (OUTPUTS[:3], "at least four outputs"),
        ((OUTPUTS[0], OUTPUTS[1], ("mel", "tensor(float16)", (1, 80, 15)), OUTPUTS[3]), "mel state output.*type"),
        ((OUTPUTS[0], OUTPUTS[1], ("mel", "tensor(float)", (1, 80, 14)), OUTPUTS[3]), "mel state output.*shape"),
        ((OUTPUTS[0], OUTPUTS[1], OUTPUTS[2], ("gru", "tensor(float16)", (2, 1, 256))), "GRU state output.*type"),
        ((OUTPUTS[0], OUTPUTS[1], OUTPUTS[2], ("gru", "tensor(float)", (1, 2, 256))), "GRU state output.*shape"),
    ],
)
def test_onnx_contract_rejects_output_disagreement(
    outputs: tuple[tuple[str, str, tuple[int, ...]], ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        verify_onnx_contract(FakeSession(outputs=outputs))


def test_onnx_output_names_probability_metadata_and_extras_are_audit_only() -> None:
    outputs = (
        ("opaque_aux", "tensor(int64)", (9,)),
        ("opaque_target", "tensor(double)", (3, 5)),
        OUTPUTS[2],
        OUTPUTS[3],
        ("opaque_extra", "tensor(string)", (1,)),
    )

    contract = verify_onnx_contract(FakeSession(outputs=outputs))

    assert contract["outputs"] == [
        {"name": name, "type": type_, "shape": list(shape)}
        for name, type_, shape in outputs
    ]
    assert contract["probability_output_index"] == 1


@pytest.mark.parametrize(
    "missing",
    ["pvad.onnx", "NOTICE", "README.md"],
)
def test_missing_required_asset_is_rejected_and_destination_absent(
    tmp_path: Path,
    missing: str,
) -> None:
    root = tmp_path / "model"
    files = dict(RAW_FILES)
    del files[missing]

    with pytest.raises(ValueError, match="required model asset"):
        download_and_verify_model(
            root,
            downloader=fake_downloader([], files=files),
            session_factory=fake_session_factory,
        )

    assert not root.exists()


def test_empty_ecapa_asset_tree_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "model"
    files = {
        name: payload
        for name, payload in RAW_FILES.items()
        if not name.startswith("spkrec-ecapa-voxceleb/")
    }

    with pytest.raises(ValueError, match="below spkrec-ecapa-voxceleb"):
        download_and_verify_model(
            root,
            downloader=fake_downloader([], files=files),
            session_factory=fake_session_factory,
        )

    assert not root.exists()


def test_extra_asset_is_rejected_as_identity_ambiguity(tmp_path: Path) -> None:
    root = tmp_path / "model"

    with pytest.raises(ValueError, match="unexpected model asset"):
        download_and_verify_model(
            root,
            downloader=fake_downloader([], extra=("mystery.bin", b"foreign")),
            session_factory=fake_session_factory,
        )

    assert not root.exists()


def test_huggingface_local_dir_metadata_is_not_published(tmp_path: Path) -> None:
    root = tmp_path / "model"

    download_and_verify_model(
        root,
        downloader=fake_downloader(
            [], extra=(".cache/huggingface/download/pvad.onnx.metadata", b"cache")
        ),
        session_factory=fake_session_factory,
    )

    assert not (root / ".cache").exists()
    assert all(not name.startswith(".cache/") for name in manifest(root)["raw_sha256"])


def test_symlinked_asset_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "model"

    with pytest.raises(ValueError, match="regular non-symlink file"):
        download_and_verify_model(
            root,
            downloader=fake_downloader([], symlink="pvad.onnx"),
            session_factory=fake_session_factory,
        )

    assert not root.exists()


def test_valid_existing_root_is_reused_without_downloading(tmp_path: Path) -> None:
    root = tmp_path / "model"
    download_and_verify_model(
        root,
        downloader=fake_downloader([]),
        session_factory=fake_session_factory,
    )
    before = snapshot_tree(root)
    calls: list[dict[str, object]] = []

    paths = download_and_verify_model(
        root,
        downloader=fake_downloader(calls),
        session_factory=fake_session_factory,
    )

    assert calls == []
    assert paths.root == root
    assert snapshot_tree(root) == before


def test_existing_root_rejects_observed_onnx_metadata_disagreement(tmp_path: Path) -> None:
    root = tmp_path / "model"
    download_and_verify_model(
        root,
        downloader=fake_downloader([]),
        session_factory=fake_session_factory,
    )
    before = snapshot_tree(root)

    def changed_session_factory(_model_path: Path) -> FakeSession:
        outputs = (("changed_aux", "tensor(float)", (1, 1)), *OUTPUTS[1:])
        return FakeSession(outputs=outputs)

    with pytest.raises(ValueError, match="not a recognized pinned FireRed model root"):
        download_and_verify_model(
            root,
            downloader=fake_downloader([]),
            session_factory=changed_session_factory,
        )

    assert snapshot_tree(root) == before


@pytest.mark.parametrize("mutation", ["foreign", "partial", "manifest", "digest", "extra"])
def test_unrecognized_existing_root_is_rejected_and_preserved(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "model"
    root.mkdir()
    if mutation == "foreign":
        (root / "foreign.txt").write_bytes(b"foreign")
    else:
        download_and_verify_model(
            tmp_path / "good",
            downloader=fake_downloader([]),
            session_factory=fake_session_factory,
        )
        for source in (tmp_path / "good").rglob("*"):
            relative = source.relative_to(tmp_path / "good")
            destination = root / relative
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
        if mutation == "partial":
            (root / "NOTICE").unlink()
        elif mutation == "manifest":
            data = manifest(root)
            data["upstream"]["revision"] = "main"
            (root / "model_manifest.json").write_text(json.dumps(data), encoding="utf-8")
        elif mutation == "digest":
            (root / "README.md").write_bytes(b"tampered")
        else:
            (root / "extra.bin").write_bytes(b"ambiguous")
    before = snapshot_tree(root)
    calls: list[dict[str, object]] = []

    with pytest.raises(ValueError, match="not a recognized pinned FireRed model root"):
        download_and_verify_model(
            root,
            downloader=fake_downloader(calls),
            session_factory=fake_session_factory,
        )

    assert calls == []
    assert snapshot_tree(root) == before


def test_existing_symlinked_root_is_rejected_and_preserved(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "marker").write_bytes(b"foreign")
    root = tmp_path / "model"
    try:
        os.symlink(target, root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(ValueError, match="not a recognized pinned FireRed model root"):
        download_and_verify_model(
            root,
            downloader=fake_downloader([]),
            session_factory=fake_session_factory,
        )

    assert root.is_symlink()
    assert (target / "marker").read_bytes() == b"foreign"


def test_staging_directory_is_unique_sibling_and_removed_after_publish(tmp_path: Path) -> None:
    root = tmp_path / "model"
    stale = tmp_path / "model.staging.stale"
    stale.mkdir()
    (stale / "marker").write_bytes(b"keep")
    calls: list[dict[str, object]] = []

    download_and_verify_model(
        root,
        downloader=fake_downloader(calls),
        session_factory=fake_session_factory,
    )

    staging = Path(calls[0]["local_dir"])
    assert staging.parent == root.parent
    assert staging != stale
    assert staging.name.startswith("model.staging.")
    assert not staging.exists()
    assert (stale / "marker").read_bytes() == b"keep"


def test_competing_destination_is_preserved_by_no_replace_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "model"
    from xh202615 import firered_model_assets

    original_move = firered_model_assets._rename_no_replace

    def competing_move(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "foreign.txt").write_bytes(b"competitor")
        original_move(source, destination)

    monkeypatch.setattr(firered_model_assets, "_rename_no_replace", competing_move)

    with pytest.raises(ValueError, match="was preserved but is not a recognized"):
        download_and_verify_model(
            root,
            downloader=fake_downloader([]),
            session_factory=fake_session_factory,
        )

    assert (root / "foreign.txt").read_bytes() == b"competitor"
