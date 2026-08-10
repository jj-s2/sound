"""Contract tests for the resumable label-free FireRed feature cache."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import pytest

from xh202615.firered_model_assets import _RAW_SNAPSHOT_FILES, _REQUIRED_DEPENDENCY_VERSIONS, _UPSTREAM_IDENTITY, FireRedModelPaths
from xh202615.firered_pvad import PVAD_GATE_FEATURE_SCHEMA, PvadUtteranceFeatures
from xh202615.pvad_cache import build_pvad_cache


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _audio(path: Path, payload: bytes = b"generated-audio") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _row(sample_id: object, wake: str, command: str, **extra: object) -> dict[str, object]:
    return {"id": sample_id, "wakeup_audio": wake, "command_audio": command, **extra}


def bundle(tmp_path: Path, *, pos: list[dict[str, object]] | None = None, neg: list[dict[str, object]] | None = None) -> Path:
    root = tmp_path / "datasetA"
    root.mkdir(parents=True)
    pos = pos if pos is not None else [_row("10", "wake-10.wav", "command-10.wav", label="secret", wakeup_text="secret wake")]
    neg = neg if neg is not None else [_row("alpha", "wake-alpha.wav", "command-alpha.wav", reference="secret reference")]
    for row in [*pos, *neg]:
        _audio(root / str(row["wakeup_audio"]))
        _audio(root / str(row["command_audio"]))
    for split, rows in (("pos", pos), ("neg", neg)):
        (root / f"{split}.jsonl").write_text("".join(_json(row) + "\n" for row in rows), encoding="utf-8")
    return root


def fake_model(tmp_path: Path) -> FireRedModelPaths:
    root = tmp_path / "model"
    root.mkdir(exist_ok=True)
    manifest = root / "model_manifest.json"
    raw_sha256 = {name: hashlib.sha256(name.encode("utf-8")).hexdigest() for name in _RAW_SNAPSHOT_FILES}
    aggregate_hash = hashlib.sha256()
    for name in sorted(raw_sha256):
        aggregate_hash.update(name.encode("utf-8"))
        aggregate_hash.update(b"\0")
        aggregate_hash.update(bytes.fromhex(raw_sha256[name]))
    onnx = {"sample_rate_hz": 16000, "frame_samples": 160, "probability_output_index": 1, "mel_state_output_index": 2, "gru_state_output_index": 3, "inputs": [{"name": name, "type": "tensor(float)", "shape": list(shape)} for name, shape in (("input_audio", (1, 160)), ("spkemb", (1, 192)), ("mel_buffer", (1, 80, 15)), ("gru_buffer", (2, 1, 256)))], "outputs": [{"name": name, "type": "tensor(float)", "shape": list(shape)} for name, shape in (("output", (1, 1)), ("prob", (1, 1)), ("mel_buffer_out", (1, 80, 15)), ("gru_buffer_out", (2, 1, 256)))]}
    manifest.write_text(_json({"artifact_kind": "firered_model_assets", "schema_version": "v1", "aggregate_sha256": aggregate_hash.hexdigest(), "raw_sha256": raw_sha256, "upstream": _UPSTREAM_IDENTITY, "onnx": onnx, "required_dependency_versions": _REQUIRED_DEPENDENCY_VERSIONS}), encoding="utf-8")
    return FireRedModelPaths(root, root / "pvad.onnx", root / "ecapa", manifest)


class FakeRuntime:
    instances: list["FakeRuntime"] = []
    fail_id: str | None = None
    nonfinite = False

    def __init__(self, _paths: FireRedModelPaths, *, config: object, cuda_peak_bytes: object = None) -> None:
        self.config = config
        self.cuda_peak_bytes = cuda_peak_bytes
        self.calls: list[tuple[str, Path, Path]] = []
        FakeRuntime.instances.append(self)

    def extract(self, sample_id: str, wake: Path, command: Path) -> PvadUtteranceFeatures:
        self.calls.append((sample_id, wake, command))
        if sample_id == self.fail_id:
            raise RuntimeError("forced inference failure")
        values = OrderedDict((name, float(index + 1)) for index, name in enumerate(PVAD_GATE_FEATURE_SCHEMA))
        if self.nonfinite:
            values[PVAD_GATE_FEATURE_SCHEMA[0]] = math.nan
        audit = {"elapsed_seconds": 1.0, "audio_seconds": 2.0, "rtf": 0.5, "peak_rss_delta_bytes": 3, "dropped_tail_samples": 0, "extraction_phase": "cold", "onnx_provider": "CPUExecutionProvider", "ecapa_device": self.config.ecapa_device}
        if self.cuda_peak_bytes is not None:
            audit["cuda_peak_bytes"] = self.cuda_peak_bytes()
        return PvadUtteranceFeatures(sample_id, values, audit)


@pytest.fixture(autouse=True)
def fake_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    from xh202615 import pvad_cache

    FakeRuntime.instances = []
    FakeRuntime.fail_id = None
    FakeRuntime.nonfinite = False
    monkeypatch.setattr(pvad_cache, "FireRedPvadRuntime", FakeRuntime)
    monkeypatch.setattr(pvad_cache, "verify_existing_model", lambda paths: paths)


def call(root: Path, model: FireRedModelPaths, tmp_path: Path, **kwargs: object) -> dict[str, Path]:
    return build_pvad_cache(root, model, tmp_path / "out", resume_root=tmp_path / "resume", ecapa_device="cpu", **kwargs)


def records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_publishes_only_canonical_label_free_records_and_manifest(tmp_path: Path) -> None:
    root = bundle(tmp_path)
    files = call(root, fake_model(tmp_path), tmp_path)

    assert set(files) == {"features", "manifest", "report"}
    assert {path.name for path in (tmp_path / "out").iterdir()} == {"pvad_features.jsonl", "pvad_manifest.json", "pvad_report.md"}
    rows = records(files["features"])
    assert [row["id"] for row in rows] == ["10", "alpha"]
    assert all(tuple(row) == ("id", "features") and tuple(row["features"]) == PVAD_GATE_FEATURE_SCHEMA for row in rows)
    text = files["features"].read_text(encoding="utf-8").lower()
    assert all(forbidden not in text for forbidden in ("label", "reference", "secret", "wake", "audit"))
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "r11_e2_firered_cache"
    assert manifest["schema_version"] == "v1"
    assert manifest["parity"]["status"] == "not-run"
    assert manifest["coverage"]["selected"]["count"] == 2


@pytest.mark.parametrize("bad", ["{\"id\":\"1\",\"id\":\"2\"}", "{\"id\":NaN}", "{\"id\":Infinity}", "[]", "not-json"])
def test_raw_jsonl_rejects_duplicate_keys_nonfinite_and_nonobjects(tmp_path: Path, bad: str) -> None:
    root = bundle(tmp_path)
    (root / "pos.jsonl").write_text(bad + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        call(root, fake_model(tmp_path), tmp_path)


@pytest.mark.parametrize("pos,neg", [([_row("1", "a.wav", "b.wav"), _row("1", "c.wav", "d.wav")], []), ([_row("1", "a.wav", "b.wav")], [_row("1", "c.wav", "d.wav")]), ([_row("", "a.wav", "b.wav")], [])])
def test_ids_are_required_and_unique_across_raw_splits(tmp_path: Path, pos: list[dict[str, object]], neg: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="id"):
        call(bundle(tmp_path, pos=pos, neg=neg), fake_model(tmp_path), tmp_path)


def test_numeric_aware_order_limit_and_label_mutation_independence(tmp_path: Path) -> None:
    root = bundle(tmp_path, pos=[_row("10", "w10.wav", "c10.wav", label="before")], neg=[_row("2", "w2.wav", "c2.wav", reference="before"), _row("alpha", "wa.wav", "ca.wav")])
    first = call(root, fake_model(tmp_path), tmp_path, limit=2)
    assert [row["id"] for row in records(first["features"])] == ["2", "10"]
    manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
    assert manifest["limit"] == {"value": 2, "canonical": False, "reason": "explicit noncanonical partial cache"}
    (root / "pos.jsonl").write_text(_json(_row("10", "w10.wav", "c10.wav", label="after", wakeup_text="different")) + "\n", encoding="utf-8")
    second = call(root, fake_model(tmp_path), tmp_path, limit=2)
    assert first["features"].read_bytes() == second["features"].read_bytes()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_limit_must_be_positive_integer(tmp_path: Path, limit: object) -> None:
    with pytest.raises(ValueError, match="limit"):
        call(bundle(tmp_path), fake_model(tmp_path), tmp_path, limit=limit)


def test_traversal_id_and_symlink_or_special_audio_fail_closed(tmp_path: Path) -> None:
    root = bundle(tmp_path, pos=[_row("../../x", "wake.wav", "command.wav")], neg=[])
    with pytest.raises(ValueError, match="id"):
        call(root, fake_model(tmp_path), tmp_path)
    root = bundle(tmp_path / "other")
    target = root / "target.wav"
    _audio(target)
    try:
        os.symlink(target, root / "wake-10.wav")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="regular non-symlink"):
        call(root, fake_model(tmp_path / "other-model"), tmp_path / "other")


def test_resume_reuse_is_exactly_once_and_digest_mismatch_fails(tmp_path: Path) -> None:
    root = bundle(tmp_path)
    model = fake_model(tmp_path)
    call(root, model, tmp_path)
    assert sum(len(instance.calls) for instance in FakeRuntime.instances) == 2
    call(root, model, tmp_path)
    assert sum(len(instance.calls) for instance in FakeRuntime.instances) == 2
    resume = next(next(path for path in (tmp_path / "resume").iterdir() if path.is_dir()).glob("*.json"))
    record = json.loads(resume.read_text(encoding="utf-8"))
    record["wake_sha256"] = "0" * 64
    resume.write_text(_json(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="resume"):
        call(root, model, tmp_path)


@pytest.mark.parametrize("payload", ["not-json\n", _json({"id": "foreign"}) + "\n"])
def test_malformed_or_foreign_resume_is_preserved_and_rejected(tmp_path: Path, payload: str) -> None:
    root = bundle(tmp_path)
    resume = tmp_path / "resume"
    resume.mkdir()
    poison = resume / "poison.json"
    poison.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="resume"):
        call(root, fake_model(tmp_path), tmp_path)
    assert poison.read_text(encoding="utf-8") == payload


def test_resume_lock_and_interrupted_temp_are_fail_closed_without_overwrite(tmp_path: Path) -> None:
    root = bundle(tmp_path)
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / ".resume.lock").mkdir()
    with pytest.raises(RuntimeError, match="resume lock"):
        call(root, fake_model(tmp_path), tmp_path)
    assert (resume / ".resume.lock").is_dir()


def test_runtime_failure_nonfinite_features_and_foreign_output_never_publish(tmp_path: Path) -> None:
    root = bundle(tmp_path)
    model = fake_model(tmp_path)
    FakeRuntime.fail_id = "10"
    with pytest.raises(RuntimeError, match="forced"):
        call(root, model, tmp_path)
    assert not (tmp_path / "out").exists()
    FakeRuntime.fail_id = None
    FakeRuntime.nonfinite = True
    with pytest.raises(ValueError, match="finite"):
        call(root, model, tmp_path)
    foreign = tmp_path / "out"
    foreign.mkdir()
    marker = foreign / "keep"
    marker.write_bytes(b"foreign")
    FakeRuntime.nonfinite = False
    with pytest.raises(ValueError, match="recognizable"):
        call(root, model, tmp_path)
    assert marker.read_bytes() == b"foreign"


def test_parity_reference_pass_fail_and_schema_id_validation(tmp_path: Path) -> None:
    root = bundle(tmp_path)
    model = fake_model(tmp_path)
    reference = call(root, model, tmp_path / "reference")["features"]
    result = call(root, model, tmp_path, parity_reference=reference)
    assert json.loads(result["manifest"].read_text(encoding="utf-8"))["parity"]["passed"] is True
    bad = tmp_path / "bad.jsonl"
    bad.write_text(_json({"id": "other", "features": {}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parity"):
        call(root, model, tmp_path / "bad", parity_reference=bad)


def test_cli_defaults_are_repo_anchored_and_help_is_lightweight(monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "cache_firered_pvad_features.py"
    spec = importlib.util.spec_from_file_location("cache_firered_pvad_features", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    args = parser.parse_args([])
    assert args.dataset_root == module.REPO_ROOT / "datasetA" / "datasetA"
    assert args.output_root == module.REPO_ROOT / "output" / "r11_e2_firered_cache"
    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["--help"])


class FakeCudaAudit:
    def __init__(self) -> None:
        self.resets = 0

    def reset_peak(self) -> None:
        self.resets += 1

    def peak_bytes(self) -> int:
        return 42

    def evidence(self) -> dict[str, object]:
        return {"cuda_device_name": "fake cuda", "cuda_driver": {"status": "available", "value": "fake-driver"}, "cuda_runtime_version": "fake-runtime"}


def test_cpu_cuda_share_context_namespaced_resume_and_cuda_requires_cpu_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from xh202615 import pvad_cache

    root, model, resume = bundle(tmp_path), fake_model(tmp_path), tmp_path / "resume"
    cpu = build_pvad_cache(root, model, tmp_path / "cpu" / "out", resume_root=resume, ecapa_device="cpu", limit=2)
    with pytest.raises(ValueError, match="parity"):
        build_pvad_cache(root, model, tmp_path / "cuda-no-reference", resume_root=resume, ecapa_device="cuda:0", limit=2)
    adapter = FakeCudaAudit()
    monkeypatch.setattr(pvad_cache, "_cuda_audit_adapter", lambda _device: adapter)
    cuda = build_pvad_cache(root, model, tmp_path / "cuda", resume_root=resume, ecapa_device="cuda:0", limit=2, parity_reference=cpu["features"])
    assert len([child for child in resume.iterdir() if child.is_dir()]) == 2
    manifest = json.loads(cuda["manifest"].read_text(encoding="utf-8"))
    assert manifest["parity"]["passed"] is True
    assert manifest["cuda"]["peak_bytes"]["max"] == 42
    assert adapter.resets == 2


def test_resume_rejects_extra_label_and_poisoned_audit_before_runtime(tmp_path: Path) -> None:
    root, model = bundle(tmp_path), fake_model(tmp_path)
    call(root, model, tmp_path)
    resume = next(path for path in (tmp_path / "resume").iterdir() if path.is_dir())
    record_path = next(resume.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["label"] = "positive"
    record_path.write_text(_json(record) + "\n", encoding="utf-8")
    before = sum(len(instance.calls) for instance in FakeRuntime.instances)
    with pytest.raises(ValueError, match="resume"):
        call(root, model, tmp_path)
    assert sum(len(instance.calls) for instance in FakeRuntime.instances) == before


def test_forged_parity_and_foreign_output_fail_before_runtime(tmp_path: Path) -> None:
    root, model = bundle(tmp_path), fake_model(tmp_path)
    forged = tmp_path / "forged"
    forged.mkdir()
    (forged / "pvad_features.jsonl").write_text("", encoding="utf-8")
    before = sum(len(instance.calls) for instance in FakeRuntime.instances)
    with pytest.raises(ValueError, match="parity"):
        build_pvad_cache(root, model, tmp_path / "cuda", resume_root=tmp_path / "resume", ecapa_device="cuda", parity_reference=forged / "pvad_features.jsonl")
    assert sum(len(instance.calls) for instance in FakeRuntime.instances) == before
    foreign = tmp_path / "out"
    foreign.mkdir()
    (foreign / "foreign").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="recognizable"):
        call(root, model, tmp_path)
    assert sum(len(instance.calls) for instance in FakeRuntime.instances) == before


def test_owned_interrupted_temp_is_quarantined_but_foreign_temp_fails(tmp_path: Path) -> None:
    root, model = bundle(tmp_path), fake_model(tmp_path)
    call(root, model, tmp_path)
    namespace = next(path for path in (tmp_path / "resume").iterdir() if path.is_dir())
    record = next(path for path in namespace.glob("record-*.json"))
    owned = namespace / f".{record.name}.{'a' * 16}.tmp"
    owned.write_bytes(record.read_bytes())
    call(root, model, tmp_path)
    assert not owned.exists()
    foreign = namespace / ".not-ours.tmp"
    foreign.write_text("foreign", encoding="utf-8")
    with pytest.raises(ValueError, match="foreign resume"):
        call(root, model, tmp_path)


def test_manifest_publishes_per_id_audio_model_dependencies_and_recomputed_schema(tmp_path: Path) -> None:
    from xh202615 import pvad_cache

    root, model = bundle(tmp_path), fake_model(tmp_path)
    parsed = json.loads(model.manifest.read_text(encoding="utf-8"))
    files = call(root, model, tmp_path)
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))

    assert manifest["source"]["per_id_audio_sha256"] == {
        "10": {"wake_sha256": hashlib.sha256(b"generated-audio").hexdigest(), "command_sha256": hashlib.sha256(b"generated-audio").hexdigest()},
        "alpha": {"wake_sha256": hashlib.sha256(b"generated-audio").hexdigest(), "command_sha256": hashlib.sha256(b"generated-audio").hexdigest()},
    }
    assert manifest["model"]["required_dependency_versions"] == parsed["required_dependency_versions"]
    assert {"onnxruntime-gpu", "torchaudio", "hyperpyyaml", "PyYAML"} <= set(manifest["environment"]["observed_dependencies"])
    assert pvad_cache._schema_digest() == "610c7e711fda490405a66a01e5ca6e7b01bf230c00333d891ebbaf20140e270f"


def test_incomplete_or_digest_forged_package_is_not_a_parity_reference(tmp_path: Path) -> None:
    root, model = bundle(tmp_path), fake_model(tmp_path)
    reference = call(root, model, tmp_path / "reference")
    manifest_path = reference["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("model", "source", "runtime_config", "runtime_config_sha256", "environment", "timing", "parity"):
        del manifest[key]
    manifest_path.write_text(_json(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="recognizable"):
        call(root, model, tmp_path, parity_reference=reference["features"])

    reference = call(root, model, tmp_path / "different-reference")
    manifest_path = reference["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records_sha256"] = "0" * 64
    manifest_path.write_text(_json(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        call(root, model, tmp_path / "different-output", parity_reference=reference["features"])


def test_unverified_matching_temp_is_preserved_and_verified_duplicate_is_promoted(tmp_path: Path) -> None:
    root, model = bundle(tmp_path), fake_model(tmp_path)
    call(root, model, tmp_path)
    namespace = next(path for path in (tmp_path / "resume").iterdir() if path.is_dir())
    record = next(path for path in namespace.glob("record-*.json"))
    foreign = namespace / f".{record.name}.{'b' * 16}.tmp"
    foreign.write_bytes(b"foreign user bytes")
    with pytest.raises(ValueError, match="temp"):
        call(root, model, tmp_path)
    assert foreign.read_bytes() == b"foreign user bytes"

    foreign.unlink()
    promoted = namespace / f".{record.name}.{'c' * 16}.tmp"
    record_bytes = record.read_bytes()
    record.unlink()
    promoted.write_bytes(record_bytes)
    call(root, model, tmp_path)
    assert record.read_bytes() == record_bytes
    assert not promoted.exists()


def test_resume_context_identity_rejects_empty_and_forged_siblings(tmp_path: Path) -> None:
    root, model = bundle(tmp_path), fake_model(tmp_path)
    resume = tmp_path / "resume"
    resume.mkdir()
    sibling = resume / ("context-" + "a" * 64)
    sibling.mkdir()
    with pytest.raises(ValueError, match="namespace"):
        call(root, model, tmp_path)
    assert sibling.is_dir() and not list(sibling.iterdir())


def _rewrite_manifest(path: Path, manifest: dict[str, object]) -> None:
    from xh202615 import pvad_cache

    path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    (path.parent / "pvad_report.md").write_text(pvad_cache._report(manifest), encoding="utf-8")


def test_frozen_schema_identity_uses_literal_backslash_n_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from xh202615 import pvad_cache

    payload = r"\n".join(PVAD_GATE_FEATURE_SCHEMA) + r"\n"
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == "610c7e711fda490405a66a01e5ca6e7b01bf230c00333d891ebbaf20140e270f"
    assert pvad_cache._schema_digest() == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    monkeypatch.setattr(pvad_cache, "PVAD_GATE_FEATURE_SCHEMA", ("changed",))
    with pytest.raises(ValueError, match="frozen"):
        pvad_cache._schema_digest()


def test_atomic_namespace_creation_preserves_competing_or_malformed_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from xh202615 import pvad_cache

    paths = fake_model(tmp_path)
    model = pvad_cache._model_identity(paths)
    model["identity_sha256"] = pvad_cache._digest(model)
    config = pvad_cache._runtime_config(pvad_cache.asdict(pvad_cache._config("cpu")))
    identity = pvad_cache._context_identity(model, config)
    root = tmp_path / "resume"
    root.mkdir()
    namespace = root / pvad_cache._context_name(model, config)

    def competing_rename(_staging: Path, destination: Path) -> None:
        Path(destination).mkdir()
        (Path(destination) / "context_identity.json").write_bytes((pvad_cache._canonical(identity) + "\n").encode("utf-8"))
        raise FileExistsError(destination)

    monkeypatch.setattr(pvad_cache, "_rename_no_replace", competing_rename)
    pvad_cache._create_namespace(root, namespace.name, identity)
    assert (namespace / "context_identity.json").read_bytes() == (pvad_cache._canonical(identity) + "\n").encode("utf-8")
    assert not list(root.glob(".context-*.tmp"))

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    bad_namespace = malformed / namespace.name
    bad_namespace.mkdir()
    (bad_namespace / "context_identity.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="namespace"):
        pvad_cache._create_namespace(malformed, namespace.name, identity)
    assert (bad_namespace / "context_identity.json").read_text(encoding="utf-8") == "{}\n"


@pytest.mark.parametrize("mutation", ["model", "audio", "jsonl", "config", "coverage"])
def test_cuda_parity_requires_exact_cpu_provenance_before_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    from xh202615 import pvad_cache

    root, model = bundle(tmp_path), fake_model(tmp_path)
    reference = call(root, model, tmp_path / "reference")
    manifest_path = reference["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "model":
        manifest["model"]["aggregate_sha256"] = "f" * 64
    elif mutation == "audio":
        manifest["source"]["per_id_audio_sha256"]["10"]["wake_sha256"] = "f" * 64
    elif mutation == "jsonl":
        manifest["source"]["jsonl_sha256"]["pos"] = "f" * 64
    elif mutation == "config":
        manifest["runtime_config"]["ema_alpha"] = 0.7
        manifest["runtime_config_sha256"] = hashlib.sha256(json.dumps(manifest["runtime_config"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    else:
        manifest["coverage"]["selected"]["id_sha256"] = "f" * 64
    _rewrite_manifest(manifest_path, manifest)
    monkeypatch.setattr(pvad_cache, "_cuda_audit_adapter", lambda _device: pytest.fail("CUDA audit must not be constructed"))
    before = sum(len(instance.calls) for instance in FakeRuntime.instances)

    with pytest.raises(ValueError, match="parity"):
        build_pvad_cache(root, model, tmp_path / "runs" / mutation / "out", resume_root=tmp_path / "runs" / mutation / "resume", ecapa_device="cuda", parity_reference=reference["features"])

    assert sum(len(instance.calls) for instance in FakeRuntime.instances) == before
    assert not (tmp_path / "runs" / mutation / "out").exists()


@pytest.mark.parametrize("field", ["runtime_config_sha256", "coverage", "model", "source", "timing", "reuse", "parity", "limit", "digest_algorithms"])
def test_package_nested_contract_tampering_fails_before_runtime(tmp_path: Path, field: str) -> None:
    root, model = bundle(tmp_path), fake_model(tmp_path)
    reference = call(root, model, tmp_path / "reference")
    manifest_path = reference["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "runtime_config_sha256":
        manifest[field] = "f" * 64
    elif field == "digest_algorithms":
        manifest[field]["model_sha256"] = "wrong"
    elif field == "model":
        manifest[field]["onnx"] = []
    elif field == "source":
        manifest[field]["jsonl_sha256"]["pos"] = "invalid"
    elif field == "coverage":
        manifest[field]["source"]["count"] = -1
    elif field == "timing":
        manifest[field]["rtf"]["p50"] = -1
    elif field == "reuse":
        manifest[field]["new"] = -1
    elif field == "parity":
        manifest[field]["status"] = "unexpected"
    else:
        manifest[field]["reason"] = "wrong"
    _rewrite_manifest(manifest_path, manifest)
    before = sum(len(instance.calls) for instance in FakeRuntime.instances)

    with pytest.raises(ValueError):
        call(root, model, tmp_path / "runs" / field, parity_reference=reference["features"])

    assert sum(len(instance.calls) for instance in FakeRuntime.instances) == before


def test_poisoned_sibling_record_and_matching_temp_fail_in_place(tmp_path: Path) -> None:
    from xh202615 import pvad_cache

    root, model = bundle(tmp_path), fake_model(tmp_path)
    call(root, model, tmp_path)
    resume = tmp_path / "resume"
    namespace = next(path for path in resume.iterdir() if path.is_dir())
    identity = json.loads((namespace / "context_identity.json").read_text(encoding="utf-8"))
    identity["runtime_config"]["ecapa_device"] = "cuda"
    identity["runtime_config_sha256"] = pvad_cache._digest(identity["runtime_config"])
    sibling = resume / pvad_cache._context_name(identity["model"], identity["runtime_config"])
    identity = pvad_cache._context_identity(identity["model"], identity["runtime_config"])
    sibling.mkdir()
    (sibling / "context_identity.json").write_text(json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    record = json.loads(next(namespace.glob("record-*.json")).read_text(encoding="utf-8"))
    record["runtime_config"] = identity["runtime_config"]
    record["runtime_config_sha256"] = identity["runtime_config_sha256"]
    record["audit"]["ecapa_device"] = "cuda"
    record["values"] = {"wrong": 1}
    poison = sibling / ("record-" + hashlib.sha256(record["id"].encode()).hexdigest()[:32] + ".json")
    poison.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="namespace|forged"):
        pvad_cache._validate_context(sibling)
    assert poison.exists()

    poison.unlink()
    malformed = sibling / (".record-" + hashlib.sha256(record["id"].encode()).hexdigest()[:32] + ".json." + "d" * 16 + ".tmp")
    malformed.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="namespace|forged"):
        pvad_cache._validate_context(sibling)
    assert malformed.exists()


@pytest.mark.parametrize("mutation", ["cuda-on-cpu", "cuda-without-parity", "reuse", "parity", "timing", "provider"])
def test_package_cross_field_evidence_is_fail_closed(tmp_path: Path, mutation: str) -> None:
    from xh202615 import pvad_cache

    root, model = bundle(tmp_path), fake_model(tmp_path)
    reference = call(root, model, tmp_path)
    manifest_path = reference["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "cuda-on-cpu":
        manifest["cuda"] = {"evil": "accepted"}
    elif mutation == "cuda-without-parity":
        manifest["runtime_config"]["ecapa_device"] = "cuda"
        manifest["runtime_config_sha256"] = pvad_cache._digest(manifest["runtime_config"])
        manifest["device"] = "cuda"
        manifest["joined_state_sha256"] = pvad_cache._digest({sample_id: pvad_cache._resume_expected(sample_id, manifest["source"]["per_id_audio_sha256"][sample_id], manifest["model"], manifest["runtime_config"]) for sample_id in manifest["coverage"]["selected"]["ids"]})
    elif mutation == "reuse":
        manifest["reuse"] = {"reused": 999, "new": 999}
    elif mutation == "parity":
        manifest["parity"] = {"status": "passed", "passed": True, "max_abs_feature_delta": 1.0}
    elif mutation == "timing":
        manifest["timing"]["rtf"] = {"count": 2, "p50": None, "p95": None, "max": None}
    else:
        manifest["runtime_config"]["onnx_provider"] = "CUDAExecutionProvider"
        manifest["runtime_config_sha256"] = pvad_cache._digest(manifest["runtime_config"])
        manifest["provider"] = "CUDAExecutionProvider"
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(ValueError):
        pvad_cache._validate_package(reference["features"].parent)


def test_context_rejects_forged_model_and_swapped_staging_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from xh202615 import pvad_cache

    root, model = bundle(tmp_path), fake_model(tmp_path)
    call(root, model, tmp_path)
    resume = tmp_path / "resume"
    namespace = next(path for path in resume.iterdir() if path.is_dir())
    identity = json.loads((namespace / "context_identity.json").read_text(encoding="utf-8"))
    identity["model"]["extra_label"] = "forged"
    identity["model"]["identity_sha256"] = pvad_cache._digest({key: value for key, value in identity["model"].items() if key != "identity_sha256"})
    forged = pvad_cache._context_identity(identity["model"], identity["runtime_config"])
    sibling = resume / pvad_cache._context_name(forged["model"], forged["runtime_config"])
    sibling.mkdir()
    (sibling / "context_identity.json").write_bytes((pvad_cache._canonical(forged) + "\n").encode("utf-8"))
    with pytest.raises(ValueError):
        pvad_cache._validate_context(sibling)

    clean_identity = json.loads((namespace / "context_identity.json").read_text(encoding="utf-8"))
    name = pvad_cache._context_name(clean_identity["model"], clean_identity["runtime_config"])
    target_name = name + "f"
    staging = resume / f".{target_name}.staging.owned"
    moved = resume / ".owned-moved"
    foreign = b"competing evidence"
    original_rename = pvad_cache._rename_no_replace

    def swap_then_fail(source: Path, destination: Path) -> None:
        if Path(source) == staging:
            os.rename(source, moved)
            staging.mkdir()
            (staging / "evidence").write_bytes(foreign)
            raise FileExistsError(destination)
        original_rename(source, destination)

    monkeypatch.setattr(pvad_cache.secrets, "token_hex", lambda _size: "owned")
    monkeypatch.setattr(pvad_cache, "_rename_no_replace", swap_then_fail)
    with pytest.raises((ValueError, RuntimeError)):
        pvad_cache._create_namespace(resume, target_name, clean_identity)
    assert (staging / "evidence").read_bytes() == foreign
    assert (moved / "context_identity.json").is_file()


@pytest.mark.parametrize("mutation", ["ema", "limit", "environment", "duplicate-output", "empty-driver"])
def test_package_rejects_noncanonical_nested_domains(tmp_path: Path, mutation: str) -> None:
    from xh202615 import pvad_cache

    root, model = bundle(tmp_path), fake_model(tmp_path)
    package = call(root, model, tmp_path)
    manifest = json.loads(package["manifest"].read_text(encoding="utf-8"))
    if mutation == "ema":
        manifest["runtime_config"]["ema_alpha"] = 0.7
        manifest["runtime_config_sha256"] = pvad_cache._digest(manifest["runtime_config"])
        manifest["joined_state_sha256"] = pvad_cache._digest({sample_id: pvad_cache._resume_expected(sample_id, manifest["source"]["per_id_audio_sha256"][sample_id], manifest["model"], manifest["runtime_config"]) for sample_id in manifest["coverage"]["selected"]["ids"]})
    elif mutation == "limit":
        manifest["limit"] = {"value": 1, "canonical": False, "reason": "explicit noncanonical partial cache"}
    elif mutation == "environment":
        manifest["environment"]["observed_dependencies"]["label"] = "secret"
    elif mutation == "duplicate-output":
        manifest["model"]["onnx"]["outputs"][0]["name"] = manifest["model"]["onnx"]["outputs"][1]["name"]
        manifest["model"]["identity_sha256"] = pvad_cache._digest({key: value for key, value in manifest["model"].items() if key != "identity_sha256"})
        manifest["joined_state_sha256"] = pvad_cache._digest({sample_id: pvad_cache._resume_expected(sample_id, manifest["source"]["per_id_audio_sha256"][sample_id], manifest["model"], manifest["runtime_config"]) for sample_id in manifest["coverage"]["selected"]["ids"]})
    else:
        cpu = call(root, model, tmp_path / "cpu")
        manifest = json.loads(cpu["manifest"].read_text(encoding="utf-8"))
        manifest["runtime_config"]["ecapa_device"] = "cuda"
        manifest["runtime_config_sha256"] = pvad_cache._digest(manifest["runtime_config"])
        manifest["device"] = "cuda"
        manifest["parity"] = {"status": "passed", "passed": True, "max_abs_feature_delta": 0.0}
        manifest["timing"]["cuda_peak_bytes"] = {"count": 2, "p50": 1, "p95": 1, "max": 1}
        manifest["cuda"] = {"cuda_device_name": "fake", "cuda_runtime_version": "fake", "cuda_driver": {"status": "available", "value": ""}, "peak_bytes": manifest["timing"]["cuda_peak_bytes"]}
        manifest["joined_state_sha256"] = pvad_cache._digest({sample_id: pvad_cache._resume_expected(sample_id, manifest["source"]["per_id_audio_sha256"][sample_id], manifest["model"], manifest["runtime_config"]) for sample_id in manifest["coverage"]["selected"]["ids"]})
        package = cpu
    _rewrite_manifest(package["manifest"], manifest)
    with pytest.raises(ValueError):
        pvad_cache._validate_package(package["features"].parent)


def test_cpu_parity_package_is_recognized_and_staging_content_mutation_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from xh202615 import pvad_cache

    root, model = bundle(tmp_path), fake_model(tmp_path)
    reference = call(root, model, tmp_path / "reference")
    package = call(root, model, tmp_path, parity_reference=reference["features"])
    pvad_cache._validate_package(package["features"].parent)

    identity = json.loads(next((tmp_path / "resume").glob("context-*/context_identity.json")).read_text(encoding="utf-8"))
    target = "context-" + "f" * 64
    staging = tmp_path / "resume" / f".{target}.staging.owned"
    original = pvad_cache._rename_no_replace

    def mutate_then_fail(source: Path, destination: Path) -> None:
        if source == staging:
            (staging / "foreign").write_bytes(b"preserve")
            raise FileExistsError(destination)
        original(source, destination)

    monkeypatch.setattr(pvad_cache.secrets, "token_hex", lambda _size: "owned")
    monkeypatch.setattr(pvad_cache, "_rename_no_replace", mutate_then_fail)
    with pytest.raises(RuntimeError, match="cleanup"):
        pvad_cache._create_namespace(tmp_path / "resume", target, identity)
    assert (staging / "foreign").read_bytes() == b"preserve"


@pytest.mark.parametrize("mutation", ["audio", "model", "config"])
def test_cpu_parity_requires_current_provenance_before_runtime(tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from xh202615 import pvad_cache

    root, model = bundle(tmp_path), fake_model(tmp_path)
    reference = call(root, model, tmp_path / "reference")
    if mutation == "audio":
        (root / "command-10.wav").write_bytes(b"different command")
    elif mutation == "model":
        parsed = json.loads(model.manifest.read_text(encoding="utf-8"))
        parsed["raw_sha256"]["NOTICE"] = "c" * 64
        aggregate = hashlib.sha256()
        for name in sorted(parsed["raw_sha256"]):
            aggregate.update(name.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(bytes.fromhex(parsed["raw_sha256"][name]))
        parsed["aggregate_sha256"] = aggregate.hexdigest()
        model.manifest.write_text(_json(parsed), encoding="utf-8")
    else:
        original = pvad_cache._config
        monkeypatch.setattr(pvad_cache, "_config", lambda device: type(original(device))(**{**pvad_cache.asdict(original(device)), "ema_alpha": 0.7}))
    before = sum(len(instance.calls) for instance in FakeRuntime.instances)
    with pytest.raises(ValueError, match="parity"):
        call(root, model, tmp_path, parity_reference=reference["features"])
    assert sum(len(instance.calls) for instance in FakeRuntime.instances) == before


@pytest.mark.parametrize("field,value", [("dropped_tail_samples", 1.5), ("peak_rss_delta_bytes", 1.5), ("audio_seconds", 0.0), ("rtf", 0.1)])
def test_reused_audit_numeric_domains_are_exact_and_preserved(tmp_path: Path, field: str, value: object) -> None:
    root, model = bundle(tmp_path), fake_model(tmp_path)
    call(root, model, tmp_path)
    record_path = next(next((tmp_path / "resume").glob("context-*")).glob("record-*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["audit"][field] = value
    payload = json.dumps(record, separators=(",", ":")) + "\n"
    record_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="resume"):
        call(root, model, tmp_path)
    assert record_path.read_text(encoding="utf-8") == payload


def test_cuda_reused_memory_counter_must_be_integral_and_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from xh202615 import pvad_cache

    root, model = bundle(tmp_path), fake_model(tmp_path)
    cpu = call(root, model, tmp_path / "cpu")
    monkeypatch.setattr(pvad_cache, "_cuda_audit_adapter", lambda _device: FakeCudaAudit())
    build_pvad_cache(root, model, tmp_path / "cuda" / "out", resume_root=tmp_path / "cuda" / "resume", ecapa_device="cuda", parity_reference=cpu["features"])
    record_path = next(next((tmp_path / "cuda" / "resume").glob("context-*")).glob("record-*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["audit"]["cuda_peak_bytes"] = 1.5
    payload = json.dumps(record, separators=(",", ":")) + "\n"
    record_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="resume"):
        build_pvad_cache(root, model, tmp_path / "cuda" / "next", resume_root=tmp_path / "cuda" / "resume", ecapa_device="cuda", parity_reference=cpu["features"])
    assert record_path.read_text(encoding="utf-8") == payload


def test_nonregular_staging_replacement_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from xh202615 import pvad_cache

    root, model = bundle(tmp_path), fake_model(tmp_path)
    call(root, model, tmp_path)
    resume = tmp_path / "resume"
    identity = json.loads(next(resume.glob("context-*/context_identity.json")).read_text(encoding="utf-8"))
    target = "context-" + "e" * 64
    staging = resume / f".{target}.staging.owned"
    moved = resume / ".moved-owned"
    original = pvad_cache._rename_no_replace

    def replace_then_publish(source: Path, destination: Path) -> None:
        if source == staging:
            os.rename(source, moved)
            staging.write_bytes(b"foreign")
            destination.mkdir()
            (destination / "context_identity.json").write_bytes((pvad_cache._canonical(identity) + "\n").encode("utf-8"))
            raise FileExistsError(destination)
        original(source, destination)

    monkeypatch.setattr(pvad_cache.secrets, "token_hex", lambda _size: "owned")
    monkeypatch.setattr(pvad_cache, "_rename_no_replace", replace_then_publish)
    with pytest.raises(RuntimeError, match="cleanup"):
        pvad_cache._create_namespace(resume, target, identity)
    assert staging.read_bytes() == b"foreign"
    assert (moved / "context_identity.json").is_file()
