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

from xh202615.firered_model_assets import FireRedModelPaths
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
    manifest.write_text(_json({"artifact_kind": "firered_model_assets", "schema_version": "v1", "aggregate_sha256": "a" * 64, "raw_sha256": {}, "upstream": {"repo_id": "fake", "revision": "r"}, "onnx": {}}), encoding="utf-8")
    return FireRedModelPaths(root, root / "pvad.onnx", root / "ecapa", manifest)


class FakeRuntime:
    instances: list["FakeRuntime"] = []
    fail_id: str | None = None
    nonfinite = False

    def __init__(self, _paths: FireRedModelPaths, *, config: object) -> None:
        self.config = config
        self.calls: list[tuple[str, Path, Path]] = []
        FakeRuntime.instances.append(self)

    def extract(self, sample_id: str, wake: Path, command: Path) -> PvadUtteranceFeatures:
        self.calls.append((sample_id, wake, command))
        if sample_id == self.fail_id:
            raise RuntimeError("forced inference failure")
        values = OrderedDict((name, float(index + 1)) for index, name in enumerate(PVAD_GATE_FEATURE_SCHEMA))
        if self.nonfinite:
            values[PVAD_GATE_FEATURE_SCHEMA[0]] = math.nan
        return PvadUtteranceFeatures(sample_id, values, {"elapsed_seconds": 1.0, "audio_seconds": 2.0, "rtf": 0.5, "peak_rss_delta_bytes": 3, "extraction_phase": "cold", "onnx_provider": "CPUExecutionProvider", "ecapa_device": "cpu"})


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
    resume = next((tmp_path / "resume").glob("*.json"))
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
