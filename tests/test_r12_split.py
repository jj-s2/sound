"""Tests for the frozen R12 60/20/20 group split manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pytest


def _make_balanced_data(n_groups: int = 10):
    """Return deterministic ids, labels, groups with one pos/neg per group."""
    ids_in_order: list[str] = []
    labels: dict[str, str | None] = {}
    groups: dict[str, str] = {}
    for g in range(n_groups):
        group = f"group_{g:02d}"
        pos_id = f"s{2 * g:03d}"
        neg_id = f"s{2 * g + 1:03d}"
        ids_in_order.extend([pos_id, neg_id])
        labels[pos_id] = f"label for {pos_id}"
        labels[neg_id] = None
        groups[pos_id] = group
        groups[neg_id] = group
    return ids_in_order, labels, groups


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _recompute_manifest_sha256(data: dict[str, Any]) -> str:
    payload = {k: v for k, v in data.items() if k != "manifest_sha256"}
    return _sha256_hex(_canonical_json(payload))


def _write_canonical_jsonl(path: Path, ids_in_order: Sequence[str]) -> None:
    required = {"id", "split", "r3_text", "primary_text", "energy_text", "tse_text", "audio_features", "source_digest"}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sid in ids_in_order:
            row = {
                "id": sid,
                "split": "train",
                "r3_text": "",
                "primary_text": "",
                "energy_text": "",
                "tse_text": "",
                "audio_features": {},
                "source_digest": "a" * 64,
            }
            assert set(row) == required
            handle.write(json.dumps(row, sort_keys=True) + "\n")


@pytest.fixture
def sample_data():
    return _make_balanced_data(10)


class TestBuildR12Split:
    def test_module_and_class_exist(self):
        from xh202615.r12_split import R12SplitManifest, build_r12_split

        assert R12SplitManifest is not None
        assert build_r12_split is not None

    def test_deterministic_output(self, sample_data):
        from xh202615.r12_split import build_r12_split

        ids_in_order, labels, groups = sample_data
        manifest_a = build_r12_split(ids_in_order, labels, groups)
        manifest_b = build_r12_split(ids_in_order, labels, groups)
        assert manifest_a == manifest_b
        assert manifest_a.manifest_sha256 == manifest_b.manifest_sha256

    def test_schema_version_and_seed(self, sample_data):
        from xh202615.r12_split import build_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        assert manifest.schema_version == "r12_split_v1"
        assert manifest.seed == 20260807

    def test_fold_to_role_mapping_60_20_20(self, sample_data):
        from xh202615.r12_split import build_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        role_counts: dict[str, int] = {}
        for role in manifest.roles_by_id.values():
            role_counts[role] = role_counts.get(role, 0) + 1
        assert role_counts.get("train", 0) == 12
        assert role_counts.get("validation", 0) == 4
        assert role_counts.get("held_out_test", 0) == 4

    def test_every_role_has_both_classes(self, sample_data):
        from xh202615.r12_split import build_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        for role in ("train", "validation", "held_out_test"):
            role_ids = [sid for sid, r in manifest.roles_by_id.items() if r == role]
            role_labels = {labels[sid] is not None for sid in role_ids}
            assert role_labels == {True, False}, f"{role} lacks both classes"

    def test_no_group_occurs_in_multiple_roles(self, sample_data):
        from xh202615.r12_split import build_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        role_by_group: dict[str, set[str]] = {}
        for sid, role in manifest.roles_by_id.items():
            group = manifest.groups_by_id[sid]
            role_by_group.setdefault(group, set()).add(role)
        for group, roles in role_by_group.items():
            assert len(roles) == 1, f"group {group} spans {roles}"

    def test_exact_id_coverage_required(self, sample_data):
        from xh202615.r12_split import build_r12_split

        ids_in_order, labels, groups = sample_data
        labels_extra = {**labels, "extra": None}
        with pytest.raises(ValueError, match="label"):
            build_r12_split(ids_in_order, labels_extra, groups)

        groups_extra = {**groups, "extra": "group_extra"}
        with pytest.raises(ValueError, match="group"):
            build_r12_split(ids_in_order, labels, groups_extra)

        ids_extra = [*ids_in_order, "extra"]
        with pytest.raises(ValueError, match="ID"):
            build_r12_split(ids_extra, labels, groups)

    def test_nonempty_group_strings(self, sample_data):
        from xh202615.r12_split import build_r12_split

        ids_in_order, labels, groups = sample_data
        groups_bad = dict(groups)
        groups_bad[ids_in_order[0]] = ""
        with pytest.raises(ValueError, match="group"):
            build_r12_split(ids_in_order, labels, groups_bad)

    def test_only_allowed_seed(self, sample_data):
        from xh202615.r12_split import build_r12_split

        ids_in_order, labels, groups = sample_data
        with pytest.raises(ValueError, match="seed"):
            build_r12_split(ids_in_order, labels, groups, seed=1234)

    def test_roles_are_literal_names_only(self, sample_data):
        from xh202615.r12_split import build_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        for role in manifest.roles_by_id.values():
            assert role in ("train", "validation", "held_out_test")

    def test_label_values_must_be_str_or_none(self, sample_data):
        from xh202615.r12_split import build_r12_split

        ids_in_order, labels, groups = sample_data
        labels_bad = dict(labels)
        labels_bad[ids_in_order[0]] = 123
        with pytest.raises(ValueError, match="label"):
            build_r12_split(ids_in_order, labels_bad, groups)

    def test_groups_must_be_nonempty_strings(self, sample_data):
        from xh202615.r12_split import build_r12_split

        ids_in_order, labels, groups = sample_data
        groups_bad = dict(groups)
        groups_bad[ids_in_order[0]] = 123
        with pytest.raises(ValueError, match="group"):
            build_r12_split(ids_in_order, labels, groups_bad)

    def test_five_feasible_folds_required(self):
        from xh202615.r12_split import build_r12_split

        ids_in_order, labels, groups = _make_balanced_data(2)
        with pytest.raises(ValueError):
            build_r12_split(ids_in_order, labels, groups)


class TestWriteAndLoadR12Split:
    def test_round_trip(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        loaded = load_r12_split(path, expected_ids=ids_in_order)
        assert loaded == manifest

    def test_serial_form_has_no_private_fields(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        text = path.read_text(encoding="utf-8")
        lower = text.lower()
        for forbidden in ("label", "reference", "target", "text"):
            assert forbidden not in lower, f"serial form leaks {forbidden}"

    def test_serial_form_is_deterministic_bytes(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        write_r12_split(path_a, manifest)
        write_r12_split(path_b, manifest)
        assert path_a.read_bytes() == path_b.read_bytes()

    def test_loader_rejects_malformed_json(self, tmp_path: Path):
        from xh202615.r12_split import load_r12_split

        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_r12_split(path, expected_ids=[])

    def test_loader_rejects_unrecognized_schema(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["schema_version"] = "evil"
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="schema"):
            load_r12_split(path, expected_ids=ids_in_order)

    def test_loader_rejects_incomplete_ids(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        with pytest.raises(ValueError, match="ID"):
            load_r12_split(path, expected_ids=ids_in_order[:-1])

    def test_loader_rejects_extra_ids(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        with pytest.raises(ValueError, match="ID"):
            load_r12_split(path, expected_ids=[*ids_in_order, "extra"])

    def test_loader_rejects_duplicate_nested_keys(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        base = json.loads(_canonical_json(manifest.__dict__))
        base["manifest_sha256"] = _recompute_manifest_sha256(base)
        sid = ids_in_order[0]
        role = base["roles_by_id"][sid]
        original_roles = json.dumps(base["roles_by_id"], sort_keys=True)
        duplicate_roles = (
            "{"
            + f"{json.dumps(sid)}: {json.dumps(role)}, "
            + f"{json.dumps(sid)}: {json.dumps(role)}"
            + "}"
        )
        text = json.dumps(base, sort_keys=True).replace(original_roles, duplicate_roles, 1)
        path = tmp_path / "dup.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            load_r12_split(path, expected_ids=ids_in_order)

    def test_loader_rejects_invalid_role_name(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["roles_by_id"] = {**data["roles_by_id"], ids_in_order[0]: "evil"}
        data["manifest_sha256"] = _recompute_manifest_sha256(data)
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="role"):
            load_r12_split(path, expected_ids=ids_in_order)

    def test_loader_rejects_group_cross_role_leakage(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["roles_by_id"] = {**data["roles_by_id"], ids_in_order[1]: "validation"}
        data["manifest_sha256"] = _recompute_manifest_sha256(data)
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="group"):
            load_r12_split(path, expected_ids=ids_in_order)

    def test_loader_rejects_extra_groups(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["groups_by_id"]["extra"] = "group_extra"
        data["manifest_sha256"] = _recompute_manifest_sha256(data)
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="group|ID"):
            load_r12_split(path, expected_ids=ids_in_order)

    def test_loader_rejects_missing_groups(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["groups_by_id"][ids_in_order[0]]
        data["manifest_sha256"] = _recompute_manifest_sha256(data)
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="group|ID"):
            load_r12_split(path, expected_ids=ids_in_order)

    def test_loader_rejects_bad_role_counts(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["role_counts"]["train"] += 1
        data["manifest_sha256"] = _recompute_manifest_sha256(data)
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="role_counts"):
            load_r12_split(path, expected_ids=ids_in_order)

    def test_loader_rejects_bad_group_counts(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        first_group = next(iter(data["group_counts"]))
        data["group_counts"][first_group] += 1
        data["manifest_sha256"] = _recompute_manifest_sha256(data)
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="group_counts"):
            load_r12_split(path, expected_ids=ids_in_order)

    def test_loader_rejects_bad_source_digests_schema(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["source_digests"]["evil"] = "a" * 64
        data["manifest_sha256"] = _recompute_manifest_sha256(data)
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="source_digests"):
            load_r12_split(path, expected_ids=ids_in_order)

    def test_loader_rejects_invalid_source_digest_hex(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["source_digests"]["ids"] = "G" * 64
        data["manifest_sha256"] = _recompute_manifest_sha256(data)
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="source_digests|hex"):
            load_r12_split(path, expected_ids=ids_in_order)

    def test_loader_rejects_bad_manifest_sha256(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["manifest_sha256"] = "0" * 64
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="sha256|SHA"):
            load_r12_split(path, expected_ids=ids_in_order)

    def test_loader_rejects_private_label_field(self, sample_data, tmp_path: Path):
        from xh202615.r12_split import build_r12_split, load_r12_split, write_r12_split

        ids_in_order, labels, groups = sample_data
        manifest = build_r12_split(ids_in_order, labels, groups)
        path = tmp_path / "manifest.json"
        write_r12_split(path, manifest)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["labels_by_id"] = {sid: "x" for sid in ids_in_order}
        path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        with pytest.raises(ValueError, match="label|private|field"):
            load_r12_split(path, expected_ids=ids_in_order)


class TestPrepareSplitCLI:
    def test_cli_module_exists(self):
        import scripts.r12_prepare_split as cli

        assert hasattr(cli, "main")
        assert hasattr(cli, "parse_args")

    def test_cli_builds_manifest_from_inputs(self, tmp_path: Path):
        import scripts.r12_prepare_split as cli

        ids_in_order, labels, groups = _make_balanced_data(10)
        jsonl_path = tmp_path / "input.jsonl"
        labels_path = tmp_path / "labels.json"
        groups_path = tmp_path / "groups.json"
        output_path = tmp_path / "manifest.json"

        _write_canonical_jsonl(jsonl_path, ids_in_order)
        labels_path.write_text(
            json.dumps(labels, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        groups_path.write_text(
            json.dumps(groups, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

        summary = cli.main(
            [
                "--canonical-input-jsonl", str(jsonl_path),
                "--labels", str(labels_path),
                "--groups", str(groups_path),
                "--output", str(output_path),
            ]
        )
        assert output_path.exists()
        assert summary["row_count"] == len(ids_in_order)

    def test_cli_requires_exact_canonical_schema(self, tmp_path: Path):
        import scripts.r12_prepare_split as cli

        ids_in_order, labels, groups = _make_balanced_data(10)
        jsonl_path = tmp_path / "input.jsonl"
        labels_path = tmp_path / "labels.json"
        groups_path = tmp_path / "groups.json"
        output_path = tmp_path / "manifest.json"

        with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
            for sid in ids_in_order:
                handle.write(json.dumps({"id": sid, "label": "secret"}, sort_keys=True) + "\n")
        labels_path.write_text(json.dumps(labels, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        groups_path.write_text(json.dumps(groups, sort_keys=True, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(ValueError, match="label-free|exact"):
            cli.main(
                [
                    "--canonical-input-jsonl", str(jsonl_path),
                    "--labels", str(labels_path),
                    "--groups", str(groups_path),
                    "--output", str(output_path),
                ]
            )

    def test_cli_rejects_invalid_label_value(self, tmp_path: Path):
        import scripts.r12_prepare_split as cli

        ids_in_order, labels, groups = _make_balanced_data(10)
        jsonl_path = tmp_path / "input.jsonl"
        labels_path = tmp_path / "labels.json"
        groups_path = tmp_path / "groups.json"
        output_path = tmp_path / "manifest.json"

        _write_canonical_jsonl(jsonl_path, ids_in_order)
        labels_bad = dict(labels)
        labels_bad[ids_in_order[0]] = 123
        labels_path.write_text(json.dumps(labels_bad, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        groups_path.write_text(json.dumps(groups, sort_keys=True, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(ValueError, match="label"):
            cli.main(
                [
                    "--canonical-input-jsonl", str(jsonl_path),
                    "--labels", str(labels_path),
                    "--groups", str(groups_path),
                    "--output", str(output_path),
                ]
            )

    def test_cli_rejects_invalid_group_value_empty(self, tmp_path: Path):
        import scripts.r12_prepare_split as cli

        ids_in_order, labels, groups = _make_balanced_data(10)
        jsonl_path = tmp_path / "input.jsonl"
        labels_path = tmp_path / "labels.json"
        groups_path = tmp_path / "groups.json"
        output_path = tmp_path / "manifest.json"

        _write_canonical_jsonl(jsonl_path, ids_in_order)
        labels_path.write_text(json.dumps(labels, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        groups_bad = dict(groups)
        groups_bad[ids_in_order[0]] = ""
        groups_path.write_text(json.dumps(groups_bad, sort_keys=True, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(ValueError, match="group"):
            cli.main(
                [
                    "--canonical-input-jsonl", str(jsonl_path),
                    "--labels", str(labels_path),
                    "--groups", str(groups_path),
                    "--output", str(output_path),
                ]
            )

    def test_cli_rejects_invalid_group_value_int(self, tmp_path: Path):
        import scripts.r12_prepare_split as cli

        ids_in_order, labels, groups = _make_balanced_data(10)
        jsonl_path = tmp_path / "input.jsonl"
        labels_path = tmp_path / "labels.json"
        groups_path = tmp_path / "groups.json"
        output_path = tmp_path / "manifest.json"

        _write_canonical_jsonl(jsonl_path, ids_in_order)
        labels_path.write_text(json.dumps(labels, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        groups_bad = dict(groups)
        groups_bad[ids_in_order[0]] = 123
        groups_path.write_text(json.dumps(groups_bad, sort_keys=True, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(ValueError, match="group"):
            cli.main(
                [
                    "--canonical-input-jsonl", str(jsonl_path),
                    "--labels", str(labels_path),
                    "--groups", str(groups_path),
                    "--output", str(output_path),
                ]
            )

    def test_cli_rejects_mismatched_ids(self, tmp_path: Path):
        import scripts.r12_prepare_split as cli

        ids_in_order, labels, groups = _make_balanced_data(10)
        jsonl_path = tmp_path / "input.jsonl"
        labels_path = tmp_path / "labels.json"
        groups_path = tmp_path / "groups.json"
        output_path = tmp_path / "manifest.json"

        _write_canonical_jsonl(jsonl_path, ids_in_order)
        labels_path.write_text(
            json.dumps({**labels, "extra": None}, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        groups_path.write_text(
            json.dumps(groups, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="ID"):
            cli.main(
                [
                    "--canonical-input-jsonl", str(jsonl_path),
                    "--labels", str(labels_path),
                    "--groups", str(groups_path),
                    "--output", str(output_path),
                ]
            )
