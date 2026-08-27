"""Tests for the frozen R12 Dataset-A 70/15/15 group split."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _data(n_groups: int = 40) -> tuple[list[str], dict[str, str | None], dict[str, str]]:
    ids: list[str] = []
    labels: dict[str, str | None] = {}
    groups: dict[str, str] = {}
    for index in range(n_groups):
        group = f"wake-{index:03d}"
        for suffix, label in (("p", f"text-{index}"), ("n", None)):
            sample_id = f"{index:03d}-{suffix}"
            ids.append(sample_id)
            labels[sample_id] = label
            groups[sample_id] = group
    return ids, labels, groups


def _counts(mapping: dict[str, str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for role in mapping.values():
        result[role] = result.get(role, 0) + 1
    return result


def test_twenty_fold_mapping_is_70_15_15_and_group_disjoint() -> None:
    from xh202615.r12_dataa_augmented_split import build_augmented_internal_split

    ids, labels, groups = _data()
    manifest = build_augmented_internal_split(ids, labels, groups)

    assert manifest.schema_version == "r12_dataa_augmented_split_v1"
    assert manifest.seed == 20260812
    assert _counts(dict(manifest.roles_by_id)) == {
        "train": 56,
        "validation": 12,
        "internal_test": 12,
    }
    for group in set(groups.values()):
        roles = {manifest.roles_by_id[sid] for sid, value in groups.items() if value == group}
        assert len(roles) == 1


def test_serialized_manifest_has_no_private_fields(tmp_path: Path) -> None:
    from xh202615.r12_dataa_augmented_split import (
        build_augmented_internal_split,
        write_augmented_internal_split,
    )

    ids, labels, groups = _data()
    path = tmp_path / "split.json"
    write_augmented_internal_split(path, build_augmented_internal_split(ids, labels, groups))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {
        "schema_version",
        "seed",
        "roles_by_id",
        "groups_by_id",
        "role_counts",
        "group_counts",
        "source_digests",
        "manifest_sha256",
    }
    rendered = path.read_text(encoding="utf-8").lower()
    assert all(word not in rendered for word in ("label", "reference", "target", "recognition"))


def test_loader_rejects_cross_role_group_after_digest_is_recomputed(tmp_path: Path) -> None:
    from xh202615.r12_dataa_augmented_split import (
        build_augmented_internal_split,
        load_augmented_internal_split,
        manifest_sha256,
        write_augmented_internal_split,
    )

    ids, labels, groups = _data()
    path = tmp_path / "split.json"
    write_augmented_internal_split(path, build_augmented_internal_split(ids, labels, groups))
    data = json.loads(path.read_text(encoding="utf-8"))
    first_train = next(sid for sid, role in data["roles_by_id"].items() if role == "train")
    data["roles_by_id"][first_train] = "validation"
    data["role_counts"] = _counts(data["roles_by_id"])
    data["manifest_sha256"] = manifest_sha256(data)
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="group"):
        load_augmented_internal_split(path, ids)


def test_fixed_seed_and_exact_input_coverage_are_required() -> None:
    from xh202615.r12_dataa_augmented_split import build_augmented_internal_split

    ids, labels, groups = _data()
    with pytest.raises(ValueError, match="seed"):
        build_augmented_internal_split(ids, labels, groups, seed=1)
    with pytest.raises(ValueError, match="label"):
        build_augmented_internal_split(ids, {**labels, "extra": None}, groups)
