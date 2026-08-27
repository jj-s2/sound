"""Compose raw Dataset-A inputs into train-only R12 ASR artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .r12_asr_folds import build_asr_folds, write_asr_folds
from .r12_asr_hotword import prepare_hotword_candidates, rank_hotword_phrases
from .r12_asr_manifest import prepare_asr_manifests
from .r12_dataa_augmentation import build_augmented_dataset, load_lineage
from .r12_dataa_augmented_split import (
    AugmentedInternalSplitManifest,
    build_augmented_internal_split,
    write_augmented_internal_split,
)


@dataclass(frozen=True)
class BootstrapConfig:
    dataset_root: Path
    labels_path: Path
    groups_path: Path
    output_root: Path
    inner_valid_fraction: float = 0.1
    seed: int = 20260814


@dataclass(frozen=True)
class BootstrapPlan:
    split: AugmentedInternalSplitManifest
    labels: Mapping[str, str | None]
    train_parent_ids: frozenset[str]
    inner_valid_parent_ids: frozenset[str]
    inner_valid_groups: frozenset[str]
    fit_groups: frozenset[str]


@dataclass(frozen=True)
class BootstrapResult:
    output_root: Path
    split_path: Path
    lineage_path: Path
    train_jsonl: Path
    inner_valid_jsonl: Path
    folds_path: Path
    hotword_summary: Path


def _load_object(path: Path, name: str) -> dict[str, object]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a JSON object")
    return {str(key): value for key, value in raw.items()}


def _raw_ids(root: Path) -> list[str]:
    values: list[str] = []
    for split in ("pos", "neg"):
        path = root / f"{split}.jsonl"
        if not path.is_file():
            raise ValueError(f"missing raw Dataset-A manifest: {path}")
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), (str, int)):
                raise ValueError(f"{path}:{number} has invalid id")
            values.append(str(row["id"]))
    if len(values) != len(set(values)):
        raise ValueError("raw Dataset-A has duplicate IDs")
    return values


def _select_inner_valid(split: AugmentedInternalSplitManifest, seed: int, fraction: float) -> tuple[frozenset[str], frozenset[str]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("inner_valid_fraction must be between zero and one")
    groups = sorted({split.groups_by_id[sid] for sid, role in split.roles_by_id.items() if role == "train"})
    if len(groups) < 2:
        return frozenset(), frozenset(groups)
    count = max(1, round(fraction * len(groups)))
    count = min(count, len(groups) - 1)
    ranked = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest())
    selected = frozenset(ranked[:count])
    return selected, frozenset(groups) - selected


def plan_bootstrap(config: BootstrapConfig) -> BootstrapPlan:
    root = Path(config.dataset_root)
    ids = _raw_ids(root)
    raw_labels = _load_object(config.labels_path, "labels")
    raw_groups = _load_object(config.groups_path, "groups")
    labels = {key: value if value is None or isinstance(value, str) else (_ for _ in ()).throw(ValueError("labels must be strings or null")) for key, value in raw_labels.items()}
    groups = {key: value if isinstance(value, str) else (_ for _ in ()).throw(ValueError("groups must be strings")) for key, value in raw_groups.items()}
    split = build_augmented_internal_split(ids, labels, groups)  # type: ignore[arg-type]
    train_ids = frozenset(sample_id for sample_id, role in split.roles_by_id.items() if role == "train")
    inner_groups, fit_groups = _select_inner_valid(split, config.seed, config.inner_valid_fraction)
    inner_ids = frozenset(sample_id for sample_id in train_ids if split.groups_by_id[sample_id] in inner_groups)
    return BootstrapPlan(split, labels, train_ids, inner_ids, inner_groups, fit_groups)


def dry_run_bootstrap(config: BootstrapConfig) -> BootstrapPlan:
    if Path(config.output_root).exists():
        raise FileExistsError(config.output_root)
    return plan_bootstrap(config)


def materialize_bootstrap(config: BootstrapConfig) -> BootstrapResult:
    plan = dry_run_bootstrap(config)
    output = Path(config.output_root)
    stage = output.with_name(f"{output.name}.bootstrap-stage-{config.seed}")
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    split_path = stage / "split.json"
    write_augmented_internal_split(split_path, plan.split)
    augmented = build_augmented_dataset(Path(config.dataset_root), plan.split, stage / "augmented")
    train_labels = {sample_id: plan.labels[sample_id] for sample_id in plan.train_parent_ids}
    labels_path = stage / "private" / "train_labels.json"
    labels_path.parent.mkdir()
    labels_path.write_text(json.dumps(train_labels, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    manifests = prepare_asr_manifests(augmented.lineage_path, labels_path, stage / "asr", inner_valid_parent_ids=plan.inner_valid_parent_ids)
    lineage = load_lineage(augmented.lineage_path)
    folds_path = stage / "folds.json"
    write_asr_folds(folds_path, build_asr_folds(lineage))
    phrases = rank_hotword_phrases(train_labels)
    hotwords = prepare_hotword_candidates(labels_path, plan.train_parent_ids, stage / "hotwords", (min(10, len(phrases)),) if phrases else ())
    stage.rename(output)
    return BootstrapResult(output, output / "split.json", output / "augmented" / "augmentation_manifest.jsonl", output / "asr" / "private" / "asr_train.jsonl", output / "asr" / "private" / "asr_inner_valid.jsonl", output / "folds.json", output / "hotwords" / "hotword_summary.json")
