"""Build the private-label R12 Dataset-A 70/15/15 split manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.r12_dataa_augmented_split import build_augmented_internal_split, write_augmented_internal_split


def _mapping(path: Path, name: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _raw_ids(dataset_root: Path) -> list[str]:
    ids: list[str] = []
    for split in ("pos", "neg"):
        path = dataset_root / f"{split}.jsonl"
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), (str, int)):
                raise ValueError(f"{path}:{number} has invalid id")
            ids.append(str(row["id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("raw Dataset-A has duplicate IDs")
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasetA" / "datasetA")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ids = _raw_ids(args.dataset_root)
    labels = _mapping(args.labels, "labels")
    groups = _mapping(args.groups, "groups")
    manifest = build_augmented_internal_split(ids, labels, groups)  # type: ignore[arg-type]
    write_augmented_internal_split(args.output, manifest)
    print(json.dumps({"output": str(args.output), "manifest_sha256": manifest.manifest_sha256, "role_counts": {role: list(manifest.roles_by_id.values()).count(role) for role in ("train", "validation", "internal_test")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
