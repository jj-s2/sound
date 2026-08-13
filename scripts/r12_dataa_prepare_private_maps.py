"""Recreate private Dataset-A labels and frozen wake-component maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.r12_dataa_private_maps import build_private_maps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasetA" / "datasetA")
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path, required=True)
    parser.add_argument("--groups-output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_private_maps(args.dataset_root, args.group_manifest, args.labels_output, args.groups_output)
    print(json.dumps({"count": summary.count, "labels": str(summary.labels_output), "labels_sha256": summary.labels_sha256, "groups": str(summary.groups_output), "groups_sha256": summary.groups_sha256}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
