"""Build the resumable label-free FireRed pVAD feature cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasetA" / "datasetA")
    parser.add_argument("--model-root", type=Path, default=REPO_ROOT / "output" / "models" / "FireRedChat-pvad" / "74561b17a50fbe9d8f84dacc453f175cb97f567c")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output" / "r11_e2_firered_cache")
    parser.add_argument("--resume-root", type=Path, default=REPO_ROOT / "tmp" / "r11_e2_firered_frames")
    parser.add_argument("--ecapa-device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parity-reference", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Imports stay below parse_args so --help never imports neural runtimes.
    from xh202615.firered_model_assets import FireRedModelPaths
    from xh202615.pvad_cache import build_pvad_cache

    root = args.model_root
    paths = FireRedModelPaths(root, root / "pvad.onnx", root / "spkrec-ecapa-voxceleb", root / "model_manifest.json")
    files = build_pvad_cache(args.dataset_root, paths, args.output_root, resume_root=args.resume_root, ecapa_device=args.ecapa_device, limit=args.limit, parity_reference=args.parity_reference)
    print(files["manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
