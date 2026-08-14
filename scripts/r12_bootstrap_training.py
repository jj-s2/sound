"""Build all local R12 ASR prerequisites from raw Dataset-A."""
from __future__ import annotations
import argparse, dataclasses, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from xh202615.r12_training_bootstrap import BootstrapConfig, dry_run_bootstrap, materialize_bootstrap
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True); parser.add_argument('--labels', type=Path, required=True); parser.add_argument('--groups', type=Path, required=True); parser.add_argument('--output-root', type=Path, required=True); parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv); config = BootstrapConfig(args.dataset_root, args.labels, args.groups, args.output_root)
    result = dry_run_bootstrap(config) if args.dry_run else materialize_bootstrap(config)
    print(json.dumps(dataclasses.asdict(result), ensure_ascii=False, default=str, sort_keys=True)); return 0
if __name__ == '__main__': raise SystemExit(main())
