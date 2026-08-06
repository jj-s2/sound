"""Build the R7 training manifest: R3 counterfactual + impostor hard negatives.

Reads the public R3 counterfactual manifest and writes a balanced (1:1
present/absent) training manifest where a deterministic fraction of negatives
are **impostor** hard negatives (a different-speaker enrollment over the same
mixture) and the rest are the R3 counterfactual negatives (interferer+noise on
the same acoustic grid). Dataset-A is never read.

See ``xh202615/r7_hard_negatives`` for the construction contract and the R7
design doc (``docs/arena/2026-08-06-r7-speaker-score-design.md``) for the
rationale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.r7_hard_negatives import prepare_r7_manifest  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r3-manifest", required=True,
        help="public R3 counterfactual manifest JSONL (R3MixtureRow)",
    )
    parser.add_argument("--output", required=True, help="output training manifest JSONL")
    parser.add_argument("--dataset-a-root", default="datasetA/datasetA")
    parser.add_argument(
        "--impostor-fraction", type=float, default=0.5,
        help="fraction of pairs whose negative is an impostor (0..1; rest are counterfactual)",
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--transcript", default=None,
        help="public AISHELL transcript file (UTT_ID token ...) whose text populates "
        "positive rows; required for the --public-manifest Overall calibration path. "
        "Defaults to data_aishell (1)/data_aishell/transcript/aishell_transcript_v0.8.txt "
        "when omitted and that file exists.",
    )
    parser.add_argument(
        "--skip-audio-check", action="store_true",
        help="skip the audio-existence fail-closed check (not recommended)",
    )
    return parser.parse_args(argv)


def _resolve_transcript(arg: str | None) -> str | None:
    """Resolve the transcript path: explicit arg, then the default AISHELL path."""
    if arg:
        return arg
    default = Path("data_aishell (1)/data_aishell/transcript/aishell_transcript_v0.8.txt")
    return str(default) if default.is_file() else None


def main(argv: Iterable[str] | None = None) -> dict:
    args = parse_args(argv)
    transcript = _resolve_transcript(args.transcript)
    summary = prepare_r7_manifest(
        args.r3_manifest,
        args.output,
        args.dataset_a_root,
        impostor_fraction=args.impostor_fraction,
        seed=args.seed,
        check_audio=not args.skip_audio_check,
        transcript_path=transcript,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
