"""Convert an R3 public counterfactual manifest to the training JSONL contract.

Never reads Dataset-A. The R3 manifest is loaded with ``read_r3_manifest`` and
fail-closed with ``assert_r3_manifest_safe`` (Dataset-A containment + R3
structural validity) before any conversion. Each ``R3MixtureRow`` is mapped to a
``TrainingManifestRow`` without reading Dataset-A audio, labels, or text. The
converted rows are validated with ``assert_valid_training_manifest`` (Dataset-A
forbidden root, duplicate IDs, speaker split leakage, field ranges) *before* the
output file is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.r3_data import R3MixtureRow, assert_r3_manifest_safe, read_r3_manifest
from xh202615.training_data import (
    TrainingManifestRow,
    assert_valid_training_manifest,
)


PUBLIC_SOURCE = "r3_public_pilot_v1"
# 31-bit non-negative int; stable across processes (blake2b, not salted hash()).
_SEED_MASK = 0x7FFFFFFF

_AUDIO_FIELDS = ("enrollment_audio", "mixture_audio", "clean_target_audio")


def deterministic_seed(row_id: str) -> int:
    """Deterministic non-negative integer seed derived from the row identity.

    ``hashlib.blake2b`` is used (not the process-salted ``hash`` builtin) so the
    seed is reproducible across runs and machines. The row_id encodes both the
    pair and the positive/negative polarity, so siblings share a pair prefix but
    receive distinct seeds.
    """
    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError("row_id must be a non-empty string to derive a seed")
    digest = hashlib.blake2b(row_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _SEED_MASK


def convert_r3_row(row: R3MixtureRow) -> TrainingManifestRow:
    """Map one R3 mixture row to a training manifest row.

    Field mapping (per the R3 training handoff contract):
      clean_target_audio       -> target_audio
      mixture_audio            -> mixture_audio
      enrollment_audio         -> enrollment_audio (unchanged)
      target_source_id         -> target_speaker_id
      interferer_source_ids[0] -> interferer_speaker_id (or null)
    split, target_present, overlap_ratio, snr_db, sir_db are preserved. R3
    carries no command text, so ``text`` is null. ``source`` is a nonempty public
    string and ``seed`` is a deterministic per-row integer.
    """
    interferer = row.interferer_source_ids[0] if row.interferer_source_ids else None
    return TrainingManifestRow(
        row_id=row.row_id,
        split=row.split,
        source=PUBLIC_SOURCE,
        enrollment_audio=row.enrollment_audio,
        target_audio=row.clean_target_audio,
        mixture_audio=row.mixture_audio,
        target_speaker_id=row.target_source_id,
        interferer_speaker_id=interferer,
        target_present=row.target_present,
        overlap_ratio=row.overlap_ratio,
        snr_db=row.snr_db,
        sir_db=row.sir_db,
        text=None,
        seed=deterministic_seed(row.row_id),
    )


def _missing_audio(rows: Iterable[R3MixtureRow]) -> list[tuple[str, str, str]]:
    """Return (row_id, field, path) tuples for every missing audio file."""
    missing: list[tuple[str, str, str]] = []
    for row in rows:
        for field_name in _AUDIO_FIELDS:
            path = Path(getattr(row, field_name))
            if not path.is_file():
                missing.append((row.row_id, field_name, str(path)))
    return missing


def build_training_rows(
    r3_rows: Iterable[R3MixtureRow],
    *,
    dataset_a_root: str | Path,
    check_audio: bool = True,
) -> tuple[TrainingManifestRow, ...]:
    """Convert R3 rows to validated training rows without reading Dataset-A.

    Fail-closed guards, in order:
      1. ``assert_r3_manifest_safe`` - Dataset-A containment + R3 structural
         validity (malformed/unknown rows, duplicate IDs, entity split leakage).
      2. audio existence - every enrollment/mixture/clean_target file must exist.
      3. ``assert_valid_training_manifest`` - Dataset-A forbidden root, duplicate
         IDs, speaker split leakage, and field ranges on the converted rows.
    """
    safe_rows = assert_r3_manifest_safe(r3_rows, dataset_a_root)
    if check_audio:
        missing = _missing_audio(safe_rows)
        if missing:
            sample = ", ".join(f"{rid}:{fld}" for rid, fld, _ in missing[:8])
            raise ValueError(
                f"missing audio for {len(missing)} row(s)/field(s): {sample}"
            )
    training_rows = tuple(convert_r3_row(row) for row in safe_rows)
    return assert_valid_training_manifest(
        training_rows,
        forbidden_roots=(Path(dataset_a_root),),
    )


def _manifest_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_training_manifest(
    r3_manifest: str | Path,
    output: str | Path,
    dataset_a_root: str | Path,
    *,
    check_audio: bool = True,
) -> dict:
    """Read an R3 manifest, convert it, validate, and write the training JSONL.

    Returns a summary dict with row counts, per-split counts, and a SHA-256
    digest of the written file. No Dataset-A file is ever read; the Dataset-A
    root is used only as a forbidden containment root.
    """
    r3_rows = read_r3_manifest(r3_manifest)
    training_rows = build_training_rows(
        r3_rows, dataset_a_root=dataset_a_root, check_audio=check_audio
    )
    output_path = Path(output).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True)
        for row in training_rows
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    split_rows = {"train": 0, "val": 0, "test": 0}
    for row in training_rows:
        if row.split in split_rows:
            split_rows[row.split] += 1
    summary = {
        "r3_manifest": str(Path(r3_manifest).resolve(strict=False)),
        "output": str(output_path),
        "dataset_a_root": str(Path(dataset_a_root).resolve(strict=False)),
        "dataset_a_used_for_training": False,
        "source": PUBLIC_SOURCE,
        "row_count": len(training_rows),
        "split_rows": split_rows,
        "manifest_digest": _manifest_digest(output_path),
    }
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-a-root", default="datasetA/datasetA")
    parser.add_argument(
        "--skip-audio-check",
        action="store_true",
        help="skip the audio-existence fail-closed check (not recommended)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> dict:
    args = parse_args(argv)
    summary = prepare_training_manifest(
        args.r3_manifest,
        args.output,
        args.dataset_a_root,
        check_audio=not args.skip_audio_check,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
