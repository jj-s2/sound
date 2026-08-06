"""R7 impostor hard-negative construction for speaker-conditioned rejection.

R6's absent rows (from ``aishell1_phase2_v2``) were noise-only, so the presence
head learned an energy detector that collapsed on Dataset-A. The R3
counterfactual manifest (``r3_public_pilot_v1``) already fixes the easy half:
its negatives are interferer+noise on the *same* RIR/SNR/SIR/overlap grid as
the positive (target removed but non-target interferer speech retained).

This module adds the **hardest** negative: an *impostor*. For a positive pair
whose mixture contains speaker *A* (the target) + interferers, an impostor
negative reuses **the same mixture** but swaps the enrollment to a **different
same-split speaker *B***. The mixture now prominently contains a non-target
speaker (*A* is the impostor) while the enrolled speaker (*B*) is absent - so
the reject decision must rest on **speaker identity**, not energy. This is the
worst case the rejector must survive at test time.

Construction is manifest-only: both the mixture file (pair *A*'s positive) and
the swapped enrollment file (pair *B*'s enrollment) already exist in the R3
output, so **no new audio is rendered**. Per pair we keep exactly one negative
- counterfactual or impostor, chosen deterministically - so present/absent
stays balanced 1:1 and both negative kinds are present.

Data boundary
-------------
Reads ONLY the public R3 manifest. Dataset-A is never read; the Dataset-A root
is used only as a forbidden containment root. Speaker-disjointness is inherited
from the R3 split (both *A* and *B* are in the same split) and re-validated by
``assert_valid_training_manifest``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from xh202615.r3_data import R3MixtureRow, read_r3_manifest
from xh202615.training_data import (
    TrainingManifestRow,
    assert_valid_training_manifest,
)

PUBLIC_SOURCE = "r3_public_pilot_v1"
IMPOSTOR_SOURCE = "r7_impostor_hard_neg"
R7_GENERATOR_VERSION = "r7-speaker-hardneg-v1"
_SEED_MASK = 0x7FFFFFFF

# AISHELL utterance IDs look like "BAC009S0002W0421"; the speaker is "S0002".
_SPEAKER_RE = re.compile(r"S\d+")


def speaker_of_utterance(utterance_id: str) -> str:
    """Extract the AISHELL speaker prefix (``S####``) from an utterance ID.

    Falls back to the whole id if no ``S\\d+`` token is found, so non-AISHELL
    ids degrade gracefully rather than crashing the build.
    """
    if not isinstance(utterance_id, str) or not utterance_id.strip():
        raise ValueError("utterance_id must be a non-empty string")
    match = _SPEAKER_RE.search(utterance_id)
    return match.group(0) if match else utterance_id


def deterministic_seed(row_id: str) -> int:
    """Reproducible non-negative integer seed from a row id (blake2b, no salt)."""
    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError("row_id must be a non-empty string to derive a seed")
    digest = hashlib.blake2b(row_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _SEED_MASK


def read_aishell_transcripts(path: str | Path) -> dict[str, str]:
    """Read an AISHELL transcript file into ``{utterance_id: text}``.

    Each line is ``UTT_ID token token ...`` (tab- or space-separated). The text is
    the space-joined token remainder, matching the existing phase2 manifest
    convention; :func:`xh202615.metrics.normalize_text` strips all whitespace
    before CER, so the separator is CER-neutral. Fails closed on malformed lines,
    duplicate ids, and empty files. Only public AISHELL transcripts are read here
    - never Dataset-A.
    """
    transcripts: dict[str, str] = {}
    transcript_path = Path(path)
    with transcript_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(
                    f"malformed AISHELL transcript line {line_no} in "
                    f"{transcript_path}: expected utterance ID followed by text"
                )
            utt_id = parts[0]
            if utt_id in transcripts:
                raise ValueError(f"duplicate utterance id {utt_id!r} in {transcript_path}")
            transcripts[utt_id] = " ".join(parts[1:])
    if not transcripts:
        raise ValueError(f"no transcript entries in {transcript_path}")
    return transcripts


@dataclass(frozen=True)
class _Pair:
    """The positive-side facts of an R3 counterfactual pair needed for impostors."""

    pair_id: str
    split: str
    target_utterance: str  # target_source_id of the positive (speaker A)
    enrollment_audio: Path  # positive enrollment (speaker A reference)
    positive_mixture: Path  # mixture containing A + interferers
    silence_target: Path  # negative clean_target (zeros) - reused for absent rows
    overlap_ratio: float
    snr_db: float | None
    sir_db: float | None


def _pairs_from_r3(rows: Iterable[R3MixtureRow]) -> list[_Pair]:
    """Group R3 rows into pairs, capturing the positive-side facts."""
    by_pair: dict[str, list[R3MixtureRow]] = {}
    for row in rows:
        by_pair.setdefault(row.pair_id, []).append(row)
    pairs: list[_Pair] = []
    for pair_id, group in by_pair.items():
        if len(group) != 2:
            raise ValueError(f"pair {pair_id!r} must have 2 rows, found {len(group)}")
        positives = [r for r in group if r.target_present]
        negatives = [r for r in group if not r.target_present]
        if len(positives) != 1 or len(negatives) != 1:
            raise ValueError(
                f"pair {pair_id!r} must have one present and one absent row"
            )
        pos, neg = positives[0], negatives[0]
        if pos.split != neg.split:
            raise ValueError(f"pair {pair_id!r} rows span splits {pos.split}/{neg.split}")
        pairs.append(
            _Pair(
                pair_id=pair_id,
                split=pos.split,
                target_utterance=pos.target_source_id,
                enrollment_audio=pos.enrollment_audio,
                positive_mixture=pos.mixture_audio,
                silence_target=neg.clean_target_audio,
                overlap_ratio=pos.overlap_ratio,
                snr_db=pos.snr_db,
                sir_db=pos.sir_db,
            )
        )
    return pairs


def _stable_int(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(p) for p in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _assign_impostor_partners(
    pairs: list[_Pair], *, impostor_fraction: float, seed: int
) -> dict[str, _Pair | None]:
    """Deterministically assign each pair a counterfactual-or-impostor negative.

    Returns ``{pair_id: partner_pair_or_None}`` where a non-None partner means
    "build an impostor negative using this partner's enrollment" and ``None``
    means "keep the counterfactual negative". Impostor assignment is stratified
    per split: a pair is an impostor candidate iff ``_stable_int(seed, pair_id)
    < impostor_fraction * 2**64``; its partner is a *different-speaker* pair in
    the same split, chosen deterministically. Fails loudly if an impostor pair
    has no different-speaker partner in its split.
    """
    if not 0.0 <= impostor_fraction <= 1.0:
        raise ValueError(f"impostor_fraction must be in [0, 1], got {impostor_fraction}")
    by_split: dict[str, list[_Pair]] = {}
    for pair in pairs:
        by_split.setdefault(pair.split, []).append(pair)
    partners: dict[str, _Pair | None] = {}
    for split, group in by_split.items():
        # Speaker -> list of pairs (sorted for determinism).
        by_speaker: dict[str, list[_Pair]] = {}
        for pair in group:
            by_speaker.setdefault(speaker_of_utterance(pair.target_utterance), []).append(pair)
        speakers = sorted(by_speaker)
        other_pairs: list[_Pair] = []
        for pair in group:
            is_impostor = (
                _stable_int(seed, pair.pair_id, "impostor") / float(2 ** 64)
            ) < impostor_fraction
            if not is_impostor:
                partners[pair.pair_id] = None
                continue
            spk = speaker_of_utterance(pair.target_utterance)
            others = [p for s in speakers if s != spk for p in by_speaker[s]]
            if not others:
                raise ValueError(
                    f"split {split!r} has no different-speaker partner for pair "
                    f"{pair.pair_id!r} (speaker {spk!r}); reduce --impostor-fraction "
                    f"or add more speakers to the split"
                )
            partner = others[_stable_int(seed, pair.pair_id, "partner") % len(others)]
            partners[pair.pair_id] = partner
            other_pairs.append(pair)
    return partners


def _counterfactual_training_row(pos: R3MixtureRow, neg: R3MixtureRow) -> TrainingManifestRow:
    """Convert an R3 counterfactual negative to a training row (present=False)."""
    interferer = neg.interferer_source_ids[0] if neg.interferer_source_ids else None
    return TrainingManifestRow(
        row_id=neg.row_id,
        split=neg.split,
        source=PUBLIC_SOURCE,
        enrollment_audio=neg.enrollment_audio,
        target_audio=neg.clean_target_audio,
        mixture_audio=neg.mixture_audio,
        target_speaker_id=neg.target_source_id,
        interferer_speaker_id=interferer,
        target_present=False,
        overlap_ratio=neg.overlap_ratio,
        snr_db=neg.snr_db,
        sir_db=neg.sir_db,
        text=None,
        seed=deterministic_seed(neg.row_id),
    )


def _positive_training_row(pos: R3MixtureRow, text: str | None = None) -> TrainingManifestRow:
    """Convert an R3 positive to a training row (present=True).

    ``text`` is the public AISHELL transcript for the target utterance
    (``pos.target_source_id``); it is the CER reference for positive rows and is
    ``None`` when no transcript path was supplied (backward-compatible).
    """
    interferer = pos.interferer_source_ids[0] if pos.interferer_source_ids else None
    return TrainingManifestRow(
        row_id=pos.row_id,
        split=pos.split,
        source=PUBLIC_SOURCE,
        enrollment_audio=pos.enrollment_audio,
        target_audio=pos.clean_target_audio,
        mixture_audio=pos.mixture_audio,
        target_speaker_id=pos.target_source_id,
        interferer_speaker_id=interferer,
        target_present=True,
        overlap_ratio=pos.overlap_ratio,
        snr_db=pos.snr_db,
        sir_db=pos.sir_db,
        text=text,
        seed=deterministic_seed(pos.row_id),
    )


def _impostor_training_row(pair_a: _Pair, partner_b: _Pair) -> TrainingManifestRow:
    """Build the impostor absent row: B's enrollment + A's positive mixture.

    The enrolled speaker is *B* (``target_speaker_id`` = B's target utterance);
    the mixture prominently contains *A* (the impostor, recorded as the
    interferer) plus A's interferers. *B* is absent, so the row is
    ``target_present=False``. The target_audio points at A's existing silence
    file (absent rows synthesise silence in the trainer; the path must merely
    exist and be non-empty).
    """
    if pair_a.split != partner_b.split:
        raise ValueError(
            f"impostor partners must share a split: {pair_a.pair_id!r} "
            f"({pair_a.split}) vs {partner_b.pair_id!r} ({partner_b.split})"
        )
    if speaker_of_utterance(pair_a.target_utterance) == speaker_of_utterance(
        partner_b.target_utterance
    ):
        raise ValueError(
            f"impostor partner {partner_b.pair_id!r} shares speaker with "
            f"{pair_a.pair_id!r}; an impostor must be a different speaker"
        )
    row_id = f"{pair_a.pair_id}-impostor"
    return TrainingManifestRow(
        row_id=row_id,
        split=pair_a.split,
        source=IMPOSTOR_SOURCE,
        enrollment_audio=partner_b.enrollment_audio,  # B's reference (absent)
        target_audio=pair_a.silence_target,  # A's silence file (absent target)
        mixture_audio=pair_a.positive_mixture,  # contains A (impostor) + interferers
        target_speaker_id=partner_b.target_utterance,  # enrolled speaker B
        interferer_speaker_id=pair_a.target_utterance,  # impostor A (present)
        target_present=False,
        overlap_ratio=pair_a.overlap_ratio,
        snr_db=pair_a.snr_db,
        sir_db=pair_a.sir_db,
        text=None,
        seed=deterministic_seed(row_id),
    )


def _audio_exists(rows: Iterable[TrainingManifestRow]) -> list[tuple[str, str]]:
    """Return (row_id, field) for missing enrollment/mixture/target files."""
    missing: list[tuple[str, str]] = []
    for row in rows:
        for field in ("enrollment_audio", "mixture_audio", "target_audio"):
            if not Path(getattr(row, field)).is_file():
                missing.append((row.row_id, field))
    return missing


def build_r7_training_rows(
    r3_rows: Iterable[R3MixtureRow],
    *,
    dataset_a_root: str | Path,
    impostor_fraction: float = 0.5,
    seed: int = 20260806,
    check_audio: bool = True,
    transcript_path: str | Path | None = None,
) -> tuple[tuple[TrainingManifestRow, ...], dict]:
    """Build a balanced R7 training manifest from R3 counterfactual pairs.

    Each pair contributes one positive and one negative. The negative is the
    R3 counterfactual (interferer+noise) for a deterministic ``1 -
    impostor_fraction`` of pairs and an **impostor** (different-speaker
    enrollment over the same mixture) for the rest. Present/absent is therefore
    balanced 1:1, with both negative kinds present.

    When ``transcript_path`` is given, every positive row's ``text`` is populated
    from the public AISHELL transcript keyed by the target utterance id
    (``target_source_id``); absent rows stay ``text=None``. This is required for
    :func:`xh202615.tse_presence.samples_from_manifest` to label positives for
    public-val Overall calibration. Fails closed if a positive's utterance id is
    missing from the transcript. The transcript is optional only for backward
    compatibility (without it, positive ``text`` is ``None`` and the
    ``--public-manifest`` calibration path will not see labeled positives).

    Fail-closed guards: R3 pair integrity, impostor partner speaker-disjointness,
    transcript coverage, audio existence, and ``assert_valid_training_manifest``
    (Dataset-A forbidden root, duplicate IDs, speaker split leakage, field ranges).
    """
    rows_list = list(r3_rows)
    pairs = _pairs_from_r3(rows_list)
    by_pair_id: dict[str, list[R3MixtureRow]] = {}
    for row in rows_list:
        by_pair_id.setdefault(row.pair_id, []).append(row)
    partners = _assign_impostor_partners(pairs, impostor_fraction=impostor_fraction, seed=seed)

    transcripts = read_aishell_transcripts(transcript_path) if transcript_path else None

    training_rows: list[TrainingManifestRow] = []
    impostor_count = {"train": 0, "val": 0, "test": 0}
    counterfactual_count = {"train": 0, "val": 0, "test": 0}
    missing_text: list[str] = []
    for pair in pairs:
        group = by_pair_id[pair.pair_id]
        pos = next(r for r in group if r.target_present)
        neg = next(r for r in group if not r.target_present)
        text: str | None = None
        if transcripts is not None:
            if pos.target_source_id not in transcripts:
                missing_text.append(pos.target_source_id)
                text = None
            else:
                text = transcripts[pos.target_source_id]
        training_rows.append(_positive_training_row(pos, text=text))
        partner = partners[pair.pair_id]
        if partner is None:
            training_rows.append(_counterfactual_training_row(pos, neg))
            counterfactual_count[pair.split] += 1
        else:
            training_rows.append(_impostor_training_row(pair, partner))
            impostor_count[pair.split] += 1

    if missing_text:
        raise ValueError(
            f"{len(missing_text)} positive target utterance id(s) missing from the "
            f"transcript; first: {missing_text[:5]}"
        )

    if check_audio:
        missing = _audio_exists(training_rows)
        if missing:
            sample = ", ".join(f"{rid}:{fld}" for rid, fld in missing[:8])
            raise ValueError(f"missing audio for {len(missing)} row(s)/field(s): {sample}")

    validated = assert_valid_training_manifest(
        training_rows,
        forbidden_roots=(Path(dataset_a_root),),
    )
    positive_with_text = sum(1 for r in validated if r.target_present and r.text)
    return validated, {
        "impostor_negatives": impostor_count,
        "counterfactual_negatives": counterfactual_count,
        "positive_text_coverage": (
            f"{positive_with_text}/{sum(1 for r in validated if r.target_present)}"
            if transcripts is not None
            else "disabled (no transcript)"
        ),
        "transcript": (str(Path(transcript_path).resolve(strict=False)) if transcript_path else None),
    }


def prepare_r7_manifest(
    r3_manifest: str | Path,
    output: str | Path,
    dataset_a_root: str | Path,
    *,
    impostor_fraction: float = 0.5,
    seed: int = 20260806,
    check_audio: bool = True,
    transcript_path: str | Path | None = None,
) -> dict:
    """Read an R3 manifest, build impostor+counterfactual negatives, write JSONL.

    Returns a summary with row counts, per-split present/absent counts, the
    impostor/counterfactual split, transcript coverage, and a SHA-256 digest of
    the written file. No Dataset-A file is ever read.

    When ``transcript_path`` is given, positive rows carry the public AISHELL
    transcript as ``text`` (the CER reference for public-val calibration); pass
    it for the documented ``--public-manifest`` calibration path to see labeled
    positives. Without it, positive ``text`` is ``None`` (backward compatible,
    but calibration via ``samples_from_manifest`` will see no labeled positives).
    """
    r3_rows = read_r3_manifest(r3_manifest)
    training_rows, neg_counts = build_r7_training_rows(
        r3_rows,
        dataset_a_root=dataset_a_root,
        impostor_fraction=impostor_fraction,
        seed=seed,
        check_audio=check_audio,
        transcript_path=transcript_path,
    )
    output_path = Path(output).expanduser().resolve(strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True)
        for row in training_rows
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    split_rows: dict[str, dict[str, int]] = {
        s: {"present": 0, "absent": 0} for s in ("train", "val", "test")
    }
    for row in training_rows:
        if row.split in split_rows:
            key = "present" if row.target_present else "absent"
            split_rows[row.split][key] += 1
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    summary = {
        "generator_version": R7_GENERATOR_VERSION,
        "r3_manifest": str(Path(r3_manifest).resolve(strict=False)),
        "output": str(output_path),
        "dataset_a_root": str(Path(dataset_a_root).resolve(strict=False)),
        "dataset_a_used_for_training": False,
        "data_boundary": (
            "public-only; Dataset-A forbidden as a training source; "
            "speaker-disjoint splits validated; target-present + target-absent rows "
            "(impostor + counterfactual hard negatives); positive text from public "
            "AISHELL transcript only"
        ),
        "impostor_fraction": impostor_fraction,
        "seed": seed,
        "transcript": neg_counts.get("transcript"),
        "positive_text_coverage": neg_counts.get("positive_text_coverage"),
        "row_count": len(training_rows),
        "split_rows": split_rows,
        "negative_counts": neg_counts,
        "manifest_digest": digest,
        "sources": sorted({row.source for row in training_rows}),
    }
    return summary
