"""Train-only private domain-hotword candidate builder for R12 M0 Paraformer.

Hotword candidates are whole normalized command phrases drawn exclusively from
the private Dataset-A-train parent labels.  Candidate strings are written only
below ``output_root / "private"``, while the public summary records capacities,
token counts, the source-label SHA-256, and each hotword SHA-256 digest -- never
the phrase text.  No validation or internal-test label is ever read here.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Mapping, Sequence

from .r12_asr_manifest import normalize_asr_target


_ARTIFACT_KIND = "r12_asr_hotword"
_SCHEMA_VERSION = "v1"
_PRIVATE_DIR = "private"
_HOTWORDS_FILENAME = "domain_hotwords.json"
_SUMMARY_FILENAME = "hotword_summary.json"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class HotwordSummary:
    private_hotwords: Path
    public_summary: Path
    phrase_count: int
    source_labels_sha256: str
    summary_sha256: str


def _read_train_labels(path: Path) -> dict[str, str | None]:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("train labels must be a JSON object")
    if any(
        not isinstance(key, str) or (value is not None and not isinstance(value, str))
        for key, value in raw.items()
    ):
        raise ValueError("train labels must map IDs to strings or null")
    return raw


def rank_hotword_phrases(labels: Mapping[str, str | None]) -> tuple[str, ...]:
    """Return NFKC-normalized, nonempty whole-command phrases by frequency then UTF-8 order.

    Null and whitespace-only labels are excluded; each surviving label is
    normalized as a whole via ``normalize_asr_target`` before counting.
    """
    counts: Counter[str] = Counter()
    for value in labels.values():
        if value is None:
            continue
        try:
            phrase = normalize_asr_target(value)
        except ValueError:
            continue  # empty or whitespace-only after normalization
        counts[phrase] += 1
    return tuple(sorted(counts, key=lambda phrase: (-counts[phrase], phrase.encode("utf-8"))))


def _validate_capacities(capacities: Sequence[int], phrase_count: int) -> tuple[int, ...]:
    validated: list[int] = []
    seen: set[int] = set()
    for capacity in capacities:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
        if capacity in seen:
            raise ValueError(f"duplicate capacity {capacity!r}")
        seen.add(capacity)
        validated.append(capacity)
    for capacity in validated:
        if capacity > phrase_count:
            raise ValueError(f"capacity {capacity} exceeds ranked phrase count {phrase_count}")
    return tuple(validated)


def prepare_hotword_candidates(
    train_labels_path: Path, train_parent_ids: Collection[str], output_root: Path, capacities: Sequence[int],
) -> HotwordSummary:
    """Stage private candidate strings and a text-free public digest summary, then publish once.

    Each candidate for capacity ``k`` is the space-joined string of the top-``k``
    ranked phrases; ``token_count`` records the number of whole phrases emitted
    (equal to ``k``).  Capacities greater than the number of ranked phrases are
    rejected rather than truncated or back-filled.
    """
    labels = _read_train_labels(train_labels_path)
    if set(labels) != set(train_parent_ids):
        raise ValueError("train labels must exactly cover declared train parents")

    phrases = rank_hotword_phrases(labels)
    capacities = _validate_capacities(capacities, len(phrases))

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    private_dir = output_root / _PRIVATE_DIR
    private_hotwords = private_dir / _HOTWORDS_FILENAME
    public_summary = output_root / _SUMMARY_FILENAME

    source_labels_sha256 = _sha_file(train_labels_path)
    candidates = {str(capacity): " ".join(phrases[:capacity]) for capacity in capacities}
    hotword_entries = [
        {
            "capacity": capacity,
            "token_count": capacity,
            "hotword_sha256": _sha256_hex(candidates[str(capacity)].encode("utf-8")),
        }
        for capacity in capacities
    ]
    summary_fields = {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "phrase_count": len(phrases),
        "source_labels_sha256": source_labels_sha256,
        "hotwords": hotword_entries,
    }
    summary_sha256 = _sha256_hex(_canonical(summary_fields))
    summary_fields["summary_sha256"] = summary_sha256

    private_dir.mkdir(parents=True, exist_ok=False)
    private_hotwords.write_text(
        json.dumps(candidates, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n",
    )
    public_summary.write_text(
        json.dumps(summary_fields, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return HotwordSummary(
        private_hotwords=private_hotwords,
        public_summary=public_summary,
        phrase_count=len(phrases),
        source_labels_sha256=source_labels_sha256,
        summary_sha256=summary_sha256,
    )
