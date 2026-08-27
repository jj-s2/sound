"""R10 multi-candidate grouped OOF selector.

Joins canonical R3, primary/energy candidates, and TSE-ASR hypotheses, builds
inference-only post-ASR features, and evaluates grouped nested-OOF policies.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data import Sample
from .evaluation import evaluate_rows
from .metrics import cer_stats, is_rejection
from .r12_personal_vad import PERSONAL_VAD_FEATURE_SCHEMA
from .text import normalize_text

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from scipy import sparse
except ImportError as _err:  # pragma: no cover
    raise ImportError("R10 selector requires scikit-learn and scipy") from _err


ACTION_ORDER = ["reject", "r3", "primary", "energy", "tse"]
CANDIDATE_ACTIONS = ["r3", "primary", "energy", "tse"]

_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_DIGIT_RE = re.compile(r"\d")
_PUNCT_CHARS = set(
    r" 　，。！？、；：,.!?;:\"'“”‘’（）()[]{}<>《》【】·…—_-"
)

_VOCAB_MIN_IDX = -1
_VOCAB_MARGIN_IDX = -1


def _finite_or_nan(value: object) -> float:
    if value is None:
        return math.nan
    try:
        f = float(value)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


@dataclass(frozen=True)
class CandidateRow:
    """Joined inference-time row with all candidate texts and acoustic features."""

    id: str
    split: str
    label: str | None
    r3_text: str
    primary_text: str
    energy_text: str
    tse_text: str
    audio_features: dict[str, float]
    original_command_audio: Path | None
    source_digest: str
    dedup_sources: dict[str, list[str]]

    @property
    def texts(self) -> dict[str, str]:
        return {
            "r3": self.r3_text,
            "primary": self.primary_text,
            "energy": self.energy_text,
            "tse": self.tse_text,
        }


@functools.lru_cache(maxsize=None)
def _normalized(text: str | None) -> str:
    return normalize_text(text)


@functools.lru_cache(maxsize=None)
def _char_cer(a: str | None, b: str | None) -> float:
    return cer_stats(_normalized(a), _normalized(b)).cer


def _wav_duration_sec(path: str | Path) -> float:
    """Lightweight WAV duration from header only (no frame decoding)."""
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            return frames / sr if sr else math.nan
    except (wave.Error, OSError, EOFError):
        return math.nan


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_text_by_id(
    path: str | Path,
    text_field: str,
    alt_field: str | None = None,
) -> dict[str, str]:
    texts: dict[str, str] = {}
    for record in _read_jsonl(path):
        sid = str(record.get("id", ""))
        if sid in texts:
            raise ValueError(f"duplicate id {sid!r} in {path}")
        text = record.get(text_field)
        if text is None and alt_field is not None:
            text = record.get(alt_field)
        texts[sid] = "" if text is None else str(text)
    return texts


def load_candidate_bundle(
    candidate_fusion_path: str | Path,
    tse_asr_path: str | Path,
    audio_map_path: str | Path,
    group_manifest_path: str | Path,
    *,
    r3_predictions_path: str | Path | None = None,
) -> tuple[dict[str, CandidateRow], dict[str, str], dict[str, str | None]]:
    """Join R3/primary/energy/TSE candidates and acoustic features by string id.

    Returns (rows_by_id, groups_by_id, labels_by_id).
    Raises ValueError on duplicate ids or missing required rows.
    """
    manifest = json.loads(Path(group_manifest_path).read_text(encoding="utf-8-sig"))
    labels: dict[str, str | None] = {}
    groups: dict[str, str] = {}
    for row in manifest.get("rows", []):
        sid = str(row["id"])
        labels[sid] = row.get("label")
        groups[sid] = row.get("wake_component", sid)

    sample_ids = sorted(labels, key=lambda x: int(x) if x.isdigit() else x)

    audio_records = _read_jsonl(audio_map_path)
    audio_by_id: dict[str, dict] = {}
    for record in audio_records:
        sid = str(record.get("id", ""))
        if sid in audio_by_id:
            raise ValueError(f"duplicate id {sid!r} in audio_map")
        audio_by_id[sid] = record

    fusion_records = _read_jsonl(candidate_fusion_path)
    fusion_by_id: dict[str, dict] = {}
    for record in fusion_records:
        sid = str(record.get("id", ""))
        if sid in fusion_by_id:
            raise ValueError(f"duplicate id {sid!r} in candidate_fusion")
        fusion_by_id[sid] = record

    tse_by_id = _load_text_by_id(tse_asr_path, "text", "recognition_text")

    r3_by_id: dict[str, str] | None = None
    if r3_predictions_path is not None:
        r3_by_id = _load_text_by_id(r3_predictions_path, "recognition_text", "text")

    rows: dict[str, CandidateRow] = {}
    for sid in sample_ids:
        if sid not in fusion_by_id:
            raise ValueError(f"missing candidate_fusion row for id {sid}")
        if sid not in tse_by_id:
            raise ValueError(f"missing TSE ASR row for id {sid}")
        if sid not in audio_by_id:
            raise ValueError(f"missing audio_map row for id {sid}")

        fusion = fusion_by_id[sid]
        candidates = fusion.get("candidate_texts", {})
        primary = str(candidates.get("primary", ""))
        energy = str(candidates.get("energy", ""))
        raw_r3 = str(fusion.get("recognition_text", fusion.get("text", "")))
        r3_text = r3_by_id[sid] if r3_by_id is not None else raw_r3
        tse_text = tse_by_id[sid]

        audio = audio_by_id[sid]
        audio_features: dict[str, float] = {
            "presence_score": _finite_or_nan(audio.get("presence_score")),
            "enhanced_cosine": _finite_or_nan(audio.get("enhanced_cosine")),
            "mixture_cosine": _finite_or_nan(audio.get("mixture_cosine")),
            "max_cosine": _finite_or_nan(audio.get("max_cosine")),
            "latency_ms": _finite_or_nan(audio.get("latency_ms")),
        }

        original_audio = audio.get("original_command_audio")
        original_command_audio = Path(original_audio) if original_audio else None
        duration = _wav_duration_sec(original_command_audio) if original_command_audio else math.nan
        audio_features["cmd_duration_sec"] = duration
        # RMS is omitted from cached features to keep loading fast; missing flag handles it.
        audio_features["cmd_rms"] = math.nan
        for name in PERSONAL_VAD_FEATURE_SCHEMA:
            audio_features[name] = _finite_or_nan(audio.get(name))

        # Deduplicate candidate texts while remembering source identities.
        text_to_sources: dict[str, list[str]] = defaultdict(list)
        for src in CANDIDATE_ACTIONS:
            text_to_sources[_normalized({"r3": r3_text, "primary": primary, "energy": energy, "tse": tse_text}[src])].append(src)
        dedup_sources = {text: sorted(sources) for text, sources in text_to_sources.items()}

        source_parts = [sid, r3_text, primary, energy, tse_text]
        source_digest = hashlib.sha256(
            json.dumps(source_parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

        rows[sid] = CandidateRow(
            id=sid,
            split=labels[sid] is not None and "pos" or "neg",
            label=labels[sid],
            r3_text=r3_text,
            primary_text=primary,
            energy_text=energy,
            tse_text=tse_text,
            audio_features=audio_features,
            original_command_audio=original_command_audio,
            source_digest=source_digest,
            dedup_sources=dedup_sources,
        )

    return rows, groups, labels


def _text_shape_features(text: str) -> dict[str, float]:
    norm = _normalized(text)
    length = len(norm)
    if length == 0:
        return {
            "length": 0.0,
            "empty": 1.0,
            "punct_ratio": 0.0,
            "digit_ratio": 0.0,
            "chinese_ratio": 0.0,
        }
    punct = sum(1 for ch in norm if ch in _PUNCT_CHARS)
    digits = len(_DIGIT_RE.findall(norm))
    chinese = len(_CHINESE_RE.findall(norm))
    return {
        "length": float(length),
        "empty": 0.0,
        "punct_ratio": punct / length,
        "digit_ratio": digits / length,
        "chinese_ratio": chinese / length,
    }


def score_action_policy(
    rows: Sequence[CandidateRow],
    labels: Mapping[str, str | None],
    action_func: callable,
) -> dict[str, float]:
    """Score a deterministic policy that maps each row to an action."""
    preds = []
    for row in rows:
        action = action_func(row)
        if action == "reject":
            text = ""
        else:
            text = row.texts.get(action, "")
        preds.append({"id": row.id, "recognition_text": text})
    samples = [
        Sample(
            id=row.id,
            split=row.split,
            wakeup_audio=Path("."),
            wakeup_text="",
            command_audio=row.original_command_audio or Path("."),
            label=labels[row.id],
        )
        for row in rows
    ]
    metrics = dict(evaluate_rows(samples, preds, missing_policy="empty").metrics)
    metrics["overall"] = ((1.0 - metrics["avg_cer"]) + metrics["avg_rr"]) / 2.0
    return metrics


def compute_oracle_metrics(
    rows: Sequence[CandidateRow],
    labels: Mapping[str, str | None],
    actions: Sequence[str],
) -> dict[str, float]:
    """Oracle: for positives choose best action; all negatives rejected."""
    def action_func(row: CandidateRow) -> str:
        label = labels[row.id]
        if label is None:
            return "reject"
        best_action = "reject"
        best_cer = float("inf")
        for action in actions:
            text = row.texts.get(action, "")
            c = _char_cer(label, text)
            # Ties prefer earlier action in the supplied list.
            if c < best_cer:
                best_cer = c
                best_action = action
        return best_action

    return score_action_policy(rows, labels, action_func)


def agreement_rescue_action(row: CandidateRow) -> str:
    """Deterministic rescue: use exact candidate agreement, else R3."""
    texts = {src: row.texts.get(src, "") for src in CANDIDATE_ACTIONS}
    nonempty = {src: txt for src, txt in texts.items() if _normalized(txt) != ""}
    # Count exact normalized matches among non-empty candidates.
    counts: dict[str, list[str]] = defaultdict(list)
    for src, txt in nonempty.items():
        counts[_normalized(txt)].append(src)
    for norm_text in sorted(counts):
        sources = counts[norm_text]
        if len(sources) >= 2:
            # Deterministic tie-break by CANDIDATE_ACTIONS order.
            chosen = min(sources, key=CANDIDATE_ACTIONS.index)
            return chosen
    return "r3"


# ---------------------------------------------------------------------------
# Task 2: inference-only post-ASR features
# ---------------------------------------------------------------------------

ACOUSTIC_FIELDS = [
    "presence_score",
    "enhanced_cosine",
    "mixture_cosine",
    "max_cosine",
    "latency_ms",
    "cmd_duration_sec",
    "cmd_rms",
]


def _build_feature_schema() -> list[str]:
    """Return the ordered list of inference-only feature names."""
    # Action identity.
    schema = ["is_reject", "is_r3", "is_primary", "is_energy", "is_tse"]
    # Per-candidate text shape.
    schema += [
        "candidate_length",
        "candidate_empty",
        "candidate_norm_length",
        "candidate_punct_ratio",
        "candidate_digit_ratio",
        "candidate_chinese_ratio",
        "candidate_exact_dup_count",
        "candidate_mean_cer_to_others",
        "candidate_min_cer_to_others",
    ]
    # Fold-local vocabulary distance.
    schema += ["vocab_min_cer", "vocab_margin"]
    # Row-level acoustic/presence features and missingness flags.
    for field in ACOUSTIC_FIELDS:
        schema.append(field)
        schema.append(f"{field}_missing")
    # Pairwise candidate agreement.
    pairs = [("r3", "primary"), ("r3", "energy"), ("r3", "tse"),
             ("primary", "energy"), ("primary", "tse"), ("energy", "tse")]
    for a, b in pairs:
        schema.append(f"cer_{a}_{b}")
    schema += ["n_nonempty_candidates", "any_exact_agreement"]
    return schema


FEATURE_SCHEMA = _build_feature_schema()
_VOCAB_MIN_IDX = FEATURE_SCHEMA.index("vocab_min_cer")
_VOCAB_MARGIN_IDX = FEATURE_SCHEMA.index("vocab_margin")


def _pairwise_cer_features(texts: dict[str, str]) -> dict[str, float]:
    """Pairwise CER and agreement descriptors for the four candidate texts."""
    pairs = [("r3", "primary"), ("r3", "energy"), ("r3", "tse"),
             ("primary", "energy"), ("primary", "tse"), ("energy", "tse")]
    features: dict[str, float] = {}
    for a, b in pairs:
        features[f"cer_{a}_{b}"] = _char_cer(texts.get(a, ""), texts.get(b, ""))
    return features


def _row_audio_features(row: CandidateRow) -> dict[str, float]:
    """Return cached command-audio metadata merged with audio_map features."""
    features: dict[str, float] = {}
    for field in ACOUSTIC_FIELDS:
        features[field] = row.audio_features.get(field, math.nan)
    return features


def _action_invariant_features(
    row: CandidateRow,
    action: str,
    max_len: int,
    pairwise: dict[str, float],
    audio: dict[str, float],
) -> dict[str, float]:
    """Inference-only features for one (row, action) that do not depend on fold vocabulary."""
    feat: dict[str, float] = {}
    # Action identity.
    for a in ACTION_ORDER:
        feat[f"is_{a}"] = 1.0 if action == a else 0.0

    # Candidate text shape.
    text = "" if action == "reject" else row.texts.get(action, "")
    shape = _text_shape_features(text)
    norm = _normalized(text)
    feat["candidate_length"] = shape["length"]
    feat["candidate_empty"] = shape["empty"]
    feat["candidate_norm_length"] = shape["length"] / (max_len + 1e-8)
    feat["candidate_punct_ratio"] = shape["punct_ratio"]
    feat["candidate_digit_ratio"] = shape["digit_ratio"]
    feat["candidate_chinese_ratio"] = shape["chinese_ratio"]

    # Exact duplicate count within row for this action's normalized text.
    dup_count = 0
    if action != "reject" and norm:
        for other_src in CANDIDATE_ACTIONS:
            if other_src != action and _normalized(row.texts.get(other_src, "")) == norm:
                dup_count += 1
    feat["candidate_exact_dup_count"] = float(dup_count)

    # Mean/min CER to other candidates.
    cers_to_others = []
    if action != "reject":
        for other_src in CANDIDATE_ACTIONS:
            if other_src != action:
                cers_to_others.append(_char_cer(text, row.texts.get(other_src, "")))
    feat["candidate_mean_cer_to_others"] = float(np.mean(cers_to_others)) if cers_to_others else 0.0
    feat["candidate_min_cer_to_others"] = float(min(cers_to_others)) if cers_to_others else 0.0

    # Fold-local vocabulary distance placeholders.
    feat["vocab_min_cer"] = 0.0
    feat["vocab_margin"] = 0.0

    # Acoustic features with missingness flags.
    for field in ACOUSTIC_FIELDS:
        value = audio.get(field, math.nan)
        feat[field] = value
        feat[f"{field}_missing"] = 1.0 if math.isnan(value) else 0.0

    # Pairwise and row-level descriptors.
    feat.update(pairwise)
    nonempty = sum(1 for src in CANDIDATE_ACTIONS if _normalized(row.texts.get(src, "")) != "")
    feat["n_nonempty_candidates"] = float(nonempty)
    any_agree = any(
        _normalized(row.texts.get(a, "")) == _normalized(row.texts.get(b, ""))
        and _normalized(row.texts.get(a, "")) != ""
        for a, b in [("r3", "primary"), ("r3", "energy"), ("r3", "tse"),
                     ("primary", "energy"), ("primary", "tse"), ("energy", "tse")]
    )
    feat["any_exact_agreement"] = 1.0 if any_agree else 0.0
    return feat


def _row_invariant_feature_vectors(row: CandidateRow) -> dict[str, np.ndarray]:
    """Return per-action invariant feature dict for a row."""
    texts = row.texts
    max_len = max(len(_normalized(t)) for t in texts.values()) if any(texts.values()) else 0
    pairwise = _pairwise_cer_features(texts)
    audio = _row_audio_features(row)
    return {
        action: np.array([
            _action_invariant_features(row, action, max_len, pairwise, audio)[name]
            for name in FEATURE_SCHEMA
        ], dtype=np.float64)
        for action in ACTION_ORDER
    }


def _build_invariant_cache(
    rows: Mapping[str, CandidateRow],
) -> dict[str, dict[str, np.ndarray]]:
    """Build once-per-row invariant feature vectors reused across every fold."""
    return {sid: _row_invariant_feature_vectors(row) for sid, row in rows.items()}


def _fit_bigram_index(
    vocab: Sequence[str],
) -> tuple[CountVectorizer | None, sparse.csr_matrix | None, np.ndarray]:
    """Fit a binary character-bigram index on the fold-local training vocabulary.

    Returns (vectorizer, vocab_matrix, vocab_norms).  When the vocabulary is
    empty or every token has fewer than two characters, the vectorizer is None
    and callers treat every distance as 1.0 with margin 0.0.
    """
    vocab_list = sorted({v for v in vocab if v != ""})
    if not vocab_list or max(len(v) for v in vocab_list) < 2:
        return None, None, np.zeros(0, dtype=np.int64)
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(2, 2),
        binary=True,
        lowercase=False,
        dtype=np.float32,
    )
    vocab_matrix = vectorizer.fit_transform(vocab_list)
    if vocab_matrix.shape[1] == 0:
        return None, None, np.zeros(0, dtype=np.int64)
    vocab_norms = np.diff(vocab_matrix.indptr)
    return vectorizer, vocab_matrix, vocab_norms


def _vocab_distance_bigrams_fast(
    texts: Sequence[str],
    vectorizer: CountVectorizer | None,
    vocab_matrix: sparse.csr_matrix | None,
    vocab_norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute min and margin Jaccard distance to the fold-local vocabulary.

    Uses one sparse matrix multiplication and evaluates Jaccard only for pairs
    with nonzero bigram overlap, so there is no dense all-vocabulary
    materialization.  Empty or single-character texts deterministically yield
    distance 1.0 and margin 0.0.
    """
    n_texts = len(texts)
    min_d = np.ones(n_texts, dtype=np.float64)
    second_d = np.ones(n_texts, dtype=np.float64)

    if vocab_matrix is None or vocab_norms.size == 0:
        return min_d, second_d - min_d

    cand_matrix = vectorizer.transform(texts)
    cand_norms = np.diff(cand_matrix.indptr)

    intersections = (cand_matrix @ vocab_matrix.T).tocsr()
    indptr = intersections.indptr
    indices = intersections.indices
    data = intersections.data

    for i in range(n_texts):
        if cand_norms[i] == 0:
            continue
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue
        n = end - start
        distances = np.empty(n, dtype=np.float64)
        for k, idx in enumerate(range(start, end)):
            inter = data[idx]
            j = indices[idx]
            union = cand_norms[i] + vocab_norms[j] - inter
            distances[k] = 0.0 if union == 0 else 1.0 - inter / union
        if distances.size >= 2:
            two = np.partition(distances, 1)[:2]
            d0, d1 = float(two[0]), float(two[1])
            if d0 > d1:
                d0, d1 = d1, d0
            min_d[i] = d0
            second_d[i] = d1
        else:
            min_d[i] = distances[0]

    return min_d, second_d - min_d


def _row_feature_vectors(
    row: CandidateRow,
    vocab: Sequence[str] | None,
) -> dict[str, np.ndarray]:
    """Return per-action feature dict for a row (backward-compatible raw features)."""
    normalized_vocab = sorted({v for v in (vocab or []) if v})
    vectorizer, vocab_matrix, vocab_norms = _fit_bigram_index(normalized_vocab)
    artifacts = FoldArtifacts(
        vocab=normalized_vocab,
        vectorizer=vectorizer,
        vocab_matrix=vocab_matrix,
        vocab_norms=vocab_norms,
        impute_values=np.zeros(len(FEATURE_SCHEMA), dtype=np.float64),
        mean=np.zeros(len(FEATURE_SCHEMA), dtype=np.float64),
        std=np.ones(len(FEATURE_SCHEMA), dtype=np.float64),
        feature_schema=list(FEATURE_SCHEMA),
    )
    invariant_cache = _build_invariant_cache({row.id: row})
    action_features = _build_fold_features([row], artifacts, invariant_cache)
    return {action: action_features[action][0] for action in ACTION_ORDER}


@dataclass
class FoldArtifacts:
    """Features and scaling statistics fitted on an outer-training fold."""

    vocab: list[str]
    vectorizer: CountVectorizer | None
    vocab_matrix: sparse.csr_matrix | None
    vocab_norms: np.ndarray
    impute_values: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    feature_schema: list[str]


def _impute_features(mat: np.ndarray, values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise median imputation; returns (imputed_matrix, impute_values)."""
    mat = np.asarray(mat, dtype=np.float64)
    if values is None:
        values = np.empty(mat.shape[1], dtype=np.float64)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            valid = col[~np.isnan(col)]
            values[j] = float(np.median(valid)) if valid.size else 0.0
    out = mat.copy()
    for j in range(mat.shape[1]):
        col = out[:, j]
        col[np.isnan(col)] = values[j]
    return out, values


def _standardize(mat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (mat - mean) / (std + 1e-8)


def _build_fold_features(
    rows: Sequence[CandidateRow],
    artifacts: FoldArtifacts,
    invariant_cache: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Build per-action feature matrices for rows using fold-local vocabulary.

    Invariant features are taken from the cache; only the vocabulary distance
    features are recomputed for this fold.
    """
    n_rows = len(rows)
    all_texts: list[str] = []
    for action in ACTION_ORDER:
        for row in rows:
            text = "" if action == "reject" else row.texts.get(action, "")
            all_texts.append(_normalized(text))

    min_d, margin = _vocab_distance_bigrams_fast(
        all_texts, artifacts.vectorizer, artifacts.vocab_matrix, artifacts.vocab_norms
    )

    result: dict[str, np.ndarray] = {}
    for a_idx, action in enumerate(ACTION_ORDER):
        start = a_idx * n_rows
        end = start + n_rows
        X = np.vstack([invariant_cache[row.id][action].copy() for row in rows])
        X[:, _VOCAB_MIN_IDX] = min_d[start:end]
        X[:, _VOCAB_MARGIN_IDX] = margin[start:end]
        result[action] = X
    return result


def fit_fold_artifacts(
    train_ids: Sequence[str],
    rows: Mapping[str, CandidateRow],
    invariant_cache: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> FoldArtifacts:
    """Fit vocabulary, imputation, and standardization on outer-training rows."""
    vocab = sorted({
        _normalized(rows[sid].label)
        for sid in train_ids
        if rows[sid].label is not None
    })
    vocab = [v for v in vocab if v != ""]
    vectorizer, vocab_matrix, vocab_norms = _fit_bigram_index(vocab)

    if invariant_cache is None:
        invariant_cache = _build_invariant_cache({sid: rows[sid] for sid in train_ids})

    # Placeholder artifacts let _build_fold_features use the fitted vocabulary.
    artifacts = FoldArtifacts(
        vocab=vocab,
        vectorizer=vectorizer,
        vocab_matrix=vocab_matrix,
        vocab_norms=vocab_norms,
        impute_values=np.zeros(len(FEATURE_SCHEMA), dtype=np.float64),
        mean=np.zeros(len(FEATURE_SCHEMA), dtype=np.float64),
        std=np.ones(len(FEATURE_SCHEMA), dtype=np.float64),
        feature_schema=list(FEATURE_SCHEMA),
    )

    train_rows = [rows[sid] for sid in train_ids]
    action_features = _build_fold_features(train_rows, artifacts, invariant_cache)
    X = np.vstack([action_features[a] for a in ACTION_ORDER])
    X_imp, impute_values = _impute_features(X)
    mean = X_imp.mean(axis=0)
    std = X_imp.std(axis=0)

    artifacts.impute_values = impute_values
    artifacts.mean = mean
    artifacts.std = std
    return artifacts


def build_inference_features(
    rows: Sequence[CandidateRow],
    artifacts: FoldArtifacts,
    action: str,
    invariant_cache: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> np.ndarray:
    """Return standardized feature matrix for a specific action."""
    if action not in ACTION_ORDER:
        raise ValueError(f"unknown action {action!r}")
    if invariant_cache is None:
        invariant_cache = _build_invariant_cache({row.id: row for row in rows})
    action_features = _build_fold_features(rows, artifacts, invariant_cache)
    X = action_features[action]
    X_imp, _ = _impute_features(X, artifacts.impute_values)
    return _standardize(X_imp, artifacts.mean, artifacts.std)


def build_all_action_features(
    rows: Sequence[CandidateRow],
    artifacts: FoldArtifacts,
    invariant_cache: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    """Return standardized feature matrices for every action."""
    if invariant_cache is None:
        invariant_cache = _build_invariant_cache({row.id: row for row in rows})
    action_features = _build_fold_features(rows, artifacts, invariant_cache)
    return {
        action: _standardize(_impute_features(X, artifacts.impute_values)[0], artifacts.mean, artifacts.std)
        for action, X in action_features.items()
    }


# ---------------------------------------------------------------------------
# Task 3: grouped nested OOF policies
# ---------------------------------------------------------------------------


def _optimal_action(row: CandidateRow) -> str:
    """Best action for a training row given its reference label."""
    if row.label is None:
        return "reject"
    best_action = "reject"
    best_cer = float("inf")
    for action in CANDIDATE_ACTIONS:
        c = _char_cer(row.label, row.texts.get(action, ""))
        if c < best_cer:
            best_cer = c
            best_action = action
    return best_action


def _action_to_index(action: str) -> int:
    return ACTION_ORDER.index(action)


def _group_to_folds(
    sample_ids: Sequence[str],
    groups: Mapping[str, str],
    n_folds: int,
    seed: int,
) -> list[list[str]]:
    """Assign group-disjoint folds balanced by total size (deterministic)."""
    group_to_ids: dict[str, list[str]] = defaultdict(list)
    for sid in sample_ids:
        group_to_ids[groups[sid]].append(sid)
    ordered = sorted(
        group_to_ids.items(),
        key=lambda item: hashlib.sha256(item[0].encode()).hexdigest(),
    )
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for _gname, members in ordered:
        target = int(np.argmin([len(folds[i]) for i in range(n_folds)]))
        folds[target].extend(members)
    return folds


def _build_folds(
    sample_ids: Sequence[str],
    groups: Mapping[str, str],
    n_outer: int,
    n_inner: int,
    seed: int,
) -> list[dict]:
    """Build nested grouped folds: list of {outer_idx, test, train, inner_folds}."""
    outer_folds = _group_to_folds(sample_ids, groups, n_outer, seed)
    result: list[dict] = []
    for outer_idx, test_ids in enumerate(outer_folds):
        train_ids = [sid for sid in sample_ids if sid not in set(test_ids)]
        inner_folds = _group_to_folds(train_ids, groups, n_inner, seed + 1000 + outer_idx)
        result.append({
            "outer_idx": outer_idx,
            "test": test_ids,
            "train": train_ids,
            "inner_folds": inner_folds,
        })
    return result


def _build_stacked_training_data(
    train_ids: Sequence[str],
    rows: Mapping[str, CandidateRow],
    invariant_cache: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, FoldArtifacts]:
    """Build standardized (X, y, artifacts) for multiclass action selection."""
    artifacts = fit_fold_artifacts(train_ids, rows, invariant_cache)
    train_rows = [rows[sid] for sid in train_ids]
    action_features = _build_fold_features(train_rows, artifacts, invariant_cache)
    X_actions = np.stack([action_features[a] for a in ACTION_ORDER], axis=1)
    X = X_actions.reshape(-1, X_actions.shape[-1])
    y = np.repeat(
        np.array([_action_to_index(_optimal_action(row)) for row in train_rows], dtype=np.int64),
        len(ACTION_ORDER),
    )
    X_imp, _ = _impute_features(X, artifacts.impute_values)
    X_std = _standardize(X_imp, artifacts.mean, artifacts.std)
    return X_std, y, artifacts


def _build_stacked_inference_data(
    rows: Sequence[CandidateRow],
    artifacts: FoldArtifacts,
    invariant_cache: Mapping[str, Mapping[str, np.ndarray]],
) -> np.ndarray:
    """Build standardized stacked feature matrix for inference rows."""
    action_features = _build_fold_features(rows, artifacts, invariant_cache)
    X_actions = np.stack([action_features[a] for a in ACTION_ORDER], axis=1)
    X = X_actions.reshape(-1, X_actions.shape[-1])
    X_imp, _ = _impute_features(X, artifacts.impute_values)
    return _standardize(X_imp, artifacts.mean, artifacts.std)


def _train_model(X: np.ndarray, y: np.ndarray, C: float, seed: int) -> LogisticRegression:
    model = LogisticRegression(
        C=C,
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X, y)
    return model


def _predict_actions_from_stacked(
    model: LogisticRegression,
    X_stacked: np.ndarray,
    n_rows: int,
    tau: float,
    n_classes: int = len(ACTION_ORDER),
) -> np.ndarray:
    """Predict action indices from stacked (n_rows*5, n_features) matrix."""
    proba = model.predict_proba(X_stacked)  # (n_rows*5, n_seen)
    seen_classes = model.classes_
    proba = proba.reshape(n_rows, len(ACTION_ORDER), proba.shape[1])
    # Map seen-class probabilities back to the full action space.
    full = np.zeros((n_rows, len(ACTION_ORDER), n_classes), dtype=np.float64)
    for j, cls in enumerate(seen_classes):
        full[:, :, int(cls)] = proba[:, :, j]
    row_proba = full.mean(axis=1)  # (n_rows, n_classes)
    p_reject = row_proba[:, 0]
    actions = np.empty(n_rows, dtype=np.int64)
    for i in range(n_rows):
        if p_reject[i] >= tau:
            actions[i] = 0
        else:
            actions[i] = int(np.argmax(row_proba[i, 1:])) + 1
    return actions


def _actions_to_predictions(
    rows: Sequence[CandidateRow],
    actions: Sequence[int],
    fallback_to_r3: bool = True,
) -> list[dict]:
    """Map action indices to recognition_text predictions with optional R3 fallback."""
    preds = []
    for row, action_idx in zip(rows, actions):
        action = ACTION_ORDER[action_idx]
        if action == "reject":
            text = ""
        else:
            text = row.texts.get(action, "")
            if fallback_to_r3 and _normalized(text) == "":
                text = row.r3_text
        preds.append({"id": row.id, "recognition_text": text})
    return preds


def _score_predictions(
    rows: Sequence[CandidateRow],
    labels: Mapping[str, str | None],
    predictions: Sequence[dict],
) -> dict[str, float]:
    """Score predictions with the official evaluator."""
    samples = [
        Sample(
            id=row.id,
            split=row.split,
            wakeup_audio=Path("."),
            wakeup_text="",
            command_audio=row.original_command_audio or Path("."),
            label=labels[row.id],
        )
        for row in rows
    ]
    metrics = dict(evaluate_rows(samples, predictions, missing_policy="empty").metrics)
    metrics["overall"] = ((1.0 - metrics["avg_cer"]) + metrics["avg_rr"]) / 2.0
    return metrics


def _inner_cv(
    train_ids: list[str],
    inner_folds: list[list[str]],
    rows: Mapping[str, CandidateRow],
    invariant_cache: Mapping[str, Mapping[str, np.ndarray]],
    C_values: Sequence[float],
    tau_values: Sequence[float],
    seed: int,
) -> dict:
    """Select (C, tau) maximizing pooled inner Overall subject to RR >= 0.95.

    Returns a dict with keys:
      - feasible (bool): whether any candidate met the RR >= 0.95 floor.
      - C (float | None): selected C when feasible, else None.
      - tau (float | None): selected tau when feasible, else None.
      - diagnostics (list): per-(C, tau) inner metrics and feasibility flags.
    """
    # Precompute feature matrices once per inner fold.
    fold_data: list[tuple] = []
    for val_ids in inner_folds:
        inner_train = [sid for sid in train_ids if sid not in set(val_ids)]
        if len(inner_train) < 10:
            continue
        X_tr, y_tr, artifacts_tr = _build_stacked_training_data(inner_train, rows, invariant_cache)
        val_rows = [rows[sid] for sid in val_ids]
        X_val = _build_stacked_inference_data(val_rows, artifacts_tr, invariant_cache)
        fold_data.append((X_tr, y_tr, X_val, val_rows))

    best_overall = -float("inf")
    best_params: tuple[float, float] | None = None
    diagnostics: list[dict] = []
    for C in C_values:
        for tau in tau_values:
            all_preds: list[dict] = []
            for X_tr, y_tr, X_val, val_rows in fold_data:
                classes = np.unique(y_tr)
                if classes.size < 2:
                    preds = [{"id": row.id, "recognition_text": row.r3_text} for row in val_rows]
                    all_preds.extend(preds)
                    continue
                model = _train_model(X_tr, y_tr, C, seed)
                actions = _predict_actions_from_stacked(model, X_val, len(val_rows), tau)
                preds = _actions_to_predictions(val_rows, actions, fallback_to_r3=True)
                all_preds.extend(preds)
            if not all_preds:
                continue
            val_rows_all = [rows[p["id"]] for p in all_preds]
            metrics = _score_predictions(val_rows_all, {r.id: r.label for r in val_rows_all}, all_preds)
            feasible = metrics["avg_rr"] >= 0.95
            diagnostics.append({
                "C": C,
                "tau": tau,
                "overall": metrics["overall"],
                "avg_rr": metrics["avg_rr"],
                "avg_cer": metrics["avg_cer"],
                "feasible": feasible,
            })
            if feasible and metrics["overall"] > best_overall:
                best_overall = metrics["overall"]
                best_params = (float(C), float(tau))
    return {
        "feasible": best_params is not None,
        "C": best_params[0] if best_params else None,
        "tau": best_params[1] if best_params else None,
        "diagnostics": diagnostics,
    }


def run_grouped_nested_oof(
    rows: Mapping[str, CandidateRow],
    labels: Mapping[str, str | None],
    groups: Mapping[str, str],
    *,
    n_outer: int = 5,
    n_inner: int = 3,
    C_values: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    tau_values: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
    seed: int = 20260807,
) -> dict:
    """Run grouped nested OOF for R3, agreement rescue, and learned selector."""
    sample_ids = sorted(rows, key=lambda x: int(x) if x.isdigit() else x)
    folds = _build_folds(sample_ids, groups, n_outer, n_inner, seed)

    # Invariant features are built once and reused across every fold and (C, tau).
    invariant_cache = _build_invariant_cache(rows)

    all_rows = [rows[sid] for sid in sample_ids]
    r3_metrics = score_action_policy(all_rows, labels, lambda row: "r3")
    rescue_metrics = score_action_policy(all_rows, labels, agreement_rescue_action)
    oracle_2_metrics = compute_oracle_metrics(all_rows, labels, ["r3", "tse"])
    oracle_all_metrics = compute_oracle_metrics(all_rows, labels, CANDIDATE_ACTIONS)

    oof_preds: list[dict] = []
    fold_reports: list[dict] = []
    for fold in folds:
        train_ids = fold["train"]
        test_ids = fold["test"]
        inner_result = _inner_cv(
            train_ids, fold["inner_folds"], rows, invariant_cache, C_values, tau_values, seed,
        )
        test_rows = [rows[sid] for sid in test_ids]
        train_group_names = sorted({groups[sid] for sid in train_ids})
        test_group_names = sorted({groups[sid] for sid in test_ids})
        group_disjoint = len(set(train_group_names) & set(test_group_names)) == 0

        if not inner_result["feasible"]:
            # Fail closed: no learned policy met the RR >= 0.95 floor; emit exact R3.
            preds = [{"id": row.id, "recognition_text": row.r3_text, "outer_fold": fold["outer_idx"]} for row in test_rows]
            fold_metrics = _score_predictions(test_rows, labels, preds)
            fold_reports.append({
                "outer_idx": fold["outer_idx"],
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "selected_C": None,
                "selected_tau": None,
                "fallback": "r3_no_feasible_inner_policy",
                "inner_cv_diagnostics": inner_result["diagnostics"],
                "train_group_names": train_group_names,
                "test_group_names": test_group_names,
                "group_disjoint": group_disjoint,
                "metrics": fold_metrics,
            })
            oof_preds.extend(preds)
            continue

        # Retrain on full outer train with the feasible selected (C, tau).
        C = inner_result["C"]
        tau = inner_result["tau"]
        X_train, y_train, artifacts = _build_stacked_training_data(train_ids, rows, invariant_cache)
        classes = np.unique(y_train)
        if classes.size < 2:
            # Fallback to R3 if training target is degenerate.
            preds = [{"id": row.id, "recognition_text": row.r3_text, "outer_fold": fold["outer_idx"]} for row in test_rows]
            fold_metrics = _score_predictions(test_rows, labels, preds)
            fold_reports.append({
                "outer_idx": fold["outer_idx"],
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "selected_C": None,
                "selected_tau": None,
                "fallback": "r3_degenerate",
                "inner_cv_diagnostics": inner_result["diagnostics"],
                "train_group_names": train_group_names,
                "test_group_names": test_group_names,
                "group_disjoint": group_disjoint,
                "metrics": fold_metrics,
            })
        else:
            model = _train_model(X_train, y_train, C, seed)
            X_test = _build_stacked_inference_data(test_rows, artifacts, invariant_cache)
            actions = _predict_actions_from_stacked(model, X_test, len(test_rows), tau)
            preds = _actions_to_predictions(test_rows, actions, fallback_to_r3=True)
            for p, action_idx in zip(preds, actions):
                p["outer_fold"] = fold["outer_idx"]
                p["selected_action"] = ACTION_ORDER[action_idx]
            fold_metrics = _score_predictions(test_rows, labels, preds)
            fold_reports.append({
                "outer_idx": fold["outer_idx"],
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "selected_C": C,
                "selected_tau": tau,
                "fallback": None,
                "inner_cv_diagnostics": inner_result["diagnostics"],
                "train_group_names": train_group_names,
                "test_group_names": test_group_names,
                "group_disjoint": group_disjoint,
                "metrics": fold_metrics,
                "model_coef": model.coef_.tolist(),
                "model_intercept": model.intercept_.tolist(),
            })
        oof_preds.extend(preds)

    # Pooled OOF metrics for learned selector.
    all_test_rows = [rows[p["id"]] for p in oof_preds]
    pooled_metrics = _score_predictions(all_test_rows, labels, oof_preds)
    n_infeasible_folds = sum(
        1 for f in fold_reports if f.get("fallback") == "r3_no_feasible_inner_policy"
    )

    return {
        "oof_predictions": oof_preds,
        "pooled_metrics": pooled_metrics,
        "fold_reports": fold_reports,
        "n_infeasible_folds": n_infeasible_folds,
        "r3_metrics": r3_metrics,
        "agreement_rescue_metrics": rescue_metrics,
        "oracle_2_metrics": oracle_2_metrics,
        "oracle_all_metrics": oracle_all_metrics,
    }


def bootstrap_grouped_ci(
    oof_predictions: Sequence[dict],
    labels: Mapping[str, str | None],
    groups: Mapping[str, str],
    *,
    n_boot: int = 2000,
    seed: int = 20260807,
) -> dict:
    """Bootstrap 95% CI for Overall over leakage groups."""
    rng = np.random.default_rng(seed)
    n = len(oof_predictions)
    is_pos = np.zeros(n, dtype=np.int64)
    ref_chars = np.zeros(n, dtype=np.int64)
    subs = np.zeros(n, dtype=np.int64)
    ins = np.zeros(n, dtype=np.int64)
    dels = np.zeros(n, dtype=np.int64)
    correct_reject = np.zeros(n, dtype=np.int64)
    for i, p in enumerate(oof_predictions):
        sid = p["id"]
        label = labels[sid]
        hyp = p.get("recognition_text", "")
        if label is None:
            is_pos[i] = 0
            if _normalized(hyp) == "":
                correct_reject[i] = 1
        else:
            is_pos[i] = 1
            st = cer_stats(_normalized(label), _normalized(hyp))
            subs[i] = st.substitutions
            ins[i] = st.insertions
            dels[i] = st.deletions
            ref_chars[i] = st.ref_chars

    group_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(oof_predictions):
        group_to_idx[groups[p["id"]]].append(i)
    group_names = np.array(sorted(group_to_idx), dtype=object)
    group_arrays = {g: np.array(idx, dtype=np.int64) for g, idx in group_to_idx.items()}
    n_groups = len(group_names)

    overalls = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled_groups = rng.choice(group_names, size=n_groups, replace=True)
        sampled_idx = np.concatenate([group_arrays[g] for g in sampled_groups])
        total_ref = int(ref_chars[sampled_idx].sum())
        total_err = int((subs[sampled_idx] + ins[sampled_idx] + dels[sampled_idx]).sum())
        avg_cer = total_err / total_ref if total_ref else 0.0
        n_neg = int((1 - is_pos)[sampled_idx].sum())
        avg_rr = int(correct_reject[sampled_idx].sum()) / n_neg if n_neg else 0.0
        overalls[b] = ((1.0 - avg_cer) + avg_rr) / 2.0

    return {
        "n_boot": n_boot,
        "n_groups": len(group_names),
        "overall_mean": float(overalls.mean()),
        "ci_low": float(np.quantile(overalls, 0.025)),
        "ci_high": float(np.quantile(overalls, 0.975)),
    }
