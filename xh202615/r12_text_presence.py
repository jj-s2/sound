"""Train-only text presence score used by the deployable R12 gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pickle
from typing import Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .r10_selector import CandidateRow


_INPUT_FIELDS = ("r3_text", "primary_text")
_PARAMETERS = {
    "analyzer": "char",
    "ngram_range": (1, 3),
    "sublinear_tf": True,
    "C": 10.0,
    "class_weight": "balanced",
    "max_iter": 1000,
}


def _text(row: CandidateRow) -> str:
    return f"{row.r3_text} [SEP] {row.primary_text}"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TrainTextPresence:
    model: Pipeline
    fit_row_count: int
    seed: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "input_fields": list(_INPUT_FIELDS),
            "parameters_digest": _digest({**_PARAMETERS, "random_state": self.seed}),
            "model_digest": hashlib.sha256(pickle.dumps(self.model, protocol=5)).hexdigest(),
        }


def fit_train_text_presence(
    rows: Sequence[CandidateRow], labels: Mapping[str, str | None], *, seed: int
) -> TrainTextPresence:
    """Fit solely on train-role labels and the two inference-time ASR texts."""
    if not rows or set(labels) != {row.id for row in rows}:
        raise ValueError("text presence labels must exactly cover nonempty train rows")
    target = np.asarray([labels[row.id] is not None for row in rows], dtype=np.int64)
    if set(target.tolist()) != {0, 1}:
        raise ValueError("text presence training requires both target classes")
    vectorizer = TfidfVectorizer(
        analyzer=_PARAMETERS["analyzer"],
        ngram_range=_PARAMETERS["ngram_range"],
        sublinear_tf=_PARAMETERS["sublinear_tf"],
    )
    classifier = LogisticRegression(
        C=_PARAMETERS["C"],
        class_weight=_PARAMETERS["class_weight"],
        max_iter=_PARAMETERS["max_iter"],
        random_state=seed,
    )
    model = Pipeline((("vectorizer", vectorizer), ("classifier", classifier))).fit(
        [_text(row) for row in rows], target
    )
    return TrainTextPresence(model, len(rows), seed)


def predict_text_presence(model: TrainTextPresence, rows: Sequence[CandidateRow]) -> np.ndarray:
    scores = np.asarray(model.model.predict_proba([_text(row) for row in rows])[:, 1], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("text presence scores must be finite")
    return scores
