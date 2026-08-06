"""Text normalization utilities for Chinese command CER evaluation."""

from __future__ import annotations

import re
import unicodedata


_ASR_TAG_RE = re.compile(r"<\|[^|<>]+\|>")
_PUNCT_RE = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()\[\]{}<>《》【】·…—_-]+")


def clean_asr_text(text: str | None) -> str:
    """Remove generic decoder artifacts before saving ASR text."""

    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _ASR_TAG_RE.sub("", text)
    return text.strip()


def normalize_text(text: str | None) -> str:
    """Normalize recognition text before CER/RR calculation.

    This intentionally applies only generic normalization. Do not add
    Dataset-A-specific command templates or replacement rules here.
    """

    if text is None:
        return ""
    text = clean_asr_text(text)
    text = text.strip().lower()
    text = _PUNCT_RE.sub("", text)
    return text
