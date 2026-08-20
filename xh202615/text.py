"""Text normalization utilities for Chinese command CER evaluation."""

from __future__ import annotations

import re
import unicodedata


_ASR_TAG_RE = re.compile(r"<\|[^|<>]+\|>")
_PUNCT_RE = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()\[\]{}<>《》【】·…—_-]+")
_CJK_SPACE_RE = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")
_CJK_SINGLE_REPEAT_RE = re.compile(r"([\u3400-\u9fff])\1{2,}")
_REPEAT_PHRASE_RE = re.compile(r"(.{1,12})\1{2,}")


def _remove_asr_repetition(text: str) -> str:
    """Remove decoder loops while retaining ordinary two-character repeats."""

    # A single CJK token repeated many times is a common autoregressive loop;
    # retain two copies so legitimate emphasis such as ``哈哈`` survives.
    text = _CJK_SINGLE_REPEAT_RE.sub(r"\1\1", text)

    def replace_phrase(match: re.Match[str]) -> str:
        phrase = match.group(1)
        if any("\u3400" <= char <= "\u9fff" for char in phrase):
            return phrase
        return match.group(0)

    # Run a few bounded passes because one loop can expose another loop after
    # its outer phrase is removed. Only phrases containing CJK are changed.
    for _ in range(4):
        updated = _REPEAT_PHRASE_RE.sub(replace_phrase, text)
        if updated == text:
            break
        text = updated
    return text


def clean_asr_text(text: str | None, *, smart_cleanup: bool = True) -> str:
    """Remove decoder artifacts and conservative CJK repetition loops."""

    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _ASR_TAG_RE.sub("", text)
    if smart_cleanup:
        text = _CJK_SPACE_RE.sub("", text)
        text = _remove_asr_repetition(text)
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
