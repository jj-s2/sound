"""CER and RR metrics for XH-202615."""

from __future__ import annotations

from dataclasses import dataclass

from .text import normalize_text


@dataclass(frozen=True)
class CerStats:
    substitutions: int
    insertions: int
    deletions: int
    ref_chars: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def cer(self) -> float:
        return self.errors / self.ref_chars if self.ref_chars else 0.0


def _edit_stats(ref: str, hyp: str) -> CerStats:
    """Character-level Levenshtein with S/I/D backtrace."""

    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
                continue
            candidates = [
                (dp[i - 1][j - 1] + 1, "S"),
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            ]
            dp[i][j], op[i][j] = min(candidates, key=lambda x: x[0])

    i, j = n, m
    s = ins = d = 0
    while i > 0 or j > 0:
        cur = op[i][j]
        if cur == "M":
            i -= 1
            j -= 1
        elif cur == "S":
            s += 1
            i -= 1
            j -= 1
        elif cur == "D":
            d += 1
            i -= 1
        elif cur == "I":
            ins += 1
            j -= 1
        else:
            break
    return CerStats(s, ins, d, n)


def cer_stats(ref: str | None, hyp: str | None) -> CerStats:
    return _edit_stats(normalize_text(ref), normalize_text(hyp))


def is_rejection(text: str | None) -> bool:
    return normalize_text(text) == ""

