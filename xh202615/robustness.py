"""No-label robustness triggers for enhancement/fallback routing."""

from __future__ import annotations

from dataclasses import dataclass

from .backends import SpeakerScores
from .text_router import TextEvidence, analyze_text


@dataclass(frozen=True)
class EnhancementDecision:
    enhance: bool
    reason: str
    evidence: TextEvidence


def _speaker_weak(scores: SpeakerScores, similarity_max: float | None, probability_max: float | None) -> bool:
    votes = 0
    available = 0
    if similarity_max is not None:
        if scores.global_similarity is not None:
            available += 1
            votes += int(scores.global_similarity <= similarity_max)
        if scores.topk_similarity is not None:
            available += 1
            votes += int(scores.topk_similarity <= similarity_max)
    if probability_max is not None and scores.target_probability is not None:
        available += 1
        votes += int(scores.target_probability <= probability_max)
    return available > 0 and votes >= max(1, min(2, available))


def should_enhance_for_robustness(
    text: str | None,
    scores: SpeakerScores | None = None,
    *,
    min_text_length: int = 8,
    max_domain_score: int = 0,
    short_text_length: int = 6,
    enable_short_non_domain: bool = False,
    incomplete_text_length: int = 12,
    max_incomplete_domain_score: int = 2,
    long_text_length: int = 14,
    long_text_max_domain_score: int = 1,
    very_long_text_length: int = 18,
    min_audio_duration_sec: float = 0.0,
    audio_duration_sec: float = 0.0,
    speaker_similarity_max: float | None = None,
    target_probability_max: float | None = None,
) -> EnhancementDecision:
    """Return whether a sample should get enhanced ASR without using labels."""

    evidence = analyze_text(text)
    if evidence.text_length == 0:
        return EnhancementDecision(False, "empty_text_keep", evidence)

    weak_speaker = _speaker_weak(scores or SpeakerScores(), speaker_similarity_max, target_probability_max)
    enough_audio = audio_duration_sec >= min_audio_duration_sec
    has_action = evidence.action_hits > 0
    has_target = evidence.device_hits + evidence.setting_hits > 0
    looks_like_command = has_action and has_target

    if evidence.text_length >= min_text_length and evidence.domain_score <= max_domain_score:
        if enough_audio:
            return EnhancementDecision(
                True,
                f"non_domain_text:len={evidence.text_length},domain={evidence.domain_score},speaker_weak={weak_speaker}",
                evidence,
            )

    if (
        enable_short_non_domain
        and evidence.text_length >= short_text_length
        and evidence.domain_score <= max_domain_score
    ):
        if enough_audio or weak_speaker:
            return EnhancementDecision(
                True,
                (
                    "short_non_domain_text"
                    f":len={evidence.text_length},domain={evidence.domain_score},speaker_weak={weak_speaker}"
                ),
                evidence,
            )

    if (
        evidence.text_length >= incomplete_text_length
        and evidence.domain_score <= max_incomplete_domain_score
        and not looks_like_command
    ):
        if enough_audio or weak_speaker:
            return EnhancementDecision(
                True,
                (
                    "incomplete_command_text"
                    f":len={evidence.text_length},domain={evidence.domain_score},"
                    f"action={evidence.action_hits},target={evidence.device_hits + evidence.setting_hits},"
                    f"speaker_weak={weak_speaker}"
                ),
                evidence,
            )

    if evidence.text_length >= long_text_length and evidence.domain_score <= long_text_max_domain_score:
        if enough_audio or weak_speaker:
            return EnhancementDecision(
                True,
                f"long_low_domain_text:len={evidence.text_length},domain={evidence.domain_score},speaker_weak={weak_speaker}",
                evidence,
            )

    if evidence.text_length >= very_long_text_length and not looks_like_command:
        if enough_audio or weak_speaker:
            return EnhancementDecision(
                True,
                (
                    "very_long_incomplete_text"
                    f":len={evidence.text_length},domain={evidence.domain_score},"
                    f"action={evidence.action_hits},target={evidence.device_hits + evidence.setting_hits},"
                    f"speaker_weak={weak_speaker}"
                ),
                evidence,
            )

    return EnhancementDecision(
        False,
        f"keep:len={evidence.text_length},domain={evidence.domain_score},speaker_weak={weak_speaker}",
        evidence,
    )
