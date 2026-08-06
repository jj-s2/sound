"""Conservative dynamic routing rules."""

from __future__ import annotations

from dataclasses import dataclass

from .backends import SpeakerScores


@dataclass(frozen=True)
class RouteResult:
    route: str
    reason: str


def _leq(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def _geq(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def route_sample(scores: SpeakerScores, config: dict) -> RouteResult:
    """Route to reject/direct_asr/tse.

    Missing speaker evidence defaults to direct_asr to protect CER. This is a
    deliberate safe baseline behavior.
    """

    policy = config.get("router", {})
    mode = policy.get("mode", "asr_only")
    if mode == "asr_only":
        return RouteResult("direct_asr", "v0_asr_only")

    reject = policy.get("reject", {})
    reject_votes = [
        _leq(scores.target_probability, reject.get("target_probability_max", -1.0)),
        _leq(scores.global_similarity, reject.get("global_similarity_max", -1.0)),
        _leq(scores.topk_similarity, reject.get("topk_similarity_max", -1.0)),
        _leq(scores.target_frame_ratio, reject.get("target_frame_ratio_max", -1.0)),
    ]
    required_votes = int(reject.get("min_votes", 3))
    quality_ok = scores.audio_quality is None or scores.audio_quality >= reject.get("audio_quality_min", 0.0)
    if sum(reject_votes) >= required_votes and quality_ok:
        return RouteResult("reject", f"conservative_reject_votes={sum(reject_votes)}")

    if mode == "safe":
        return RouteResult("direct_asr", "safe_non_reject")

    direct = policy.get("direct", {})
    direct_votes = [
        _geq(scores.target_probability, direct.get("target_probability_min", 1.1)),
        _geq(scores.global_similarity, direct.get("global_similarity_min", 1.1)),
        _geq(scores.topk_similarity, direct.get("topk_similarity_min", 1.1)),
        _geq(scores.target_frame_ratio, direct.get("target_frame_ratio_min", 1.1)),
    ]
    low_noise = scores.noise_score is None or scores.noise_score <= direct.get("noise_score_max", 1.0)
    low_overlap = scores.overlap_probability is None or scores.overlap_probability <= direct.get("overlap_probability_max", 1.0)
    if sum(direct_votes) >= int(direct.get("min_votes", 2)) and low_noise and low_overlap:
        return RouteResult("direct_asr", f"clear_owner_votes={sum(direct_votes)}")

    return RouteResult("tse", "gray_or_difficult_fallback_to_raw_until_separator_ready")

