"""Deterministic, provenance-aware three-action routing policy."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Mapping

from .contracts import RouteAction, RouteDecision, TemporalSpeakerEvidence


@dataclass(frozen=True)
class PolicyConfig:
    """Versioned thresholds for the fixture-safe routing policy."""

    version: str
    reject_probability_max: float
    raw_probability_min: float
    enhancement_overlap_min: float

    def __post_init__(self) -> None:
        if not isinstance(self.version, str):
            raise ValueError("version must be a string")
        for field_name in (
            "reject_probability_max",
            "raw_probability_min",
            "enhancement_overlap_min",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a finite probability in [0, 1]")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be a finite probability in [0, 1]")

        if self.reject_probability_max >= self.raw_probability_min:
            raise ValueError(
                "reject_probability_max must be less than raw_probability_min"
            )

    @classmethod
    def from_dict(cls, value: dict) -> "PolicyConfig":
        """Build a policy config from its explicit serialized representation."""
        if not isinstance(value, Mapping):
            raise ValueError("PolicyConfig must be a dict")

        expected = {field.name for field in fields(cls)}
        unknown = sorted(set(value) - expected)
        if unknown:
            raise ValueError(f"PolicyConfig field {unknown[0]!r} is not recognized")
        missing = sorted(expected - set(value))
        if missing:
            raise ValueError(f"PolicyConfig field {missing[0]!r} is missing")

        return cls(
            version=value["version"],
            reject_probability_max=value["reject_probability_max"],
            raw_probability_min=value["raw_probability_min"],
            enhancement_overlap_min=value["enhancement_overlap_min"],
        )


class ThreeActionPolicy:
    """Route evidence to reject, raw ASR, or an optional enhancement stage."""

    def __init__(self, config: PolicyConfig):
        if not isinstance(config, PolicyConfig):
            raise TypeError("config must be a PolicyConfig")
        self.config = config

    @staticmethod
    def _evidence_version(evidence: TemporalSpeakerEvidence) -> str:
        backend = evidence.backend
        config_hash = backend.config_hash or "none"
        return f"{backend.name}:{backend.model_id}:{config_hash}"

    def decide(
        self,
        evidence: TemporalSpeakerEvidence,
        enhancement_available: bool,
    ) -> RouteDecision:
        """Choose a route using fixed safety-first decision ordering."""
        evidence_version = self._evidence_version(evidence)
        probability = evidence.target_probability

        if evidence.error or probability is None:
            action = RouteAction.RAW
            reason_code = "missing_evidence_safe_raw"
        elif probability <= self.config.reject_probability_max:
            action = RouteAction.REJECT
            reason_code = "target_probability_below_reject_bound"
        elif (
            probability >= self.config.raw_probability_min
            and (
                evidence.overlap_probability is None
                or evidence.overlap_probability < self.config.enhancement_overlap_min
            )
        ):
            action = RouteAction.RAW
            reason_code = "clear_target_raw"
        elif enhancement_available:
            action = RouteAction.ENHANCED
            reason_code = "gray_or_overlap_enhanced"
        else:
            action = RouteAction.RAW
            reason_code = "enhancement_unavailable_raw_fallback"

        return RouteDecision(
            id=evidence.id,
            action=action,
            reason_code=reason_code,
            policy_version=self.config.version,
            evidence_version=evidence_version,
            estimated_target_probability=probability,
            estimated_raw_risk=None,
            estimated_enhanced_risk=None,
            estimated_incremental_cost_ms=None,
        )
