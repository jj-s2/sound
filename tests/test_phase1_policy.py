import math
import unittest

from xh202615.contracts import BackendMetadata, RouteAction, TemporalSpeakerEvidence
from xh202615.policy import PolicyConfig, ThreeActionPolicy


class Phase1PolicyTest(unittest.TestCase):
    def setUp(self):
        self.config = PolicyConfig.from_dict(
            {
                "version": "phase1-test-v1",
                "reject_probability_max": 0.2,
                "raw_probability_min": 0.8,
                "enhancement_overlap_min": 0.5,
            }
        )
        self.policy = ThreeActionPolicy(self.config)

    @staticmethod
    def evidence(target_probability=None, overlap_probability=None, error=None):
        return TemporalSpeakerEvidence(
            id="sample-1",
            backend=BackendMetadata(
                name="speaker-fixture",
                model_id="model-1",
                config_hash="config-sha256",
            ),
            enrollment_source="wake.wav",
            command_source="command.wav",
            target_probability=target_probability,
            overlap_probability=overlap_probability,
            error=error,
        )

    def test_missing_probability_routes_to_raw_safely(self):
        decision = self.policy.decide(self.evidence(), enhancement_available=True)

        self.assertEqual(decision.action, RouteAction.RAW)
        self.assertEqual(decision.reason_code, "missing_evidence_safe_raw")
        self.assertEqual(decision.evidence_version, "speaker-fixture:model-1:config-sha256")
        self.assertIsNone(decision.estimated_target_probability)
        self.assertIsNone(decision.estimated_raw_risk)
        self.assertIsNone(decision.estimated_enhanced_risk)
        self.assertIsNone(decision.estimated_incremental_cost_ms)

    def test_evidence_error_routes_to_raw_even_when_probability_is_low(self):
        decision = self.policy.decide(
            self.evidence(target_probability=0.1, error="backend_failure"),
            enhancement_available=True,
        )

        self.assertEqual(decision.action, RouteAction.RAW)
        self.assertEqual(decision.reason_code, "missing_evidence_safe_raw")
        self.assertEqual(decision.estimated_target_probability, 0.1)

    def test_probability_at_reject_bound_routes_to_reject(self):
        decision = self.policy.decide(
            self.evidence(target_probability=0.2), enhancement_available=False
        )

        self.assertEqual(decision.action, RouteAction.REJECT)
        self.assertEqual(decision.reason_code, "target_probability_below_reject_bound")
        self.assertEqual(decision.estimated_target_probability, 0.2)

    def test_clear_target_at_raw_bound_routes_to_raw(self):
        decision = self.policy.decide(
            self.evidence(target_probability=0.8), enhancement_available=True
        )

        self.assertEqual(decision.action, RouteAction.RAW)
        self.assertEqual(decision.reason_code, "clear_target_raw")

    def test_high_target_with_low_overlap_routes_to_raw(self):
        decision = self.policy.decide(
            self.evidence(target_probability=0.9, overlap_probability=0.49),
            enhancement_available=True,
        )

        self.assertEqual(decision.action, RouteAction.RAW)
        self.assertEqual(decision.reason_code, "clear_target_raw")

    def test_missing_overlap_with_high_target_routes_to_raw(self):
        decision = self.policy.decide(
            self.evidence(target_probability=0.9, overlap_probability=None),
            enhancement_available=True,
        )

        self.assertEqual(decision.action, RouteAction.RAW)
        self.assertEqual(decision.reason_code, "clear_target_raw")

    def test_gray_or_high_overlap_routes_to_enhanced_when_available(self):
        gray = self.policy.decide(
            self.evidence(target_probability=0.5, overlap_probability=0.2),
            enhancement_available=True,
        )
        high_overlap = self.policy.decide(
            self.evidence(target_probability=0.9, overlap_probability=0.5),
            enhancement_available=True,
        )

        self.assertEqual(gray.action, RouteAction.ENHANCED)
        self.assertEqual(gray.reason_code, "gray_or_overlap_enhanced")
        self.assertEqual(high_overlap.action, RouteAction.ENHANCED)
        self.assertEqual(high_overlap.reason_code, "gray_or_overlap_enhanced")

    def test_enhancement_unavailable_falls_back_to_raw(self):
        decision = self.policy.decide(
            self.evidence(target_probability=0.5, overlap_probability=0.5),
            enhancement_available=False,
        )

        self.assertEqual(decision.action, RouteAction.RAW)
        self.assertEqual(decision.reason_code, "enhancement_unavailable_raw_fallback")

    def test_policy_version_and_target_probability_are_recorded(self):
        decision = self.policy.decide(
            self.evidence(target_probability=0.5), enhancement_available=False
        )

        self.assertEqual(decision.id, "sample-1")
        self.assertEqual(decision.policy_version, "phase1-test-v1")
        self.assertEqual(decision.estimated_target_probability, 0.5)

    def test_evidence_version_uses_none_for_missing_config_hash(self):
        evidence = self.evidence(target_probability=0.5)
        evidence = TemporalSpeakerEvidence(
            id=evidence.id,
            backend=BackendMetadata("speaker-fixture", "model-1"),
            enrollment_source=evidence.enrollment_source,
            command_source=evidence.command_source,
            target_probability=evidence.target_probability,
        )

        decision = self.policy.decide(evidence, enhancement_available=False)

        self.assertEqual(decision.evidence_version, "speaker-fixture:model-1:none")

    def test_config_rejects_probability_outside_unit_interval(self):
        for field in (
            "reject_probability_max",
            "raw_probability_min",
            "enhancement_overlap_min",
        ):
            for invalid in (-0.01, 1.01, math.inf, -math.inf, math.nan):
                values = {
                    "version": "v1",
                    "reject_probability_max": 0.2,
                    "raw_probability_min": 0.8,
                    "enhancement_overlap_min": 0.5,
                }
                values[field] = invalid
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, field):
                        PolicyConfig.from_dict(values)

    def test_config_requires_reject_bound_below_raw_bound(self):
        values = {
            "version": "v1",
            "reject_probability_max": 0.8,
            "raw_probability_min": 0.8,
            "enhancement_overlap_min": 0.5,
        }

        with self.assertRaisesRegex(ValueError, "reject_probability_max"):
            PolicyConfig.from_dict(values)

    def test_config_rejects_missing_or_unknown_fields(self):
        values = {
            "version": "v1",
            "reject_probability_max": 0.2,
            "raw_probability_min": 0.8,
            "enhancement_overlap_min": 0.5,
        }
        with self.assertRaisesRegex(ValueError, "version"):
            PolicyConfig.from_dict({key: value for key, value in values.items() if key != "version"})
        with self.assertRaisesRegex(ValueError, "unknown"):
            PolicyConfig.from_dict({**values, "unknown": 1})


if __name__ == "__main__":
    unittest.main()
