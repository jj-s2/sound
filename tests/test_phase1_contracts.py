import unittest

from xh202615.backend_protocols import (
    AsrBackend,
    EnhancementBackend,
    TemporalSpeakerBackend,
)
from xh202615.backends import AsrResult
from xh202615.contracts import (
    BackendMetadata,
    EnhancedAudioResult,
    EvidenceWindow,
    RouteAction,
    RouteDecision,
    RunTrace,
    StageTiming,
    TemporalSpeakerEvidence,
    ValidationIssue,
)


class Phase1ContractsTest(unittest.TestCase):
    def test_existing_asr_result_constructor_stays_compatible(self):
        result = AsrResult("打开空调", 0.9, 12.5, "fixture")
        self.assertEqual(result.text, "打开空调")
        self.assertIsNone(result.metadata)
        self.assertIsNone(result.error)

    def test_temporal_evidence_round_trip(self):
        evidence = TemporalSpeakerEvidence(
            id="7",
            backend=BackendMetadata("fixture", "m1", "1", True, "abc"),
            enrollment_source="wake.wav",
            command_source="cmd.wav",
            windows=(EvidenceWindow(0.0, 1.0, 0.75, 0.8),),
            global_similarity=0.7,
            topk_similarity=0.75,
            temporal_coverage=1.0,
            consistency=1.0,
            target_probability=0.8,
            overlap_probability=0.2,
            quality=0.8,
            latency_ms=4.0,
            error=None,
        )
        self.assertEqual(TemporalSpeakerEvidence.from_dict(evidence.to_dict()), evidence)

    def test_invalid_window_reports_interval(self):
        with self.assertRaisesRegex(ValueError, "end_sec"):
            EvidenceWindow(1.0, 1.0, 0.5)

    def test_route_action_serializes_as_string(self):
        route = RouteDecision("7", RouteAction.RAW, "safe", "p1", "e1")
        self.assertEqual(route.to_dict()["action"], "raw")

    def test_run_trace_round_trip_preserves_measurement_mode(self):
        trace = RunTrace(
            run_id="r1", measurement_mode="replay", device="cpu", batch_size=1,
            warmup_count=0, cuda_synchronized=False, model_load_sec=0.0,
            inference_sec=0.1, total_sec=0.2, mean_latency_ms=100.0,
            p50_latency_ms=100.0, p95_latency_ms=100.0,
            peak_gpu_memory_mb=None, peak_cpu_rss_mb=None, sample_count=1,
            stages=(StageTiming("asr", 100.0, True),), capability_notes=("gpu_unavailable",),
        )
        self.assertEqual(RunTrace.from_dict(trace.to_dict()), trace)

    def test_all_contracts_serialize_nested_values(self):
        metadata = BackendMetadata("fixture", "m1")
        enhanced = EnhancedAudioResult(None, metadata, 1.5, True, "failed")
        issue = ValidationIssue("bad_value", "invalid value")
        self.assertEqual(enhanced.to_dict()["backend"], metadata.to_dict())
        self.assertEqual(issue.to_dict()["severity"], "error")

    def test_strict_from_dict_reports_contract_and_field(self):
        cases = (
            (BackendMetadata, {}, "BackendMetadata.*name"),
            (EvidenceWindow, {"start_sec": 0.0, "end_sec": "bad", "similarity": 0.5},
             "EvidenceWindow.*end_sec"),
            (TemporalSpeakerEvidence, {"id": "7"},
             "TemporalSpeakerEvidence.*backend"),
            (RunTrace, {"run_id": "r1"}, "RunTrace.*measurement_mode"),
            (RunTrace, {**self._trace_dict(), "sample_count": "bad"},
             "RunTrace.*sample_count"),
        )
        for contract, value, message in cases:
            with self.subTest(contract=contract.__name__):
                with self.assertRaisesRegex(ValueError, message):
                    contract.from_dict(value)

    def test_temporal_evidence_malformed_window_reports_sample_and_field(self):
        value = self._evidence_dict()
        value["windows"] = [{"start_sec": 0.0, "end_sec": 0.0, "similarity": 0.5}]
        with self.assertRaisesRegex(ValueError, "TemporalSpeakerEvidence.*7.*windows.*end_sec"):
            TemporalSpeakerEvidence.from_dict(value)

    def test_run_trace_rejects_invalid_measurement_mode(self):
        value = self._trace_dict()
        value["measurement_mode"] = "estimated"
        with self.assertRaisesRegex(ValueError, "RunTrace.*measurement_mode"):
            RunTrace.from_dict(value)

    def test_backend_protocols_are_runtime_checkable(self):
        class Backend:
            metadata = BackendMetadata("fixture", "m1")

            def load(self):
                pass

            def transcribe(self, sample):
                return AsrResult("", None, 0.0, "fixture")

            def score(self, sample):
                return TemporalSpeakerEvidence(
                    id=str(sample.id), backend=self.metadata,
                    enrollment_source="wake.wav", command_source="cmd.wav",
                )

            def enhance(self, sample):
                return EnhancedAudioResult(None, self.metadata, 0.0)

        backend = Backend()
        self.assertIsInstance(backend, AsrBackend)
        self.assertIsInstance(backend, TemporalSpeakerBackend)
        self.assertIsInstance(backend, EnhancementBackend)

    @staticmethod
    def _evidence_dict():
        return TemporalSpeakerEvidence(
            id="7", backend=BackendMetadata("fixture", "m1"),
            enrollment_source="wake.wav", command_source="cmd.wav",
        ).to_dict()

    @staticmethod
    def _trace_dict():
        return RunTrace(
            run_id="r1", measurement_mode="replay", device="cpu", batch_size=1,
            warmup_count=0, cuda_synchronized=False, model_load_sec=0.0,
            inference_sec=0.1, total_sec=0.2, mean_latency_ms=None,
            p50_latency_ms=None, p95_latency_ms=None, peak_gpu_memory_mb=None,
            peak_cpu_rss_mb=None, sample_count=1,
        ).to_dict()


if __name__ == "__main__":
    unittest.main()
