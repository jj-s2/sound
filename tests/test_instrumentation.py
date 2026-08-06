import unittest

from xh202615.instrumentation import RunTraceBuilder


class FakeProbe:
    def __init__(self, *, synchronized=True, gpu_memory=123.5, rss=45.25, notes=()):
        self.events = []
        self._synchronized = synchronized
        self._gpu_memory = gpu_memory
        self._rss = rss
        self._notes = tuple(notes)

    def synchronize(self):
        self.events.append("synchronize")
        return self._synchronized

    def reset_peak_gpu_memory(self):
        self.events.append("reset_peak_gpu_memory")
        return self._gpu_memory is not None

    def peak_gpu_memory_mb(self):
        self.events.append("peak_gpu_memory_mb")
        return self._gpu_memory

    def cpu_rss_mb(self):
        self.events.append("cpu_rss_mb")
        return self._rss

    def capability_notes(self):
        return self._notes


class InstrumentationTest(unittest.TestCase):
    def test_stage_synchronizes_immediately_before_and_after_body(self):
        probe = FakeProbe()
        builder = RunTraceBuilder("run-1", "cuda:0", probe=probe)

        with builder.stage("inference", replay=False):
            self.assertEqual(probe.events[-1], "synchronize")
            probe.events.append("body")

        self.assertEqual(probe.events[-3:], ["synchronize", "body", "synchronize"])

    def test_replay_finalize_preserves_stage_replay_flags(self):
        probe = FakeProbe()
        builder = RunTraceBuilder("run-1", "cpu", probe=probe)
        with builder.stage("asr", replay=True):
            pass

        trace = builder.finalize(
            measurement_mode="replay",
            batch_size=1,
            warmup_count=0,
            model_load_sec=0.0,
            inference_sec=0.02,
            total_sec=0.03,
        )

        self.assertEqual(trace.measurement_mode, "replay")
        self.assertEqual(trace.stages[0].stage, "asr")
        self.assertTrue(trace.stages[0].replay)

    def test_nearest_rank_percentiles_are_deterministic(self):
        builder = RunTraceBuilder("run-1", "cpu", probe=FakeProbe())
        for latency in (10.0, 20.0, 30.0, 40.0):
            builder.record_sample_latency(latency)

        trace = builder.finalize(
            measurement_mode="real",
            batch_size=2,
            warmup_count=1,
            model_load_sec=0.1,
            inference_sec=0.2,
            total_sec=0.3,
        )

        self.assertEqual(trace.mean_latency_ms, 25.0)
        self.assertEqual(trace.p50_latency_ms, 20.0)
        self.assertEqual(trace.p95_latency_ms, 40.0)

    def test_missing_resources_are_none_with_capability_notes(self):
        probe = FakeProbe(
            synchronized=False,
            gpu_memory=None,
            rss=None,
            notes=("torch_unavailable", "cuda_unavailable", "psutil_unavailable"),
        )
        builder = RunTraceBuilder("run-1", "cpu", probe=probe)
        trace = builder.finalize(
            measurement_mode="replay",
            batch_size=1,
            warmup_count=0,
            model_load_sec=0.0,
            inference_sec=0.0,
            total_sec=0.0,
        )

        self.assertIsNone(trace.peak_gpu_memory_mb)
        self.assertIsNone(trace.peak_cpu_rss_mb)
        self.assertFalse(trace.cuda_synchronized)
        self.assertIn("torch_unavailable", trace.capability_notes)
        self.assertIn("cuda_unavailable", trace.capability_notes)
        self.assertIn("psutil_unavailable", trace.capability_notes)

    def test_invalid_measurement_mode_is_rejected(self):
        builder = RunTraceBuilder("run-1", "cpu", probe=FakeProbe())
        with self.assertRaisesRegex(ValueError, "measurement_mode"):
            builder.finalize(
                measurement_mode="estimated",
                batch_size=1,
                warmup_count=0,
                model_load_sec=0.0,
                inference_sec=0.0,
                total_sec=0.0,
            )


if __name__ == "__main__":
    unittest.main()
