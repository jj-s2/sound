"""Tests for competition submission validation."""

from __future__ import annotations

import unittest

from xh202615.contracts import RunTrace
from xh202615.submission_validation import validate_competition_payload


EXPECTED = {"a.wav", "b.wav"}


def payload(rows=None, duration="1.25"):
    return {
        "result": {
            "results": rows if rows is not None else [
                {"id": "a.wav", "content": "hello"},
                {"id": "b.wav", "content": ""},
            ],
            "duration": duration,
        }
    }


def trace(mode: str) -> RunTrace:
    return RunTrace(
        run_id="run",
        measurement_mode=mode,
        device="cpu",
        batch_size=1,
        warmup_count=0,
        cuda_synchronized=False,
        model_load_sec=0.1,
        inference_sec=1.0,
        total_sec=1.1,
        mean_latency_ms=None,
        p50_latency_ms=None,
        p95_latency_ms=None,
        peak_gpu_memory_mb=None,
        peak_cpu_rss_mb=None,
        sample_count=2,
    )


class SubmissionValidationTests(unittest.TestCase):
    def validate(self, value, **kwargs):
        return validate_competition_payload(
            value,
            expected_ids=EXPECTED,
            allowed_row_fields={"id", "content"},
            **kwargs,
        )

    def codes(self, issues):
        return {issue.code for issue in issues}

    def test_valid_wrapper_and_exact_ids(self):
        self.assertEqual(self.validate(payload()), ())

    def test_duplicate_missing_and_extra_ids(self):
        duplicate = self.validate(payload([
            {"id": "a.wav", "content": "one"},
            {"id": "a.wav", "content": "two"},
        ]))
        self.assertIn("duplicate_id", self.codes(duplicate))
        self.assertIn("missing_id", self.codes(duplicate))

        missing = self.validate(payload([{"id": "a.wav", "content": "one"}]))
        self.assertIn("missing_id", self.codes(missing))

        extra = self.validate(payload([
            {"id": "a.wav", "content": "one"},
            {"id": "b.wav", "content": "two"},
            {"id": "c.wav", "content": "three"},
        ]))
        self.assertIn("extra_id", self.codes(extra))

    def test_non_string_content_is_rejected_except_empty_string(self):
        issues = self.validate(payload([
            {"id": "a.wav", "content": 1},
            {"id": "b.wav", "content": None},
        ]))
        self.assertIn("content_not_string", self.codes(issues))

    def test_disallowed_row_fields_are_rejected(self):
        issues = self.validate(payload([
            {"id": "a.wav", "content": "one", "label": "truth"},
            {"id": "b.wav", "content": "two"},
        ]))
        self.assertIn("disallowed_row_field", self.codes(issues))

    def test_invalid_duration_is_rejected(self):
        for duration in ("not-a-number", -1, None):
            with self.subTest(duration=duration):
                self.assertIn("invalid_duration", self.codes(self.validate(payload(duration=duration))))

    def test_official_without_trace_requires_real_trace(self):
        issues = self.validate(payload(), official=True)
        self.assertIn("official_duration_requires_real_trace", self.codes(issues))

    def test_official_replay_trace_requires_real_trace(self):
        issues = self.validate(payload(), trace=trace("replay"), official=True)
        self.assertIn("official_duration_requires_real_trace", self.codes(issues))

    def test_official_real_trace_is_accepted(self):
        self.assertEqual(self.validate(payload(), trace=trace("real"), official=True), ())


if __name__ == "__main__":
    unittest.main()
