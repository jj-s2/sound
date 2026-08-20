"""Honest stage timing and optional resource measurement."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Iterator, Protocol

from .contracts import RunTrace, StageTiming


class ResourceProbe(Protocol):
    def synchronize(self) -> bool: ...

    def reset_peak_gpu_memory(self) -> bool: ...

    def peak_gpu_memory_mb(self) -> float | None: ...

    def cpu_rss_mb(self) -> float | None: ...

    def capability_notes(self) -> tuple[str, ...]: ...


class DefaultResourceProbe:
    """Probe CUDA and process resources without mandatory dependencies."""

    def __init__(self) -> None:
        self._notes: list[str] = []

    def _note(self, note: str) -> None:
        if note not in self._notes:
            self._notes.append(note)

    def synchronize(self) -> bool:
        try:
            import torch
        except ImportError:
            self._note("torch_unavailable")
            self._note("cuda_unavailable")
            return False

        try:
            if not torch.cuda.is_available():
                self._note("cuda_unavailable")
                return False
            torch.cuda.synchronize()
        except (AttributeError, RuntimeError):
            self._note("cuda_unavailable")
            return False
        return True

    def reset_peak_gpu_memory(self) -> bool:
        try:
            import torch
        except ImportError:
            self._note("torch_unavailable")
            self._note("cuda_unavailable")
            return False

        try:
            if not torch.cuda.is_available():
                self._note("cuda_unavailable")
                return False
            torch.cuda.reset_peak_memory_stats()
        except (AttributeError, RuntimeError):
            self._note("cuda_unavailable")
            return False
        return True

    def peak_gpu_memory_mb(self) -> float | None:
        try:
            import torch
        except ImportError:
            self._note("torch_unavailable")
            self._note("cuda_unavailable")
            return None

        try:
            if not torch.cuda.is_available():
                self._note("cuda_unavailable")
                return None
            return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            self._note("cuda_unavailable")
            return None

    def cpu_rss_mb(self) -> float | None:
        try:
            import psutil
        except ImportError:
            self._note("psutil_unavailable")
            return None

        try:
            return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
        except (AttributeError, OSError, TypeError, ValueError):
            self._note("psutil_measurement_unavailable")
            return None

    def capability_notes(self) -> tuple[str, ...]:
        return tuple(self._notes)


class RunTraceBuilder:
    def __init__(
        self,
        run_id: str,
        device: str,
        probe: ResourceProbe | None = None,
    ) -> None:
        self.run_id = run_id
        self.device = device
        self.probe = probe if probe is not None else DefaultResourceProbe()
        self._stages: list[StageTiming] = []
        self._latencies_ms: list[float] = []
        self._cuda_synchronized = False
        self.probe.reset_peak_gpu_memory()

    @contextmanager
    def stage(self, name: str, replay: bool = False) -> Iterator[None]:
        synchronized_before = self.probe.synchronize()
        self._cuda_synchronized = self._cuda_synchronized or synchronized_before
        started = time.perf_counter()
        try:
            yield
        finally:
            synchronized_after = self.probe.synchronize()
            self._cuda_synchronized = self._cuda_synchronized or synchronized_after
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._stages.append(StageTiming(name, elapsed_ms, replay))

    def record_sample_latency(self, latency_ms: float) -> None:
        if isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)):
            raise ValueError("latency_ms must be a finite non-negative number")
        latency = float(latency_ms)
        if not math.isfinite(latency) or latency < 0.0:
            raise ValueError("latency_ms must be a finite non-negative number")
        self._latencies_ms.append(latency)

    @staticmethod
    def _nearest_rank(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = math.ceil(percentile * len(ordered)) - 1
        return ordered[index]

    def finalize(
        self,
        *,
        measurement_mode: str,
        batch_size: int,
        warmup_count: int,
        model_load_sec: float,
        inference_sec: float,
        total_sec: float,
    ) -> RunTrace:
        if measurement_mode not in {"replay", "real"}:
            raise ValueError("measurement_mode must be 'replay' or 'real'")

        gpu_memory = self.probe.peak_gpu_memory_mb()
        cpu_rss = self.probe.cpu_rss_mb()
        notes = self.probe.capability_notes()
        mean_latency = (
            sum(self._latencies_ms) / len(self._latencies_ms)
            if self._latencies_ms
            else None
        )

        return RunTrace(
            run_id=self.run_id,
            measurement_mode=measurement_mode,
            device=self.device,
            batch_size=batch_size,
            warmup_count=warmup_count,
            cuda_synchronized=self._cuda_synchronized,
            model_load_sec=model_load_sec,
            inference_sec=inference_sec,
            total_sec=total_sec,
            mean_latency_ms=mean_latency,
            p50_latency_ms=self._nearest_rank(self._latencies_ms, 0.50),
            p95_latency_ms=self._nearest_rank(self._latencies_ms, 0.95),
            peak_gpu_memory_mb=gpu_memory,
            peak_cpu_rss_mb=cpu_rss,
            sample_count=len(self._latencies_ms),
            stages=tuple(self._stages),
            capability_notes=notes,
        )
