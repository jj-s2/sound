"""Lifecycle-aware interfaces for inference backend implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .backends import AsrResult
from .contracts import BackendMetadata, EnhancedAudioResult, TemporalSpeakerEvidence
from .data import Sample


@runtime_checkable
class AsrBackend(Protocol):
    metadata: BackendMetadata

    def load(self) -> None: ...

    def transcribe(self, sample: Sample) -> AsrResult: ...


@runtime_checkable
class TemporalSpeakerBackend(Protocol):
    metadata: BackendMetadata

    def load(self) -> None: ...

    def score(self, sample: Sample) -> TemporalSpeakerEvidence: ...


@runtime_checkable
class EnhancementBackend(Protocol):
    metadata: BackendMetadata

    def load(self) -> None: ...

    def enhance(self, sample: Sample) -> EnhancedAudioResult: ...
