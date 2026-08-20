"""Collect environment and optional inference capability information."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from .instrumentation import DefaultResourceProbe


VersionResolver = Callable[[str], str]


def _package_report(name: str, version_resolver: VersionResolver) -> dict:
    try:
        version = version_resolver(name)
    except Exception:
        return {"installed": False, "version": None}
    return {"installed": True, "version": str(version)}


def _torch_report() -> tuple[dict, dict, str]:
    try:
        import torch
    except ImportError:
        return (
            {"installed": False, "version": None},
            {"available": False, "version": None},
            "cpu",
        )

    version = getattr(torch, "__version__", None)
    torch_info = {"installed": True, "version": None if version is None else str(version)}
    try:
        cuda_available = bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        cuda_available = False

    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    cuda_info: dict[str, object] = {
        "available": cuda_available,
        "version": None if cuda_version is None else str(cuda_version),
    }
    if not cuda_available:
        return torch_info, cuda_info, "cpu"

    try:
        device_index = int(torch.cuda.current_device())
        device_name = str(torch.cuda.get_device_name(device_index))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        device_index = 0
        device_name = None
    cuda_info["device_index"] = device_index
    cuda_info["device_name"] = device_name
    return torch_info, cuda_info, f"cuda:{device_index}"


def collect_environment(
    package_names: Iterable[str],
    artifact_paths: Iterable[str | Path],
    *,
    version_resolver: VersionResolver | None = None,
) -> dict:
    """Return a JSON-serializable environment report without changing the system."""

    resolver = version_resolver or importlib.metadata.version
    packages = {
        str(name): _package_report(str(name), resolver)
        for name in package_names
    }
    torch_info, cuda_info, device = _torch_report()

    probe = DefaultResourceProbe()
    gpu_memory = probe.peak_gpu_memory_mb()
    cpu_rss = probe.cpu_rss_mb()
    resource_capabilities = {
        "cuda_synchronization": probe.synchronize(),
        "gpu_peak_memory": gpu_memory is not None,
        "cpu_rss": cpu_rss is not None,
        "capability_notes": list(probe.capability_notes()),
    }

    artifact_checks = {}
    for artifact in artifact_paths:
        path_text = str(artifact)
        path = Path(artifact)
        artifact_checks[path_text] = {
            "exists": path.exists(),
            "is_file": path.is_file(),
        }

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "torch": torch_info,
        "cuda": cuda_info,
        "device": device,
        "resource_capabilities": resource_capabilities,
        "artifact_checks": artifact_checks,
    }
