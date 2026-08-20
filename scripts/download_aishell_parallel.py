"""Resumable, range-based AISHELL-1 downloader.

Completed 128-MB parts are reused. The downloader is intentionally independent of
the training code so it can be stopped and restarted safely.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path


TOTAL_BYTES = 15_582_913_665
DEFAULT_URL = "https://openslr.magicdatatech.com/resources/33/data_aishell.tgz"
DEFAULT_RESOURCE_URL = "https://openslr.magicdatatech.com/resources/33/resource_aishell.tgz"
DEFAULT_CHUNK_BYTES = 128 * 1024 * 1024


class DownloadError(RuntimeError):
    pass


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _part_ranges(total: int, chunk: int) -> list[tuple[int, int, int]]:
    return [
        (index, index * chunk, min(total - 1, (index + 1) * chunk - 1))
        for index in range((total + chunk - 1) // chunk)
    ]


def _complete_bytes(parts: Path, ranges: list[tuple[int, int, int]]) -> tuple[int, int]:
    count = 0
    done = 0
    for index, start, end in ranges:
        path = parts / f"part-{index:04d}.bin"
        expected = end - start + 1
        if path.exists() and path.stat().st_size == expected:
            count += 1
            done += expected
    return count, done


def _write_status(
    status_path: Path,
    parts: Path,
    ranges: list[tuple[int, int, int]],
    *,
    workers: int,
    started_utc: str,
    extra: dict | None = None,
) -> None:
    count, done = _complete_bytes(parts, ranges)
    total = ranges[-1][2] + 1
    payload = {
        "phase": "parts",
        "total_bytes": total,
        "bytes_done": done,
        "parts_done": count,
        "parts_total": len(ranges),
        "percent": round(done / total * 100, 4),
        "workers": workers,
        "started_utc": started_utc,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    _atomic_json(status_path, payload)


def _download_part(
    item: tuple[int, int, int],
    *,
    parts: Path,
    url: str,
    status_path: Path,
    ranges: list[tuple[int, int, int]],
    workers: int,
    started_utc: str,
    lock: threading.Lock,
) -> tuple[int, bool, str]:
    index, start, end = item
    expected = end - start + 1
    final = parts / f"part-{index:04d}.bin"
    temp = parts / f"part-{index:04d}.tmp"
    range_temp = parts / f"part-{index:04d}.range.tmp"
    if final.exists() and final.stat().st_size == expected:
        return index, True, "already_complete"
    if final.exists():
        final.unlink()
    for attempt in range(1, 21):
        have = temp.stat().st_size if temp.exists() else 0
        if have > expected:
            temp.unlink()
            have = 0
        # A previous curl process may have been interrupted while writing the
        # remainder.  Treat those bytes as a committed prefix before opening a
        # new request; this keeps Ctrl-C, network timeouts, and process restarts
        # genuinely resumable.
        pending = range_temp.stat().st_size if range_temp.exists() else 0
        if pending:
            if have + pending <= expected:
                with range_temp.open("rb") as source, temp.open("ab") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                range_temp.unlink()
                have += pending
            else:
                range_temp.unlink()
        if have == expected:
            os.replace(temp, final)
            with lock:
                _write_status(
                    status_path, parts, ranges, workers=workers, started_utc=started_utc
                )
            return index, True, "completed_temp"
        request_start = start + have
        # Write each HTTP response to a separate file.  ``curl --output``
        # truncates its target; writing directly to ``temp`` would silently
        # discard already-downloaded bytes when resuming a partial part.
        if range_temp.exists():
            range_temp.unlink()
        command = [
            "curl.exe",
            "--location",
            "--fail",
            "--insecure",
            "--retry",
            "10",
            "--retry-delay",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--max-time",
            "7200",
            # The mirror can leave a connection open without sending bytes.
            # Abort that request after two minutes so the next attempt can
            # continue from the bytes already received instead of hanging for
            # hours on one range.
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
            "--range",
            f"{request_start}-{end}",
            "--output",
            str(range_temp),
            url,
        ]
        # A range request is written to a temporary file, so an interrupted
        # transfer cannot make a complete part look valid.
        try:
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            got = range_temp.stat().st_size if range_temp.exists() else 0
            remaining = expected - have
            if got <= remaining and got:
                with range_temp.open("rb") as source, temp.open("ab") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                range_temp.unlink()
                have += got
            if result.returncode == 0 and have == expected:
                if temp.stat().st_size != expected:
                    raise DownloadError(
                        f"part {index} resume size {temp.stat().st_size} != {expected}"
                    )
                os.replace(temp, final)
                with lock:
                    _write_status(
                        status_path, parts, ranges, workers=workers, started_utc=started_utc
                    )
                return index, True, "downloaded"
            if attempt in (1, 5, 10, 20):
                message = (result.stderr or "").strip().replace("\n", " ")[-240:]
                print(
                    f"part {index} attempt {attempt} failed rc={result.returncode} "
                    f"got={got}/{remaining} prefix={have}/{expected} {message}",
                    flush=True,
                )
        except Exception as exc:
            print(f"part {index} attempt {attempt} exception {exc}", flush=True)
        time.sleep(min(30, attempt * 2))
    return index, False, "failed"


def _merge_archive(parts: Path, archive: Path, ranges: list[tuple[int, int, int]]) -> None:
    with archive.open("wb") as output:
        for index, start, end in ranges:
            part = parts / f"part-{index:04d}.bin"
            expected = end - start + 1
            if not part.exists() or part.stat().st_size != expected:
                raise DownloadError(f"missing or short part {index}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    total = ranges[-1][2] + 1
    if archive.stat().st_size != total:
        raise DownloadError(f"archive size {archive.stat().st_size} != {total}")


def download(
    root: str | Path,
    *,
    workers: int = 8,
    url: str = DEFAULT_URL,
    resource_url: str = DEFAULT_RESOURCE_URL,
    chunk_mb: int = 128,
    download_resource: bool = True,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    if chunk_mb < 1:
        raise ValueError("chunk_mb must be positive")
    root = Path(root).expanduser().resolve(strict=False)
    parts = root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    status_path = root / "download_status.json"
    archive = root / "data_aishell.tgz"
    resource = root / "resource_aishell.tgz"
    ranges = _part_ranges(TOTAL_BYTES, chunk_mb * 1024 * 1024)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lock = threading.Lock()
    _write_status(
        status_path, parts, ranges, workers=workers, started_utc=started_utc
    )
    failures: list[tuple[int, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _download_part,
                item,
                parts=parts,
                url=url,
                status_path=status_path,
                ranges=ranges,
                workers=workers,
                started_utc=started_utc,
                lock=lock,
            )
            for item in ranges
        ]
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            index, ok, message = future.result()
            if not ok:
                failures.append((index, message))
            if number % 4 == 0 or number == len(futures):
                with lock:
                    _write_status(
                        status_path,
                        parts,
                        ranges,
                        workers=workers,
                        started_utc=started_utc,
                        extra={"completed_tasks": number, "failures": failures},
                    )
                print(
                    f"tasks {number}/{len(futures)} failures={len(failures)}",
                    flush=True,
                )
    if failures:
        _atomic_json(status_path, {"phase": "failed_parts", "failures": failures})
        raise DownloadError(str(failures))
    _atomic_json(status_path, {"phase": "merging"})
    _merge_archive(parts, archive, ranges)
    if download_resource:
        _atomic_json(status_path, {"phase": "resource"})
        command = [
            "curl.exe",
            "--location",
            "--fail",
            "--insecure",
            "--retry",
            "20",
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--max-time",
            "7200",
            "--output",
            str(resource),
            resource_url,
        ]
        result = subprocess.run(command, text=True)
        if result.returncode != 0:
            raise DownloadError(f"resource download failed: {result.returncode}")
    _atomic_json(
        status_path,
        {
            "phase": "complete",
            "archive_bytes": archive.stat().st_size,
            "resource_bytes": resource.stat().st_size if resource.exists() else 0,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/external/aishell1/download")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=128)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--resource-url", default=DEFAULT_RESOURCE_URL)
    parser.add_argument("--skip-resource", action="store_true")
    args = parser.parse_args()
    download(
        args.root,
        workers=args.workers,
        url=args.url,
        resource_url=args.resource_url,
        chunk_mb=args.chunk_mb,
        download_resource=not args.skip_resource,
    )


if __name__ == "__main__":
    main()
