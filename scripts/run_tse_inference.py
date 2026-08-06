"""Run the trained enrollment-conditioned TSE model and write an ASR audio map.

The input contract intentionally uses only ``id``, ``split``, enrollment audio,
and command audio. Other JSON fields (including labels/text) are ignored. The
resulting JSONL is accepted by ``scripts/run_funasr_asr.py`` via
``--command-audio-map``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.target_extractor import FiLMCRNExtractor, enhance_waveform


SAMPLE_RATE = 16_000
EMBEDDING_DIM = 256
EmbeddingProvider = Callable[[Path], np.ndarray]


@dataclass(frozen=True)
class InputRow:
    sample_id: str
    split: str
    wakeup_audio: Path
    command_audio: Path


_ID_KEYS = ("id", "row_id", "utt_id", "sample_id")
_WAKE_KEYS = (
    "wakeup_audio", "enrollment_audio", "kws_path", "唤醒音频", "鍞ら啋闊抽"
)
_COMMAND_KEYS = (
    "command_audio", "mixture_audio", "cmd_path", "识别音频", "璇嗗埆闊抽"
)


def _first(row: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in row:
            return row[key]
    return None


def _resolve_audio(value: object, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("audio path must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.expanduser().resolve(strict=False)


def parse_input_row(row: dict, *, base_dir: Path, default_split: str) -> InputRow:
    """Extract input-side fields only; label/text fields are deliberately ignored."""
    sample_id = _first(row, _ID_KEYS)
    wake = _first(row, _WAKE_KEYS)
    command = _first(row, _COMMAND_KEYS)
    if sample_id is None or not str(sample_id).strip():
        raise ValueError("row is missing id")
    if wake is None or command is None:
        raise ValueError(f"row {sample_id!r} is missing wakeup_audio or command_audio")
    split = row.get("split", default_split)
    return InputRow(
        sample_id=str(sample_id),
        split=str(split or default_split),
        wakeup_audio=_resolve_audio(wake, base_dir),
        command_audio=_resolve_audio(command, base_dir),
    )


def read_input_jsonl(path: str | Path, *, default_split: str = "") -> list[InputRow]:
    """Read JSONL without depending on Dataset-A's label schema."""
    path = Path(path).expanduser().resolve(strict=False)
    rows: list[InputRow] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"row at {path}:{line_no} must be an object")
            rows.append(parse_input_row(raw, base_dir=path.parent, default_split=default_split))
    return rows


def load_input_rows(
    *, input_jsonl: Iterable[str | Path] = (), dataset_root: str | Path | None = None,
    splits: Iterable[str] = ("pos", "neg"),
) -> list[InputRow]:
    paths = [Path(p) for p in input_jsonl]
    if dataset_root is not None:
        root = Path(dataset_root).expanduser().resolve(strict=False)
        paths.extend(root / f"{split}.jsonl" for split in splits)
    if not paths:
        raise ValueError("provide --input-jsonl or --dataset-root")
    rows: list[InputRow] = []
    for path in paths:
        split = path.stem if dataset_root is not None else ""
        rows.extend(read_input_jsonl(path, default_split=split))
    ids = [row.sample_id for row in rows]
    duplicates = sorted({x for x in ids if ids.count(x) > 1})
    if duplicates:
        raise ValueError(f"duplicate ids in input: {duplicates[:8]}")
    return rows


def input_manifest_digest(paths: Iterable[str | Path]) -> str:
    """Digest full input manifest bytes and names for cache invalidation."""
    digest = hashlib.sha256()
    for path in paths:
        resolved = Path(path).expanduser().resolve(strict=False)
        digest.update(str(resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_audio(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing audio: {path}")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}: {path}")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"audio is empty or non-finite: {path}")
    return np.ascontiguousarray(mono)


def write_audio(path: Path, audio: np.ndarray) -> None:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"enhanced output is empty or non-finite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, subtype="FLOAT")


def load_checkpoint(path: str | Path, device: torch.device) -> FiLMCRNExtractor:
    checkpoint_path = Path(path).expanduser().resolve(strict=False)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config")
    state = checkpoint.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint must contain model_config and model_state_dict")
    required = {"embedding_dim", "channels", "n_fft", "hop_length", "win_length", "gru_hidden", "gru_layers"}
    if not required.issubset(config):
        raise ValueError(f"checkpoint model_config missing: {sorted(required - set(config))}")
    model = FiLMCRNExtractor(
        embedding_dim=int(config["embedding_dim"]),
        channels=tuple(int(x) for x in config["channels"]),
        n_fft=int(config["n_fft"]),
        hop_length=int(config["hop_length"]),
        win_length=int(config["win_length"]),
        gru_hidden=int(config["gru_hidden"]),
        gru_layers=int(config["gru_layers"]),
    ).to(device)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all() for value in model.parameters()):
        raise ValueError("checkpoint contains non-finite parameters")
    model.eval()
    return model


def _extract_embedding(model, audio: np.ndarray) -> np.ndarray:
    pcm = torch.from_numpy(audio).reshape(1, -1)
    with torch.no_grad():
        model_device = torch.device(getattr(model, "device", "cpu"))
        if (
            model_device.type != "cpu"
            and callable(getattr(model, "compute_features", None))
            and callable(getattr(model, "model", None))
        ):
            features = model.compute_features(pcm, sample_rate=SAMPLE_RATE, cmn=True)
            outputs = model.model(features.to(model_device))
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
            value = outputs[0].detach().cpu().numpy()
        else:
            value = model.extract_embedding_from_pcm(pcm, SAMPLE_RATE)
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.size != EMBEDDING_DIM or not np.isfinite(value).all():
        raise ValueError(f"unexpected enrollment embedding shape: {value.shape}")
    return value


def build_wespeaker_provider(model_name: str, device: torch.device) -> EmbeddingProvider:
    try:
        import wespeaker
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("WeSpeaker is required for TSE inference") from exc
    speaker_model = wespeaker.load_model(model_name)
    if callable(getattr(speaker_model, "set_device", None)):
        speaker_model.set_device(str(device))
    if callable(getattr(speaker_model, "eval", None)):
        speaker_model.eval()

    def provider(path: Path) -> np.ndarray:
        return _extract_embedding(speaker_model, read_audio(path))

    return provider


def load_or_build_cache(
    rows: Iterable[InputRow], *, cache_path: Path, manifest_digest: str,
    model_name: str, provider: EmbeddingProvider, reuse: bool = True,
) -> dict[str, np.ndarray]:
    if reuse and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("manifest_digest") == manifest_digest and payload.get("model_name") == model_name:
            values = payload.get("embeddings")
            if isinstance(values, dict):
                converted = {str(k): np.asarray(v, dtype=np.float32).reshape(-1) for k, v in values.items()}
                if all(value.size == EMBEDDING_DIM and np.isfinite(value).all() for value in converted.values()):
                    return converted
    embeddings: dict[str, np.ndarray] = {}
    for row in rows:
        key = str(row.wakeup_audio)
        if key not in embeddings:
            value = np.asarray(provider(row.wakeup_audio), dtype=np.float32).reshape(-1)
            if value.size != EMBEDDING_DIM or not np.isfinite(value).all():
                raise ValueError(f"invalid enrollment embedding for {row.wakeup_audio}: {value.shape}")
            embeddings[key] = value
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest_digest": manifest_digest, "model_name": model_name, "embeddings": embeddings}, cache_path)
    return embeddings


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "sample"


def _done_records(path: Path, rows_by_id: dict[str, InputRow]) -> tuple[set[str], list[str]]:
    if not path.is_file():
        return set(), []
    done: set[str] = set()
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            sample_id = str(record["id"])
            output = Path(str(record["enhanced_command_audio"]))
            if sample_id in rows_by_id and not record.get("error") and output.is_file() and sample_id not in done:
                done.add(sample_id)
                kept.append(json.dumps(record, ensure_ascii=False))
        except (ValueError, OSError, json.JSONDecodeError, KeyError):
            continue
    return done, kept


def run_inference(
    rows: list[InputRow], *, checkpoint: str | Path, output_root: str | Path,
    output_map: str | Path, embedding_cache: str | Path, device: str = "cuda",
    model_name: str = "chinese", limit: int | None = None, resume: bool = False,
    on_error: str = "raise", embedding_provider: EmbeddingProvider | None = None,
    manifest_digest: str = "",
) -> dict:
    if not rows:
        raise ValueError("no input rows")
    if limit is not None:
        rows = rows[:limit]
    rows_by_id = {row.sample_id: row for row in rows}
    for row in rows:
        read_audio(row.wakeup_audio)
        read_audio(row.command_audio)
    torch_device = torch.device(device)
    model = load_checkpoint(checkpoint, torch_device)
    provider = embedding_provider or build_wespeaker_provider(model_name, torch_device)
    cache = load_or_build_cache(
        rows, cache_path=Path(embedding_cache), manifest_digest=manifest_digest,
        model_name=model_name, provider=provider,
    )
    out_map = Path(output_map)
    out_map.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    kept: list[str] = []
    if resume:
        done, kept = _done_records(out_map, rows_by_id)
        tmp = out_map.with_suffix(out_map.suffix + ".tmp")
        tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        tmp.replace(out_map)
    mode = "a" if resume else "w"
    output_root = Path(output_root)
    count = errors = 0
    with out_map.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if row.sample_id in done:
                continue
            output_path = output_root / row.split / f"cmd_{_safe_id(row.sample_id)}.wav"
            record = {
                "id": row.sample_id,
                "split": row.split,
                "wakeup_audio": str(row.wakeup_audio),
                "original_command_audio": str(row.command_audio),
                "enhanced_command_audio": str(output_path),
                "latency_ms": 0.0,
                "error": "",
            }
            if output_path.resolve(strict=False) == row.command_audio.resolve(strict=False):
                raise ValueError("refusing to overwrite original command audio")
            try:
                mixture = read_audio(row.command_audio)
                embedding = torch.from_numpy(cache[str(row.wakeup_audio)]).reshape(1, -1).to(torch_device)
                waveform = torch.from_numpy(mixture).reshape(1, -1).to(torch_device)
                start = time.perf_counter()
                with torch.inference_mode():
                    enhanced = enhance_waveform(model, waveform, embedding)
                result = enhanced.detach().float().cpu().numpy().reshape(-1)
                if result.size != mixture.size or not np.isfinite(result).all():
                    raise ValueError("non-finite or length-changing enhanced output")
                write_audio(output_path, result)
                record["latency_ms"] = (time.perf_counter() - start) * 1000.0
            except Exception as exc:  # noqa: BLE001
                errors += 1
                if on_error == "raise":
                    raise
                record["error"] = str(exc)
                record["enhanced_command_audio"] = str(row.command_audio)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
    return {"rows": count, "errors": errors, "skipped": len(done), "output_map": str(out_map), "embedding_cache": str(embedding_cache)}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-jsonl", action="append", default=[])
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--splits", default="pos,neg")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-map", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="chinese")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--on-error", choices=("raise", "copy"), default="raise")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> dict:
    args = parse_args(argv)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    rows = load_input_rows(input_jsonl=args.input_jsonl, dataset_root=args.dataset_root, splits=splits)
    manifest_paths = list(args.input_jsonl)
    if args.dataset_root:
        root = Path(args.dataset_root)
        manifest_paths.extend(root / f"{split}.jsonl" for split in splits)
    summary = run_inference(
        rows, checkpoint=args.checkpoint, output_root=args.output_root,
        output_map=args.output_map, embedding_cache=args.embedding_cache,
        device=args.device, model_name=args.model, limit=args.limit,
        resume=args.resume, on_error=args.on_error,
        manifest_digest=input_manifest_digest(manifest_paths),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
