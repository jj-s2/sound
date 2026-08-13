"""Run the frozen temporal head from label-free input and candidate JSONL maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_tse_inference import InputRow, read_input_jsonl


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        raw = json.loads(line)
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), (str, int)):
            raise ValueError(f"candidate row {number} has invalid id")
        sample_id = str(raw["id"])
        if sample_id in result:
            raise ValueError(f"duplicate candidate ID {sample_id!r}")
        text = raw.get("recognition_text", raw.get("text", ""))
        result[sample_id] = "" if text is None else str(text)
    return result


def run_inference(
    input_jsonl: Path,
    candidate_asr: Path,
    *,
    threshold: float,
    probability_for: Callable[[InputRow], float],
) -> list[dict[str, object]]:
    """Gate a complete candidate map using a label-free probability provider."""
    rows = read_input_jsonl(input_jsonl)
    ids = [row.sample_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("input JSONL has duplicate IDs")
    candidates = _candidate_map(candidate_asr)
    if set(candidates) != set(ids):
        raise ValueError("candidate IDs must exactly cover input IDs")
    output: list[dict[str, object]] = []
    for row in rows:
        probability = float(probability_for(row))
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("temporal probability must be finite and in [0, 1]")
        accepted = probability >= threshold
        output.append({
            "id": row.sample_id,
            "recognition_text": candidates[row.sample_id] if accepted else "",
            "temporal_probability": probability,
            "accepted": accepted,
            "route": "accepted_fusion" if accepted else "rejected",
            "command_audio_sha256": _sha_file(row.command_audio),
        })
    return output


def run_inference_many(
    input_jsonls: Sequence[Path],
    candidate_asr: Path,
    *,
    threshold: float,
    probability_for: Callable[[InputRow], float],
) -> list[dict[str, object]]:
    """Run label-free temporal inference across disjoint input manifests."""
    if not input_jsonls:
        raise ValueError("provide at least one input JSONL")
    merged = [row for path in input_jsonls for row in read_input_jsonl(path)]
    ids = [row.sample_id for row in merged]
    if len(ids) != len(set(ids)):
        raise ValueError("input JSONLs have duplicate IDs")
    candidates = _candidate_map(candidate_asr)
    if set(candidates) != set(ids):
        raise ValueError("candidate IDs must exactly cover input IDs")
    output: list[dict[str, object]] = []
    for row in merged:
        probability = float(probability_for(row))
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("temporal probability must be finite and in [0, 1]")
        accepted = probability >= threshold
        output.append({
            "id": row.sample_id,
            "recognition_text": candidates[row.sample_id] if accepted else "",
            "temporal_probability": probability,
            "accepted": accepted,
            "route": "accepted_fusion" if accepted else "rejected",
            "command_audio_sha256": _sha_file(row.command_audio),
        })
    return output


def _model_probability_provider(checkpoint_path: Path, device: str) -> tuple[float, Callable[[InputRow], float]]:
    import torch

    from scripts.train_temporal_head import _feature_sequence, prepare_frozen_encoder
    from xh202615.temporal_head import TemporalSpeakerHead

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    threshold = float(checkpoint.get("presence_threshold", 0.5))
    model = TemporalSpeakerHead(
        input_dim=int(checkpoint["input_dim"]), hidden_dim=int(checkpoint["hidden_dim"]), mode=str(checkpoint["mode"])
    )
    model.load_state_dict(checkpoint["model"])
    torch_device = torch.device(device)
    model.to(torch_device).eval()
    try:
        import wespeaker
    except ImportError as exc:  # pragma: no cover - depends on the inference environment
        raise RuntimeError("WeSpeaker is required for temporal inference") from exc
    encoder = prepare_frozen_encoder(wespeaker.load_model("chinese"), device=str(torch_device))
    windows = int(checkpoint["window_count"])

    def probability_for(row: InputRow) -> float:
        sequence = _feature_sequence(encoder, row.wakeup_audio, row.command_audio, windows)
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(sequence).unsqueeze(0).to(torch_device))
        return float(torch.sigmoid(logits).item())

    return threshold, probability_for


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--candidate-asr", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    threshold, provider = _model_probability_provider(args.checkpoint, args.device)
    rows = run_inference_many(args.input_jsonl, args.candidate_asr, threshold=threshold, probability_for=provider)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
