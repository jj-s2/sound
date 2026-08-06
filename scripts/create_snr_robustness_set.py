"""Create Dataset-A style SNR robustness evaluation data.

The competition document defines SNR robustness as CER measured under several
SNR conditions such as 5 dB, 0 dB and -5 dB, then averaged. This script creates
a labeled positive-only evaluation set for that protocol.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xh202615.data import load_dataset
from xh202615.snr import ensure_sample_rate, estimate_snr_db, mix_at_snr, read_mono_audio, write_mono_audio


def parse_args():
    parser = argparse.ArgumentParser(description="Create SNR robustness eval set")
    parser.add_argument("--dataset-root", default="datasetA")
    parser.add_argument("--output-root", default="output/snr_eval/snr_5_0_-5")
    parser.add_argument("--splits", default="pos", help="Usually pos, because CER needs labels")
    parser.add_argument("--snrs", default="5,0,-5", help="Comma-separated SNR dB values")
    parser.add_argument("--noise-source", choices=("white", "neg", "directory"), default="white")
    parser.add_argument("--noise-dir", default=None, help="Directory of wav/flac/mp3 noise files for --noise-source directory")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-split-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=202615)
    parser.add_argument("--copy-wakeup", action="store_true", help="Copy wakeup wav files into output-root instead of using absolute paths")
    return parser.parse_args()


def parse_snrs(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def snr_tag(snr_db: float) -> str:
    sign = "p" if snr_db >= 0 else "m"
    value = str(abs(snr_db)).replace(".", "d")
    return f"{sign}{value}"


def collect_directory_noise(noise_dir: str | Path) -> list[Path]:
    root = Path(noise_dir)
    suffixes = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    return [path for path in root.rglob("*") if path.suffix.lower() in suffixes]


def load_noise_pool(args, rng: random.Random) -> list[Path] | None:
    if args.noise_source == "white":
        return None
    if args.noise_source == "directory":
        if not args.noise_dir:
            raise SystemExit("--noise-dir is required for --noise-source directory")
        pool = collect_directory_noise(args.noise_dir)
    else:
        pool = [sample.command_audio for sample in load_dataset(args.dataset_root, ["neg"])]
    if not pool:
        raise SystemExit(f"No noise files found for noise source: {args.noise_source}")
    rng.shuffle(pool)
    return pool


def choose_noise(noise_pool: list[Path] | None, clean_len: int, rng_np: np.random.Generator, rng_py: random.Random) -> tuple[np.ndarray, int, str]:
    if noise_pool is None:
        return rng_np.standard_normal(clean_len).astype(np.float32), 0, "white"
    path = rng_py.choice(noise_pool)
    noise, sr = read_mono_audio(path)
    return noise, sr, str(path)


def main() -> None:
    args = parse_args()
    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)
    snrs = parse_snrs(args.snrs)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples = [sample for sample in load_dataset(args.dataset_root, splits) if sample.label is not None]
    if args.per_split_limit is not None:
        selected = []
        for split in splits:
            selected.extend([sample for sample in samples if sample.split == split][: args.per_split_limit])
        samples = selected
    if args.limit is not None:
        samples = samples[: args.limit]

    out_root = Path(args.output_root)
    audio_dir = out_root / "pos"
    wakeup_dir = out_root / "wakeup"
    out_root.mkdir(parents=True, exist_ok=True)
    noise_pool = load_noise_pool(args, rng_py)

    rows = []
    meta_rows = []
    for sample in samples:
        clean, sr = read_mono_audio(sample.command_audio)
        wakeup_path = sample.wakeup_audio
        if args.copy_wakeup:
            wakeup_path = wakeup_dir / f"kws_{sample.id}.wav"
            wakeup_path.parent.mkdir(parents=True, exist_ok=True)
            if not wakeup_path.exists():
                shutil.copy2(sample.wakeup_audio, wakeup_path)
        for snr_db in snrs:
            tag = snr_tag(snr_db)
            new_id = f"{sample.id}_snr_{tag}"
            noise, noise_sr, noise_name = choose_noise(noise_pool, clean.size, rng_np, rng_py)
            if noise_sr and noise_sr != sr:
                noise = ensure_sample_rate(noise, noise_sr, sr)
            mixed = mix_at_snr(clean, noise, snr_db, rng_np)
            rel_cmd = Path("pos") / f"cmd_{new_id}.wav"
            out_wav = out_root / rel_cmd
            write_mono_audio(out_wav, mixed, sr)
            rows.append(
                {
                    "id": new_id,
                    "wakeup_audio": str(wakeup_path.resolve()) if not args.copy_wakeup else str(Path("wakeup") / f"kws_{sample.id}.wav"),
                    "wakeup_text": sample.wakeup_text,
                    "command_audio": str(rel_cmd),
                    "recognition_text": sample.label,
                }
            )
            meta_rows.append(
                {
                    "id": new_id,
                    "source_id": sample.id,
                    "snr_db": snr_db,
                    "estimated_snr_db": estimate_snr_db(clean, mixed),
                    "noise_source": args.noise_source,
                    "noise_audio": noise_name,
                    "sample_rate": sr,
                    "duration_sec": len(clean) / sr if sr else 0.0,
                    "command_audio": str(rel_cmd),
                    "reference_text": sample.label,
                }
            )

    jsonl_path = out_root / "pos.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta_path = out_root / "snr_metadata.jsonl"
    with meta_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    readme = out_root / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "SNR robustness evaluation set",
                f"source_dataset={args.dataset_root}",
                f"snrs={','.join(str(x) for x in snrs)}",
                f"noise_source={args.noise_source}",
                f"samples={len(samples)}",
                f"rows={len(rows)}",
                "Use --dataset-root pointing to this directory and --splits pos.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote SNR eval set: {out_root}")
    print(f"Rows: {len(rows)}; metadata: {meta_path}; jsonl: {jsonl_path}")


if __name__ == "__main__":
    main()
