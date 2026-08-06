"""List downloaded public data files for XH-202615."""

from __future__ import annotations

from pathlib import Path


def fmt_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024
    return f"{size}B"


def main() -> None:
    root = Path("data/raw_public")
    if not root.exists():
        print("No data/raw_public directory.")
        return
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [p for p in dataset_dir.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        print(f"\n{dataset_dir.name}: {len(files)} files, {fmt_size(total)}")
        for file in files[:20]:
            print(f"  - {file.relative_to(root)} ({fmt_size(file.stat().st_size)})")
        if len(files) > 20:
            print(f"  ... {len(files) - 20} more files")


if __name__ == "__main__":
    main()

