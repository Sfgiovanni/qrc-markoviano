#!/usr/bin/env python3
"""Item 10 — sha256 of every CSV that feeds a final figure or the verdict.

Covers the immutable source tables (results_abc_comparison_v2/, results_extra_v*/,
results_corrections_v6/) and the review outputs (results_review/). Writes
results_review/csv_hashes.txt (sorted, `sha256  relpath  bytes`). Run from repo root.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRS = ["results/v2_abc_comparison", "results/extra_v3", "results/extra_v4",
        "results/extra_v5", "results/corrections_v6", "results/review"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    lines = []
    for d in DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for csv in sorted(base.rglob("*.csv")):
            if ".ipynb_checkpoints" in csv.parts:
                continue
            rel = csv.relative_to(ROOT).as_posix()
            lines.append(f"{sha256(csv)}  {rel}  {csv.stat().st_size}")
    header = ("# sha256 of CSVs feeding final figures / verdict (item 10)\n"
              "# format: <sha256>  <path>  <bytes>\n"
              f"# {len(lines)} files\n")
    (ROOT / "results_review" / "csv_hashes.txt").write_text(
        header + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"hashed {len(lines)} CSVs -> results_review/csv_hashes.txt")


if __name__ == "__main__":
    main()
