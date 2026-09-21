#!/usr/bin/env python3
"""Regenerate Figures 2-5 and S1-S3 without changing locked publication PDFs."""
from __future__ import annotations
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RENDERER = ROOT / "code" / "final_publication_figures.py"
SOURCE = ROOT / "data" / "source_data"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def snapshot() -> dict[str, str]:
    return {p.relative_to(SOURCE).as_posix(): sha256(p) for p in SOURCE.rglob("*") if p.is_file()}

parser = argparse.ArgumentParser()
parser.add_argument("--dpi", type=int, default=200)
parser.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "svg"])
parser.add_argument("--allow-font-fallback-for-preview", action="store_true")
args = parser.parse_args()

before = snapshot()
cmd = [sys.executable, "-B", str(RENDERER), "--figures", "2", "3", "4", "5", "S1", "S2", "S3", "--main-grid", "mixed", "--dpi", str(args.dpi), "--formats", *args.formats]
if args.allow_font_fallback_for_preview:
    cmd.append("--allow-font-fallback-for-preview")
subprocess.run(cmd, cwd=ROOT, check=True)
after = snapshot()
if before != after:
    raise SystemExit("ERROR: source-data hashes changed during plotting.")
print("PASS: source tables were unchanged.")
print("Figure 1 remains the locked static publication asset in ../figures/main/Figure_1.pdf")
