#!/usr/bin/env python3
"""Check regenerated scratch outputs without comparing them to locked PDFs."""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "outputs" / "main"
SI = ROOT / "outputs" / "si"
expected = {MAIN: ["Figure_2", "Figure_3", "Figure_4", "Figure_5"], SI: ["Figure_S1", "Figure_S2", "Figure_S3"]}
errors=[]
for root, stems in expected.items():
    for stem in stems:
        for ext in ("pdf", "png"):
            p=root/f"{stem}.{ext}"
            if not p.exists() or p.stat().st_size == 0:
                errors.append(f"missing/empty: {p.relative_to(ROOT)}")
if errors:
    print("OUTPUT QA FAILED")
    for error in errors:
        print(" -", error)
    sys.exit(1)
print("OUTPUT QA PASS")
print("Regenerated files are scratch review outputs; publication PDFs under ../figures remain authoritative.")
