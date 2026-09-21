#!/usr/bin/env python3
"""Regenerate one publication figure from saved source tables."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RENDERER = ROOT / "code" / "final_publication_figures.py"

parser = argparse.ArgumentParser()
parser.add_argument("figure", choices=["2", "3", "4", "5", "S1", "S2", "S3"])
parser.add_argument("--dpi", type=int, default=200)
parser.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "svg"])
parser.add_argument("--allow-font-fallback-for-preview", action="store_true")
args = parser.parse_args()

cmd = [sys.executable, "-B", str(RENDERER), "--figures", args.figure, "--main-grid", "mixed", "--dpi", str(args.dpi), "--formats", *args.formats]
if args.allow_font_fallback_for_preview:
    cmd.append("--allow-font-fallback-for-preview")
subprocess.run(cmd, cwd=ROOT, check=True)
