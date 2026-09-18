#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from figure_style import configure_matplotlib
from fewshot_figures import FIGURE_FUNCTIONS

p = argparse.ArgumentParser()
p.add_argument("figure", choices=list(FIGURE_FUNCTIONS))
p.add_argument("--source-root", type=Path, default=ROOT / "data")
p.add_argument("--output-root", type=Path, default=ROOT / "outputs")
p.add_argument("--dpi", type=int, default=200)
p.add_argument("--allow-font-fallback-for-preview", action="store_true")
a = p.parse_args()
configure_matplotlib(strict_font=not a.allow_font_fallback_for_preview)
out = a.output_root / ("held_s3" if a.figure == "S3" else "final")
FIGURE_FUNCTIONS[a.figure](a.source_root, out, a.dpi)
print(out)
