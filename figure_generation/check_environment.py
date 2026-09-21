#!/usr/bin/env python3
"""Check plotting dependencies and Arial availability."""
from __future__ import annotations
import sys

for name in ("numpy", "pandas", "matplotlib", "PIL"):
    try:
        __import__(name)
    except Exception as exc:
        raise SystemExit(f"Missing dependency {name}: {exc}")

from matplotlib import font_manager
try:
    arial = font_manager.findfont("Arial", fallback_to_default=False)
except Exception:
    raise SystemExit("Arial is not visible to Matplotlib. Final regeneration requires Arial.")

import matplotlib, numpy, pandas
print("Python     :", sys.version.split()[0])
print("matplotlib :", matplotlib.__version__)
print("numpy      :", numpy.__version__)
print("pandas     :", pandas.__version__)
print("Arial      :", arial)
print("Status     : PASS")
