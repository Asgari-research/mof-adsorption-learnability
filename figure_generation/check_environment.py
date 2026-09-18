#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from figure_style import configure_matplotlib, resolve_arial, assert_arial_family

font = resolve_arial(strict=True)
configure_matplotlib(strict_font=True)
styles = assert_arial_family()

import matplotlib
import numpy
import pandas

print("FewShot figure environment")
print("  Python      :", sys.version.split()[0])
print("  matplotlib  :", matplotlib.__version__)
print("  numpy       :", numpy.__version__)
print("  pandas      :", pandas.__version__)
print("  Arial file  :", font)
for k, v in styles.items():
    print(f"  Arial {k:11s}: {v}")
print("  Status      : PASS - actual Arial located")
