#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re, shutil, subprocess, sys
from PIL import Image

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "outputs" / "final"
HELD = ROOT / "outputs" / "held_s3"
EXPECTED_FINAL = [
    "Figure_1_redesign_hybrid_screening_evidence",
    "Figure_2_redesign_learning_curves",
    "Figure_3_redesign_descriptor_transfer",
    "Figure_4_redesign_empirical_calibration",
    "Figure_5_redesign_retrospective_candidates",
    "Figure_S1_redesign_target_distributions",
    "Figure_S2_redesign_quality_gate_audit",
]
EXPECTED_HELD = ["Figure_S3_redesign_external_domain_overlap_HOLD"]
EXPECTED_WIDTH_PT = 178 / 25.4 * 72.0
EXPECTED_WIDTH_PX = 178 / 25.4 * 200.0
errors=[]

def check_pair(root: Path, stem: str):
    pdf=root/f"{stem}.pdf"; png=root/f"{stem}.png"
    for p in [pdf,png]:
        if not p.exists() or p.stat().st_size == 0: errors.append(f"missing/empty: {p}")
    if png.exists():
        with Image.open(png) as im:
            if abs(im.width-EXPECTED_WIDTH_PX)>2: errors.append(f"wrong PNG width: {png.name}={im.width}px")
            dpi=im.info.get("dpi")
            if dpi and abs(float(dpi[0])-200)>0.5: errors.append(f"wrong PNG dpi: {png.name}={dpi}")
    if pdf.exists() and shutil.which("pdfinfo"):
        txt=subprocess.run(["pdfinfo",str(pdf)],text=True,capture_output=True,check=True).stdout
        m=re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts",txt)
        if not m: errors.append(f"cannot parse PDF size: {pdf.name}")
        elif abs(float(m.group(1))-EXPECTED_WIDTH_PT)>0.75: errors.append(f"wrong PDF width: {pdf.name}={m.group(1)} pt")
    if pdf.exists() and shutil.which("pdffonts"):
        txt=subprocess.run(["pdffonts",str(pdf)],text=True,capture_output=True,check=True).stdout
        if "Arial" not in txt: errors.append(f"Arial not detected: {pdf.name}")
        if "DejaVu" in txt: errors.append(f"DejaVu fallback detected: {pdf.name}")

for s in EXPECTED_FINAL: check_pair(FINAL,s)
for s in EXPECTED_HELD: check_pair(HELD,s)

if errors:
    print("OUTPUT QA FAILED")
    for e in errors: print(" -",e)
    sys.exit(1)
print("OUTPUT QA PASS")
print(" - 7 final figures + 1 held S3 figure have PDF and 200-dpi PNG exports")
print(" - all figure canvases are 178 mm wide")
if shutil.which("pdffonts"): print(" - Arial detected and DejaVu absent in every final/held PDF")
print(" - S3 remains in outputs/held_s3 and is not certified for submission")
