#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from figure_style import configure_matplotlib
from fewshot_figures import FIGURE_FUNCTIONS

FINAL_DEFAULT = ["1", "2", "3", "4", "5", "S1", "S2"]
HELD_DEFAULT = ["S3"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files(source_root: Path):
    return sorted(p for p in source_root.rglob("*") if p.is_file())


def write_manifest(paths, root: Path, out: Path):
    rows = ["relative_path,bytes,sha256"]
    for p in paths:
        rows.append(f'"{p.relative_to(root).as_posix()}",{p.stat().st_size},{sha256(p)}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def contact_sheet(pngs: list[Path], out: Path, columns: int = 2):
    if not pngs:
        return
    cells = []
    cell_w = 900
    for p in pngs:
        im = Image.open(p).convert("RGB")
        scale = cell_w / im.width
        im = im.resize((cell_w, int(im.height * scale)))
        cell = Image.new("RGB", (cell_w + 30, im.height + 65), "white")
        cell.paste(im, (15, 15))
        ImageDraw.Draw(cell).text((15, im.height + 35), p.stem, fill="black")
        cells.append(cell)
    row_h = max(c.height for c in cells)
    sheet = Image.new("RGB", (columns * (cell_w + 30), math.ceil(len(cells)/columns) * row_h), "white")
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i % columns)*(cell_w+30), (i // columns)*row_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, dpi=(200, 200))


def main():
    parser = argparse.ArgumentParser(description="Regenerate redesigned FewShot figures from supplied saved values only.")
    parser.add_argument("--source-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--figures", nargs="*", default=FINAL_DEFAULT)
    parser.add_argument("--include-held-s3", action="store_true", help="Also regenerate S3 into outputs/held_s3 (scientific HOLD).")
    parser.add_argument("--allow-font-fallback-for-preview", action="store_true", help="Testing only; final WSL exports must use actual Arial.")
    args = parser.parse_args()

    configure_matplotlib(strict_font=not args.allow_font_fallback_for_preview)
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    final_root = output_root / "final"
    held_root = output_root / "held_s3"
    final_root.mkdir(parents=True, exist_ok=True)
    held_root.mkdir(parents=True, exist_ok=True)

    before = {p.relative_to(source_root).as_posix(): sha256(p) for p in source_files(source_root)}
    write_manifest(source_files(source_root), source_root, output_root / "SOURCE_SHA256_BEFORE.csv")

    final_generated: list[Path] = []
    held_generated: list[Path] = []
    for key in args.figures:
        if key == "S3":
            raise SystemExit("S3 is held. Use --include-held-s3 instead of placing S3 in --figures.")
        if key not in FIGURE_FUNCTIONS:
            raise SystemExit(f"Unknown figure {key}. Valid: {', '.join(FIGURE_FUNCTIONS)}")
        print(f"Generating final Figure {key}...")
        final_generated.extend(FIGURE_FUNCTIONS[key](source_root, final_root, args.dpi))

    if args.include_held_s3:
        print("Generating Figure S3 into held_s3 (scientific HOLD)...")
        held_generated.extend(FIGURE_FUNCTIONS["S3"](source_root, held_root, args.dpi))

    after = {p.relative_to(source_root).as_posix(): sha256(p) for p in source_files(source_root)}
    write_manifest(source_files(source_root), source_root, output_root / "SOURCE_SHA256_AFTER.csv")
    if before != after:
        raise RuntimeError("Source-data hashes changed during plotting. Plotting must be read-only.")

    final_pngs = [p for p in final_generated if p.suffix.lower() == ".png"]
    held_pngs = [p for p in held_generated if p.suffix.lower() == ".png"]
    contact_sheet(final_pngs, output_root / "CONTACT_SHEET_FINAL_200dpi.png")
    if held_pngs:
        contact_sheet(final_pngs + held_pngs, output_root / "CONTACT_SHEET_ALL_WITH_HELD_S3_200dpi.png")

    out_files = sorted(p for p in output_root.rglob("*") if p.is_file() and p.name != "OUTPUT_SHA256.csv")
    write_manifest(out_files, output_root, output_root / "OUTPUT_SHA256.csv")

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "dpi": args.dpi,
        "final_figures": args.figures,
        "held_s3_generated": bool(args.include_held_s3),
        "source_hashes_unchanged": True,
        "figure1_basis": "conceptual workflow plus audited benchmark information roles",
        "s3_status": "HOLD: exact stored v4.9 values redrawn, but upstream external-geometry mapping provenance remains unresolved",
    }
    (output_root / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("PASS: source hashes unchanged.")
    print(f"Final figures: {final_root}")
    if args.include_held_s3:
        print(f"Held S3: {held_root}")


if __name__ == "__main__":
    main()
