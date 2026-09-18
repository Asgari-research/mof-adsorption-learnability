#!/usr/bin/env python3
"""Read-only integrity/static check for the repository.

This script does not import the historical modelling module, fit models,
regenerate figures, or modify repository files.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []
warnings: list[str] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(path: Path) -> None:
    if not path.exists():
        errors.append(f"missing required path: {path.relative_to(ROOT)}")


required = [
    ROOT / "README.md",
    ROOT / "LICENSE",
    ROOT / "AUTHORS.md",
    ROOT / "docs" / "WORKFLOW_AND_REPRODUCIBILITY.md",
    ROOT / "docs" / "modeling" / "SOURCE_MANIFEST.csv",
    ROOT / "docs" / "modeling" / "STATIC_ARCHITECTURE.md",
    ROOT / "docs" / "modeling" / "EXECUTION_SEMANTICS.md",
    ROOT / "docs" / "figures" / "FINAL_FIGURE_MANIFEST.csv",
    ROOT / "docs" / "figures" / "FINAL_FIGURE_SOURCE_MANIFEST.csv",
    ROOT / "modeling_pipeline" / "README.md",
    ROOT / "figure_generation" / "README.md",
]
for p in required:
    require(p)

# Historical modelling source hashes.
manifest = ROOT / "docs" / "modeling" / "SOURCE_MANIFEST.csv"
if manifest.exists():
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = ROOT / row["path"]
            if not path.exists():
                errors.append(f"source-manifest path missing: {row['path']}")
                continue
            got = sha256(path)
            if got != row["sha256"]:
                errors.append(f"source hash mismatch: {row['path']}")

# Final figure hashes.
fig_manifest = ROOT / "docs" / "figures" / "FINAL_FIGURE_MANIFEST.csv"
if fig_manifest.exists():
    with fig_manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["file"]
            candidates = [ROOT / "figures" / "main" / name, ROOT / "figures" / "si" / name]
            existing = [p for p in candidates if p.exists()]
            if len(existing) != 1:
                errors.append(f"figure path ambiguity/missing for {name}")
                continue
            if sha256(existing[0]) != row["sha256"]:
                errors.append(f"final figure hash mismatch: {name}")

# Figure source hashes.
src_manifest = ROOT / "docs" / "figures" / "FINAL_FIGURE_SOURCE_MANIFEST.csv"
if src_manifest.exists():
    with src_manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = ROOT / "figure_generation" / "data" / row["relative_path"]
            if not path.exists():
                errors.append(f"figure source missing: {row['relative_path']}")
                continue
            if sha256(path) != row["sha256"]:
                errors.append(f"figure source hash mismatch: {row['relative_path']}")

# Parse Python without importing it.
py_files = []
for rel in ["modeling_pipeline", "figure_generation", "tools"]:
    d = ROOT / rel
    if d.exists():
        py_files.extend(d.rglob("*.py"))
for p in sorted(py_files):
    try:
        ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
    except SyntaxError as e:
        errors.append(f"Python syntax error in {p.relative_to(ROOT)}: {e}")

# Basic release hygiene.
import re
conflict_re = re.compile(r"^(?:<<<<<<< .+|=======|>>>>>>> .+)$", re.MULTILINE)
private_key_markers = (
    "-----BEGIN " + "PRIVATE KEY-----",
    "-----BEGIN " + "RSA PRIVATE KEY-----",
    "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
)
for p in ROOT.rglob("*"):
    if not p.is_file() or ".git" in p.parts:
        continue
    try:
        if p.stat().st_size > 25 * 1024 * 1024:
            warnings.append(f"large file >25 MiB: {p.relative_to(ROOT)}")
    except OSError:
        continue
    if p.suffix.lower() in {".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".toml", ".sh"}:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if conflict_re.search(txt):
            errors.append(f"merge-conflict marker found: {p.relative_to(ROOT)}")
        if any(m in txt for m in private_key_markers):
            errors.append(f"private-key marker found: {p.relative_to(ROOT)}")

# Report (not reject) known non-finite values in the included saved QC table.
qc = ROOT / "figure_generation" / "data" / "tables" / "si" / "Table_S3_model_stability_quality_gate.csv"
if qc.exists():
    by_model = defaultdict(lambda: {"rows": 0, "rmse_inf": 0, "rmse_nan": 0,
                                    "spearman_inf": 0, "spearman_nan": 0})
    with qc.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row.get("model", "")
            d = by_model[model]
            d["rows"] += 1
            for col, key_inf, key_nan in [
                ("rmse_top5row", "rmse_inf", "rmse_nan"),
                ("spearman_top5row", "spearman_inf", "spearman_nan"),
            ]:
                try:
                    x = float(row.get(col, "nan"))
                except Exception:
                    x = math.nan
                if math.isinf(x):
                    d[key_inf] += 1
                elif math.isnan(x):
                    d[key_nan] += 1
    print("Saved Table S3 non-finite diagnostics (reported, not treated as verifier failure):")
    for model in sorted(by_model):
        d = by_model[model]
        if d["rmse_inf"] or d["rmse_nan"] or d["spearman_inf"] or d["spearman_nan"]:
            print(
                f"  {model}: n={d['rows']} "
                f"rmse_inf={d['rmse_inf']} rmse_nan={d['rmse_nan']} "
                f"spearman_inf={d['spearman_inf']} spearman_nan={d['spearman_nan']}"
            )

if warnings:
    print("\nWARNINGS:")
    for x in warnings:
        print(" -", x)

if errors:
    print("\nVERIFY FAIL:")
    for x in errors:
        print(" -", x)
    sys.exit(1)

print("\nREPOSITORY VERIFY PASS")
print(f"Parsed Python files: {len(py_files)}")
print("No models were fit and no figures were regenerated.")
