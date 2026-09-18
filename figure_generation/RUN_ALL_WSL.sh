#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if command -v python >/dev/null 2>&1; then PYTHON_BIN="${FEWSHOT_PYTHON:-python}"; else PYTHON_BIN="${FEWSHOT_PYTHON:-python3}"; fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }
echo "Using current Python: $(command -v "$PYTHON_BIN")"
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then echo "Active Conda env: $CONDA_DEFAULT_ENV (reused)"; fi
rm -rf outputs/final outputs/held_s3
mkdir -p outputs/final outputs/held_s3
"$PYTHON_BIN" check_environment.py
"$PYTHON_BIN" run_all.py --include-held-s3
"$PYTHON_BIN" verify_outputs.py

echo
echo "Finished successfully."
echo "Regenerated review outputs: $HERE/outputs/final"
echo "S3 redraw (SCIENTIFIC HOLD): $HERE/outputs/held_s3"
echo "Review contact sheet: $HERE/outputs/CONTACT_SHEET_ALL_WITH_HELD_S3_200dpi.png"
