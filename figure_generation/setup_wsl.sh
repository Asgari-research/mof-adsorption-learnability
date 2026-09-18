#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if command -v python >/dev/null 2>&1; then PYTHON_BIN="${FEWSHOT_PYTHON:-python}"; else PYTHON_BIN="${FEWSHOT_PYTHON:-python3}"; fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: Python not found: $PYTHON_BIN"; exit 1; }

echo "[1/4] Using the Python already active in this shell"
echo "  Python: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" --version
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then echo "  Active Conda env: $CONDA_DEFAULT_ENV (reused; not created or changed)"; fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then echo "  Active virtualenv: $VIRTUAL_ENV (reused; not created or changed)"; fi

echo "[2/4] Checking plotting dependencies (no environment will be created)"
if ! "$PYTHON_BIN" - <<'PY'
mods = ["numpy", "pandas", "matplotlib", "PIL"]
missing=[]
for m in mods:
    try: __import__(m)
    except Exception: missing.append(m)
if missing:
    print("Missing:", ", ".join(missing))
    raise SystemExit(1)
print("Required plotting imports: PASS")
PY
then
  echo
  echo "Dependencies are missing from your CURRENT Python environment."
  echo "This script will not create another environment."
  echo "If you want to install into the current environment, run:"
  echo "  $PYTHON_BIN -m pip install -r requirements.txt"
  exit 2
fi

echo "[3/4] Making Windows Arial visible to WSL2 (font files are not distributed)"
FONT_DIR="$HOME/.local/share/fonts/fewshot_windows_arial"
mkdir -p "$FONT_DIR"
found=0
for f in arial.ttf arialbd.ttf ariali.ttf arialbi.ttf; do
  src="/mnt/c/Windows/Fonts/$f"
  if [[ -f "$src" ]]; then ln -sfn "$src" "$FONT_DIR/$f"; found=$((found + 1)); fi
done
if [[ "$found" -lt 4 ]]; then
  echo "ERROR: Expected Arial files under /mnt/c/Windows/Fonts/. Found $found/4."
  exit 3
fi
if command -v fc-cache >/dev/null 2>&1; then fc-cache -f "$FONT_DIR" >/dev/null 2>&1 || true; fi
rm -f "$HOME"/.cache/matplotlib/fontlist-v*.json 2>/dev/null || true

echo "[4/4] Final environment check"
"$PYTHON_BIN" check_environment.py

echo
echo "Setup/check complete. No Conda environment or venv was created."
echo "Next: bash RUN_ALL_WSL.sh"
