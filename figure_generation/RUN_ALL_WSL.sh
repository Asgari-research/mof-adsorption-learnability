#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python check_environment.py
python run_all.py
python verify_outputs.py
