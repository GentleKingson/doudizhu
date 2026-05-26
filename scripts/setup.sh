#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel cmake ninja
python -m pip install -e ".[dev]"
python -m pip install -v --no-build-isolation -e nanopokerlib
