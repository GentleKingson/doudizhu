#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ $# -ne 1 ]]; then
  echo "Usage: bash evaluations/doudizhu/agents/perfectdou.sh <socket_path>" >&2
  exit 2
fi

SOCKET_PATH="$1"

PYTHON37_BIN="${PYTHON37:-${PY37:-}}"

PYTHON_CMD=()
if [[ -n "$PYTHON37_BIN" ]]; then
  if [[ "$PYTHON37_BIN" == *" "* ]]; then
    echo "PYTHON37 must be a path to a Python 3.7 executable (no spaces)." >&2
    exit 2
  fi
  PYTHON_CMD=("$PYTHON37_BIN")
elif command -v python3.7 >/dev/null 2>&1; then
  PYTHON_CMD=("python3.7")
else
  echo "PerfectDou requires Python 3.7." >&2
  echo "Set PYTHON37=/path/to/python3.7 (or PY37=...), or install python3.7 on PATH." >&2
  exit 2
fi

cd "$REPO_ROOT"
"${PYTHON_CMD[@]}" -u -m evaluations.doudizhu.agents.perfectdou "$SOCKET_PATH"
