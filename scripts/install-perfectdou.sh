#!/usr/bin/env bash
# Clones and bootstraps the upstream PerfectDou evaluator (Python 3.7).

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]:-$0}")"
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

log() {
  printf '[%s] %s\n' "$SCRIPT_NAME" "$*"
}

fatal() {
  printf '[%s] ERROR: %s\n' "$SCRIPT_NAME" "$*" >&2
  if [[ "${BASH_SOURCE[0]:-$0}" != "$0" ]]; then
    return 1
  fi
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

setup_python_command() {
  local target_env="$1"
  local python_bin="${PERFECTDOU_PYTHON:-python}"

  if [[ -n "$target_env" && "${CONDA_DEFAULT_ENV:-}" != "$target_env" ]]; then
    if command_exists conda; then
      local conda_base
      conda_base="$(conda info --base 2>/dev/null)" || conda_base=""
      if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1090
        source "$conda_base/etc/profile.d/conda.sh"
        if conda activate "$target_env" >/dev/null 2>&1; then
          log "Activated Conda environment '$target_env'."
        else
          log "Unable to activate Conda environment '$target_env'; continuing with current interpreter."
        fi
      else
        log "Conda profile script not found; continuing with current interpreter."
      fi
    else
      log "Conda command not available; continuing with current interpreter."
    fi
  elif [[ "${CONDA_DEFAULT_ENV:-}" == "$target_env" && -n "$target_env" ]]; then
    log "Detected active Conda environment '$target_env'."
  fi

  if ! command_exists "$python_bin"; then
    fatal "Python interpreter '$python_bin' not found. Set PERFECTDOU_PYTHON to override."
  fi

  PYTHON_BIN=("$python_bin")
}

run_python() {
  "${PYTHON_BIN[@]}" "$@"
}

clone_or_update_repo() {
  local url="$1"
  local dest="$2"
  local ref="$3"
  local depth="${PERFECTDOU_CLONE_DEPTH:-1}"

  if [[ ! -d "$dest/.git" ]]; then
    log "Cloning $url into $dest"
    mkdir -p "$(dirname "$dest")"
    if [[ -n "$depth" ]]; then
      if ! GIT_TERMINAL_PROMPT=0 git clone --depth "$depth" "$url" "$dest"; then
        log "Clone from $url failed; removing incomplete directory."
        rm -rf "$dest"
        return 1
      fi
    else
      if ! GIT_TERMINAL_PROMPT=0 git clone "$url" "$dest"; then
        log "Clone from $url failed; removing incomplete directory."
        rm -rf "$dest"
        return 1
      fi
    fi
  else
    log "Updating existing repository at $dest"
    git -C "$dest" fetch --tags --prune
  fi

  if [[ -n "$ref" ]]; then
    log "Checking out ref $ref"
    git -C "$dest" checkout "$ref"
  else
    git -C "$dest" pull --ff-only || log "Skipping fast-forward pull (local changes?)"
  fi
}

ensure_pip() {
  if ! run_python -m pip --version >/dev/null 2>&1; then
    log "Bootstrapping pip via ensurepip"
    run_python -m ensurepip --upgrade
  fi
}

ensure_python_37() {
  if [[ "${PERFECTDOU_REQUIRE_PY37:-1}" -eq 0 ]]; then
    return 0
  fi

  local pyver
  pyver="$(run_python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
  if [[ "$pyver" != "3.7" ]]; then
    fatal "PerfectDou requires Python 3.7 (detected $pyver). Set PERFECTDOU_PYTHON, or set PERFECTDOU_REQUIRE_PY37=0."
  fi
}

install_deps() {
  if [[ "${PERFECTDOU_NO_DEPS:-0}" -ne 0 ]]; then
    log "Skipping dependency installation (PERFECTDOU_NO_DEPS=1)."
    return 0
  fi

  ensure_pip

  log "Upgrading pip"
  run_python -m pip install --upgrade pip

  local repo_dir="$1"
  local minimal="${PERFECTDOU_MINIMAL:-0}"
  local -a pip_args=()

  if [[ -n "${PERFECTDOU_PIP_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206 # Intentional splitting for user-supplied args.
    pip_args+=(${PERFECTDOU_PIP_EXTRA_ARGS})
  fi

  if [[ "$minimal" -ne 0 ]]; then
    log "Installing minimal deps (onnxruntime + numpy) into current environment"
    run_python -m pip install --upgrade "${pip_args[@]}" numpy "onnxruntime==1.7.0"
  else
    log "Installing PerfectDou requirements into current environment"
    run_python -m pip install --upgrade "${pip_args[@]}" -r "$repo_dir/requirements.txt" numpy
  fi

  if [[ "${PERFECTDOU_INSTALL_RLCARD:-0}" -ne 0 ]]; then
    log "Installing optional dependency: rlcard"
    run_python -m pip install --upgrade "${pip_args[@]}" rlcard
  fi
}

generate_eval_data() {
  local repo_dir="$1"
  local generate="${PERFECTDOU_GENERATE_EVAL_DATA:-1}"
  if [[ "$generate" -eq 0 ]]; then
    return 0
  fi

  local force="${PERFECTDOU_FORCE_EVAL_DATA:-0}"

  local tiny_path="$repo_dir/eval_data_tiny.pkl"
  local small_path="$repo_dir/eval_data_small.pkl"

  if [[ "$force" -eq 0 && -f "$tiny_path" && -f "$small_path" ]]; then
    log "Eval data already present (eval_data_tiny.pkl, eval_data_small.pkl); skipping generation."
    return 0
  fi

  log "Generating eval data (protocol 4; run with Python 3.7)"
  pushd "$repo_dir" >/dev/null
  if [[ "$force" -ne 0 || ! -f "$tiny_path" ]]; then
    run_python generate_eval_data.py --output eval_data_tiny --num_games 10
  fi
  if [[ "$force" -ne 0 || ! -f "$small_path" ]]; then
    run_python generate_eval_data.py --output eval_data_small --num_games 100
  fi
  popd >/dev/null
}

verify_install() {
  local repo_dir="$1"

  log "Verifying checkout and imports"
  run_python - <<PY || fatal "Failed to import PerfectDou. Ensure external/PerfectDou is present and Python 3.7 deps installed."
import sys
from pathlib import Path

repo = Path(r"$repo_dir")
if not repo.exists():
    raise SystemExit(f"missing repo dir: {repo}")
sys.path.insert(0, str(repo))

import perfectdou  # noqa: F401
print("perfectdou import: ok")
PY

  if [[ "${PERFECTDOU_VERIFY_ENCODE:-1}" -ne 0 ]]; then
    run_python - <<PY || fatal "PerfectDou encode extension import failed; PerfectDou baseline requires Python 3.7 on x86_64 Linux."
import os
import sys
from pathlib import Path

repo = Path(r"$repo_dir").resolve()
os.chdir(repo)
sys.path.insert(0, str(repo))

from perfectdou.env.encode import encode_obs_landlord  # noqa: F401
print("perfectdou.env.encode import: ok")
PY
  fi

  if [[ "${PERFECTDOU_VERIFY_ONNXRUNTIME:-1}" -ne 0 && "${PERFECTDOU_NO_DEPS:-0}" -eq 0 ]]; then
    run_python - <<'PY' || fatal "onnxruntime import failed; verify requirements.txt installation."
import onnxruntime as ort
print(f"onnxruntime version: {getattr(ort, '__version__', 'unknown')}")
PY
  fi
}

main() {
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    emulate -L bash
  fi

  set -euo pipefail

  if [[ "${1:-}" =~ ^(-h|--help)$ ]]; then
    cat <<'EOF'
Usage:
  bash scripts/install-perfectdou.sh

This clones https://github.com/Netease-Games-AI-Lab-Guangzhou/PerfectDou into external/PerfectDou
and installs dependencies into the active Python environment.

Environment variables:
  PERFECTDOU_REPO_URL           Git URL for PerfectDou (default: https://github.com/Netease-Games-AI-Lab-Guangzhou/PerfectDou.git)
  PERFECTDOU_REPO_DIR           Target directory for the clone (default: <repo_root>/external/PerfectDou)
  PERFECTDOU_REPO_REF           Optional git ref (branch/tag/commit) to checkout.
  PERFECTDOU_CLONE_DEPTH        Depth for git clone (default: 1, set empty to clone full history).
  PERFECTDOU_CONDA_ENV          Conda environment name to activate (default: perfectdou37; set empty to skip).
  PERFECTDOU_PYTHON             Python interpreter to use (default: python).
  PERFECTDOU_REQUIRE_PY37       Set to 0 to allow non-3.7 interpreters (default: 1).
  PERFECTDOU_NO_DEPS            Set to 1 to skip pip installs (default: 0).
  PERFECTDOU_MINIMAL            Set to 1 to skip torch and only install onnxruntime + numpy (default: 0).
  PERFECTDOU_INSTALL_RLCARD     Set to 1 to install rlcard baseline dependency (default: 0).
  PERFECTDOU_PIP_EXTRA_ARGS     Additional arguments forwarded to pip.
  PERFECTDOU_GENERATE_EVAL_DATA Set to 0 to skip generating eval_data_{tiny,small}.pkl (default: 1).
  PERFECTDOU_FORCE_EVAL_DATA    Set to 1 to regenerate eval_data_{tiny,small}.pkl (default: 0).
  PERFECTDOU_VERIFY_ENCODE      Set to 0 to skip the encode extension import check (default: 1).
  PERFECTDOU_VERIFY_ONNXRUNTIME Set to 0 to skip the onnxruntime import check (default: 1).
EOF
    return 0
  fi

  if ! command_exists git; then
    fatal "git is required but missing. Please install git."
  fi

  local repo_url="${PERFECTDOU_REPO_URL:-https://github.com/Netease-Games-AI-Lab-Guangzhou/PerfectDou.git}"
  local repo_dir="${PERFECTDOU_REPO_DIR:-$REPO_ROOT/external/PerfectDou}"
  local repo_ref="${PERFECTDOU_REPO_REF:-}"
  local conda_env="${PERFECTDOU_CONDA_ENV:-perfectdou37}"

  setup_python_command "$conda_env"
  ensure_python_37

  clone_or_update_repo "$repo_url" "$repo_dir" "$repo_ref" \
    || fatal "Unable to clone PerfectDou repository. Set PERFECTDOU_REPO_URL to a reachable Git URL."

  install_deps "$repo_dir"
  generate_eval_data "$repo_dir"
  verify_install "$repo_dir"

  log "Installation complete."
  log "Repo: $repo_dir"
  log "Run:  (cd \"$repo_dir\" && python generate_eval_data.py --output eval_data_big --num_games 10000)"
  log "Run:  (cd \"$repo_dir\" && python evaluate.py --landlord perfectdou --landlord_up douzero --landlord_down douzero)"
}

main "$@"

exit_code=$?
if [[ "${BASH_SOURCE[0]:-$0}" != "$0" ]]; then
  return "$exit_code"
fi
exit "$exit_code"
