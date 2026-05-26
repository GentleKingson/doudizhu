#!/usr/bin/env bash
# Creates a dedicated Conda env for PerfectDou and runs install-perfectdou.sh.

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

setup_conda_shell() {
  local conda_base
  conda_base="$(conda info --base 2>/dev/null)" || conda_base=""
  if [[ -z "$conda_base" || ! -f "$conda_base/etc/profile.d/conda.sh" ]]; then
    fatal "Unable to locate conda profile script from 'conda info --base'."
  fi
  # shellcheck disable=SC1090
  source "$conda_base/etc/profile.d/conda.sh"
}

conda_env_exists() {
  local env_name="$1"
  conda env list 2>/dev/null | awk 'NF > 0 && $1 !~ /^#/ {print $1}' | grep -Fxq "$env_name"
}

ensure_env_python_version() {
  local env_name="$1"
  local expected="$2"
  local have
  have="$(conda run -n "$env_name" python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
  if [[ "$have" != "$expected" ]]; then
    fatal "Conda env '$env_name' uses Python $have (expected $expected). Recreate the env or use --name."
  fi
}

create_or_reuse_env() {
  local env_name="$1"
  local py_version="$2"
  local recreate="$3"

  if [[ "$recreate" -ne 0 ]] && conda_env_exists "$env_name"; then
    log "Removing existing Conda env '$env_name' (--recreate)."
    conda remove -n "$env_name" --all -y
  fi

  if conda_env_exists "$env_name"; then
    log "Reusing existing Conda env '$env_name'."
  else
    log "Creating Conda env '$env_name' with Python $py_version."
    conda create -n "$env_name" "python=$py_version" -y
  fi

  ensure_env_python_version "$env_name" "$py_version"
}

run_installer() {
  local env_name="$1"
  local installer="$SCRIPT_DIR/install-perfectdou.sh"
  if [[ ! -f "$installer" ]]; then
    fatal "Missing installer script: $installer"
  fi

  log "Running install-perfectdou.sh in Conda env '$env_name'."
  PERFECTDOU_CONDA_ENV="$env_name" bash "$installer"
}

main() {
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    emulate -L bash
  fi

  set -euo pipefail

  local env_name="${SETUP_PERFECTDOU_ENV:-perfectdou37}"
  local py_version="${SETUP_PERFECTDOU_PYTHON_VERSION:-3.7}"
  local skip_install=0
  local recreate=0

  if [[ "${1:-}" =~ ^(-h|--help)$ ]]; then
    cat <<'EOF'
Usage:
  bash scripts/setup-perfectdou.sh [--name ENV] [--python-version VER] [--recreate] [--skip-install]

Creates (or reuses) a dedicated Conda env for PerfectDou, verifies Python version,
then runs scripts/install-perfectdou.sh with PERFECTDOU_CONDA_ENV=<ENV>.

Flags:
  --name             Conda env name (default: perfectdou37).
  --python-version   Python major.minor version for the env (default: 3.7).
  --recreate         Remove and recreate the Conda env if it already exists.
  --skip-install     Only create/verify the Conda env, skip install-perfectdou.sh.

Environment overrides:
  SETUP_PERFECTDOU_ENV              Default for --name.
  SETUP_PERFECTDOU_PYTHON_VERSION   Default for --python-version.
  PERFECTDOU_*                      Forwarded to scripts/install-perfectdou.sh.
EOF
    return 0
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --name)
        env_name="${2:-}"
        shift 2
        ;;
      --python-version)
        py_version="${2:-}"
        shift 2
        ;;
      --recreate)
        recreate=1
        shift
        ;;
      --skip-install)
        skip_install=1
        shift
        ;;
      *)
        fatal "Unknown argument '$1'. Run with --help for usage."
        ;;
    esac
  done

  if [[ -z "$env_name" ]]; then
    fatal "--name must be non-empty."
  fi
  if [[ -z "$py_version" ]]; then
    fatal "--python-version must be non-empty."
  fi
  if ! command_exists conda; then
    fatal "conda not found. Activate conda/miniforge and retry."
  fi

  setup_conda_shell
  create_or_reuse_env "$env_name" "$py_version" "$recreate"

  if [[ "$skip_install" -eq 0 ]]; then
    run_installer "$env_name"
  else
    log "Skipped install-perfectdou.sh (--skip-install)."
  fi

  log "Done. Conda env '$env_name' is ready."
  log "Example: PY37_CONDA_ENV=$env_name python -u -m evaluations.run doudizhu perfectdou douzero --num-games 200 --batch-size 128"
}

main "$@"

exit_code=$?
if [[ "${BASH_SOURCE[0]:-$0}" != "$0" ]]; then
  return "$exit_code"
fi
exit "$exit_code"
