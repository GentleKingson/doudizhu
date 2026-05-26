#!/usr/bin/env bash
# Downloads and bootstraps the FPDou baseline bundle (Python 3.7).

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
  local python_bin="${FPDOU_PYTHON:-python}"

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
    fatal "Python interpreter '$python_bin' not found. Set FPDOU_PYTHON to override."
  fi

  PYTHON_BIN=("$python_bin")
}

run_python() {
  "${PYTHON_BIN[@]}" "$@"
}

ensure_pip() {
  if ! run_python -m pip --version >/dev/null 2>&1; then
    log "Bootstrapping pip via ensurepip"
    run_python -m ensurepip --upgrade
  fi
}

ensure_python_37() {
  if [[ "${FPDOU_REQUIRE_PY37:-1}" -eq 0 ]]; then
    return 0
  fi

  local pyver
  pyver="$(run_python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
  if [[ "$pyver" != "3.7" ]]; then
    fatal "FPDou requires Python 3.7 (detected $pyver). Set FPDOU_PYTHON, or set FPDOU_REQUIRE_PY37=0."
  fi
}

download_google_drive_zip() (
  set -euo pipefail

  local file_id="$1"
  local out_path="$2"

  if ! command_exists curl; then
    fatal "curl is required but missing. Please install curl."
  fi

  local tmp_cookie tmp_html
  tmp_cookie="$(mktemp)"
  tmp_html="$(mktemp)"
  trap 'rm -f "$tmp_cookie" "$tmp_html"' EXIT

  local initial_url="https://drive.google.com/uc?export=download&id=${file_id}"

  log "Fetching Google Drive download page..."
  curl -L --fail --retry 3 --retry-delay 1 -c "$tmp_cookie" -o "$tmp_html" "$initial_url"

  local confirm uuid filename
  confirm="$(sed -nE 's/.*name="confirm" value="([^"]+)".*/\1/p' "$tmp_html" | head -n 1 || true)"
  uuid="$(sed -nE 's/.*name="uuid" value="([^"]+)".*/\1/p' "$tmp_html" | head -n 1 || true)"
  filename="$(sed -nE 's/.*<a href="\\/open\\?id=[^"]+">([^<]+)<\\/a>.*/\1/p' "$tmp_html" | head -n 1 || true)"

  if [[ -n "$confirm" && -n "$uuid" ]]; then
    local dl_url="https://drive.usercontent.google.com/download?export=download&id=${file_id}&confirm=${confirm}&uuid=${uuid}"
    log "Downloading ${filename:-fpdou.zip}..."
    curl -L --fail --retry 3 --retry-delay 2 -b "$tmp_cookie" -o "$out_path" "$dl_url"
    exit 0
  fi

  log "Confirm token not found; attempting direct download..."
  curl -L --fail --retry 3 --retry-delay 2 -o "$out_path" "$initial_url"
)

verify_zip() {
  local zip_path="$1"
  if ! command_exists unzip; then
    fatal "unzip is required but missing. Please install unzip."
  fi
  unzip -t "$zip_path" >/dev/null
}

extract_fpdou_zip() (
  set -euo pipefail

  local zip_path="$1"
  local target_dir="$2"

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT

  unzip -q "$zip_path" -d "$tmp_dir"
  rm -rf "$tmp_dir/__MACOSX" "$tmp_dir/fpdou/__MACOSX"
  find "$tmp_dir" -name '.DS_Store' -delete 2>/dev/null || true

  if [[ ! -d "$tmp_dir/fpdou" ]]; then
    fatal "Unexpected zip layout: missing 'fpdou/' directory at archive root."
  fi

  mkdir -p "$(dirname "$target_dir")"
  mv "$tmp_dir/fpdou" "$target_dir"
)

install_deps() {
  local repo_dir="$1"
  if [[ "${FPDOU_NO_DEPS:-0}" -ne 0 ]]; then
    log "Skipping dependency installation (FPDOU_NO_DEPS=1)."
    return 0
  fi

  ensure_pip

  log "Upgrading pip"
  run_python -m pip install --upgrade pip

  local minimal="${FPDOU_MINIMAL:-0}"
  local -a pip_args=()

  if [[ -n "${FPDOU_PIP_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206 # Intentional splitting for user-supplied args.
    pip_args+=(${FPDOU_PIP_EXTRA_ARGS})
  fi

  if [[ "$minimal" -ne 0 ]]; then
    log "Installing minimal deps (onnxruntime + numpy) into current environment"
    run_python -m pip install --upgrade "${pip_args[@]}" numpy "onnxruntime==1.7.0"
  else
    log "Installing FPDou requirements into current environment"
    run_python -m pip install --upgrade "${pip_args[@]}" -r "$repo_dir/requirements.txt"
  fi
}

verify_install() {
  local repo_dir="$1"

  log "Verifying checkout and imports"
  run_python - <<PY || fatal "Failed to import FPDou. Ensure it was extracted and Python 3.7 deps installed."
import os
import sys
from pathlib import Path

repo = Path(r"$repo_dir").resolve()
if not repo.exists():
    raise SystemExit(f"missing repo dir: {repo}")

os.chdir(repo)
sys.path.insert(0, str(repo))

import fpdou  # noqa: F401
print("fpdou import: ok")
PY

  if [[ "${FPDOU_VERIFY_ENCODE:-1}" -ne 0 ]]; then
    run_python - <<PY || fatal "FPDou encode extension import failed; FPDou baseline requires Python 3.7 on x86_64 Linux."
import os
import sys
from pathlib import Path

repo = Path(r"$repo_dir").resolve()
os.chdir(repo)
sys.path.insert(0, str(repo))

from fpdou.env.encode import encode_obs_landlord  # noqa: F401
print("fpdou.env.encode import: ok")
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
  bash scripts/install-fpdou.sh

This downloads the FPDou baseline bundle from Google Drive, extracts it under:
  external/fpdou

Environment variables:
  FPDOU_FILE_ID            Google Drive file id (default: 12NyPl7aOivItoPMnNei2A-F0KMHVsa-U)
  FPDOU_ZIP_PATH           Where to cache the downloaded zip (default: <repo_root>/external/fpdou.zip)
  FPDOU_DIR                Where to extract FPDou (default: <repo_root>/external/fpdou)
  FPDOU_CONDA_ENV          Conda env name to activate (default: perfectdou37; set empty to skip).
  FPDOU_PYTHON             Python interpreter to use (default: python).
  FPDOU_REQUIRE_PY37        Set to 0 to allow non-3.7 interpreters (default: 1).
  FPDOU_FORCE              Set to 1 to re-download and re-extract (default: 0).
  FPDOU_NO_DEPS            Set to 1 to skip pip installs (default: 0).
  FPDOU_MINIMAL            Set to 1 to skip torch and only install onnxruntime + numpy (default: 0).
  FPDOU_PIP_EXTRA_ARGS     Additional arguments forwarded to pip.
  FPDOU_VERIFY_ENCODE      Set to 0 to skip the encode extension import check (default: 1).
EOF
    return 0
  fi

  local file_id="${FPDOU_FILE_ID:-12NyPl7aOivItoPMnNei2A-F0KMHVsa-U}"
  local zip_path="${FPDOU_ZIP_PATH:-$REPO_ROOT/external/fpdou.zip}"
  local target_dir="${FPDOU_DIR:-$REPO_ROOT/external/fpdou}"
  local conda_env="${FPDOU_CONDA_ENV:-perfectdou37}"
  local force="${FPDOU_FORCE:-0}"

  setup_python_command "$conda_env"
  ensure_python_37

  mkdir -p "$(dirname "$zip_path")"
  if [[ -f "$zip_path" && "$force" -eq 0 ]]; then
    log "Zip already present at $zip_path; skipping download."
  else
    log "Downloading fpdou.zip to $zip_path"
    download_google_drive_zip "$file_id" "$zip_path"
  fi

  verify_zip "$zip_path"

  if [[ -d "$target_dir" ]]; then
    if [[ "$force" -ne 0 ]]; then
      log "Removing existing $target_dir (FPDOU_FORCE=1)."
      rm -rf "$target_dir"
    else
      log "FPDou already extracted at $target_dir; skipping unzip."
    fi
  fi

  if [[ ! -d "$target_dir" ]]; then
    log "Extracting $zip_path into $target_dir"
    extract_fpdou_zip "$zip_path" "$target_dir"
  fi

  install_deps "$target_dir"
  verify_install "$target_dir"

  log "Installation complete."
  log "FPDou: $target_dir"
  log "Help:  (cd \"$REPO_ROOT\" && python -u -m evaluations.run doudizhu --help)"
  log "Run:   (cd \"$REPO_ROOT\" && PYTHON37=/path/to/python3.7 python -u -m evaluations.run doudizhu perfectdou fpdou --num-games 1000 --batch-size 256 --workers-per-gpu 2)"
}

main "$@"

exit_code=$?
if [[ "${BASH_SOURCE[0]:-$0}" != "$0" ]]; then
  return "$exit_code"
fi
exit "$exit_code"
