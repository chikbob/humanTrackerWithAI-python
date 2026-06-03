#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
FRONTEND_DIR="$PROJECT_DIR/frontend"

find_python311() {
  if [[ -x "$VENV_DIR/bin/python" ]] && "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
    printf '%s\n' "$VENV_DIR/bin/python"
    return 0
  fi

  local candidate
  for candidate in python3.11 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$1"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd npm

if [[ ! -d "$FRONTEND_DIR" ]]; then
  echo "Frontend directory not found: $FRONTEND_DIR" >&2
  exit 1
fi

PYTHON_BIN="$(find_python311 || true)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "Python 3.11 is required." >&2
  echo "Install it first, for example: brew install python@3.11" >&2
  exit 1
fi

log "Preparing Python virtual environment"
if [[ ! -x "$VENV_DIR/bin/python" ]] || ! "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

log "Upgrading pip"
python -m pip install --upgrade pip

log "Installing backend dependencies"
python -m pip install -r "$PROJECT_DIR/requirements.txt"
python -m pip install -r "$PROJECT_DIR/requirements-api.txt"

log "Installing frontend dependencies"
if [[ -f "$FRONTEND_DIR/package-lock.json" ]]; then
  npm ci --prefix "$FRONTEND_DIR"
else
  npm install --prefix "$FRONTEND_DIR"
fi

log "Building frontend"
npm run build --prefix "$FRONTEND_DIR"

cat <<EOF

Dev build complete.

Python venv: $VENV_DIR
Frontend build: $FRONTEND_DIR/dist

Run API:
  source "$VENV_DIR/bin/activate" && python "$PROJECT_DIR/run_api.py"

Run frontend dev server:
  npm run dev --prefix "$FRONTEND_DIR"

Optional Streamlit UI:
  source "$VENV_DIR/bin/activate" && streamlit run "$PROJECT_DIR/app.py"
EOF
