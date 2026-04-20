#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY311_BIN="${PY311_BIN:-/opt/homebrew/bin/python3.11}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv311}"

if [[ ! -x "$PY311_BIN" ]]; then
  echo "python3.11 not found at $PY311_BIN"
  echo "Install it first, for example: brew install python@3.11"
  exit 1
fi

"$PY311_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo "Python 3.11 environment is ready: $VENV_DIR"
