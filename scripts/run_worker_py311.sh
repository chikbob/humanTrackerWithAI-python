#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv311}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Missing $VENV_DIR/bin/python"
  echo "Run scripts/setup_py311_env.sh first."
  exit 1
fi

exec "$VENV_DIR/bin/python" "$PROJECT_DIR/run_worker.py" "$@"
