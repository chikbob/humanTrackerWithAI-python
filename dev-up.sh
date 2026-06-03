#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
FRONTEND_DIR="$PROJECT_DIR/frontend"
LOG_DIR="$PROJECT_DIR/.dev-logs"
API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
API_LOG="$LOG_DIR/api.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
API_PID=""
FRONTEND_PID=""
TAIL_PID=""
SEED_DEMO=0
DEMO_EMPLOYEES="${DEMO_EMPLOYEES:-120}"
DEMO_VISITS="${DEMO_VISITS:-900}"
DEMO_SEED="${DEMO_SEED:-42}"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$1"
}

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

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

kill_port() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
}

wait_for_url() {
  local url="$1"
  local label="$2"
  local attempt
  for attempt in $(seq 1 60); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "$label did not become ready: $url" >&2
  return 1
}

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$TAIL_PID" ]] && kill -0 "$TAIL_PID" >/dev/null 2>&1; then
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
  fi
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    kill "$FRONTEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" >/dev/null 2>&1; then
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed-demo)
      SEED_DEMO=1
      shift
      ;;
    --help|-h)
      cat <<EOF
Usage: ./dev-up.sh [--seed-demo]

Options:
  --seed-demo   reset monitoring.db and fill it with demo data before startup

Environment:
  DEMO_EMPLOYEES  default: 120
  DEMO_VISITS     default: 900
  DEMO_SEED       default: 42
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

require_cmd npm
require_cmd curl
require_cmd lsof

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

mkdir -p "$LOG_DIR"

log "Preparing Python 3.11 environment"
if [[ ! -x "$VENV_DIR/bin/python" ]] || ! "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

log "Installing backend dependencies"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r "$PROJECT_DIR/requirements.txt" >/dev/null
python -m pip install -r "$PROJECT_DIR/requirements-api.txt" >/dev/null

if [[ "$SEED_DEMO" == "1" ]]; then
  log "Resetting and seeding demo database"
  python "$PROJECT_DIR/scripts/reset_seed_database.py" \
    --employees "$DEMO_EMPLOYEES" \
    --visits "$DEMO_VISITS" \
    --seed "$DEMO_SEED"
fi

log "Installing frontend dependencies"
if [[ -f "$FRONTEND_DIR/package-lock.json" ]]; then
  npm ci --prefix "$FRONTEND_DIR" >/dev/null
else
  npm install --prefix "$FRONTEND_DIR" >/dev/null
fi

log "Clearing old dev processes on ports $API_PORT and $FRONTEND_PORT"
kill_port "$API_PORT"
kill_port "$FRONTEND_PORT"

: > "$API_LOG"
: > "$FRONTEND_LOG"

log "Starting API on http://127.0.0.1:$API_PORT"
(
  cd "$PROJECT_DIR"
  source "$VENV_DIR/bin/activate"
  exec python run_api.py
) >>"$API_LOG" 2>&1 &
API_PID=$!

wait_for_url "http://127.0.0.1:$API_PORT/health/live" "API"

log "Starting frontend on http://127.0.0.1:$FRONTEND_PORT"
(
  cd "$PROJECT_DIR"
  exec npm run dev --prefix frontend -- --host 0.0.0.0 --port "$FRONTEND_PORT"
) >>"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

wait_for_url "http://127.0.0.1:$FRONTEND_PORT" "Frontend"

printf '\nDev environment is ready.\n\n'
printf 'Frontend: http://localhost:%s\n' "$FRONTEND_PORT"
printf 'API:      http://localhost:%s\n' "$API_PORT"
printf 'Alt URL:  http://127.0.0.1:%s\n' "$FRONTEND_PORT"
printf 'API log:  %s\n' "$API_LOG"
printf 'UI log:   %s\n\n' "$FRONTEND_LOG"
printf 'Press Ctrl+C to stop both processes.\n\n'

tail -n 20 -f "$API_LOG" "$FRONTEND_LOG" &
TAIL_PID=$!
wait "$API_PID" "$FRONTEND_PID" 2>/dev/null || true
kill "$TAIL_PID" 2>/dev/null || true
