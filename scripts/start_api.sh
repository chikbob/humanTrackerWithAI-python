#!/usr/bin/env bash
set -euo pipefail

python /app/scripts/bootstrap_runtime.py
exec uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
