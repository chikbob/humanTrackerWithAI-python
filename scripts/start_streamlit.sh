#!/usr/bin/env bash
set -euo pipefail

python /app/scripts/bootstrap_runtime.py
exec streamlit run /app/app.py --server.address=0.0.0.0 --server.port="${PORT:-8501}"
