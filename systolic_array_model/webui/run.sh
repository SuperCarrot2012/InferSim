#!/usr/bin/env bash
# Start systolic_array_model Web UI (FastAPI + React static build)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PORT="${PORT:-8766}"
echo "Starting systolic array model simulator on http://127.0.0.1:${PORT}"
exec python -m uvicorn systolic_array_model.webui.server:app --host 0.0.0.0 --port "$PORT"
