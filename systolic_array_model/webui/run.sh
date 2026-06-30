#!/usr/bin/env bash
# Start systolic_array_model Web UI (FastAPI + React static build)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PORT="${PORT:-8766}"
FRONTEND="$ROOT/systolic_array_model/webui/frontend"
if [[ -d "$FRONTEND/node_modules/.bin" ]]; then
  if [[ ! -f "$FRONTEND/dist/index.html" ]] \
    || find "$FRONTEND/src" -newer "$FRONTEND/dist/index.html" -print -quit | grep -q .; then
    echo "Building frontend..."
    (cd "$FRONTEND" && ./node_modules/.bin/tsc -b && ./node_modules/.bin/vite build)
  fi
fi
echo "Starting systolic array model simulator on http://127.0.0.1:${PORT}"
exec python -m uvicorn systolic_array_model.webui.server:app --host 0.0.0.0 --port "$PORT"
