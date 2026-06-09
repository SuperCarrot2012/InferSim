#!/usr/bin/env bash
# Start systolic array Web UI (FastAPI + React static build)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PORT="${PORT:-8765}"
echo "Starting systolic array simulator on http://127.0.0.1:${PORT}"
exec python -m uvicorn systolic_array.webui.server:app --host 0.0.0.0 --port "$PORT"
