#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
AI_GATEWAY_PORT="${AI_GATEWAY_PORT:-8787}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
REUSE_EXISTING="${REUSE_EXISTING:-0}"

PIDS=()

cleanup() {
  echo
  echo "Stopping dashboard services..."
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  wait >/dev/null 2>&1 || true
}

require_file() {
  local path="$1"
  local message="$2"
  if [[ ! -e "$path" ]]; then
    echo "$message" >&2
    exit 1
  fi
}

if [[ -x "$ROOT_DIR/backend/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/backend/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

port_open() {
  local port="$1"
  "$PYTHON_BIN" - "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.25)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

http_ok() {
  local url="$1"
  curl -fsS "$url" >/dev/null 2>&1
}

kill_port() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return
  fi
  echo "Stopping existing process on port ${port}: ${pids//$'\n'/ }"
  kill $pids >/dev/null 2>&1 || true
  sleep 0.6
  if lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    kill -9 $pids >/dev/null 2>&1 || true
    sleep 0.2
  fi
}

trap cleanup EXIT INT TERM

require_file "$ROOT_DIR/backend/.env" "Missing backend/.env. Copy backend/.env.example and fill in Leonardo settings."
require_file "$ROOT_DIR/ai-gateway/.env" "Missing ai-gateway/.env. Add OPENAI_API_KEY there."
require_file "$ROOT_DIR/frontend/node_modules" "Missing frontend/node_modules. Run: cd dashboard/frontend && npm install"
require_file "$ROOT_DIR/ai-gateway/node_modules" "Missing ai-gateway/node_modules. Run: cd dashboard/ai-gateway && npm install"

echo "Starting Leonardo dashboard stack"
echo "Backend:    http://127.0.0.1:${BACKEND_PORT}"
echo "AI gateway: http://127.0.0.1:${AI_GATEWAY_PORT}"
echo "Frontend:   http://127.0.0.1:${FRONTEND_PORT}"
echo

if [[ "$REUSE_EXISTING" != "1" ]]; then
  kill_port "$BACKEND_PORT"
  kill_port "$AI_GATEWAY_PORT"
  kill_port "$FRONTEND_PORT"
fi

if port_open "$BACKEND_PORT"; then
  if http_ok "http://127.0.0.1:${BACKEND_PORT}/api/health"; then
    echo "Backend already running on ${BACKEND_PORT}; reusing it."
  else
    echo "Port ${BACKEND_PORT} is already in use, but it is not the dashboard backend." >&2
    echo "Stop that process or run with BACKEND_PORT=<free-port>." >&2
    exit 1
  fi
else
  (
    cd "$ROOT_DIR/backend"
    "$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port "$BACKEND_PORT" --reload
  ) &
  PIDS+=("$!")
fi

if port_open "$AI_GATEWAY_PORT"; then
  if http_ok "http://127.0.0.1:${AI_GATEWAY_PORT}/api/ai/health"; then
    echo "AI gateway already running on ${AI_GATEWAY_PORT}; reusing it."
  else
    echo "Port ${AI_GATEWAY_PORT} is already in use, but it is not the AI gateway." >&2
    echo "Stop that process or run with AI_GATEWAY_PORT=<free-port>." >&2
    exit 1
  fi
else
  (
    cd "$ROOT_DIR/ai-gateway"
    AI_GATEWAY_PORT="$AI_GATEWAY_PORT" npm run dev
  ) &
  PIDS+=("$!")
fi

if port_open "$FRONTEND_PORT"; then
  if http_ok "http://127.0.0.1:${FRONTEND_PORT}/"; then
    echo "Frontend already running on ${FRONTEND_PORT}; reusing it."
  else
    echo "Port ${FRONTEND_PORT} is already in use, but it is not responding as a web server." >&2
    echo "Stop that process or run with FRONTEND_PORT=<free-port>." >&2
    exit 1
  fi
else
  (
    cd "$ROOT_DIR/frontend"
    npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort
  ) &
  PIDS+=("$!")
fi

echo "All services launched. Press Ctrl+C to stop."
wait
