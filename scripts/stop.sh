#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Stopping SVGML Studio services..."

if [ -d "$ROOT_DIR/.run" ]; then
  for pidfile in "$ROOT_DIR/.run"/*.pid; do
    [ -e "$pidfile" ] || continue
    PID=$(cat "$pidfile")
    if kill -0 "$PID" 2>/dev/null; then
      echo "Stopping local process $PID ($(basename "$pidfile" .pid))..."
      kill "$PID" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  done
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker compose down 2>/dev/null || true
fi

echo "✅ All services stopped."
