#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:---docker}"

echo "========================================================"
echo "  SVGML Studio — Run Script (Mode: $MODE)"
echo "========================================================"

run_local() {
  mkdir -p "$ROOT_DIR/.run" "$ROOT_DIR/data"

  if [ ! -x "$ROOT_DIR/bin/svgml-rust-engine" ] || [ ! -x "$ROOT_DIR/bin/svgml-go-gateway" ]; then
    echo "Local binaries not found. Running './scripts/build.sh --local' first..."
    "$ROOT_DIR/scripts/build.sh" --local
  fi

  echo "Starting Rust SVG & vtracer Engine on :18081..."
  RUST_PORT=18081 "$ROOT_DIR/bin/svgml-rust-engine" > "$ROOT_DIR/.run/rust.log" 2>&1 &
  echo $! > "$ROOT_DIR/.run/rust.pid"

  echo "Starting Python PyTorch/LiteRT ML Worker on :18082..."
  ML_PORT=18082 "$ROOT_DIR/ml-worker/.venv/bin/python" "$ROOT_DIR/ml-worker/app.py" > "$ROOT_DIR/.run/ml.log" 2>&1 &
  echo $! > "$ROOT_DIR/.run/ml.pid"

  echo "Starting Go API Gateway & HF Hub Manager on :18080..."
  GO_PORT=18080 \
  RUST_ENGINE_URL="http://localhost:18081" \
  ML_WORKER_URL="http://localhost:18082" \
  DATA_DIR="$ROOT_DIR/data" \
  "$ROOT_DIR/bin/svgml-go-gateway" > "$ROOT_DIR/.run/go.log" 2>&1 &
  echo $! > "$ROOT_DIR/.run/go.pid"

  echo "Starting React Frontend Dev Server on :5173..."
  (cd "$ROOT_DIR/frontend" && npm run dev > "$ROOT_DIR/.run/frontend.log" 2>&1) &
  echo $! > "$ROOT_DIR/.run/frontend.pid"

  echo ""
  echo "✅ All services started locally on 0.0.0.0!"
  echo "   • React Studio UI:        http://0.0.0.0:5173"
  echo "   • Go API Gateway:         http://0.0.0.0:18080/api/health"
  echo "   • Rust SVG/vtracer API:   http://0.0.0.0:18081/health"
  echo "   • Python ML Worker API:   http://0.0.0.0:18082/health"
  echo "   • Logs directory:         $ROOT_DIR/.run/"
  echo "   • To stop all services:   ./scripts/stop.sh"

  if [ "${WAIT_FOREGROUND:-0}" = "1" ]; then
    wait
  fi
}

case "$MODE" in
  --local)
    run_local
    ;;
  --gpu)
    echo "Starting Docker Compose stack with NVIDIA GPU acceleration..."
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
    echo "✅ Stack running at http://localhost:5173"
    ;;
  --docker|*)
    echo "Starting Docker Compose stack (CPU-compatible mode)..."
    docker compose up -d --build
    echo "✅ Stack running at http://localhost:5173"
    ;;
esac
