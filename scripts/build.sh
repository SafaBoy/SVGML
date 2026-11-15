#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:---all}"

echo "========================================================"
echo "  SVGML Studio — Build Script (Mode: $MODE)"
echo "========================================================"

build_local() {
  echo "[1/4] Building Rust SVG & Color Vectorization Engine (vtracer)..."
  (cd "$ROOT_DIR/backend-rust" && cargo build --release)

  echo "[2/4] Building Go API Gateway & Hugging Face Hub Manager..."
  mkdir -p "$ROOT_DIR/bin"
  (cd "$ROOT_DIR/backend-go" && go build -o "$ROOT_DIR/bin/svgml-go-gateway" .)
  cp "$ROOT_DIR/backend-rust/target/release/svgml-rust-engine" "$ROOT_DIR/bin/svgml-rust-engine"

  echo "[3/4] Setting up Python PyTorch/LiteRT ML Worker virtual environment..."
  if [ ! -d "$ROOT_DIR/ml-worker/.venv" ]; then
    python3 -m venv "$ROOT_DIR/ml-worker/.venv"
  fi
  "$ROOT_DIR/ml-worker/.venv/bin/pip" install --upgrade pip >/dev/null
  "$ROOT_DIR/ml-worker/.venv/bin/pip" install --extra-index-url https://download.pytorch.org/whl/cu126 \
    "torch==2.14.1+cu126" transformers diffusers accelerate fastapi "uvicorn[standard]" pydantic Pillow numpy huggingface-hub playwright

  echo "[4/4] Building React + TypeScript Frontend..."
  (cd "$ROOT_DIR/frontend" && npm install && npm run build)

  echo "✅ Local build complete! Binaries placed in ./bin/"
}

build_docker() {
  echo "[Docker] Building all container images via docker compose..."
  docker compose build
  echo "✅ Docker Compose images built successfully!"
}

case "$MODE" in
  --local)
    build_local
    ;;
  --docker)
    build_docker
    ;;
  --all|*)
    build_local
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
      build_docker
    else
      echo "ℹ️  Skipping Docker image build (Docker daemon not active or accessible; use './scripts/build.sh --docker' when ready)."
    fi
    ;;
esac
