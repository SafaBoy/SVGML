#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: ./scripts/build.sh [--local|--docker|--all] [--cpu|--gpu]

  --local     Build the Rust engine, Go gateway, Python venv and frontend
  --docker    Build the Docker Compose images
  --all       Local build, then Docker build when the daemon is reachable (default)

  --cpu       Install CPU-only PyTorch (smaller, works on machines without a GPU)
  --gpu       Install CUDA 12.6 PyTorch (needed for NVIDIA GPUs such as the GTX 1050 Ti)

Torch selection defaults to auto-detection via `nvidia-smi`. Override it with
SVGML_TORCH=cpu|cu126|none. Set SVGML_SKIP_PLAYWRIGHT=1 to skip the browser
automation dependency used by scripts/test_ui_playwright.py.
EOF
}

MODE="--all"
TORCH_VARIANT="${SVGML_TORCH:-auto}"

while [ $# -gt 0 ]; do
  case "$1" in
    --local|--docker|--all) MODE="$1" ;;
    --cpu) TORCH_VARIANT="cpu" ;;
    --gpu) TORCH_VARIANT="cu126" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; echo >&2; usage >&2; exit 2 ;;
  esac
  shift
done

echo "========================================================"
echo "  SVGML Studio — Build Script (Mode: $MODE)"
echo "========================================================"

# CUDA 13 wheels no longer ship Pascal (sm_61) kernels, so the newest PyPI torch
# silently falls back to CPU on cards like the GTX 1050 Ti. Install torch from an
# explicit index instead of letting requirements.txt pull the default wheel.
detect_torch_variant() {
  if [ "$TORCH_VARIANT" != "auto" ]; then
    echo "$TORCH_VARIANT"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "cu126"
  else
    echo "cpu"
  fi
}

torch_index_url() {
  case "$1" in
    cpu) echo "https://download.pytorch.org/whl/cpu" ;;
    cu126) echo "https://download.pytorch.org/whl/cu126" ;;
    *) echo "" ;;
  esac
}

build_local() {
  echo "[1/4] Building Rust SVG & Color Vectorization Engine (vtracer)..."
  (cd "$ROOT_DIR/backend-rust" && cargo build --release)

  echo "[2/4] Building Go API Gateway & Hugging Face Hub Manager..."
  mkdir -p "$ROOT_DIR/bin"
  (cd "$ROOT_DIR/backend-go" && go build -o "$ROOT_DIR/bin/svgml-go-gateway" .)
  cp "$ROOT_DIR/backend-rust/target/release/svgml-rust-engine" "$ROOT_DIR/bin/svgml-rust-engine"

  local variant
  variant="$(detect_torch_variant)"
  local index_url
  index_url="$(torch_index_url "$variant")"

  echo "[3/4] Setting up Python PyTorch/LiteRT ML Worker (torch: $variant)..."
  if [ ! -d "$ROOT_DIR/ml-worker/.venv" ]; then
    python3 -m venv "$ROOT_DIR/ml-worker/.venv"
  fi
  local venv_pip="$ROOT_DIR/ml-worker/.venv/bin/pip"
  "$venv_pip" install --upgrade pip >/dev/null

  if [ "$variant" != "none" ]; then
    "$venv_pip" install --index-url "$index_url" torch
  fi

  # Install everything else from requirements.txt, minus the torch line: leaving
  # it in would let pip swap in the default PyPI wheel and undo the choice above.
  local reqs_no_torch
  reqs_no_torch="$(mktemp)"
  grep -v -E '^torch([^a-zA-Z0-9_.-]|$)' "$ROOT_DIR/ml-worker/requirements.txt" > "$reqs_no_torch"
  "$venv_pip" install -r "$reqs_no_torch"
  rm -f "$reqs_no_torch"

  if [ "${SVGML_SKIP_PLAYWRIGHT:-0}" != "1" ]; then
    "$venv_pip" install playwright
  fi

  if ! "$ROOT_DIR/ml-worker/.venv/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info < (3, 13) else 1)'; then
    echo "ℹ️  $( "$ROOT_DIR/ml-worker/.venv/bin/python" -V ) — ai-edge-litert requires Python < 3.13,"
    echo "    so the LiteRT backend is unavailable in this environment (PyTorch paths still work)."
  fi

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
      echo "ℹ️  Skipping Docker image build (Docker daemon not active or inaccessible; use './scripts/build.sh --docker' when ready)."
    fi
    ;;
esac
