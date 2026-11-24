#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -z "${HF_CACHE_DIR:-}" ] && [ -d "$HOME/.cache/huggingface" ]; then
  export HF_CACHE_DIR="$HOME/.cache/huggingface"
  echo "ℹ️  Reusing the host Hugging Face cache: $HF_CACHE_DIR"
  echo "   (set HF_CACHE_DIR to override, or HF_CACHE_DIR=hf_cache for a named volume)"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "⚠️  nvidia-smi not found; the GPU device may not be available to Docker."
fi

echo "Starting Docker Compose stack with NVIDIA GPU acceleration (no rebuild)..."
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --no-build
echo "✅ Stack running at http://localhost:5173"
