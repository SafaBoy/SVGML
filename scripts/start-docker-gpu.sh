#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BUILD_MISSING=0
case "${1:-}" in
  "") ;;
  --build-missing) BUILD_MISSING=1 ;;
  *)
    echo "Usage: $0 [--build-missing]" >&2
    exit 2
    ;;
esac

if [ -z "${HF_CACHE_DIR:-}" ] && [ -d "$HOME/.cache/huggingface" ]; then
  export HF_CACHE_DIR="$HOME/.cache/huggingface"
  echo "ℹ️  Reusing the host Hugging Face cache: $HF_CACHE_DIR"
  echo "   (set HF_CACHE_DIR to override, or HF_CACHE_DIR=hf_cache for a named volume)"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "⚠️  nvidia-smi not found; the GPU device may not be available to Docker."
fi

COMPOSE_ARGS=(-f docker-compose.yml -f docker-compose.gpu.yml)
# Compose's default image names are <project>-<service>.
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-svgml}"

image_list="$(docker compose "${COMPOSE_ARGS[@]}" config --images)" || {
  echo "❌ Unable to read the compose image list." >&2
  exit 1
}
service_list="$(docker compose "${COMPOSE_ARGS[@]}" config --services)" || {
  echo "❌ Unable to read the compose service list." >&2
  exit 1
}
if [ -z "$image_list" ]; then
  echo "❌ No images configured for this compose stack." >&2
  exit 1
fi

missing_images=()
while IFS= read -r image; do
  [ -n "$image" ] || continue
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    missing_images+=("$image")
  fi
done <<< "$image_list"

if [ "${#missing_images[@]}" -gt 0 ]; then
  if [ "$BUILD_MISSING" -ne 1 ]; then
    echo "❌ Missing Docker images for this stack:" >&2
    printf '   - %s\n' "${missing_images[@]}" >&2
    echo "Build them once with './scripts/run.sh --gpu', or rerun with '--build-missing' to build only the missing images." >&2
    exit 1
  fi

  missing_services=()
  for image in "${missing_images[@]}"; do
    service="${image#"${PROJECT_NAME}-"}"
    service="${service%%:*}"
    if ! grep -qx -- "$service" <<< "$service_list"; then
      echo "❌ Cannot map image '$image' to a compose service; build it manually." >&2
      exit 1
    fi
    missing_services+=("$service")
  done

  echo "Building missing images: ${missing_services[*]}..."
  docker compose "${COMPOSE_ARGS[@]}" build "${missing_services[@]}"
fi

echo "Starting Docker Compose stack with NVIDIA GPU acceleration (no rebuild)..."
docker compose "${COMPOSE_ARGS[@]}" up -d --no-build
echo "✅ Stack running at http://localhost:5173"
