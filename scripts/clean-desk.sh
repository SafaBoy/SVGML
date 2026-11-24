#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ASSUME_YES=0
case "${1:-}" in
  --yes) ASSUME_YES=1 ;;
  "") ;;
  *)
    echo "Usage: $0 [--yes]" >&2
    exit 2
    ;;
esac

if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "Stop SVGML containers and remove local build artifacts? [y/N] " reply
  case "$reply" in
    y | Y | yes | YES) ;;
    *)
      echo "Aborted. Nothing was removed."
      exit 0
      ;;
  esac
fi

echo "Stopping SVGML project containers..."
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml down --remove-orphans || true
else
  echo "Docker is unavailable; skipping container cleanup."
fi

echo "Removing local build and runtime artifacts..."
rm -rf \
  "$ROOT_DIR/.run" \
  "$ROOT_DIR/frontend/dist" \
  "$ROOT_DIR/frontend/node_modules" \
  "$ROOT_DIR/frontend/tsconfig.tsbuildinfo" \
  "$ROOT_DIR/backend-rust/target" \
  "$ROOT_DIR/backend-go/backend-go" \
  "$ROOT_DIR/bin" \
  "$ROOT_DIR/dist" \
  "$ROOT_DIR/ml-worker/.venv"
rm -f "$ROOT_DIR"/*.log "$ROOT_DIR"/*.pid "$ROOT_DIR/data/gallery.json"

find "$ROOT_DIR" -type d -name '__pycache__' -not -path '*/.git/*' -prune -exec rm -rf {} + 2>/dev/null || true
find "$ROOT_DIR" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.pyd' \) -not -path '*/.git/*' -delete 2>/dev/null || true

for dir in \
  "$ROOT_DIR/backend-go/embedded/frontend" \
  "$ROOT_DIR/backend-go/embedded/ml-worker" \
  "$ROOT_DIR/backend-go/embedded/bin"; do
  [ -d "$dir" ] || continue
  find "$dir" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' ! -name '.keep' -exec rm -rf {} +
done

echo "✅ Desk clean. Project containers stopped and local artifacts removed."
echo "   The Hugging Face cache, local env files, volumes and unrelated Docker data were preserved."
