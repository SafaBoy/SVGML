#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "========================================================"
echo "  SVGML Studio — End-to-End Verification Suite"
echo "========================================================"

BASE_URL="${BASE_URL:-http://localhost:5173}"

echo "[1/4] Checking Gateway + Rust Engine + Python ML Worker Health via $BASE_URL/api/health..."
curl -fsS "$BASE_URL/api/health" | python3 -m json.tool

echo "[2/4] Testing Direct Text-to-SVG Pipeline (Go -> PyTorch/HF -> Rust SVG Colorizer)..."
RESP_DIRECT=$(curl -fsS -X POST "$BASE_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A vibrant sunset over twin geometric mountains and ocean waves",
    "model_id": "vinoku89/svg-code-generator",
    "pipeline": "direct-svg",
    "style": "vibrant-gradient",
    "palette": ["#6366F1", "#EC4899", "#06B6D4", "#F59E0B", "#10B981"]
  }')
echo "$RESP_DIRECT" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'  -> Generated ID: {d[\"id\"]}, Colors: {d[\"extracted_colors\"]}, Paths: {d[\"path_count\"]}, TotalMs: {d[\"total_ms\"]}ms')"

echo "[3/4] Testing LiteRT/Diffusion + Rust vtracer Color Vectorization Pipeline..."
RESP_VTRACER=$(curl -fsS -X POST "$BASE_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic AI robot head with glowing neon cyan and magenta visor",
    "model_id": "segmind/tiny-sd",
    "pipeline": "diffusion-vtracer",
    "style": "cyberpunk",
    "color_precision": 6,
    "filter_speckle": 4
  }')
echo "$RESP_VTRACER" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'  -> Generated ID: {d[\"id\"]}, Colors: {d[\"extracted_colors\"]}, Paths: {d[\"path_count\"]}, TotalMs: {d[\"total_ms\"]}ms')"

echo "[4/4] Running Playwright Headless Chromium Web UI Tests..."
"$ROOT_DIR/ml-worker/.venv/bin/python" "$ROOT_DIR/scripts/test_ui_playwright.py"

echo "✅ All End-to-End API & Playwright Web UI tests passed!"
