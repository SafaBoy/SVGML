#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
EMBED_DIR="$ROOT_DIR/backend-go/embedded"

echo "========================================================"
echo "  SVGML Studio — Single-Binary Packager (Linux & Windows)"
echo "========================================================"

mkdir -p "$DIST_DIR"
rm -rf "$EMBED_DIR"
mkdir -p "$EMBED_DIR/frontend" "$EMBED_DIR/ml-worker" "$EMBED_DIR/bin"

# 1. Build React Production Frontend & stage into backend-go/embedded/frontend
echo ""
echo "[1/5] Building React Studio Frontend (production bundle)..."
(
  cd "$ROOT_DIR/frontend"
  if [ ! -d "node_modules" ]; then
    npm install
  fi
  npm run build
)
cp -r "$ROOT_DIR/frontend/dist/"* "$EMBED_DIR/frontend/"
echo "  ✔ React SPA staged into backend-go/embedded/frontend"

# 2. Stage Python PyTorch/CUDA ML Worker into backend-go/embedded/ml-worker
echo ""
echo "[2/5] Staging Python PyTorch/CUDA ML Worker..."
cp "$ROOT_DIR/ml-worker/app.py" "$EMBED_DIR/ml-worker/app.py"
cp "$ROOT_DIR/ml-worker/requirements.txt" "$EMBED_DIR/ml-worker/requirements.txt"
echo "  ✔ ML Worker staged into backend-go/embedded/ml-worker"

# 3. Build Rust vtracer Engine for Linux (and Windows if mingw-w64 is installed)
echo ""
echo "[3/5] Building Rust SVG & vtracer Engine..."
if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi
(
  cd "$ROOT_DIR/backend-rust"
  cargo build --release
  cp target/release/svgml-rust-engine "$EMBED_DIR/bin/svgml-rust-engine-linux-amd64"
  strip "$EMBED_DIR/bin/svgml-rust-engine-linux-amd64" 2>/dev/null || true
)
echo "  ✔ Linux Rust vtracer binary staged into backend-go/embedded/bin/svgml-rust-engine-linux-amd64"

if command -v x86_64-w64-mingw32-gcc >/dev/null 2>&1; then
  echo "  • Detected mingw-w64; cross-compiling Rust vtracer engine for Windows..."
  (
    cd "$ROOT_DIR/backend-rust"
    rustup target add x86_64-pc-windows-gnu >/dev/null 2>&1 || true
    cargo build --release --target x86_64-pc-windows-gnu && \
      cp target/x86_64-pc-windows-gnu/release/svgml-rust-engine.exe "$EMBED_DIR/bin/svgml-rust-engine-windows-amd64.exe" || true
  )
else
  echo "  • Note: x86_64-w64-mingw32-gcc not installed; Windows .exe will use the built-in pure-Go native SVG/Line-Art/Vectorizer engine + embedded Python CUDA worker."
fi

# 4. Build Linux Single Binary (x86_64)
echo ""
echo "[4/5] Packaging Single Binary for Linux (dist/svgml-studio-linux-amd64)..."
(
  cd "$ROOT_DIR/backend-go"
  CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -trimpath -ldflags="-s -w" \
    -o "$DIST_DIR/svgml-studio-linux-amd64" .
  chmod +x "$DIST_DIR/svgml-studio-linux-amd64"
)
echo "  ✔ Created: $DIST_DIR/svgml-studio-linux-amd64"

# 5. Build Windows Single Binary (x86_64 .exe)
echo ""
echo "[5/5] Packaging Single Binary for Windows (dist/svgml-studio-windows-amd64.exe)..."
(
  # Temporarily move the Linux ELF Rust binary out of embedded/bin during the Windows build so the .exe doesn't carry unused Linux ELF bytes
  TMP_LINUX_RUST="$(mktemp)"
  mv "$EMBED_DIR/bin/svgml-rust-engine-linux-amd64" "$TMP_LINUX_RUST"
  # Ensure embedded/bin is non-empty for go:embed
  touch "$EMBED_DIR/bin/.keep"

  cd "$ROOT_DIR/backend-go"
  CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build \
    -trimpath -ldflags="-s -w" \
    -o "$DIST_DIR/svgml-studio-windows-amd64.exe" .

  mv "$TMP_LINUX_RUST" "$EMBED_DIR/bin/svgml-rust-engine-linux-amd64"
)
echo "  ✔ Created: $DIST_DIR/svgml-studio-windows-amd64.exe"

# Generate SHA256 checksums
(
  cd "$DIST_DIR"
  sha256sum svgml-studio-linux-amd64 svgml-studio-windows-amd64.exe > SHA256SUMS.txt
)

echo ""
echo "========================================================"
echo "✅ Single-Binary Packaging Complete!"
echo "========================================================"
ls -lh "$DIST_DIR/svgml-studio-linux-amd64" "$DIST_DIR/svgml-studio-windows-amd64.exe"
echo ""
echo "SHA-256 Checksums ($DIST_DIR/SHA256SUMS.txt):"
cat "$DIST_DIR/SHA256SUMS.txt"
echo ""
echo "How to run:"
echo "  • Linux:   ./dist/svgml-studio-linux-amd64"
echo "  • Windows: .\\dist\\svgml-studio-windows-amd64.exe"
echo "  (Opens Web UI & API on http://0.0.0.0:5173)"
