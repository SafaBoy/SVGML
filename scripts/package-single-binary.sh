#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
EMBED_DIR="$ROOT_DIR/backend-go/embedded"
STAGE_DIR="$ROOT_DIR/.run/pkgstage"
MANIFEST="$DIST_DIR/BUILD-INFO.txt"

ALLOW_DEGRADED=0
for arg in "$@"; do
  case "$arg" in
    --allow-degraded) ALLOW_DEGRADED=1 ;;
    --help|-h)
      cat <<'USAGE'
Usage: scripts/package-single-binary.sh [--allow-degraded]

Builds dist/svgml-studio-linux-amd64 and dist/svgml-studio-windows-amd64.exe,
each with its own Rust vtracer engine embedded.

A Windows Rust engine requires a mingw-w64 toolchain. Put it on PATH or point
MINGW_PREFIX at its install prefix (the directory containing bin/x86_64-w64-mingw32-gcc).
Without it the script stops, because the Windows build would silently ship the
coarse pure-Go fallback vectorizer instead of the real engine.

  --allow-degraded   Build the Windows binary anyway, without the Rust engine,
                     and mark it degraded in dist/BUILD-INFO.txt.
USAGE
      exit 0
      ;;
    *) echo "Unknown argument: $arg (try --help)" >&2; exit 2 ;;
  esac
done

echo "========================================================"
echo "  SVGML Studio — Single-Binary Packager (Linux & Windows)"
echo "========================================================"

mkdir -p "$DIST_DIR"
rm -rf "$EMBED_DIR" "$STAGE_DIR"
mkdir -p "$EMBED_DIR/frontend" "$EMBED_DIR/ml-worker" "$EMBED_DIR/bin"
mkdir -p "$STAGE_DIR/linux" "$STAGE_DIR/windows"
touch "$EMBED_DIR/bin/.keep" "$EMBED_DIR/frontend/.gitkeep" "$EMBED_DIR/ml-worker/.gitkeep"

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

# 3. Build the Rust vtracer engine for Linux, and for Windows when a mingw-w64
#    toolchain is available. Both are staged outside embedded/ so each packaged
#    binary can carry only the engine that matches its own platform.
echo ""
echo "[3/5] Building Rust SVG & vtracer Engine..."
if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

LINUX_ENGINE="$STAGE_DIR/linux/svgml-rust-engine-linux-amd64"
WINDOWS_ENGINE="$STAGE_DIR/windows/svgml-rust-engine-windows-amd64.exe"

strip_quiet() { strip "$1" 2>/dev/null || true; }

(
  cd "$ROOT_DIR/backend-rust"
  cargo build --release
  cp target/release/svgml-rust-engine "$LINUX_ENGINE"
)
strip_quiet "$LINUX_ENGINE"
echo "  ✔ Linux Rust engine: $LINUX_ENGINE ($(du -h "$LINUX_ENGINE" | cut -f1))"

MINGW_GCC=""
if command -v x86_64-w64-mingw32-gcc >/dev/null 2>&1; then
  MINGW_GCC="$(command -v x86_64-w64-mingw32-gcc)"
elif [ -n "${MINGW_PREFIX:-}" ] && [ -x "$MINGW_PREFIX/bin/x86_64-w64-mingw32-gcc" ]; then
  MINGW_GCC="$(cd "$MINGW_PREFIX/bin" && pwd)/x86_64-w64-mingw32-gcc"
fi

if [ -n "$MINGW_GCC" ]; then
  MINGW_BIN="$(dirname "$MINGW_GCC")"
  echo "  • Cross-compiling Rust engine for Windows with $MINGW_GCC"
  (
    cd "$ROOT_DIR/backend-rust"
    PATH="$MINGW_BIN:$PATH" rustup target add x86_64-pc-windows-gnu >/dev/null 2>&1 || true
    PATH="$MINGW_BIN:$PATH" CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="$MINGW_GCC" \
      cargo build --release --target x86_64-pc-windows-gnu
    cp target/x86_64-pc-windows-gnu/release/svgml-rust-engine.exe "$WINDOWS_ENGINE"
  )
  if [ -x "$MINGW_BIN/x86_64-w64-mingw32-strip" ]; then
    "$MINGW_BIN/x86_64-w64-mingw32-strip" "$WINDOWS_ENGINE" 2>/dev/null || true
  else
    strip_quiet "$WINDOWS_ENGINE"
  fi
  echo "  ✔ Windows Rust engine: $WINDOWS_ENGINE ($(du -h "$WINDOWS_ENGINE" | cut -f1))"
else
  echo ""
  echo "  ✖ No mingw-w64 toolchain found — cannot build the Windows Rust engine."
  echo "    Install it (Debian/Ubuntu: sudo apt-get install -y mingw-w64) or set"
  echo "    MINGW_PREFIX to a prefix containing bin/x86_64-w64-mingw32-gcc."
  echo "    Without it the Windows .exe only has the coarse pure-Go fallback"
  echo "    vectorizer, so packaging stops here."
  if [ "$ALLOW_DEGRADED" -ne 1 ]; then
    echo ""
    echo "    Re-run with --allow-degraded to build that reduced Windows binary"
    echo "    anyway (it will be marked degraded in $MANIFEST)."
    exit 1
  fi
  echo "    Continuing because --allow-degraded was passed."
fi

# Confirm a slice of the staged engine really ended up inside the packaged
# binary, so a broken packaging step can never masquerade as a complete build.
verify_embedded_engine() {
  local binary="$1" engine="$2"
  python3 - "$binary" "$engine" <<'PY'
import sys

data = open(sys.argv[1], "rb").read()
engine = open(sys.argv[2], "rb").read()
if len(engine) < 65536:
    sys.exit("staged engine is suspiciously small: %d bytes" % len(engine))
probe = engine[len(engine) // 2:len(engine) // 2 + 256]
sys.exit(0 if probe in data else 1)
PY
}

MANIFEST_BODY="SVGML Studio single-binary build manifest
generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')
git: $(cd "$ROOT_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)$(cd "$ROOT_DIR" && { [ -z "$(git status --porcelain 2>/dev/null)" ] || echo ' (dirty)'; } 2>/dev/null || true)
script: scripts/package-single-binary.sh
"

record_artifact() {
  local out="$1" engine_name="$2" engine_src="$3" degraded="$4"
  local path="$DIST_DIR/$out"
  MANIFEST_BODY+="
artifact: $out
  size_bytes: $(stat -c '%s' "$path")
  sha256: $(sha256sum "$path" | cut -d' ' -f1)
  degraded: $degraded
  embedded_rust_engine: $engine_name"
  if [ -n "$engine_src" ]; then
    MANIFEST_BODY+=" ($(stat -c '%s' "$engine_src") bytes, verified)"
  else
    MANIFEST_BODY+=" (none — pure-Go fallback engine only)"
  fi
  MANIFEST_BODY+="
  embedded_frontend: yes
  embedded_ml_worker: yes
"
}

# Stage exactly one engine in embedded/bin, build, then prove the engine bytes
# are in the artifact.
package_target() {
  local goos="$1" out="$2" engine_name="$3" engine_src="$4" degraded="$5"

  rm -rf "$EMBED_DIR/bin"
  mkdir -p "$EMBED_DIR/bin"
  touch "$EMBED_DIR/bin/.keep"
  if [ -n "$engine_src" ]; then
    cp "$engine_src" "$EMBED_DIR/bin/$engine_name"
  fi

  (
    cd "$ROOT_DIR/backend-go"
    CGO_ENABLED=0 GOOS="$goos" GOARCH=amd64 go build \
      -trimpath -ldflags="-s -w" \
      -o "$DIST_DIR/$out" .
  )
  chmod +x "$DIST_DIR/$out" 2>/dev/null || true

  if [ -n "$engine_src" ]; then
    if ! verify_embedded_engine "$DIST_DIR/$out" "$engine_src"; then
      echo "  ✖ $out does not contain the staged $engine_name — packaging is broken" >&2
      exit 1
    fi
    echo "  ✔ Created: $DIST_DIR/$out ($(du -h "$DIST_DIR/$out" | cut -f1), $engine_name verified inside)"
  else
    echo "  ⚠ Created: $DIST_DIR/$out ($(du -h "$DIST_DIR/$out" | cut -f1), NO Rust engine — degraded)"
  fi
  record_artifact "$out" "$engine_name" "$engine_src" "$degraded"
}

echo ""
echo "[4/5] Packaging Single Binary for Linux (dist/svgml-studio-linux-amd64)..."
package_target linux svgml-studio-linux-amd64 svgml-rust-engine-linux-amd64 "$LINUX_ENGINE" false

echo ""
echo "[5/5] Packaging Single Binary for Windows (dist/svgml-studio-windows-amd64.exe)..."
if [ -n "$MINGW_GCC" ]; then
  package_target windows svgml-studio-windows-amd64.exe svgml-rust-engine-windows-amd64.exe "$WINDOWS_ENGINE" false
else
  package_target windows svgml-studio-windows-amd64.exe svgml-rust-engine-windows-amd64.exe "" true
fi

# Leave embedded/bin holding both engines so a plain `go build` in backend-go
# produces the same complete bundle as the packaged artifacts.
rm -rf "$EMBED_DIR/bin"
mkdir -p "$EMBED_DIR/bin"
touch "$EMBED_DIR/bin/.keep"
cp "$LINUX_ENGINE" "$EMBED_DIR/bin/svgml-rust-engine-linux-amd64"
if [ -n "$MINGW_GCC" ]; then
  cp "$WINDOWS_ENGINE" "$EMBED_DIR/bin/svgml-rust-engine-windows-amd64.exe"
fi

printf '%s' "$MANIFEST_BODY" > "$MANIFEST"

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
echo "Build manifest ($MANIFEST):"
cat "$MANIFEST"
echo ""
echo "SHA-256 Checksums ($DIST_DIR/SHA256SUMS.txt):"
cat "$DIST_DIR/SHA256SUMS.txt"
echo ""
echo "How to run:"
echo "  • Linux:   ./dist/svgml-studio-linux-amd64 --self-check"
echo "  • Windows: .\\dist\\svgml-studio-windows-amd64.exe --self-check"
echo "  (Then start without --self-check: Web UI & API on http://0.0.0.0:5173)"
