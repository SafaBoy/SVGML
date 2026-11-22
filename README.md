# SVGML Studio — PyTorch Color SVG Generator

Full-stack AI Scalable Vector Graphics (SVG) generation studio combining **Hugging Face PyTorch** models, a **Go** API Gateway & Hugging Face Hub Manager, a **Rust** Color Vectorization (`vtracer`) & SVG Post-Processing Engine, and a **React + TypeScript + Tailwind CSS** frontend.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     React + TypeScript Studio (:5173)                   │
│  Prompt Studio • HF Model Explorer • Live Rust Color Remapper • Canvas  │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ HTTP / JSON (/api/*)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               Go API Gateway & HF Hub Manager (:18080)                  │
│  • Live Hugging Face Hub Search (/api/models/search) & Curated Registry │
│  • Pipeline Orchestration (/api/generate) & Gallery Store (data/)       │
└──────────────────┬───────────────────────────────────┬──────────────────┘
                   │ 1. ML Inference (/infer)          │ 2. Vectorize / Colorize
                   ▼                                   ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────┐
│ Python PyTorch Worker (:18082)       │  │ Rust SVG Engine (:18081)      │
│ • Direct Text-to-SVG HF Transformers │  │ • Hierarchical Color vtracer  │
│ • Text-to-Image HF Diffusers         │  │ • SVG Sanitization & ViewBox  │
│ • PyTorch Neural Color-Field Raster  │  │ • Multi-Color Palette Injector│
└──────────────────────────────────────┘  └───────────────────────────────┘
```

## Ports & Configuration

| Service | Default port | Override |
| --- | --- | --- |
| React studio (Vite dev / nginx) | `5173` | `VITE_API_PROXY_TARGET` for the dev proxy |
| Go API Gateway | `18080` | `GO_PORT` (or `PORT`) |
| Rust SVG engine | `18081` | `RUST_PORT` |
| Python ML worker | `18082` | `ML_PORT` |

The gateway locates its dependencies through `RUST_ENGINE_URL` and `ML_WORKER_URL`
(defaults `http://localhost:18081` and `http://localhost:18082`); the Docker network overrides them
with service names. Other variables: `DATA_DIR` (gallery storage, default `./data`), `HF_TOKEN`,
`HF_EAGER_DOWNLOAD` (`true` lets the ML worker pull the first uncached multi-GB checkpoint on
demand; cached checkpoints are used regardless), and `SINGLE_BINARY=1` to run the packaged
single-binary mode, which serves the UI on `5173` and starts the embedded sub-services itself.
See `.env.example` for a copy-pasteable set of defaults.

## Supported Hugging Face Models & Pipelines

1. **Direct Text-to-SVG Code Generation (`direct-svg`)** — the ML worker loads the checkpoint with
   `transformers` and decodes SVG markup directly:
   - `vinoku89/svg-code-generator` (Qwen3-0.6B SVG fine-tune, chat template) — works out of the box
   - `Qwen/Qwen2.5-Coder-1.5B-Instruct` (instructed colored SVG XML) — works out of the box
   - `Norod78/gpt-fluentui-flat-svg` (completion-only GPT-2, emits Fluent UI icon markup)
   - `mohannad-tazi/Llama-3.1-8B-Instruct-text-to-svg` — the Hub repo only stores a LoRA adapter;
     download the Llama-3.1-8B base model and merge it first
   - `OmniSVG/OmniSVG1.1_4B` — **not loadable as a plain HF checkpoint**: the repo ships weights only
     and inference needs OmniSVG's own `SketchDecoder` + `SVGTokenizer` (path tokens, not XML).
     Selecting it returns an explicit diagnostic instead of a fake result.
   Both of those repositories stay in `/api/models/curated` with an `unsupported_reason` so search
   results can explain them; the model hub marks them **Unsupported**, refuses to select or download
   them, and shows the reason. Direct API calls with either id still get the explicit diagnostic.
   If no `direct-svg` model can be loaded (not cached, out of memory, or unsupported), the worker
   generates a deterministic procedural SVG and reports that honestly in `backend_used`
   (`... | hf skipped: <reason>`), with `model_used = svgml-procedural-templates`.
2. **Diffusion + Rust Color Vectorization (`diffusion-vtracer`)**:
   - `segmind/tiny-sd` (PyTorch Diffusers -> Rust `vtracer` stacked color splines)
   - `stabilityai/sdxl-turbo` (1-step Diffusion -> Rust `vtracer`)
3. **PyTorch Neural Color Field (`neural-vtracer`)**:
   - Seeds a wide neural color field with PyTorch, rasterises it with Pillow, then traces it with Rust
     `vtracer`. Fully local, no model download; the old `litert-vtracer` id is still accepted as an alias.
   - No LiteRT runtime or `.tflite` weights ship with this project — earlier versions implied one.

## Quick Start

### Option 1: Docker Compose (CPU out-of-the-box)
```bash
./scripts/build.sh --docker
./scripts/run.sh --docker
```
### Option 2: Docker Compose with NVIDIA GPU (CUDA)
```bash
./scripts/run.sh --gpu
```
Both Docker modes reuse your host Hugging Face cache when `~/.cache/huggingface` exists: `run.sh`
exports `HF_CACHE_DIR` and compose bind-mounts it into the ML worker, so checkpoints you already
downloaded are available immediately. Set `HF_CACHE_DIR=/some/other/cache` to point elsewhere, or
`HF_CACHE_DIR=hf_cache` to fall back to the project's named volume.

### Option 3: Local Native Execution (Go + Rust + Python Venv + Vite)
```bash
./scripts/build.sh --local
./scripts/run.sh --local
```

`build.sh` picks the PyTorch wheel by itself: `nvidia-smi` present means the CUDA 12.6 build (the only
one that still ships Pascal kernels for cards like the GTX 1050 Ti), otherwise CPU-only. Force it with
`--cpu` / `--gpu` or `SVGML_TORCH=cpu|cu126|none`. Everything else comes from
`ml-worker/requirements.txt`.

### Verification & Stop
```bash
# Go gateway middleware tests (auth / CORS / rate limit / body limits)
cd backend-go && go test ./...

# Rust engine tests (SVG sanitising/colorising, recolor, real vtracer round-trip)
cd backend-rust && cargo test

# ML worker unit tests (SVG extraction / model classification, stdlib unittest)
cd ml-worker && .venv/bin/python -m unittest discover -s tests -t .

# Run end-to-end health and pipeline tests
./scripts/test-e2e.sh

# Stop all running containers or local background processes
./scripts/stop.sh
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same checks on every push and pull request:
`go build`/`go vet`/`go test`, `cargo build --release --locked`/`cargo test`, the frontend production
build, and the ML worker unit tests.

### Degraded mode is explicit
If the Rust engine or Python ML worker is unreachable, the gateway keeps serving from its in-process Go
fallbacks but says so: `/api/health` reports `status: "degraded"`, each unavailable service carries
`fallback: true` with the connection error, `/api/models/status` marks cached models `unavailable` and
sets `source: "native-fallback"`, and `/api/models/download` returns HTTP 503 instead of pretending a
download succeeded.

### Gateway security defaults
The gateway is a proxy that can run expensive models, so it is not wide open by default:

- **CORS** — no `Access-Control-Allow-Origin` header is emitted unless the origin is listed in
  `CORS_ALLOWED_ORIGINS` (comma separated; `*` restores the old allow-any behaviour). Same-origin
  deployments — the nginx proxy and the single binary — need no configuration. Cross-origin callers now
  get a browser-side rejection instead of an open API.
- **Rate limit** — per-IP token bucket on the routes that cost real work (`/api/generate`,
  `/api/image-to-svg`, `/api/svg/recolor`, `/api/svg/enhance`, `/api/models/download`,
  `/api/models/search`): `RATE_LIMIT_PER_MIN` requests per minute (default 60, burst is a quarter of that
  plus five). Exceeding it returns `429` with `Retry-After` and `retry_after_seconds`. Health checks and
  model-status polling are never limited, so the studio UI keeps working. `RATE_LIMIT_PER_MIN=0` disables it.
- **Body limits** — 1 MiB by default, 8 MiB for SVG payloads, 25 MiB for photos. Oversized requests get
  `413` naming the limit instead of a generic `400`.
- **Optional token** — set `SVGML_API_TOKEN` and `/api/*` (except `/api/health`, so container probes keep
  working) requires `Authorization: Bearer <token>`; failures return `401` with `WWW-Authenticate`.
  The bundled studio sends `VITE_API_TOKEN` when it is baked in at build time.
- **Headers** — `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`,
  `Referrer-Policy: no-referrer`.

There is no user accounts system: the token is a single shared secret, and the gallery is a single JSON
file with no per-user separation. Do not expose the stack to the internet without putting it behind an
authenticating reverse proxy.

### Gallery storage

`GET /api/gallery?limit=24&offset=0` returns one page plus `total`, `offset`, `limit` and `has_more`, and
the studio shows a "Load older SVGs" button while more remain. `limit` defaults to 24 and is clamped to
100; the store keeps the newest 100 items and drops the rest.

`data/gallery.json` is rewritten through a temp file plus `fsync` and `rename`, so a crash mid-write cannot
truncate it. If the file is unparseable at startup it is moved aside to `gallery.json.corrupt-<unix>`
instead of being overwritten.

### Studio source layout

`frontend/src/App.tsx` holds the studio component; shared pieces live beside it so they can be read and
changed on their own: `types.ts` (API payload types), `presets.ts` (style, line-art, enhance and system
prompt presets plus sample prompts), `api.ts` (`API_BASE`, `VITE_API_TOKEN`, the page size and the
`apiFetch` helper that attaches the bearer token) and `samplePhoto.ts` (the demo photos). Failures in
user-initiated actions (recolour, save, delete, loading another gallery page) raise a dismissible toast;
background health/model polls stay silent because the header pills and the model-hub banner already
report that state.

## Single-Binary Builds

```bash
./scripts/package-single-binary.sh        # -> dist/svgml-studio-linux-amd64
                                          #    dist/svgml-studio-windows-amd64.exe
```

Each artifact embeds the production React studio, the ML worker source, and **its own platform's** Rust
vtracer engine; it serves the UI and `/api` on the same port (`5173` by default). The ML worker still
needs a local Python environment with `uvicorn` and PyTorch — without one the binary starts in the
degraded mode described above. The embedded Rust engine is started automatically.

The Windows artifact needs a mingw-w64 toolchain to cross-compile that engine
(`sudo apt-get install -y mingw-w64`, or set `MINGW_PREFIX` to a prefix containing
`bin/x86_64-w64-mingw32-gcc`). The packager stops rather than silently shipping a Windows `.exe` whose
vectorizer is the coarse pure-Go fallback; pass `--allow-degraded` if you genuinely want that reduced
build.

Before reporting success the packager proves the staged engine bytes are present inside each artifact,
and writes `dist/BUILD-INFO.txt` (per-artifact size, SHA-256, which engine is embedded, whether it was
verified, and `degraded: true|false`). Any shipped binary can be inspected the same way:

```bash
./dist/svgml-studio-linux-amd64 --self-check
# SVGML Studio embedded bundle (linux/amd64)
#   rust engine for this platform: svgml-rust-engine-linux-amd64
#   embedded/bin/svgml-rust-engine-linux-amd64          5.50 MB
#   ...
```

Pushing a `v*` tag runs `.github/workflows/release.yml`, which installs mingw-w64, packages both
binaries, and attaches them plus `SHA256SUMS.txt` and `BUILD-INFO.txt` to the GitHub release.

## License

MIT — see [LICENSE](LICENSE).
