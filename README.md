# SVGML Studio — PyTorch & LiteRT Color SVG Generator

Full-stack AI Scalable Vector Graphics (SVG) generation studio combining **Hugging Face (PyTorch & LiteRT)** models, a **Go** API Gateway & Hugging Face Hub Manager, a **Rust** Color Vectorization (`vtracer`) & SVG Post-Processing Engine, and a **React + TypeScript + Tailwind CSS** frontend.

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
│ Python PyTorch & LiteRT (:18082)     │  │ Rust SVG Engine (:18081)      │
│ • Direct Text-to-SVG HF Transformers │  │ • Hierarchical Color vtracer  │
│ • Text-to-Image HF Diffusers         │  │ • SVG Sanitization & ViewBox  │
│ • LiteRT / PyTorch Neural CPPN Field │  │ • Multi-Color Palette Injector│
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
   If no `direct-svg` model can be loaded (not cached, out of memory, or unsupported), the worker
   generates a deterministic procedural SVG and reports that honestly in `backend_used`
   (`... | hf skipped: <reason>`), with `model_used = svgml-procedural-templates`.
2. **Diffusion + Rust Color Vectorization (`diffusion-vtracer`)**:
   - `segmind/tiny-sd` (PyTorch Diffusers -> Rust `vtracer` stacked color splines)
   - `stabilityai/sdxl-turbo` (1-step Diffusion -> Rust `vtracer`)
3. **LiteRT / PyTorch Neural Color Field (`litert-vtracer`)**:
   - `litert/neural-svg-synthesizer-v1` (Sub-second CPU/Edge neural color synthesis -> Rust `vtracer`)

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
### Option 3: Local Native Execution (Go + Rust + Python Venv + Vite)
```bash
./scripts/build.sh --local
./scripts/run.sh --local
```

### Verification & Stop
```bash
# ML worker unit tests (SVG extraction / model classification, stdlib unittest)
cd ml-worker && .venv/bin/python -m unittest discover -s tests -t .

# Run end-to-end health and pipeline tests
./scripts/test-e2e.sh

# Stop all running containers or local background processes
./scripts/stop.sh
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same checks on every push and pull request:
`go build`/`go vet`, `cargo build --release --locked`, the frontend production build, and the ML worker
unit tests.

### Degraded mode is explicit
If the Rust engine or Python ML worker is unreachable, the gateway keeps serving from its in-process Go
fallbacks but says so: `/api/health` reports `status: "degraded"`, each unavailable service carries
`fallback: true` with the connection error, `/api/models/status` marks cached models `unavailable` and
sets `source: "native-fallback"`, and `/api/models/download` returns HTTP 503 instead of pretending a
download succeeded.

## License

MIT — see [LICENSE](LICENSE).
