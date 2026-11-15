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
│                Go API Gateway & HF Hub Manager (:8080)                  │
│  • Live Hugging Face Hub Search (/api/models/search) & Curated Registry │
│  • Pipeline Orchestration (/api/generate) & Gallery Store (data/)       │
└──────────────────┬───────────────────────────────────┬──────────────────┘
                   │ 1. ML Inference (/infer)          │ 2. Vectorize / Colorize
                   ▼                                   ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────┐
│ Python PyTorch & LiteRT Worker (:8082│  │   Rust SVG Engine (:8081)     │
│ • Direct Text-to-SVG HF Transformers │  │ • Hierarchical Color vtracer  │
│ • Text-to-Image HF Diffusers         │  │ • SVG Sanitization & ViewBox  │
│ • LiteRT / PyTorch Neural CPPN Field │  │ • Multi-Color Palette Injector│
└──────────────────────────────────────┘  └───────────────────────────────┘
```

## Supported Hugging Face Models & Pipelines

1. **Direct Text-to-SVG Code Generation (`direct-svg`)**:
   - `vinoku89/svg-code-generator` (PyTorch / Transformers Causal LM)
   - `OmniSVG/OmniSVG1.1_4B` (Multimodal 4B SVG Foundation Model)
   - `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Structured Colored SVG XML)
   - `mohannad-tazi/Llama-3.1-8B-Instruct-text-to-svg`
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
# Run end-to-end health and pipeline tests
./scripts/test-e2e.sh

# Stop all running containers or local background processes
./scripts/stop.sh
```
