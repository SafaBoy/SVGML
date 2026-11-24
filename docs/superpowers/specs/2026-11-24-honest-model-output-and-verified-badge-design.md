# Honest model output and a verified-SVG badge

Date: 2026-11-24

## Intent

A cruise-ship prompt with explicit navy/white colours produced either a generic
flower or a five-shape purple/white box. Neither result was honest: the flower
came from the Go procedural fallback while the gallery still named the selected
model, and the box came from a general-purpose code model that the catalogue
described as SVG-trained. The user wants bad outputs to be visible and wants a
badge on searched models that are *actually* SVG-trained and fit the GPU.

## Goals / non-goals

- Goals: report fallbacks and weak outputs instead of silently presenting them
  as model success; stop advertising unsuitable models as SVG-capable or
  CPU-friendly; badge verified SVG-trained models that fit; make explicit
  prompt colours win over the fixed style palette.
- Non-goals: teaching small general LLMs to draw; changing the renderer;
  new endpoints; removing live Hugging Face search; changing which weights are
  downloaded.

## Root causes (confirmed)

1. `backend-go/main.go` `/api/generate` replaces any failed worker call with
   `nativeMLInfer` and still writes a normal 200 GalleryItem — the flower.
2. `Qwen/Qwen2.5-Coder-1.5B-Instruct` is a general code model but the curated
   list calls it SVG-trained; `_svg_instruction` forces a fixed vibrant palette,
   so "navy blue" becomes indigo; `run_hf_direct_svg` accepts any XML-valid SVG.
3. Live search labels models CPU-friendly unless the name contains "8b"/"7b",
   so a 15 GB gemma was offered to a 4 GB GPU; it killed the worker mid-load.

## Approaches

1. **Honest pipeline + fit gating (chosen).** Surface fallbacks and weak output
   to the user, fix the catalogue and prompt, add a verification/fit badge.
2. **Output quality gate alone.** Validate colours and complexity and retry or
   relabel; brittle, and leaves the silent fallback and bogus catalogue in place.
3. **Curated-only models.** Drop live search; simplest but hides failures and
   removes useful discovery.

The user chose approach 1 plus the output checks from approach 2, and asked for
a verified-SVG/GPU-fit badge on searched models.

## Design

- **Gateway honesty.** Add optional `warning: string` and `fallback_used: bool`
  to the `/api/generate` response and GalleryItem. Set them when the worker or
  Rust call falls back, with a specific reason (e.g. "ML worker unavailable;
  used built-in procedural generator"). Keep status 200 and the existing
  fields. Frontend shows an amber notice beside the generated SVG and stores
  the fields in the gallery; no silent substitution.
- **Model fitness and badge.** Add `verified_svg: bool`, `estimated_vram_mb: int`
  and `fits_gpu: bool` to `CuratedModel`. Verification is an explicit backend
  allowlist of genuinely SVG-trained models (`Norod78/gpt-fluentui-flat-svg`,
  `vinoku89/svg-code-generator`), not free-text tags. Live search already
  supports `expand[]=safetensors`; estimate fp16 weights as `total * 2` bytes
  plus a small overhead, and compare against a GPU budget reported by the
  worker in `/api/health` (`vram_total_mb`) and cached by the gateway. Badge
  text: "Verified SVG · fits GPU"; show it only when both flags are true.
  Replace the "8b"/"7b" CPU-friendly heuristic with the fit result, fix the
  false Qwen SVG description, and mark oversized/unknown models with an
  `unsupported_reason` so the studio blocks them before download.
- **Prompt and quality check.** In `ml-worker/app.py`, append explicit colour
  words parsed from the prompt to `_svg_instruction` and state that they take
  precedence over the style palette. After XML validation, run a lightweight
  content check for those requested colour words and a minimum shape count;
  when it fails, return the SVG with `quality_warning` instead of claiming
  clean success. No retry — a CPU retry costs ~85 s. Keep the embedded
  `backend-go/embedded/ml-worker/app.py` byte-identical.

## Verification

- Go tests for the fallback warning fields, search expand/size parsing, badge
  flags and fit gating; Python tests for the colour-aware instruction and the
  quality warning; frontend build and Playwright check of the badge and warning.
- Reproduce the cruise-ship prompt against the chosen Qwen model and confirm
  the response now either honours navy/white or carries a visible warning.
- Full suites: ML-worker, Go, Rust, frontend. No push or release without an
  explicit choice.
