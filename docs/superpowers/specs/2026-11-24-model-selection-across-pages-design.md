# Model selection across Prompt, Photo-SVG, and Edit-SVG — Design

Date: 2026-11-24

## Intent

The user wants to select a downloaded Hugging Face model on each of the three creation pages
(Prompt → SVG, Photo → Line-Art, Edit & AI Enhance). Selecting a model must automatically switch
the ML + Rust pipeline mode. Manual pipeline choice should not be the default UI; it belongs in an
advanced disclosure. Crucially, the choice must be real on all three pages: the selected model
must actually affect the output, not just change a label.

## Goals

- One model picker available on Prompt, Photo-SVG, and Edit-SVG, listing downloaded, runnable models.
- Picking a model sets both `selectedModel` and `pipeline` from the model's curated `pipeline` hint,
  using the existing `handleSelectModel` behavior.
- Prompt's three large pipeline buttons move into a collapsed "Advanced" disclosure; Photo and Edit
  get the same disclosure rather than no control at all.
- `/api/image-to-svg` and `/api/svg/enhance` accept `model_id` and `pipeline`, and use the selected
  model to generate or remix the SVG. When that fails, they keep the deterministic result and say so.
- Gallery/response metadata records the model and pipeline that actually produced the SVG.

## Non-goals

- No VLM image understanding: the photo model prompt is built from the line-art mode, extracted
  colors, palette, and filename — not from "looking at" the uploaded image.
- No new model downloads or dependencies, no changes to model weights, no new curated models.
- No changes to Prompt's `/api/generate` semantics (it already honors both fields).
- No rewriting of the user's prompt text or Suggestions panel behavior.

## Anti-goals (what would make this a failure)

- A picker that changes the label but not the output.
- A procedural fallback reported as if the selected model generated the SVG.
- A manual pipeline override presented as equal to the model-driven choice.

## Constraints

- Worker GPU is a GTX 1050 Ti with 4 GB VRAM; only models already reported `ready`, without
  `unsupported_reason`, and with `fits_gpu` true are selectable.
- ML inference client timeout is 600 s; generation can be slow on CPU fallback.
- The worker's existing `run_hf_direct_svg` already validates XML and returns a skip reason;
  reuse that rather than adding a second validator.
- Editing `ml-worker/app.py` requires regenerating the gitignored embedded copy
  (`backend-go/embedded/ml-worker/app.py`) with `scripts/package-single-binary.sh` before packaging.
- Frontend state `selectedModel` and `pipeline` is already persisted in `SessionState`; no new
  session keys are needed.

## Approaches considered

- **A — Frontend only.** Shared picker plus advanced disclosure, with Photo/Edit selection labeled
  as inert. Small and safe, but fails the user's request that the model actually affect output.
- **B — Full wiring (chosen).** Shared picker plus backend/worker changes so the model generates or
  remixes the SVG on all three pages, with honest fallback. Larger, touches Go and Python, but is
  the only option that makes the control real.
- **C — Prompt only.** Keep Photo/Edit as-is. Doesn't meet the request.

For the model's role on Photo/Edit, the user chose **generate/remix the SVG**: the model's output
replaces the deterministic result only when it is valid; otherwise the existing tracer/enhancer
result is kept and labeled as such. The alternative of style-only refinement was rejected; the
vision-model-only option was rejected because none of the current downloads qualify.

## Decisions

### Frontend

- Add one reusable inline JSX builder in `App.tsx` (plain function returning JSX, no new component
  file or dependency) rendered near the top of each of the three creation branches. It lists
  `models` filtered by `modelStatuses[id].status === 'ready'`, no `unsupported_reason`, and
  `fits_gpu !== false`. Each entry calls the existing `handleSelectModel(m)`.
- If the current `selectedModel` is not in the ready list, show a short warning pointing to Settings
  rather than a broken picker.
- The model-driven pipeline is `models.find((m) => m.id === selectedModel)?.pipeline`. A manual
  override is detected by `pipeline !== modelPipeline` and labeled "manual override" — no new
  session key.
- Prompt's pipeline buttons move behind a collapsed "Advanced: ML + Rust Pipeline" disclosure.
  Photo and Edit get the same disclosure (still functional, still visibly secondary).
- Photo requests add `model_id` and `pipeline` to `/api/image-to-svg`; Edit requests already send
  `model_id`, and now also send `pipeline`.

### Backend (`backend-go/main.go`)

- Add optional `ModelID` / `Pipeline` fields to `ImageToSvgRequest` and `EnhanceSvgRequest`.
- Keep the existing deterministic stages first (line-art + vtracer for Photo; `/enhance-svg` for
  Edit). Then, if a model is requested, call the worker's `/infer` with `require_hf: true`:
  - `mode: 'svg'` → sanitize `raw_svg` through Rust `/process-svg` and use it.
  - `mode: 'image'` → send `image_base64` to Rust `/vectorize`, then `/process-svg`.
  - 422 (model could not run) or any transport error → keep the deterministic SVG unchanged.
- Prompt construction:
  - Photo: line-art mode label, filename, the traced SVG's extracted colors, and the chosen palette.
    Explicitly not an image description.
  - Edit: enhancement mode label, the user's instruction, the palette, and up to 1200 characters of
    the input SVG as style context.
- Response and gallery metadata: record the model/pipeline actually used and a `model_warning`
  string when fallback occurred. Never report the requested model in `ModelID` if it did not run.
  Use the existing `GalleryItem` fields (`ModelID`, `Pipeline`, `Warning`, `FallbackUsed`) — no new
  gallery fields.
- `/api/svg/enhance` response gains `model_used`, `pipeline`, and `model_warning` fields; the
  frontend uses them for the run-status label. Builder call: the three pages' existing footer
  "Running …" label is the display point; no new panel is introduced.
- Builder call on `mode: 'image'`: vectorize the **model's** `image_base64` (not the deterministic
  line-art image), then sanitize through `/process-svg`. The deterministic line-art/tracer result
  remains the fallback when the model cannot run.
- Builder call on Edit: the deterministic `/enhance-svg` stage still runs and is the fallback;
  `/infer` generates a replacement SVG using the **original input SVG** as style context (not the
  enhanced one). If `/infer` returns `mode: 'image'`, apply the same vectorize-then-process path as
  Photo. The enhanced SVG is never fed back into the model in this version.

### ML worker (`ml-worker/app.py`)

- Add `require_hf: bool = False` to `InferRequest`. Default preserves current Prompt behavior.
- When true, every procedural fallback branch (`neural-vtracer`, `diffusion-vtracer`, and direct
  text-to-SVG) returns HTTP 422 with a reason instead of generating procedural output. This includes
  the case where `should_run_hf` is false because the model is not cached: a requested model must
  either actually run or fail loudly. HF failures are logged with the existing `short_reason`
  helper.
- No changes to `/image-to-lineart` or `/enhance-svg` themselves; reuse `/infer` and its XML
  validation, device management, and GPU→CPU retry.

## What's left to the builder

- Tests: Go tests for request plumbing, prompt building, and fallback labeling; Python tests for
  `require_hf` behavior without downloading models; frontend tests for picker filtering,
  auto-switch, and the advanced disclosure.
- Run `go test ./...`, `cargo test --locked`, the ml-worker unit suite, `npm run build`, and the
  frontend node tests.
- Regenerate the embedded worker copy before any release.
