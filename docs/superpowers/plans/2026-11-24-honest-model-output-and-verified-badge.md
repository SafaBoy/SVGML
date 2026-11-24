# Honest Model Output and Verified SVG Badge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop presenting procedural fallbacks and semantically wrong SVG as successful model output, and badge only searched models that are verified SVG-trained and fit the GPU.

**Architecture:** The ML worker parses explicit colour requests into the instruction, flags weak SVG with `quality_warning`, and publishes `vram_total_mb` on `/health`. The Go gateway adds `warning`/`fallback_used` to `/api/generate`, derives a verified-SVG + VRAM-fit catalogue from an explicit allowlist and HF `safetensors` metadata, and caches the worker's GPU budget. The frontend renders the warning and the exact "Verified SVG · fits GPU" badge. No new endpoints.

**Tech Stack:** Python 3.14/FastAPI, Go stdlib, React/TypeScript, stdlib `unittest` and `go test`.

**Spec:** `docs/superpowers/specs/2026-11-24-honest-model-output-and-verified-badge-design.md`

## Global Constraints

- Keep all existing endpoint paths and existing response fields; additions are optional fields only.
- Badge text is EXACTLY `Verified SVG · fits GPU` and appears only when `verified_svg && fits_gpu`.
- Verification comes from the backend allowlist `Norod78/gpt-fluentui-flat-svg`, `vinoku89/svg-code-generator` — never from free-text tags or descriptions.
- Never retry a slow HF generation (~85 s CPU); one warning-bearing result is enough.
- `ml-worker/app.py` and `backend-go/embedded/ml-worker/app.py` must stay byte-identical; the embedded copy is generated and gitignored, so never `git add` it.
- Never run `docker compose up -d --build <service>`; use explicit `build` then `up -d --no-build`.
- No push or release without an explicit choice from the user.

## Review Focus

- **Worker dies mid-load and Go substitutes the flower** → the response must say a fallback was used, not silently succeed. Test: Task 2 Step 1 asserts `fallback_used=true` and a reason-bearing `warning` on the GalleryItem.
- **Prompt says "navy blue", fixed palette has no navy** → explicit colour must be in the instruction and a wrong-colour SVG must not be clean success. Test: Task 1 Step 1 asserts the instruction contains "navy blue" and takes precedence, and that a 2-path indigo result gets a `quality_warning`.
- **15 GB model offered to 4 GB GPU** → search result must be marked unfit/unsupported before download. Test: Task 3 Step 1 feeds a 16 GB `safetensors.total`, a 4096 MB budget and asserts `fits_gpu=false` plus an `unsupported_reason`.
- **General code model described as SVG-trained** → the false description must be gone and no badge shown. Test: Task 3 Step 1 asserts the Qwen catalogue entry is not verified and its description no longer claims SVG training.
- **Embedded worker drifts** → packaged binary keeps the old behaviour. Test: Task 4 Step 1 runs `cmp -s` on the two `app.py` paths and expects 0.

---

### Task 1: Worker colour instruction, quality warning and VRAM health

**Files:**
- Modify: `ml-worker/app.py` (`InferResponse` ~150, `health` ~290, `_svg_instruction` ~739, `run_hf_direct_svg` ~766, `infer` ~514-540)
- Test: `ml-worker/tests/test_output_quality.py` (new)

**Interfaces:**
- Consumes: `prompt`, `style`, `palette`, `raw_svg` from existing inference flow.
- Produces: `prompt_color_terms(prompt: str) -> List[str]`; `check_svg_quality(svg: str, prompt: str) -> str` returning `""` when acceptable; `InferResponse.quality_warning: str = ""` (serialised as `quality_warning` and mirrored on Go's `MLInferResponse`); `/health` field `vram_total_mb: int`.

- [ ] **Step 1: Write the failing tests**

In `ml-worker/tests/test_output_quality.py` (`import app`, stdlib `unittest`), assert: `prompt_color_terms("A Cruise ship ... Bottom half of the hull needs to be navy blue and top half white.")` contains `navy blue` and `white`; `_svg_instruction(prompt, "vibrant-gradient", ["#6366F1", ...])` contains both requested words and says they take precedence over the palette; `check_svg_quality` on the 2-path indigo/white cruise-ship SVG returns a non-empty warning mentioning the requested colour, while a 6-path SVG containing `navy blue` and `white` returns `""`; `app.health()["vram_total_mb"]` is an int >= 0.

- [ ] **Step 2: Run to verify RED**

Run `cd ml-worker && .venv/bin/python -m unittest tests.test_output_quality -v`. Expected: FAIL on missing names/fields.

- [ ] **Step 3: Implement minimal worker changes**

`prompt_color_terms` matches a fixed dictionary of common color words including `navy blue` and `white`, longest term first. Append a sentence to `_svg_instruction` when terms are found: explicit prompt colours take precedence over `Style`/palette. `check_svg_quality` counts SVG shape tags (`path`, `rect`, `circle`, `polygon`, `ellipse`, `line`) and looks for each requested term or its canonical hex in a case-insensitive SVG text; warn when fewer than 3 shapes or a requested colour is absent. `run_hf_direct_svg` returns the same 3-tuple; `infer` sets `quality_warning=check_svg_quality(raw_svg, req.prompt)` on the HF success branch only. Add `vram_total_mb` to `/health` from `torch.cuda.get_device_properties(0).total_memory // (1024*1024)` when `CUDA_AVAILABLE`, else 0.

- [ ] **Step 4: Run to verify GREEN**

Run `cd ml-worker && .venv/bin/python -m unittest tests.test_output_quality -v`, then `cd ml-worker && .venv/bin/python -m unittest discover -s tests -t .`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ml-worker/app.py ml-worker/tests/test_output_quality.py
git commit -m "Flag weak SVG output and publish worker VRAM budget"
```

### Task 2: Gateway fallback honesty

**Files:**
- Modify: `backend-go/gallery.go` (GalleryItem ~25), `backend-go/main.go` (`/api/generate` ~398-497)
- Test: `backend-go/generate_test.go` (new)

**Interfaces:**
- Consumes: `MLInferResponse` (~80) with `QualityWarning string \`json:"quality_warning,omitempty"\``, and fallback reason strings from worker/rust failures.
- Produces: `GalleryItem.Warning string \`json:"warning,omitempty"\``, `GalleryItem.FallbackUsed bool \`json:"fallback_used,omitempty"\``; helper `fallbackWarning(stage, reason string) string`.

- [ ] **Step 1: Write the failing tests**

In `backend-go/generate_test.go`, assert `fallbackWarning("worker", "dial tcp: refused")` contains `"ML worker"` and the reason; `fallbackWarning("rust", "unavailable")` mentions the Rust engine; and a `GalleryItem` marshalled with `Warning`/`FallbackUsed` emits the JSON keys `warning` and `fallback_used`.

- [ ] **Step 2: Run to verify RED**

Run `cd backend-go && go test ./...`. Expected: compile FAIL on missing fields/helper.

- [ ] **Step 3: Implement fallback reporting**

Add the two fields to `GalleryItem`. Add `fallbackWarning(stage, reason string) string` returning a short human-readable sentence. In `/api/generate`, whenever the worker call fails and `nativeMLInfer` is used, set both fields on the item before writing it; do the same for a Rust post-process fallback (include the worker's `quality_warning` in `Warning` when present). Keep HTTP 200 and every existing field unchanged; propagate `quality_warning` from `MLInferResponse` into `Warning` on the success path too.

- [ ] **Step 4: Run to verify GREEN**

Run `cd backend-go && go test ./... && go vet ./...`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend-go/gallery.go backend-go/main.go backend-go/generate_test.go
git commit -m "Report model fallbacks and weak output in generation results"
```

### Task 3: Verified SVG catalogue, VRAM fit and badge data

**Files:**
- Modify: `backend-go/main.go` (`CuratedModel` ~30, `curatedModels` ~143, `/api/models/curated` ~314, `searchHuggingFaceHub` ~699, `filterCurated` ~774)
- Test: `backend-go/models_test.go`

**Interfaces:**
- Consumes: `MLInferResponse` unchanged; worker `/health` `vram_total_mb` from Task 1.
- Produces: `CuratedModel.VerifiedSVG bool \`json:"verified_svg"\``, `EstimatedVRAMMB int \`json:"estimated_vram_mb"\``, `FitsGPU bool \`json:"fits_gpu"\``; helpers `verifiedSVGRepos(id string) bool`, `estimateVRAMMB(totalBytes int64) int`, `gpuVRAMBudgetMB() int`.

- [ ] **Step 1: Write the failing tests**

In `backend-go/models_test.go` add assertions: `verifiedSVGRepos` is true only for the two allowlisted ids; `estimateVRAMMB(16*1024*1024*1024)` is roughly 32768 + overhead and `estimateVRAMMB(0)` is 0; with a stub health response of `{"vram_total_mb":4096}` the 16 GB model has `FitsGPU=false` and a non-empty `UnsupportedReason`, while a 1 GB model has `FitsGPU=true`; the Qwen 1.5B curated entry has `VerifiedSVG=false`, `FitsGPU=false` on a 4096 MB budget and a description that does not contain "trained" or "fine-tuned"; two allowlisted curated entries have `VerifiedSVG=true`.

- [ ] **Step 2: Run to verify RED**

Run `cd backend-go && go test ./...`. Expected: FAIL on missing fields/helpers.

- [ ] **Step 3: Implement catalogue and gating**

Add the three fields and set them for curated models from the allowlist plus a declared `EstimatedVRAMMB` per curated entry. Add `vfy` allowlist helper. In `searchHuggingFaceHub`, request `expand[]=safetensors`, parse `safetensors.total`, compute `EstimatedVRAMMB` via fp16 (`total*2` bytes + 512 MB overhead), set `FitsGPU` against a cached `gpuVRAMBudgetMB()` (fetch worker `/health` with a 3 s timeout, cache 5 minutes, default 0), replace the `8b`/`7b` heuristic with `FitsGPU`, and set `UnsupportedReason` for unfit or unknown-size results. Keep the curated fallback path and the `unsupported_reason` test for OmniSVG/mohannad-tazi green.

- [ ] **Step 4: Run to verify GREEN**

Run `cd backend-go && go test ./... && go vet ./...`. Expected: PASS, including the existing `TestCuratedModelsMarkOnlyUnsupportedRepositories` (no new curated unsupported reasons).

- [ ] **Step 5: Commit**

```bash
git add backend-go/main.go backend-go/models_test.go
git commit -m "Badge verified SVG models that fit the GPU"
```

### Task 4: Frontend notice and badge, mirror, full verification

**Files:**
- Modify: `frontend/src/types.ts` (`CuratedModel`, `GalleryItem`), `frontend/src/App.tsx` (model list ~search result rendering, gallery detail/canvas notice), `backend-go/embedded/ml-worker/app.py` (generated copy), `README.md` (one line if suitable)

**Interfaces:**
- Consumes: `warning`, `fallback_used`, `quality_warning` via `warning`, `verified_svg`, `fits_gpu`, `estimated_vram_mb`.
- Produces: visible amber notice and exactly one badge string.

- [ ] **Step 1: Write the failing drift check**

Run `cmp -s ml-worker/app.py backend-go/embedded/ml-worker/app.py; echo $?`. Expected: `1` (embedded copy is stale after Task 1).

- [ ] **Step 2: Copy worker and verify**

Copy `ml-worker/app.py` over `backend-go/embedded/ml-worker/app.py`; rerun `cmp -s ...; echo $?`. Expected: `0`. Do not `git add` the embedded copy.

- [ ] **Step 3: Render warning and badge**

Add optional fields to `CuratedModel` and `GalleryItem` in `frontend/src/types.ts`. In `App.tsx`, set a `warning` state from `data.warning` in `handleGenerate` and render an amber notice above the canvas when non-empty; show the exact `Verified SVG · fits GPU` badge next to a model only when `m.verified_svg && m.fits_gpu`; show `m.unsupported_reason` via the existing notify path. Clear the notice when a new generation starts.

- [ ] **Step 4: Full verification**

Run: ML-worker `cd ml-worker && .venv/bin/python -m unittest discover -s tests -t .`; Go `cd backend-go && go test ./... && go vet ./...`; Rust `cd backend-rust && cargo test --locked`; frontend `cd frontend && npm run build`. Then use the existing Playwright setup (`ml-worker/.venv/bin/python`) to load http://localhost:5173, open the model hub and confirm the badge appears only on allowlisted fitting models and the amber notice appears after a forced fallback. If the worker image needs rebuilding, use explicit `docker compose build ml-worker` then `up -d --no-build ml-worker`; do not rebuild other services. Expected: all green; `/api/health` still ok.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/types.ts frontend/src/App.tsx README.md
git commit -m "Show fallback warnings and verified SVG GPU-fit badge"
```
